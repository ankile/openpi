"""Real-time chunking (RTC) for pi0.5: the action prefix and inference-time guidance.

Two ways to make a new chunk continue from the actions already committed to the robot:

- Training-time RTC (Black, Ren, Equi, Levine, arXiv 2512.05964): the model is trained with the first
  `delay` actions given clean (flow time 0) and masked from the loss, so at inference the prefix is
  inpainted exactly. Implemented in `pi0.Pi0` behind `Pi0Config.simulated_delay`.
- Inference-time RTC (Black, Galliker, Levine, arXiv 2506.07339): for a model trained without that, the
  velocity is guided toward the prefix with the pseudo-inverse (PiGDM) correction of Eqs. 2-5. That is
  this module's `guided_velocity`.

openpi's time convention is the opposite of the papers': t = 1 is noise and t = 0 is data, so the
papers' tau = 1 - t and v_openpi = -v_paper. Every formula below is the paper's, rewritten in t.
"""

from collections.abc import Callable

from flax import struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

# 2506.07339 Sec. 4.2: "we clip the guidance weight to a maximum value beta = 5"
DEFAULT_MAX_GUIDANCE_WEIGHT = 5.0


@struct.dataclass
class ActionPrefix:
    """The actions already committed to the robot, at the start of the chunk being generated.

    `actions` are in model space (normalized and padded like training actions), right-padded to the
    action horizon. Index i < delay will execute before the new chunk can arrive; index i < prefix_length
    is the previous chunk's not-yet-executed remainder. Everything at i >= prefix_length is ignored.
    `delay` and `prefix_length` are arrays so one compiled graph serves every value.
    """

    actions: at.Float[at.Array, "b ah ad"]
    delay: at.Int[at.Array, " b"]
    prefix_length: at.Int[at.Array, " b"]


def delay_mask(prefix: ActionPrefix, action_horizon: int) -> at.Bool[at.Array, "b ah"]:
    """True at the indices that are inpainted (i < delay)."""
    return jnp.arange(action_horizon)[None, :] < prefix.delay[:, None]


def prefix_weights(prefix: ActionPrefix, action_horizon: int) -> at.Float[at.Array, "b ah"]:
    """2506.07339 Eq. 5, the soft mask W, with H - s = prefix_length and d = delay.

    W_i = 1                        for i < d
    W_i = c_i (e^c_i - 1) / (e - 1) for d <= i < H - s,   c_i = (H - s - i) / (H - s - d + 1)
    W_i = 0                        for i >= H - s
    """
    i = jnp.arange(action_horizon)[None, :].astype(jnp.float32)
    d = prefix.delay[:, None].astype(jnp.float32)
    remaining = prefix.prefix_length[:, None].astype(jnp.float32)
    c = (remaining - i) / (remaining - d + 1.0)
    soft = c * jnp.expm1(c) / (jnp.e - 1.0)
    return jnp.where(i < d, 1.0, jnp.where(i < remaining, soft, 0.0))


def guidance_weight(t: at.Float[at.Array, ""], max_weight: float) -> at.Float[at.Array, ""]:
    """2506.07339 Eq. 2's weight min(beta, (1 - tau) / (tau r_tau^2)), r_tau^2 = (1 - tau)^2 / (tau^2 + (1 - tau)^2).

    In openpi time (tau = 1 - t) that is ((1 - t)^2 + t^2) / ((1 - t) t). It is infinite at t = 1 (pure
    noise, the first step), where the clip gives exactly beta.
    """
    t = jnp.asarray(t, jnp.float32)
    denominator = (1.0 - t) * t
    safe = jnp.where(denominator > 0, denominator, 1.0)
    weight = ((1.0 - t) ** 2 + t**2) / safe
    return jnp.where(denominator > 0, jnp.minimum(max_weight, weight), max_weight)


def guided_velocity(
    velocity_fn: Callable[[jax.Array], jax.Array],
    x_t: at.Float[at.Array, "b ah ad"],
    t: at.Float[at.Array, ""],
    prefix: ActionPrefix,
    *,
    max_weight: float,
) -> at.Float[at.Array, "b ah ad"]:
    """2506.07339 Eqs. 2-4 in openpi time: the velocity at x_t, guided toward the prefix.

    x0_hat = x_t - t v(x_t)                       (Eq. 3: the one-step estimate of the clean chunk)
    e      = (Y - x0_hat) * W                     (Eq. 2's weighted error, W from Eq. 5)
    v_g    = v - w_t * e^T d(x0_hat)/d(x_t)       (Eq. 2; the sign flips with v_openpi = -v_paper)

    The vector-Jacobian product goes through the model's forward pass with no stop-gradient, as in the
    paper.
    """

    def denoise(x):
        v = velocity_fn(x)
        return x - t * v, v

    x0_hat, vjp_fn, v_t = jax.vjp(denoise, x_t, has_aux=True)
    weights = prefix_weights(prefix, x_t.shape[-2])[..., None]
    error = (prefix.actions - x0_hat) * weights
    (correction,) = vjp_fn(error)
    return v_t - guidance_weight(t, max_weight) * correction
