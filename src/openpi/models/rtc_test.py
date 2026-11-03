# ruff: noqa: SLF001  # tests reach into the model and policy internals they test
"""zima/rtc: real-time chunking in pi0.5 (training-time RTC, 2512.05964; inference-time guidance, 2506.07339).

A tiny pi0.5 (dummy Gemma experts, a one-layer SigLIP) so the tests run on CPU. Parameters are perturbed
away from their init so the adaRMS modulation (zero-initialized) is exercised.
"""

import dataclasses
import math
import subprocess
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import rtc
import openpi.models.siglip as _siglip_module
from openpi.shared import nnx_utils

UPSTREAM = "15a9616"  # upstream openpi commit the fork's model code is cut from
H = 8
BATCH = 2


def _tiny_siglip():
    def module(**kwargs):
        return _siglip_module.Module(**{**kwargs, "variant": "mu/14"})

    return types.SimpleNamespace(Module=module)


def _config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=H,
        max_token_len=12,
        dtype="float32",
        **kwargs,
    )


def _perturb(model, key):
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(key, len(leaves))
    leaves = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys, strict=True)]
    nnx.update(model, jax.tree.unflatten(treedef, leaves))


@pytest.fixture(scope="module")
def models():
    """(plain, rtc) pi0.5 with identical, perturbed parameters; rtc has simulated_delay=3."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pi0, "_siglip", _tiny_siglip())
        plain = _config().create(jax.random.key(0))
        rtc_model = _config(simulated_delay=3).create(jax.random.key(0))
    _perturb(plain, jax.random.key(1))
    nnx.update(rtc_model, nnx.state(plain))
    return plain, rtc_model


def _obs_act(config, key=2):
    k1, k2, k3 = jax.random.split(jax.random.key(key), 3)
    obs = config.fake_obs(BATCH)
    obs = dataclasses.replace(
        obs,
        images={n: jax.random.uniform(k1, im.shape, minval=-1, maxval=1) for n, im in obs.images.items()},
        state=jax.random.normal(k2, obs.state.shape),
    )
    act = jax.random.normal(k3, config.fake_act(BATCH).shape)
    return obs, act


def _prefix(actions, delay, prefix_length):
    return rtc.ActionPrefix(
        actions=jnp.asarray(actions, jnp.float32),
        delay=jnp.asarray(delay, jnp.int32),
        prefix_length=jnp.asarray(prefix_length, jnp.int32),
    )


# --- configuration -------------------------------------------------------------------------------------------


def test_simulated_delay_validation():
    assert _config(simulated_delay=H - 1).simulated_delay == H - 1
    for bad in (0, H, H + 3, True, 1.5):
        with pytest.raises(ValueError, match="simulated_delay"):
            _config(simulated_delay=bad)
    with pytest.raises(ValueError, match="pi0.5"):
        pi0_config.Pi0Config(pi05=False, simulated_delay=3)


# --- off: upstream's code, exactly ---------------------------------------------------------------------------


def _upstream_modules():
    def show(path):
        try:
            return subprocess.run(
                ["git", "show", f"{UPSTREAM}:{path}"], check=True, capture_output=True, text=True
            ).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            pytest.skip(f"upstream source not available: {e}")

    gemma = types.ModuleType("upstream_gemma")
    exec(compile(show("src/openpi/models/gemma.py"), "upstream_gemma.py", "exec"), gemma.__dict__)
    src = show("src/openpi/models/pi0.py")
    line = "import openpi.models.gemma as _gemma\n"
    assert line in src
    pi0 = types.ModuleType("upstream_pi0")
    pi0.__dict__["_gemma"] = gemma
    exec(compile(src.replace(line, ""), "upstream_pi0.py", "exec"), pi0.__dict__)
    return pi0


def test_off_is_upstream_exactly(models):
    plain, _ = models
    upstream = _upstream_modules()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(upstream, "_siglip", _tiny_siglip())
        reference = upstream.Pi0(_config(), rngs=nnx.Rngs(jax.random.key(0)))
    nnx.update(reference, nnx.state(plain))
    obs, act = _obs_act(_config())
    key = jax.random.key(7)
    np.testing.assert_array_equal(
        nnx_utils.module_jit(plain.compute_loss)(key, obs, act),
        nnx_utils.module_jit(reference.compute_loss)(key, obs, act),
    )
    np.testing.assert_array_equal(
        nnx_utils.module_jit(plain.sample_actions)(key, obs, num_steps=4),
        nnx_utils.module_jit(reference.sample_actions)(key, obs, num_steps=4),
    )


# --- training-time RTC -----------------------------------------------------------------------------------------


def test_rtc_loss_is_the_papers_batch_token_mean(models):
    """2512.05964 Alg. 1, re-derived here from the model's parts: prefix tokens clean and unsupervised, the loss
    the mean over the batch's supervised tokens."""
    _, model = models
    config = _config(simulated_delay=3)
    obs, act = _obs_act(config)
    key = jax.random.key(11)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(key, 4)
    delay = jax.random.randint(delay_rng, (BATCH,), 0, 3)
    noise = jax.random.normal(noise_rng, act.shape)
    t = jax.random.beta(time_rng, 1.5, 1, (BATCH,)) * 0.999 + 0.001
    supervised = np.arange(H)[None, :] >= np.asarray(delay)[:, None]
    token_t = jnp.where(supervised, t[:, None], 0.0)
    x_t = token_t[..., None] * noise + (1 - token_t[..., None]) * act
    obs_p = _pi0._model.preprocess_observation(preprocess_rng, obs, train=False)

    def per_token(module):
        prefix_tokens, prefix_mask, prefix_ar = module.embed_prefix(obs_p)
        suffix_tokens, suffix_mask, suffix_ar, cond = module.embed_suffix(obs_p, x_t, token_t)
        mask = _pi0.make_attn_mask(
            jnp.concatenate([prefix_mask, suffix_mask], 1), jnp.concatenate([prefix_ar, suffix_ar])
        )
        positions = jnp.cumsum(jnp.concatenate([prefix_mask, suffix_mask], 1), axis=1) - 1
        (_, out), _ = module.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=mask, positions=positions, adarms_cond=[None, cond]
        )
        v = module.action_out_proj(out[:, -H:])
        return jnp.mean(jnp.square(v - (noise - act)), axis=-1)

    raw = np.asarray(per_token(model))
    assert supervised.sum() < supervised.size  # this key draws at least one prefix token
    np.testing.assert_array_equal(np.asarray(loss)[~supervised], 0.0)
    expected = (raw * supervised).sum() / supervised.sum()
    np.testing.assert_allclose(np.asarray(loss).mean(), expected, rtol=1e-5)


def test_per_token_time_matches_per_sample_time(models):
    """All tokens at the same flow time through the per-token path (b, ah, emb) equals upstream's (b, emb)."""
    plain, _ = models
    obs, act = _obs_act(_config())
    obs = _pi0._model.preprocess_observation(None, obs, train=False)
    t = jnp.array([0.3, 0.8])
    _, _, _, per_sample = plain.embed_suffix(obs, act, t)
    _, _, _, per_token = plain.embed_suffix(obs, act, jnp.broadcast_to(t[:, None], (BATCH, H)))
    assert per_sample.shape == (BATCH, per_token.shape[-1])
    assert per_token.shape == (BATCH, H, per_sample.shape[-1])
    np.testing.assert_allclose(per_token, jnp.broadcast_to(per_sample[:, None], per_token.shape), atol=1e-6)


def test_inpainting(models):
    plain, model = models
    obs, _ = _obs_act(_config())
    key = jax.random.key(3)
    noise = jax.random.normal(jax.random.key(4), (BATCH, H, 32))
    sample = nnx_utils.module_jit(model.sample_actions)
    prefix_actions = jax.random.normal(jax.random.key(5), (BATCH, H, 32))
    delay, length = [2, 1], [5, 6]
    out = sample(key, obs, num_steps=4, noise=noise, action_prefix=_prefix(prefix_actions, delay, length))
    for b in range(BATCH):
        np.testing.assert_array_equal(out[b, : delay[b]], prefix_actions[b, : delay[b]])

    # padding past prefix_length cannot matter (nor, for inpainting, anything past the delay)
    padded = prefix_actions.at[:, 6:].set(1e6)
    again = sample(key, obs, num_steps=4, noise=noise, action_prefix=_prefix(padded, delay, length))
    np.testing.assert_array_equal(out, again)

    # delay = prefix_length = 0 is the first chunk: the same as no prefix (per-token path vs upstream's path)
    first = sample(key, obs, num_steps=4, noise=noise, action_prefix=_prefix(prefix_actions, [0, 0], [0, 0]))
    none = nnx_utils.module_jit(plain.sample_actions)(key, obs, num_steps=4, noise=noise)
    np.testing.assert_allclose(first, none, atol=1e-5)


def test_one_compile_serves_every_delay(models):
    _, model = models
    obs, _ = _obs_act(_config())
    graphdef, state = nnx.split(model)
    traces = []

    def fun(state, prefix):
        traces.append(1)
        return nnx.merge(graphdef, state).sample_actions(jax.random.key(0), obs, num_steps=3, action_prefix=prefix)

    jitted = jax.jit(fun)
    actions = jnp.zeros((BATCH, H, 32))
    jitted(state, _prefix(actions, [1, 1], [8, 8]))
    jitted(state, _prefix(actions, [2, 0], [5, 0]))
    assert len(traces) == 1


def test_prefix_refused_for_pi0():
    model = types.SimpleNamespace(pi05=False)
    with pytest.raises(ValueError, match="pi0.5"):
        _pi0.Pi0._sample_actions_rtc(
            model,
            None,
            None,
            num_steps=1,
            noise=None,
            action_prefix=None,
            max_guidance_weight=5.0,
        )


def test_bf16_sampling_and_training_are_finite(models):
    """The production dtype: bf16 activations through inpainting, guidance and the RTC loss."""
    plain, _ = models
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pi0, "_siglip", _tiny_siglip())
        bf16_rtc = (
            _config(simulated_delay=3)
            .__class__(**{**dataclasses.asdict(_config(simulated_delay=3)), "dtype": "bfloat16"})
            .create(jax.random.key(0))
        )
        bf16_plain = (
            _config().__class__(**{**dataclasses.asdict(_config()), "dtype": "bfloat16"}).create(jax.random.key(0))
        )
    nnx.update(bf16_rtc, nnx.state(plain))
    nnx.update(bf16_plain, nnx.state(plain))
    obs, act = _obs_act(_config())
    prefix = _prefix(jax.random.normal(jax.random.key(5), (BATCH, H, 32)), [2, 3], [5, 6])
    for model in (bf16_rtc, bf16_plain):
        out = nnx_utils.module_jit(model.sample_actions)(jax.random.key(1), obs, num_steps=3, action_prefix=prefix)
        assert np.isfinite(np.asarray(out, np.float32)).all()
    loss = nnx_utils.module_jit(bf16_rtc.compute_loss)(jax.random.key(2), obs, act)
    assert np.isfinite(np.asarray(loss, np.float32)).all()


# --- inference-time guidance (2506.07339) ------------------------------------------------------------------------


@pytest.mark.parametrize(("horizon", "s", "d"), [(16, 8, 3), (16, 8, 0), (15, 6, 5), (16, 9, 7), (10, 4, 6)])
def test_prefix_weights_are_eq5(horizon, s, d):
    remaining = horizon - s
    expected = []
    for i in range(horizon):
        if i < d:
            expected.append(1.0)
        elif i < remaining:
            c = (remaining - i) / (remaining - d + 1)
            expected.append(c * (math.exp(c) - 1) / (math.e - 1))
        else:
            expected.append(0.0)
    got = rtc.prefix_weights(_prefix(np.zeros((1, horizon, 2)), [d], [remaining]), horizon)
    np.testing.assert_allclose(got[0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("beta", [2.0, 5.0, 10.0])
def test_guidance_weight_is_eq2(beta):
    for t in [0.1, 0.25, 0.5, 0.7, 0.9, 0.95]:
        tau = 1 - t
        r2 = (1 - tau) ** 2 / (tau**2 + (1 - tau) ** 2)
        expected = min(beta, (1 - tau) / (tau * r2))
        np.testing.assert_allclose(rtc.guidance_weight(jnp.float32(t), beta), expected, rtol=1e-5)
    # t = 1 (tau = 0, pure noise): the expression is infinite, the clip gives beta
    assert float(rtc.guidance_weight(jnp.float32(1.0), beta)) == beta


def test_guided_step_on_a_linear_field_is_the_closed_form():
    """v(x) = a x + c: x0_hat = (1 - t a) x - t c, its Jacobian (1 - t a) I, so
    v_g = v - w_t (1 - t a) W (Y - x0_hat)."""
    a, c, t, beta = 0.7, 0.3, 0.6, 5.0
    x = jax.random.normal(jax.random.key(0), (1, 6, 3))
    y = jax.random.normal(jax.random.key(1), (1, 6, 3))
    prefix = _prefix(y, [2], [4])
    got = rtc.guided_velocity(lambda z: a * z + c, x, jnp.float32(t), prefix, max_weight=beta)
    v = a * x + c
    x0 = x - t * v
    w = rtc.prefix_weights(prefix, 6)[..., None]
    expected = v - rtc.guidance_weight(jnp.float32(t), beta) * (1 - t * a) * w * (y - x0)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    # finite at the first step
    assert np.isfinite(rtc.guided_velocity(lambda z: a * z + c, x, jnp.float32(1.0), prefix, max_weight=beta)).all()


def test_guidance_on_pi05(models):
    plain, _ = models
    obs, _ = _obs_act(_config())
    key = jax.random.key(3)
    noise = jax.random.normal(jax.random.key(4), (BATCH, H, 32))
    sample = nnx_utils.module_jit(plain.sample_actions)
    unguided = sample(key, obs, num_steps=4, noise=noise)
    prefix_actions = jax.random.normal(jax.random.key(5), (BATCH, H, 32))

    # delay = prefix_length = 0: zero weights, the unguided chunk
    first = sample(key, obs, num_steps=4, noise=noise, action_prefix=_prefix(prefix_actions, [0, 0], [0, 0]))
    np.testing.assert_allclose(first, unguided, atol=1e-5)

    prefix = _prefix(prefix_actions, [3, 2], [6, 5])
    guided = sample(key, obs, num_steps=4, noise=noise, action_prefix=prefix)
    assert np.isfinite(guided).all()
    w = np.asarray(rtc.prefix_weights(prefix, H))[..., None]
    err = lambda out: float((w * (np.asarray(out) - np.asarray(prefix_actions)) ** 2).sum())  # noqa: E731
    assert err(guided) < 0.5 * err(unguided)

    # padding past prefix_length cannot matter
    padded = _prefix(prefix_actions.at[:, 6:].set(1e6), [3, 2], [6, 5])
    np.testing.assert_array_equal(guided, sample(key, obs, num_steps=4, noise=noise, action_prefix=padded))

    # beta is a setting, not a constant
    weaker = sample(key, obs, num_steps=4, noise=noise, action_prefix=prefix, max_guidance_weight=1.0)
    assert err(guided) < err(weaker)
