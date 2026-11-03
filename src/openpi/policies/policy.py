from collections.abc import Callable, Sequence
import logging
import math
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        action_prefix_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        rtc: dict[str, Any] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            action_prefix_transform: zima/rtc: maps a raw action prefix (H, raw action dim) to model space with
                the action transforms only (normalize, pad). With it (and `rtc`), a request may carry
                `action_prefix` = {"actions", "delay", "prefix_length"} (real-time chunking).
            rtc: zima/rtc: what the server does with an action prefix; announced in the metadata as `rtc`.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        if (action_prefix_transform is None) != (rtc is None):
            raise ValueError("action_prefix_transform and rtc go together")
        if rtc is not None and is_pytorch:
            raise ValueError("real-time chunking is implemented for JAX models only")
        self._action_prefix_transform = action_prefix_transform
        self._rtc = rtc
        if rtc is not None:
            self._metadata = {**self._metadata, "rtc": rtc}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        raw_prefix = inputs.pop("action_prefix", None)
        action_prefix = None if raw_prefix is None else self._action_prefix(raw_prefix)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        if action_prefix is not None:
            sample_kwargs["action_prefix"] = action_prefix

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def rtc(self) -> dict[str, Any] | None:
        return self._rtc

    def _action_prefix(self, raw: Any) -> _rtc.ActionPrefix:
        """Validate a request's action prefix and bring it to model space (batch of one).

        The actions stay as the robot recorded them (raw action space, the recorded action labels); only the
        action transforms run on them, so their model-space values are what training saw for the same labels.
        """
        if self._rtc is None:
            raise ValueError("this server takes no action_prefix (real-time chunking is not enabled for it)")
        if not isinstance(raw, dict) or set(raw) != {"actions", "delay", "prefix_length"}:
            raise ValueError("action_prefix must be {'actions', 'delay', 'prefix_length'}")
        horizon, width = self._rtc["action_horizon"], self._rtc["action_dim"]
        actions = np.asarray(raw["actions"])
        real = np.issubdtype(actions.dtype, np.floating) or np.issubdtype(actions.dtype, np.integer)
        if actions.shape != (horizon, width) or not real or actions.dtype == np.bool_:
            raise ValueError(
                f"action_prefix actions must be real numbers of shape ({horizon}, {width}), got {actions.dtype} "
                f"{actions.shape}"
            )
        actions = actions.astype(np.float64)
        delay, prefix_length = (_integer(raw[k], k) for k in ("delay", "prefix_length"))
        if not 0 <= delay <= prefix_length <= horizon:
            raise ValueError(
                f"action_prefix needs 0 <= delay <= prefix_length <= {horizon}, got delay {delay}, "
                f"prefix_length {prefix_length}"
            )
        if self._rtc["mode"] == "inpaint" and delay > self._rtc["simulated_delay"] - 1:
            raise ValueError(
                f"delay {delay} is outside the trained range 0..{self._rtc['simulated_delay'] - 1} "
                "(simulated_delay); the model never saw a longer prefix"
            )
        # Only the first prefix_length rows mean anything; the rest is padding, set to zero before it can
        # overflow (a finite 1e300 becomes inf in float32, and 0 * inf is NaN in the guidance error).
        actions[prefix_length:] = 0.0
        if not np.isfinite(actions).all():
            raise ValueError("action_prefix actions must be finite")
        model_actions = np.asarray(self._action_prefix_transform(actions), np.float32)
        if not np.isfinite(model_actions[:prefix_length]).all():
            raise ValueError("action_prefix actions are not finite after normalization (out of float32 range)")
        return _rtc.ActionPrefix(
            actions=jnp.asarray(model_actions)[None],
            delay=jnp.asarray([delay], jnp.int32),
            prefix_length=jnp.asarray([prefix_length], jnp.int32),
        )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        if isinstance(value, float | np.floating) and math.isfinite(value) and float(value).is_integer():
            return int(value)
        raise ValueError(f"action_prefix {name} must be an integer, got {value!r}")
    return int(value)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
