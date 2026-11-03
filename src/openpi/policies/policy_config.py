import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from openpi.models import rtc as _rtc
import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.policies import droid_policy
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    rtc: bool = True,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        rtc: zima/rtc: accept an action prefix (real-time chunking) where supported: a JAX pi0.5 policy whose
            actions pass through the data transforms unchanged (DROID). Inpainting for a model trained with
            simulated_delay, guidance (arXiv 2506.07339) otherwise. A request without a prefix is unchanged.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    sample_kwargs = dict(sample_kwargs or {})
    _check_sample_kwargs(train_config, sample_kwargs)  # validates and normalizes to Python numbers
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    sample_kwargs = dict(sample_kwargs or {})
    prefix_transform, rtc_info = None, None
    if rtc and not is_pytorch and _actions_pass_through(train_config, data_config, repack_transforms):
        prefix_transform, rtc_info = _action_prefix_support(train_config, data_config, norm_stats, sample_kwargs)
    # Settings only Pi0.sample_actions takes; the guidance clip only matters in guidance mode.
    if rtc_info is None or rtc_info["mode"] != "guidance":
        sample_kwargs.pop("max_guidance_weight", None)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        action_prefix_transform=prefix_transform,
        rtc=rtc_info,
    )


def _check_sample_kwargs(train_config, sample_kwargs) -> None:
    """zima/rtc: refuse sampler settings the model cannot take or that break sampling, before any weights load."""
    if "num_steps" in sample_kwargs:
        steps = sample_kwargs["num_steps"]
        if not isinstance(train_config.model, _pi0_config.Pi0Config):
            raise ValueError(f"num_steps applies to flow-matching pi0/pi0.5 models only, not {train_config.name}")
        if isinstance(steps, bool | np.bool_) or not isinstance(steps, int | np.integer) or steps < 1:
            raise ValueError(f"num_steps must be a positive integer, got {steps!r}")
        sample_kwargs["num_steps"] = int(steps)
    if "max_guidance_weight" in sample_kwargs:
        weight = sample_kwargs["max_guidance_weight"]
        real = isinstance(weight, int | float | np.integer | np.floating) and not isinstance(weight, bool | np.bool_)
        if not real or not 0 <= float(weight) < float("inf"):
            raise ValueError(f"max_guidance_weight must be a finite number >= 0 (0 turns guidance off), got {weight!r}")
        sample_kwargs["max_guidance_weight"] = float(weight)


def _actions_pass_through(train_config, data_config, repack_transforms) -> bool:
    """True when a policy's input pipeline hands `actions` to Normalize unchanged and only pads them after: a
    pi0.5 DROID policy (DroidInputs, no delta actions), whose action prefix then needs the action transforms only."""
    if not isinstance(train_config.model, _pi0_config.Pi0Config) or not train_config.model.pi05:
        return False
    if repack_transforms.inputs:
        return False
    data_inputs = list(data_config.data_transforms.inputs)
    if len(data_inputs) != 1 or not isinstance(data_inputs[0], droid_policy.DroidInputs):
        return False
    if [type(t) for t in data_config.data_transforms.outputs] != [droid_policy.DroidOutputs]:
        return False
    model_inputs = [type(t) for t in data_config.model_transforms.inputs]
    allowed = {transforms.InjectDefaultPrompt, transforms.ResizeImages, transforms.TokenizePrompt}
    return model_inputs.count(transforms.PadStatesAndActions) == 1 and set(model_inputs) <= {
        *allowed,
        transforms.PadStatesAndActions,
    }


def _action_prefix_support(train_config, data_config, norm_stats, sample_kwargs):
    model_config = train_config.model
    normalize = transforms.Normalize({"actions": norm_stats["actions"]}, use_quantiles=data_config.use_quantile_norm)
    # The raw action width is what DroidOutputs returns (joint velocities + gripper position); the stats may
    # be stored padded to the model's 32 dims (pi05_droid's are), and Normalize uses their first `width`.
    width = droid_policy.DROID_ACTION_DIM

    def prefix_transform(actions):
        out = normalize({"actions": actions})["actions"]
        return transforms.pad_to_dim(out, model_config.action_dim, axis=-1)

    inpaint = model_config.simulated_delay is not None
    max_weight = float(sample_kwargs.get("max_guidance_weight", _rtc.DEFAULT_MAX_GUIDANCE_WEIGHT))
    info = {
        "protocol": 1,
        "mode": "inpaint" if inpaint else "guidance",
        "simulated_delay": model_config.simulated_delay,
        "action_horizon": model_config.action_horizon,
        "action_dim": width,
        "num_steps": int(sample_kwargs.get("num_steps", 10)),
        "guidance": None if inpaint else {"max_weight": max_weight, "schedule": "exp"},
    }
    return prefix_transform, info
