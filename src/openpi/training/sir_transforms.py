from collections.abc import Sequence
import dataclasses

import numpy as np

from openpi import transforms


def _pick_first_available(data: dict, keys: Sequence[str], label: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing {label}. Tried keys: {list(keys)}")


def _to_prompt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclasses.dataclass(frozen=True)
class SIRDroidRepackTransform(transforms.DataTransformFn):
    """Map SIR real-robot LeRobot samples into OpenPI Droid policy input keys."""

    exterior_image_keys: Sequence[str] = (
        "observation.images.exterior_image_1_left",
        "observation.images.25916956_left",
    )
    wrist_image_keys: Sequence[str] = (
        "observation.images.wrist_image_left",
        "observation.images.18650758_left",
    )
    joint_position_key: str = "observation.state.joint_position"
    gripper_position_key: str = "observation.state.gripper_position"
    action_joint_velocity_key: str = "action.joint_velocity"
    action_gripper_position_key: str = "action.gripper_position"
    # Only honor an explicit "prompt" — do NOT fall back to the LeRobot "task"
    # slug, so the config's default_prompt (a real instruction) is used instead.
    prompt_keys: Sequence[str] = ("prompt",)

    def __call__(self, data: dict) -> dict:
        out = {
            "observation/exterior_image_1_left": _pick_first_available(
                data, self.exterior_image_keys, "exterior image"
            ),
            "observation/wrist_image_left": _pick_first_available(data, self.wrist_image_keys, "wrist image"),
            "observation/joint_position": np.asarray(data[self.joint_position_key], dtype=np.float32),
            "observation/gripper_position": np.asarray(data[self.gripper_position_key], dtype=np.float32),
        }

        prompt_value = None
        for prompt_key in self.prompt_keys:
            if prompt_key in data:
                prompt_value = data[prompt_key]
                break
        if prompt_value is not None:
            out["prompt"] = _to_prompt(prompt_value)

        if self.action_joint_velocity_key in data and self.action_gripper_position_key in data:
            joint_vel = np.asarray(data[self.action_joint_velocity_key], dtype=np.float32)
            gripper = np.asarray(data[self.action_gripper_position_key], dtype=np.float32)
            if gripper.ndim == joint_vel.ndim - 1:
                gripper = gripper[..., None]
            elif gripper.ndim == joint_vel.ndim and gripper.shape[-1] == 1:
                pass
            else:
                raise ValueError(
                    "Unexpected gripper action shape for concatenation: "
                    f"{gripper.shape} vs joint velocity {joint_vel.shape}"
                )
            out["actions"] = np.concatenate([joint_vel, gripper], axis=-1)

        return out
