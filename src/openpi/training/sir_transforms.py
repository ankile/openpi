from collections.abc import Sequence
import dataclasses

import numpy as np

from openpi import transforms


def _pick_first_available(data: dict, keys: Sequence[str], label: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing {label}. Tried keys: {list(keys)}")


def _crop_chw(image, box: tuple[int, int, int, int], label: str):
    """Slice a channels-first ``(..., C, H, W)`` frame to ``box = (x0, y0, x1, y1)``
    (half-open, ``[..., y0:y1, x0:x1]``) in the STORED-frame pixel space.

    Works on torch tensors and numpy arrays alike (both index height-then-width on the
    last two axes; plain slicing is backend-agnostic). SIR real LeRobot v3 datasets store
    video frames as channels-first, exactly as the sibling ``sir.real.side_crop`` /
    ``image_preprocess`` train path assumes, so we REQUIRE channels-first (``shape[-3] ==
    3``) and fail loud on any other layout rather than silently mis-cropping an HWC frame.
    The box must lie inside the frame — a box that overruns the stored resolution is a
    config error (wrong crop reference), not something to clamp silently.
    """
    x0, y0, x1, y1 = box
    if image.shape[-3] != 3:
        raise ValueError(
            f"{label} crop expects a channels-first (...,3,H,W) frame, got shape "
            f"{tuple(image.shape)}. SIR LeRobot v3 stores frames channels-first; refusing "
            f"to guess the spatial axes."
        )
    h, w = image.shape[-2], image.shape[-1]
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
        raise ValueError(
            f"{label} crop box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) is invalid; require "
            f"nonnegative origin and x1>x0, y1>y0"
        )
    if x1 > w or y1 > h:
        raise ValueError(
            f"{label} crop box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) exceeds frame of size "
            f"HxW={h}x{w}; the box must lie inside the stored frame (wrong crop reference?)"
        )
    return image[..., y0:y1, x0:x1]


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
        "observation.images.exterior_image_1_left",  # DROID canonical name
        "observation.images.25916956_left",  # legacy serial key (marker room = side_1)
        "observation.images.side_1",  # role-keyed (routing_d1 and newer rooms)
    )
    wrist_image_keys: Sequence[str] = (
        "observation.images.wrist_image_left",  # DROID canonical name
        "observation.images.18650758_left",  # legacy serial key (marker room = wrist_left)
        "observation.images.wrist_left",  # role-keyed (routing_d1 and newer rooms)
    )
    # Optional third camera. When set, emit "observation/exterior_image_2_left" from the
    # first available key (KeyError if none present — fail loud, no silent skip). Used to
    # feed a second exterior view (e.g. routing_d1's side_2) into the model's third slot.
    exterior_image_2_keys: Sequence[str] | None = None
    # Optional per-slot STATIC crop boxes in STORED-frame (480x640) px, (x0, y0, x1, y1)
    # half-open — applied to the chosen frame BEFORE OpenPI's resize-with-pad to 224x224,
    # so the policy trains on a tighter, higher-effective-resolution ROI. None => full frame
    # (byte-identical to the pre-crop path). These live here (not in DroidInputs) to stay
    # SIR-scoped; the eval wrapper reads the SAME boxes off the resolved train config so
    # train and eval crop identically. NOT tyro-exposed (the repack is built programmatically
    # in LeRobotSIRDROIDDataConfig.create), so `tuple | None` is safe here.
    exterior_image_crop: tuple[int, int, int, int] | None = None
    wrist_image_crop: tuple[int, int, int, int] | None = None
    exterior_image_2_crop: tuple[int, int, int, int] | None = None
    joint_position_key: str = "observation.state.joint_position"
    gripper_position_key: str = "observation.state.gripper_position"
    action_joint_velocity_key: str = "action.joint_velocity"
    action_gripper_position_key: str = "action.gripper_position"
    # Only honor an explicit "prompt" — do NOT fall back to the LeRobot "task"
    # slug, so the config's default_prompt (a real instruction) is used instead.
    prompt_keys: Sequence[str] = ("prompt",)

    def __call__(self, data: dict) -> dict:
        exterior_image = _pick_first_available(data, self.exterior_image_keys, "exterior image")
        wrist_image = _pick_first_available(data, self.wrist_image_keys, "wrist image")
        if self.exterior_image_crop is not None:
            exterior_image = _crop_chw(exterior_image, self.exterior_image_crop, "exterior_image")
        if self.wrist_image_crop is not None:
            wrist_image = _crop_chw(wrist_image, self.wrist_image_crop, "wrist_image")

        out = {
            "observation/exterior_image_1_left": exterior_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": np.asarray(data[self.joint_position_key], dtype=np.float32),
            "observation/gripper_position": np.asarray(data[self.gripper_position_key], dtype=np.float32),
        }

        if self.exterior_image_2_keys is not None:
            exterior_image_2 = _pick_first_available(data, self.exterior_image_2_keys, "exterior image 2")
            if self.exterior_image_2_crop is not None:
                exterior_image_2 = _crop_chw(exterior_image_2, self.exterior_image_2_crop, "exterior_image_2")
            out["observation/exterior_image_2_left"] = exterior_image_2

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
