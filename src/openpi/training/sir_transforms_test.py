import dataclasses

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.training import config as _config
from openpi.training import sir_transforms


def test_sir_droid_repack_transform_with_legacy_camera_keys():
    transform = sir_transforms.SIRDroidRepackTransform()
    item = {
        "observation.images.25916956_left": np.zeros((3, 64, 64), dtype=np.float32),
        "observation.images.18650758_left": np.ones((3, 64, 64), dtype=np.float32),
        "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
        "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
        "action.joint_velocity": np.zeros((16, 7), dtype=np.float32),
        "action.gripper_position": np.ones((16,), dtype=np.float32),
        "task": "insert marker",
    }

    out = transform(item)

    assert "observation/exterior_image_1_left" in out
    assert "observation/wrist_image_left" in out
    assert out["actions"].shape == (16, 8)
    # The LeRobot "task" slug is intentionally NOT used as the prompt (only an explicit
    # "prompt" key is honored, so the config's default_prompt wins) -- see
    # SIRDroidRepackTransform.prompt_keys.
    assert "prompt" not in out
    # Default (no exterior_image_2_keys) => 2-camera output, no third image key.
    assert "observation/exterior_image_2_left" not in out


def test_sir_droid_repack_transform_with_role_keys_three_cameras():
    transform = sir_transforms.SIRDroidRepackTransform(
        exterior_image_2_keys=("observation.images.side_2",),
    )
    item = {
        "observation.images.side_1": np.zeros((3, 64, 64), dtype=np.float32),
        "observation.images.wrist_left": np.ones((3, 64, 64), dtype=np.float32),
        "observation.images.side_2": np.full((3, 64, 64), 0.5, dtype=np.float32),
        "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
        "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
    }

    out = transform(item)

    assert "observation/exterior_image_1_left" in out
    assert "observation/wrist_image_left" in out
    assert "observation/exterior_image_2_left" in out
    np.testing.assert_array_equal(out["observation/exterior_image_2_left"], item["observation.images.side_2"])


def test_sir_droid_repack_transform_missing_exterior_2_raises():
    transform = sir_transforms.SIRDroidRepackTransform(
        exterior_image_2_keys=("observation.images.side_2",),
    )
    item = {
        "observation.images.side_1": np.zeros((3, 64, 64), dtype=np.float32),
        "observation.images.wrist_left": np.ones((3, 64, 64), dtype=np.float32),
        "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
        "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
    }
    with pytest.raises(KeyError, match="Missing exterior image 2"):
        transform(item)


def _droid_inputs_item():
    return {
        "observation/exterior_image_1_left": np.zeros((64, 64, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.ones((64, 64, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((7,), dtype=np.float32),
        "observation/gripper_position": np.array([0.1], dtype=np.float32),
    }


def test_droid_inputs_third_slot_off_is_zeros_masked_false():
    transform = droid_policy.DroidInputs(model_type=_model.ModelType.PI05)
    out = transform(_droid_inputs_item())
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], np.zeros((64, 64, 3), dtype=np.uint8))
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is False


def test_droid_inputs_third_slot_on_uses_exterior_2_masked_true():
    transform = droid_policy.DroidInputs(model_type=_model.ModelType.PI05, use_exterior_image_2=True)
    item = _droid_inputs_item()
    third = np.full((64, 64, 3), 7, dtype=np.uint8)
    item["observation/exterior_image_2_left"] = third
    out = transform(item)
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], third)
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is True


def test_droid_inputs_third_slot_on_missing_key_raises():
    transform = droid_policy.DroidInputs(model_type=_model.ModelType.PI05, use_exterior_image_2=True)
    with pytest.raises(KeyError):
        transform(_droid_inputs_item())


def test_droid_inputs_pi0_fast_with_exterior_2_raises():
    transform = droid_policy.DroidInputs(model_type=_model.ModelType.PI0_FAST, use_exterior_image_2=True)
    item = _droid_inputs_item()
    item["observation/exterior_image_2_left"] = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not supported for PI0_FAST"):
        transform(item)


def test_routing_3cam_config_resolves_and_builds_data_config(tmp_path):
    # NB: lives here (not data_loader_test.py) so it runs without importing lerobot, which
    # requires Python >=3.12 while the available openpi venv is 3.11. openpi.training.config
    # imports cleanly under 3.11; data_loader (via lerobot) does not.
    config = _config.get_config("pi05_sir_droid_finetune_routing_3cam")
    # Third camera wired through as a role-keyed side_2 view.
    assert config.data.exterior_image_2_keys == ("observation.images.side_2",)
    assert config.data.filter_idle_frames is True

    # Build the DataConfig off the registered factory (strip the gs:// assets_dir so norm-stat
    # loading resolves to a nonexistent local path and is skipped -- no network in this test).
    data_factory = dataclasses.replace(config.data, assets=_config.AssetsConfig())
    data_config = data_factory.create(tmp_path, config.model)
    assert data_config.action_sequence_keys == ("action.joint_velocity", "action.gripper_position")
    assert data_config.filter_idle_frames is True

    # Run the CONFIG-created repack (not a hand-built transform) on a role-keyed sample:
    # the factory passes its OWN exterior/wrist key fields into the repack, so role keys
    # missing from the config defaults fail at train time even when the transform's
    # defaults list them (exactly how jobs 16091463/64 died).
    repack = data_config.repack_transforms.inputs[0]
    out = repack(
        {
            "observation.images.side_1": np.zeros((3, 64, 64), dtype=np.float32),
            "observation.images.wrist_left": np.ones((3, 64, 64), dtype=np.float32),
            "observation.images.side_2": np.full((3, 64, 64), 0.5, dtype=np.float32),
            "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
            "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
        }
    )
    assert "observation/exterior_image_1_left" in out
    assert "observation/wrist_image_left" in out
    assert "observation/exterior_image_2_left" in out


def test_sir_droid_repack_transform_with_canonical_camera_keys():
    transform = sir_transforms.SIRDroidRepackTransform()
    item = {
        "observation.images.exterior_image_1_left": np.zeros((3, 64, 64), dtype=np.float32),
        "observation.images.wrist_image_left": np.ones((3, 64, 64), dtype=np.float32),
        "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
        "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
    }

    out = transform(item)
    assert out["observation/joint_position"].shape == (7,)
    assert out["observation/gripper_position"].shape == (1,)
    assert "actions" not in out


def test_sir_droid_repack_transform_requires_camera_keys():
    transform = sir_transforms.SIRDroidRepackTransform()
    item = {
        "observation.state.joint_position": np.zeros((7,), dtype=np.float32),
        "observation.state.gripper_position": np.array([0.1], dtype=np.float32),
    }
    try:
        transform(item)
    except KeyError as exc:
        assert "Missing exterior image" in str(exc)
    else:
        raise AssertionError("Expected missing camera key error.")
