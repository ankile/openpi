import numpy as np

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
    assert out["prompt"] == "insert marker"


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
