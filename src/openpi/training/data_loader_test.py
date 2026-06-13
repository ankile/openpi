import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

    stats = loader.data_stats()
    assert stats is not None
    assert stats["repo_ids"] == ["fake"]
    assert stats["training_frames"] == 1024
    assert stats["total_frames"] == 1024
    assert stats["filtered_frames"] == 0


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_apply_repo_ids_override():
    config = _config.get_config("pi05_sir_droid_finetune")
    updated = _config.apply_repo_ids_override(config, ["a/repo1", "b/repo2"])
    assert updated.data.repo_id == "a/repo1"
    assert list(updated.data.repo_ids) == ["a/repo1", "b/repo2"]


class _TinyDataset:
    def __init__(self):
        self.features = {"success": {}, "source": {}, "is_valid": {}}
        self.hf_dataset = [
            {"success": 1, "source": 1, "is_valid": 1},
            {"success": 0, "source": 1, "is_valid": 1},
            {"success": 1, "source": 0, "is_valid": 1},
            {"success": 1, "source": 1, "is_valid": 0},
            {"success": 1, "source": 1, "is_valid": 1},
        ]

    def __getitem__(self, index):
        return self.hf_dataset[index]

    def __len__(self):
        return len(self.hf_dataset)


def test_build_filtered_indices_stats():
    dataset = _TinyDataset()
    config = _config.DataConfig(
        filter_success_only=True,
        filter_human_source=True,
        require_valid_frames=True,
        success_value=1,
        source_human_value=1,
    )

    indices, stats = _data_loader._build_filtered_indices(dataset, config)
    assert indices == [0, 4]
    assert stats["total_frames"] == 5
    assert stats["training_frames"] == 2
    assert stats["filtered_frames"] == 3
    assert stats["excluded_success_frames"] == 1
    assert stats["excluded_source_frames"] == 1
    assert stats["excluded_invalid_frames"] == 1


def test_compute_nonidle_keep_mask_matches_droid_semantics():
    import numpy as np

    rng = np.random.default_rng(0)
    dof = 7
    # Episode layout: 30 moving, 10 idle (>= min_idle_len -> dropped),
    # 40 moving, 3 idle (< min_idle_len -> kept), 27 moving.
    moving_a = rng.uniform(-0.5, 0.5, size=(30, dof))
    idle_long = np.tile(rng.uniform(-0.5, 0.5, size=(1, dof)), (10, 1))
    moving_b = rng.uniform(-0.5, 0.5, size=(40, dof))
    idle_short = np.tile(rng.uniform(-0.5, 0.5, size=(1, dof)), (3, 1))
    moving_c = rng.uniform(-0.5, 0.5, size=(27, dof))
    jv = np.concatenate([moving_a, idle_long, moving_b, idle_short, moving_c])

    mask = _data_loader._compute_nonidle_keep_mask(
        jv,
        idle_action_threshold=1e-3,
        min_idle_len=7,
        min_non_idle_len=16,
        filter_last_n_in_ranges=10,
    )

    # Idle frames are flagged on the diff, so the long idle run covers frames 31..39
    # (frame 30 differs from 29). Kept ranges before tail-trim: [0, 31) and [40, 110)
    # (the 3-frame idle run at 70+3.. is shorter than min_idle_len so it survives).
    expected = np.zeros(len(jv), dtype=bool)
    expected[0 : 31 - 10] = True
    expected[40 : 110 - 10] = True
    assert mask.tolist() == expected.tolist()

    # An all-idle episode keeps nothing.
    all_idle = np.tile(jv[:1], (50, 1))
    mask_idle = _data_loader._compute_nonidle_keep_mask(
        all_idle,
        idle_action_threshold=1e-3,
        min_idle_len=7,
        min_non_idle_len=16,
        filter_last_n_in_ranges=10,
    )
    assert not mask_idle.any()


class _TinyIdleDataset:
    def __init__(self):
        import numpy as np

        rng = np.random.default_rng(1)
        self.repo_id = "test/tiny-idle"
        self.features = {"action.joint_velocity": {}, "episode_index": {}, "frame_index": {}}
        self.hf_dataset = []
        for episode_index in range(2):
            moving = rng.uniform(-0.5, 0.5, size=(30, 7))
            idle = np.tile(rng.uniform(-0.5, 0.5, size=(1, 7)), (10, 1))
            jv = np.concatenate([moving, idle, rng.uniform(-0.5, 0.5, size=(20, 7))])
            for frame_index in range(len(jv)):
                self.hf_dataset.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "action.joint_velocity": jv[frame_index],
                    }
                )

    def __getitem__(self, index):
        return self.hf_dataset[index]

    def __len__(self):
        return len(self.hf_dataset)


def test_build_filtered_indices_idle_filter():
    dataset = _TinyIdleDataset()
    config = _config.DataConfig(filter_idle_frames=True)

    indices, stats = _data_loader._build_filtered_indices(dataset, config)
    assert stats["total_frames"] == 120
    assert stats["excluded_idle_frames"] > 0
    assert stats["training_frames"] == len(indices)
    assert stats["training_frames"] + stats["excluded_idle_frames"] == 120
    # Per-episode: frames 31..39 idle-dropped, ranges [0,31) and [40,60) tail-trimmed by 10.
    per_episode_kept = 21 + 10
    assert stats["training_frames"] == 2 * per_episode_kept
