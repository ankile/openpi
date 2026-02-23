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
