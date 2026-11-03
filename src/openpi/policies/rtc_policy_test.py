# ruff: noqa: SLF001  # tests reach into the model and policy internals they test
"""zima/rtc: the server side of real-time chunking. The real DROID transforms and pi05_droid norm stats around a
fake model that echoes the prefix it is given, so no weights are needed (the norm stats and tokenizer come
from openpi's download cache)."""

import dataclasses
import pathlib
import typing

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import rtc
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.shared import normalize
from openpi.training import config as _config
import openpi.transforms as _transforms


class EchoModel(nnx.Module):
    """Returns the model-space prefix when given one, else zeros; records nothing else."""

    def __init__(self, config: pi0_config.Pi0Config):
        self.action_horizon, self.action_dim = config.action_horizon, config.action_dim
        self.offset = nnx.Param(jnp.zeros(()))

    def sample_actions(self, rng, observation, *, action_prefix=None, **kwargs):
        batch = observation.state.shape[0]
        if action_prefix is None:
            return jnp.zeros((batch, self.action_horizon, self.action_dim)) + self.offset.value
        return action_prefix.actions + self.offset.value


def _droid_norm_stats():
    path = download.get_cache_dir() / "openpi-assets/checkpoints/pi05_droid/assets/droid"
    if not (path / "norm_stats.json").exists():
        pytest.skip(f"pi05_droid norm stats are not in the openpi cache ({path})")
    return normalize.load(path)


def _policy(monkeypatch, name, **kwargs):
    """The policy create_trained_policy serves for `name`, around the echo model, with pi05_droid's norm stats
    (which every DROID fine-tune reuses)."""
    config = _config.get_config(name)
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda self, params, **_: EchoModel(self))
    monkeypatch.setattr(_policy_config._model, "restore_params", lambda *a, **k: None)
    original = download.maybe_download
    monkeypatch.setattr(
        download,
        "maybe_download",
        lambda path, **kw: pathlib.Path(path) if path == "/nonexistent" else original(path, **kw),
    )
    return _policy_config.create_trained_policy(config, "/nonexistent", norm_stats=_droid_norm_stats(), **kwargs)


def _request(prefix=None):
    example = droid_policy.make_droid_example()
    if prefix is not None:
        example["action_prefix"] = prefix
    return example


def _raw_prefix(horizon, seed=0):
    rng = np.random.default_rng(seed)
    actions = np.concatenate([rng.uniform(-0.6, 0.6, (horizon, 7)), rng.uniform(0, 1, (horizon, 1))], axis=1)
    return actions.astype(np.float32)


def test_rtc_config_is_the_published_recipe_plus_simulated_delay():
    base, rtc_config = _config.get_config("pi05_droid_finetune"), _config.get_config("pi05_droid_finetune_rtc")
    assert rtc_config.model.simulated_delay == 8
    assert base.model.simulated_delay is None
    assert dataclasses.replace(rtc_config.model, simulated_delay=None) == base.model
    assert dataclasses.replace(rtc_config, name=base.name, model=base.model) == base


def test_metadata_says_what_the_server_does(monkeypatch):
    inpaint = _policy(monkeypatch, "pi05_droid_finetune_rtc")
    assert inpaint.metadata["rtc"] == {
        "protocol": 1,
        "mode": "inpaint",
        "simulated_delay": 8,
        "action_horizon": 16,
        "action_dim": 8,
        "num_steps": 10,
        "guidance": None,
    }
    guided = _policy(monkeypatch, "pi05_droid", sample_kwargs={"max_guidance_weight": 2.0})
    assert guided.metadata["rtc"]["mode"] == "guidance"
    assert guided.metadata["rtc"]["action_horizon"] == 15
    assert guided.metadata["rtc"]["guidance"] == {"max_weight": 2.0, "schedule": "exp"}
    assert "rtc" not in _policy(monkeypatch, "pi05_droid", rtc=False).metadata
    # a non-DROID pipeline gets no prefix support, nor does pi0
    for name in ("pi05_libero", "pi0_droid"):
        config = _config.get_config(name)
        data_config = config.data.create(config.assets_dirs, config.model)
        assert not _policy_config._actions_pass_through(config, data_config, _transforms.Group())


def test_prefix_round_trip_through_the_action_transforms(monkeypatch):
    """raw labels -> Normalize (quantile) -> pad -> [model echoes] -> Unnormalize -> DroidOutputs = raw labels."""
    policy = _policy(monkeypatch, "pi05_droid_finetune_rtc")
    raw = _raw_prefix(16)
    model_space = policy._action_prefix_transform(raw.astype(np.float64))
    assert model_space.shape == (16, 32)
    np.testing.assert_array_equal(model_space[:, 8:], 0.0)  # padded dims are zero, as in training
    out = policy.infer(_request({"actions": raw, "delay": 3, "prefix_length": 8}))
    np.testing.assert_allclose(out["actions"][:8], raw[:8], atol=1e-5)  # the prefix rows; the rest is padding
    # padding is zero in model space: un-normalized it is the quantile midpoint, whatever the robot sent
    np.testing.assert_allclose(out["actions"][8:], policy.infer(_request())["actions"][8:], atol=1e-5)
    # no prefix: the plain path (the echo model's zeros, un-normalized)
    assert policy.infer(_request())["actions"].shape == (16, 8)


@pytest.mark.parametrize(
    ("prefix", "match"),
    [
        ({"actions": np.zeros((16, 8)), "delay": 5, "prefix_length": 4}, "delay <= prefix_length"),
        ({"actions": np.zeros((16, 8)), "delay": 8, "prefix_length": 10}, "trained range"),
        ({"actions": np.zeros((16, 8)), "delay": -1, "prefix_length": 4}, "0 <= delay"),
        ({"actions": np.zeros((16, 8)), "delay": 1, "prefix_length": 17}, "prefix_length <= 16"),
        ({"actions": np.full((16, 8), np.nan), "delay": 1, "prefix_length": 4}, "finite"),
        ({"actions": np.zeros((15, 8)), "delay": 1, "prefix_length": 4}, "shape"),
        ({"actions": np.zeros((16, 7)), "delay": 1, "prefix_length": 4}, "shape"),
        ({"actions": np.zeros((16, 8)), "delay": True, "prefix_length": 4}, "integer"),
        ({"actions": np.zeros((16, 8)), "delay": 1.5, "prefix_length": 4}, "integer"),
        ({"actions": np.zeros((16, 8)), "delay": 1}, "action_prefix must be"),
        ({"actions": np.zeros((16, 8), complex), "delay": 1, "prefix_length": 4}, "real numbers"),
        ({"actions": np.zeros((16, 8), bool), "delay": 1, "prefix_length": 4}, "real numbers"),
        ({"actions": np.full((16, 8), 1e300), "delay": 1, "prefix_length": 4}, "after normalization"),
        ({"actions": np.zeros((16, 8)), "delay": 1, "prefix_length": 4, "extra": 0}, "action_prefix must be"),
    ],
)
def test_bad_prefixes_are_refused(monkeypatch, prefix, match):
    policy = _policy(monkeypatch, "pi05_droid_finetune_rtc")
    with pytest.raises(ValueError, match=match):
        policy.infer(_request(prefix))


def test_padding_cannot_overflow_into_the_guidance(monkeypatch):
    policy = _policy(monkeypatch, "pi05_droid")
    actions = _raw_prefix(15).astype(np.float64)
    actions[6:] = 1e300  # finite in float64, inf in float32, beyond prefix_length
    assert np.isfinite(actions).all()
    prefix = policy._action_prefix({"actions": actions, "delay": 2, "prefix_length": 6})
    assert np.isfinite(np.asarray(prefix.actions)).all()
    np.testing.assert_array_equal(np.asarray(prefix.actions)[0, 6:], 0.0)  # model-space padding is zero


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_steps": 0}, "positive integer"),
        ({"num_steps": -1}, "positive integer"),
        ({"num_steps": 2.5}, "positive integer"),
        ({"max_guidance_weight": float("inf")}, "finite number"),
        ({"max_guidance_weight": -1.0}, "finite number"),
        ({"max_guidance_weight": float("nan")}, "finite number"),
        ({"num_steps": True}, "positive integer"),
        ({"num_steps": np.int64(0)}, "positive integer"),
        ({"max_guidance_weight": np.True_}, "finite number"),
    ],
)
def test_bad_sampler_settings_are_refused_at_the_factory(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _policy_config.create_trained_policy(_config.get_config("pi05_droid"), "/nonexistent", sample_kwargs=kwargs)


@pytest.mark.parametrize("weight", [np.float32(0), np.float32(5), np.int64(5), np.float64(5), 5])
def test_numpy_sampler_settings_are_accepted_and_normalized(monkeypatch, weight):
    policy = _policy(
        monkeypatch, "pi05_droid", sample_kwargs={"max_guidance_weight": weight, "num_steps": np.int64(10)}
    )
    assert policy.metadata["rtc"]["guidance"]["max_weight"] == float(weight)
    assert type(policy._sample_kwargs["max_guidance_weight"]) is float
    assert type(policy._sample_kwargs["num_steps"]) is int


def test_zero_guidance_weight_is_guidance_off(monkeypatch):
    policy = _policy(monkeypatch, "pi05_droid", sample_kwargs={"max_guidance_weight": 0.0})
    assert policy.metadata["rtc"]["guidance"]["max_weight"] == 0.0


def test_num_steps_is_refused_for_models_without_it(monkeypatch):
    with pytest.raises(ValueError, match="num_steps applies"):
        _policy_config.create_trained_policy(
            _config.get_config("pi0_fast_droid"), "/nonexistent", sample_kwargs={"num_steps": 5}
        )


def _serve_policy():
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("serve_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serve_settings_and_warm_up(monkeypatch):
    serve = _serve_policy()
    assert "num_steps" not in serve._sample_kwargs(serve.Args())  # the model's own default graph

    class Recorder:
        rtc: typing.ClassVar = {"mode": "inpaint", "simulated_delay": 1, "action_horizon": 16, "action_dim": 8}

        def __init__(self):
            self.requests = []

        def infer(self, obs):
            self.requests.append(obs)
            return {}

    policy = Recorder()
    serve.warm_up(policy)  # D = 1 trains only delay 0
    assert policy.requests[1]["action_prefix"]["delay"] == 0
    policy.rtc = None
    checkpoint = serve.Args(env=serve.EnvMode.DROID, policy=serve.Checkpoint(config="pi05_libero", dir="x"))
    assert not serve.should_warm_up(checkpoint, policy)  # an explicit non-DROID checkpoint gets no DROID inputs
    assert serve.should_warm_up(serve.Args(env=serve.EnvMode.DROID), policy)
    assert not serve.should_warm_up(serve.Args(env=serve.EnvMode.LIBERO), policy)


def test_guidance_has_no_trained_range_but_keeps_the_rest(monkeypatch):
    policy = _policy(monkeypatch, "pi05_droid")
    out = policy.infer(_request({"actions": _raw_prefix(15), "delay": 9, "prefix_length": 12}))
    assert out["actions"].shape == (15, 8)
    with pytest.raises(ValueError, match="delay <= prefix_length"):
        policy.infer(_request({"actions": _raw_prefix(15), "delay": 9, "prefix_length": 8}))


def test_a_server_without_rtc_refuses_a_prefix(monkeypatch):
    policy = _policy(monkeypatch, "pi05_droid", rtc=False)
    with pytest.raises(ValueError, match="takes no action_prefix"):
        policy.infer(_request({"actions": _raw_prefix(15), "delay": 1, "prefix_length": 4}))


def test_prefix_is_passed_as_arrays(monkeypatch):
    policy = _policy(monkeypatch, "pi05_droid_finetune_rtc")
    prefix = policy._action_prefix({"actions": _raw_prefix(16), "delay": np.int64(2), "prefix_length": 9})
    assert isinstance(prefix, rtc.ActionPrefix)
    assert prefix.actions.shape == (1, 16, 32)
    assert prefix.delay.dtype == jnp.int32
    assert prefix.prefix_length.tolist() == [9]
