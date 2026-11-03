import dataclasses
import enum
import logging
import socket
import time

import numpy as np
import tyro

from openpi.models import rtc as _rtc
from openpi.policies import droid_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # zima/rtc: accept an action prefix (real-time chunking) where the policy supports it (JAX pi0.5 DROID):
    # inpainting for a model trained with simulated_delay, guidance (arXiv 2506.07339) for one trained without.
    rtc: bool = True
    # Guidance weight clip beta (2506.07339 Eq. 2; the paper's 5). Used only in guidance mode.
    max_guidance_weight: float = _rtc.DEFAULT_MAX_GUIDANCE_WEIGHT
    # Flow-matching integration steps (openpi's default 10).
    num_steps: int = 10
    # Compile the inference graphs (and, with rtc, the prefix graph) before opening the port.
    warm_up: bool = True


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _sample_kwargs(args: Args | None) -> dict:
    if args is None:
        return {}
    # openpi's default step count is left to the model, so a request without a prefix runs upstream's graph
    out = {"max_guidance_weight": args.max_guidance_weight}
    if args.num_steps != 10:
        out["num_steps"] = args.num_steps
    return out


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, args: Args | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=_sample_kwargs(args),
            rtc=True if args is None else args.rtc,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=_sample_kwargs(args),
                rtc=args.rtc,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, args=args)


def warm_up(policy: _policy.Policy) -> dict:
    """zima/rtc: run the no-prefix graph and, for an RTC policy, the prefix graph once, before serving.

    The first call of each compiles (minutes on a fresh server); a client must not pay that inside a stream.
    Only for DROID inputs, the only policies with prefix support. Returns the timings in ms.
    """
    timings = {}
    started = time.monotonic()
    policy.infer(droid_policy.make_droid_example())
    timings["no_prefix_ms"] = (time.monotonic() - started) * 1000
    if policy.rtc is not None:
        horizon, width = policy.rtc["action_horizon"], policy.rtc["action_dim"]
        example = droid_policy.make_droid_example()
        prefix_length = horizon // 2
        delay = 2 if policy.rtc["mode"] == "guidance" else policy.rtc["simulated_delay"] - 1
        example["action_prefix"] = {
            "actions": np.zeros((horizon, width), np.float32),
            "delay": min(delay, prefix_length),
            "prefix_length": prefix_length,
        }
        started = time.monotonic()
        policy.infer(example)
        timings["prefix_ms"] = (time.monotonic() - started) * 1000
    logging.info("warm-up done: %s", timings)
    return timings


def should_warm_up(args: Args, policy: _policy.Policy) -> bool:
    """Warm-up sends DROID inputs: only to a policy with prefix support (DROID by construction) or the
    default DROID policy, never to an explicit checkpoint of another kind."""
    return args.warm_up and (policy.rtc is not None or (isinstance(args.policy, Default) and args.env == EnvMode.DROID))


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if should_warm_up(args, policy):
        warm_up(policy)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
