"""Convert an OpenPI checkpoint to bfloat16 and upload to HuggingFace Hub.

Converts params from mixed float32/bfloat16 → pure bfloat16 (halves size),
then uploads params + assets + metadata + model card to an HF model repo.
Excludes train_state/ (not needed for inference).

Must be run from deps/openpi/ inside OpenPI's uv environment (needs JAX + orbax):

    # Finetuned checkpoint
    cd deps/openpi
    uv run python ../../scripts/convert_checkpoint_bf16.py \\
        --checkpoint-dir checkpoints/pi05_droid_finetune/my-exp/5000 \\
        --repo-id ankile/my-model-name \\
        --config-name pi05_droid_finetune \\
        --dataset-id ankile/my-dataset

    # Pretrained checkpoint (from OpenPI's GCS cache)
    uv run python ../../scripts/convert_checkpoint_bf16.py \\
        --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \\
        --repo-id ankile/openpi-pi05-droid-pretrained \\
        --config-name pi05_droid \\
        --pretrained

To convert without uploading (e.g. for local testing):

    uv run python ../../scripts/convert_checkpoint_bf16.py \\
        --checkpoint-dir checkpoints/pi05_droid_finetune/my-exp/5000 \\
        --output-dir /tmp/my-checkpoint-bf16 \\
        --no-upload
"""

import argparse
from datetime import datetime, timezone
import json
import re
import shutil
import tempfile
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models.model import restore_params


def convert_to_bf16(checkpoint_dir: Path, output_dir: Path, config_name: str) -> float:
    """Load checkpoint params as bfloat16 and save to output_dir. Returns size in GB."""
    params_path = checkpoint_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"No params/ directory found in {checkpoint_dir}")

    print(f"Loading params from {params_path} as bfloat16...")
    params = restore_params(params_path, dtype=jnp.bfloat16, restore_type=np.ndarray)

    # Verify all arrays are bfloat16
    leaves = jax.tree.leaves(params)
    dtypes = {a.dtype for a in leaves}
    print(f"Param dtypes after conversion: {dtypes}")
    if dtypes != {np.dtype("bfloat16")}:
        raise ValueError(f"Expected all bfloat16, got {dtypes}")

    total_bytes = sum(a.nbytes for a in leaves)
    param_size_gb = total_bytes / 1e9
    print(f"Total param size: {param_size_gb:.2f} GB")

    # Save params with orbax
    dst_params = output_dir / "params"
    print(f"Saving bfloat16 params to {dst_params}...")
    if dst_params.exists():
        shutil.rmtree(dst_params)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(dst_params), ocp.args.PyTreeSave({"params": params}))

    # Copy assets and metadata (skip train_state — not needed for inference)
    assets_src = checkpoint_dir / "assets"
    if assets_src.exists():
        dst_assets = output_dir / "assets"
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(assets_src, dst_assets)
        print(f"Copied assets/ to {dst_assets}")

    metadata_src = checkpoint_dir / "_CHECKPOINT_METADATA"
    if metadata_src.exists():
        shutil.copy2(metadata_src, output_dir / "_CHECKPOINT_METADATA")
        print("Copied _CHECKPOINT_METADATA")

    # Required by sir/real OpenPI wrapper for auto config inference with hf:// models.
    config_path = output_dir / "openpi_config.json"
    config_path.write_text(json.dumps({"config_name": config_name}, indent=2) + "\n")
    print(f"Wrote {config_path}")

    print(f"Checkpoint ready at: {output_dir}")
    return param_size_gb


def parse_wandb_run_url(wandb_run_url: str) -> tuple[str, str, str] | None:
    """Parse a W&B run URL into (entity, project, run_id)."""
    match = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", wandb_run_url)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def write_provenance_metadata(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    repo_id: str | None,
    config_name: str,
    wandb_run_url: str | None,
    slurm_job_id: str | None,
) -> Path:
    """Write training/export provenance metadata for traceability."""
    exported_at = datetime.now(timezone.utc).isoformat()
    step = checkpoint_dir.name
    metadata = {
        "config_name": config_name,
        "checkpoint_step": step,
        "source_checkpoint_dir": str(checkpoint_dir),
        "hf_repo_id": repo_id,
        "hf_model_url": f"https://huggingface.co/{repo_id}" if repo_id else None,
        "wandb_run_url": wandb_run_url,
        "slurm_job_id": slurm_job_id,
        "exported_at_utc": exported_at,
    }
    out_path = output_dir / "checkpoint_provenance.json"
    out_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote provenance metadata: {out_path}")
    return out_path


def update_wandb_run_summary(
    *,
    wandb_run_url: str,
    repo_id: str,
    config_name: str,
    checkpoint_step: str,
) -> None:
    """Back-link the uploaded HF model to the originating W&B run."""
    parsed = parse_wandb_run_url(wandb_run_url)
    if parsed is None:
        print(f"WARNING: Could not parse W&B run URL: {wandb_run_url}")
        return

    entity, project, run_id = parsed
    run_path = f"{entity}/{project}/{run_id}"

    try:
        import wandb
    except Exception as e:
        print(f"WARNING: wandb import failed; skipping W&B back-link update: {e}")
        return

    try:
        api = wandb.Api()
        run = api.run(run_path)
        run.summary["hf_model_repo"] = repo_id
        run.summary["hf_model_url"] = f"https://huggingface.co/{repo_id}"
        run.summary["hf_model_checkpoint_step"] = checkpoint_step
        run.summary["hf_model_openpi_config"] = config_name
        run.summary["hf_model_linked_at_utc"] = datetime.now(timezone.utc).isoformat()
        run.summary.update()
        print(f"Updated W&B run summary with HF model link: {run_path}")
    except Exception as e:
        print(f"WARNING: Failed to update W&B run summary ({run_path}): {e}")


def generate_model_card(
    output_dir: Path,
    repo_id: str,
    config_name: str,
    dataset_id: str | None,
    checkpoint_dir: Path,
    param_size_gb: float,
    wandb_run_url: str | None = None,
    slurm_job_id: str | None = None,
    pretrained: bool = False,
) -> None:
    """Generate a README.md model card and write it to output_dir."""
    # Extract step number from checkpoint path (e.g. .../5000/ → 5000)
    step = checkpoint_dir.name
    dataset_link = f"[`{dataset_id}`](https://huggingface.co/datasets/{dataset_id})" if dataset_id else "N/A"
    dataset_yaml = f"\ndatasets:\n  - {dataset_id}" if dataset_id else ""

    if pretrained:
        description = "The official [Pi0.5](https://www.physicalintelligence.company/blog/pi0-5) DROID pretrained model from the [OpenPI](https://github.com/Physical-Intelligence/openpi) framework, converted to bfloat16 for efficient storage and inference."
        # Indentation (8 spaces) must match the outer textwrap.dedent template
        details_rows_list = [
            f"        | OpenPI config | `{config_name}` |",
            f"        | Precision | bfloat16 |",
            f"        | Parameter size | ~{param_size_gb:.1f} GB |",
            f"        | Original source | `gs://openpi-assets/checkpoints/{config_name}` |",
        ]
    else:
        description = f"A [Pi0.5](https://www.physicalintelligence.company/blog/pi0-5) model fine-tuned using the [OpenPI](https://github.com/Physical-Intelligence/openpi) framework."
        details_rows_list = [
            f"        | OpenPI config | `{config_name}` |",
            f"        | Checkpoint step | {step} |",
            f"        | Training data | {dataset_link} |",
            f"        | Precision | bfloat16 |",
            f"        | Parameter size | ~{param_size_gb:.1f} GB |",
            f"        | Source checkpoint | `{checkpoint_dir}` |",
        ]

    details_rows_list.append(f"        | Hugging Face repo | `{repo_id}` |")
    if wandb_run_url:
        details_rows_list.append(f"        | W&B run | [link]({wandb_run_url}) |")
    if slurm_job_id:
        details_rows_list.append(f"        | SLURM job ID | `{slurm_job_id}` |")
    details_rows = "\n".join(details_rows_list)

    card = textwrap.dedent(f"""\
        ---
        license: apache-2.0
        library_name: openpi
        tags:
          - robotics
          - manipulation
          - pi0.5
          - openpi
          - jax
          - orbax{dataset_yaml}
        pipeline_tag: robotics
        ---

        # {repo_id.split("/")[-1]}

        {description}

        ## Model Details

        | Property | Value |
        |---|---|
{details_rows}

        ## Usage

        ### Download and run inference

        ```bash
        # Download checkpoint from HF Hub
        huggingface-cli download {repo_id} --local-dir <local_path>

        # Run inference server
        cd deps/openpi
        uv run python scripts/serve_policy.py {config_name} \\
          --checkpoint-dir <local_path>
        ```

        ### In-process inference (Python)

        ```python
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config as openpi_policy_config

        train_config = openpi_config.get_config("{config_name}")
        policy = openpi_policy_config.create_trained_policy(
            train_config, "<local_path>"
        )
        result = policy.infer(obs_dict)
        actions = result["actions"]
        ```

        ## Checkpoint Format

        [Orbax](https://github.com/google/orbax) format, all parameters in bfloat16.

        ```
        ├── _CHECKPOINT_METADATA
        ├── checkpoint_provenance.json
        ├── openpi_config.json
        ├── assets/
        │   └── (normalization stats)
        └── params/
            └── (orbax checkpoint files)
        ```

        ## License

        Apache 2.0
    """)

    readme_path = output_dir / "README.md"
    readme_path.write_text(card)
    print(f"Generated model card at {readme_path}")


def upload_to_hub(local_dir: Path, repo_id: str) -> None:
    """Upload a directory to HuggingFace Hub as a model repo."""
    print(f"\nUploading to https://huggingface.co/{repo_id} ...")
    print("Mirror upload enabled: removing stale remote files not present locally.")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        delete_patterns="*",
        commit_message="Upload converted checkpoint (mirror repo contents)",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenPI checkpoint to bfloat16 and upload to HuggingFace Hub."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to the checkpoint directory (containing params/, assets/, _CHECKPOINT_METADATA)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. ankile/my-model). Required unless --no-upload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for bf16 checkpoint. Default: auto temp dir (cleaned up after upload).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_droid_finetune",
        help="OpenPI config name used for training (default: pi05_droid_finetune).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="HuggingFace dataset ID used for training (for model card).",
    )
    parser.add_argument(
        "--wandb-run-url",
        type=str,
        default=None,
        help="W&B run URL that produced this checkpoint (for model card + back-linking).",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=str,
        default=None,
        help="SLURM job ID that produced this checkpoint (for provenance metadata).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Convert only, skip HF Hub upload. Requires --output-dir.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Flag this as a pretrained (not finetuned) checkpoint. Adjusts model card language.",
    )
    args = parser.parse_args()

    if not args.no_upload and args.repo_id is None:
        parser.error("--repo-id is required unless --no-upload is set")
    if args.no_upload and args.output_dir is None:
        parser.error("--output-dir is required when using --no-upload")

    # Determine output directory
    use_tempdir = args.output_dir is None
    if use_tempdir:
        tmpdir = tempfile.mkdtemp(prefix="openpi-bf16-")
        output_dir = Path(tmpdir)
    else:
        output_dir = args.output_dir

    try:
        param_size_gb = convert_to_bf16(args.checkpoint_dir, output_dir, args.config_name)
        write_provenance_metadata(
            output_dir,
            checkpoint_dir=args.checkpoint_dir,
            repo_id=args.repo_id,
            config_name=args.config_name,
            wandb_run_url=args.wandb_run_url,
            slurm_job_id=args.slurm_job_id,
        )

        if args.repo_id:
            generate_model_card(
                output_dir,
                repo_id=args.repo_id,
                config_name=args.config_name,
                dataset_id=args.dataset_id,
                checkpoint_dir=args.checkpoint_dir,
                param_size_gb=param_size_gb,
                wandb_run_url=args.wandb_run_url,
                slurm_job_id=args.slurm_job_id,
                pretrained=args.pretrained,
            )

        if not args.no_upload:
            upload_to_hub(output_dir, args.repo_id)
            if args.wandb_run_url:
                update_wandb_run_summary(
                    wandb_run_url=args.wandb_run_url,
                    repo_id=args.repo_id,
                    config_name=args.config_name,
                    checkpoint_step=args.checkpoint_dir.name,
                )
    finally:
        if use_tempdir:
            print(f"Cleaning up temp dir: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
