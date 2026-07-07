# OpenPI SIR training environment (py3.12)

The committed root `deps/openpi/pyproject.toml` is the **robot-workstation eval**
env (py3.11, pyzed numpy<2, policy-arena, SpaceMouse). The SIR **training** env is
divergent and lives here so it is reproducible from git on any cluster (iris, delta,
…) without clobbering the robot env.

`pyproject.toml` here: py3.12, `lerobot[dataset]`, `torchcodec>=0.4,<0.5`,
`transformers>=5.4,<5.6`, `[tool.uv] environments` pinned to linux/x86_64, no
robot deps. `uv.lock` is the frozen resolution that produced the iris env
(jax 0.5.3 / torch 2.7.1+cu128 / transformers 5.5.4 / numpy 2.2.6 / torchcodec 0.4.0
/ lerobot 0.5.2). The lock's editable `lerobot = { path = "../lerobot" }` resolves
relative to the checkout root, so the checkout's sibling must be the FF lerobot
(inside a full SIR clone, `deps/openpi/../lerobot` == `deps/lerobot`).

## Provision (any cluster)

Run from a full SIR clone whose submodules are at the pinned commits
(`deps/lerobot` @ 73782447, `deps/openpi` @ this commit). Keep the venv + uv cache
off any quota'd home (delta `/u` is 103G-capped) — put both on scratch/work.

```bash
export UV_CACHE_DIR=/work/.../uv-cache          # NOT ~/.cache on quota'd homes
cd <clone>/deps/openpi
cp sir_scripts/training_env/pyproject.toml .    # swap robot env -> training env
cp sir_scripts/training_env/uv.lock .
uv python install 3.12
uv sync --frozen                                # exact iris resolution
# smoke: uv run --no-sync python scripts/train.py pi05_sir_droid_finetune_routing_3cam_crop --help
```

`../lerobot` resolves to `deps/lerobot` automatically. `OPENPI_DIR` for the launch
script is then `<clone>/deps/openpi`.
