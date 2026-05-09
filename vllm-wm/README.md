# vllm-wm

`vllm-wm` is a minimal `vLLM`-style backend for interactive world models.

It starts from the same compact scheduler/runner idea as `nano-vllm-omni`, but changes the target abstraction:

- `nano-vllm-omni`: one diffusion pipeline, one offline generation request
- `vllm-wm`: many interactive world models, one unified session API

The first goal is to give you one standalone project that owns:

- one backend process
- one model registry
- one session contract
- one vendored inference code surface for the current priority world models

## Why This Shape

After reading the priority models in this repo, they fall into four execution families:

1. Latent-history diffusion
   - `Matrix-Game 2.0`
   - `Infinite-World`
   - `Vid2World`
   - `Open-Oasis`

2. Chunked image-to-video / prompt-driven world rollout
   - `YUME 1.5`
   - `Matrix-Game 3.0`

3. Autoregressive token world models
   - `WHAM`

4. Wrapped world-model environments
   - `Diamond`

The unifying fact is not that they share the same network architecture.
The unifying fact is that they all expose the same outer control loop:

1. load model
2. start a session from an image
3. feed an action
4. return the next frame plus metadata

That is the contract `vllm-wm` standardizes.

## Architecture

`vllm-wm` keeps the `request -> scheduler -> runner` boundary from `nano-vllm-omni`, but applies it at the world-model request level instead of the denoising-step level:

```text
HTTP API
  -> WorldModelEngine
    -> RequestScheduler
      -> ModelRunner
        -> ManagedServiceBackend
          -> per-model worker subprocess
            -> project-local service runtime
              -> vendored model inference code
```

This is intentional:

- it keeps one unified session API
- it isolates conflicting upstream module namespaces like `utils` and `main`
- it keeps the minimum practical amount of upstream code inside this project
- it avoids depending on the outer WMFactory `models/*` and `services/*` directories at runtime

## Current Backends

- `matrixgame`
- `matrixgame3`
- `yume`
- `diamond`
- `open-oasis`
- `wham`
- `vid2world`
- `infinite-world`
- `worldplay`
- `mineworld`

The backend registry is defined in [vllm_wm/registry.py](/mnt/server/WMFactory/vllm-wm/vllm_wm/registry.py:1).

Each backend is just one `ModelSpec` plus one internal service app. The spec controls the worker directory, env prefix, default port, timeout profile, and GPU selection policy.

The standalone code layout is:

- `services/*`: local runtime entrypoints
- `vendors/*`: vendored minimal upstream inference code
- `checkpoints/*`: local model weights after import/download
- `data/*`: local datasets when needed

## API

For compatibility with the existing WMFactory frontend contract, `vllm-wm` exposes both plain and `/api/*` routes:

- `GET /models`
- `POST /models/load`
- `POST /sessions/start`
- `POST /sessions/step`
- `POST /sessions/reset`
- `POST /sessions/progress`
- `POST /datasets/random-image`

Aliases:

- `GET /api/models`
- `POST /api/models/load`
- `POST /api/sessions/start`
- `POST /api/sessions/step`
- `POST /api/sessions/reset`
- `POST /api/sessions/progress`
- `POST /api/datasets/random-image`

## Environment

The recommended unified environment is documented in:

- [requirements/py312-cu12.txt](/mnt/server/WMFactory/vllm-wm/requirements/py312-cu12.txt:1)
- [scripts/bootstrap_py312.sh](/mnt/server/WMFactory/vllm-wm/scripts/bootstrap_py312.sh:1)

Design choices:

- Python `3.12`
- Torch `2.9.0`
- Transformers `4.57.3`
- Diffusers `0.37.1`
- local FlashAttention wheel first, PyPI fallback second

This is deliberately newer than several original model environments. The point is to converge on one modern stack first, then patch out small incompatibilities instead of preserving eight isolated environments forever.

## Quick Start

List models:

```bash
cd /mnt/server/WMFactory/vllm-wm
PYTHONPATH=. /usr/bin/python3.12 -m vllm_wm.cli list-models
```

Bootstrap the recommended env:

```bash
cd /mnt/server/WMFactory
http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= \
bash vllm-wm/scripts/bootstrap_py312.sh
```

Run the backend:

```bash
cd /mnt/server/WMFactory
source venvs/vllm-wm/bin/activate
cd vllm-wm
vllm-wm serve --host 0.0.0.0 --port 9100
```

Probe managed workers without loading model weights:

```bash
cd /mnt/server/WMFactory
PYTHONNOUSERSITE=1 venvs/vllm-wm/bin/python vllm-wm/scripts/probe_services.py
```

Import already-downloaded local checkpoints into the standalone layout:

```bash
cd /mnt/server/WMFactory
PYTHONNOUSERSITE=1 venvs/vllm-wm/bin/python vllm-wm/scripts/import_local_assets.py --dry-run
PYTHONNOUSERSITE=1 venvs/vllm-wm/bin/python vllm-wm/scripts/import_local_assets.py
```

If you also want to pull the large Vid2World dataset history into the standalone project:

```bash
cd /mnt/server/WMFactory
PYTHONNOUSERSITE=1 venvs/vllm-wm/bin/python vllm-wm/scripts/import_local_assets.py --include-data
```

## Standalone Scope

`vllm-wm` now contains the inference-time code surface for:

- Matrix-Game 2.0
- Matrix-Game 3.0
- YUME 1.5
- Diamond
- Open-Oasis
- WHAM
- Vid2World
- Infinite-World
- HY-WorldPlay 5B
- MineWorld

The vendored subset list is documented in [VENDORS.md](/mnt/server/WMFactory/vllm-wm/VENDORS.md:1).

## Why Workers Instead Of In-Process Imports

The first in-process prototype worked for model listing, but it broke once multiple upstream repos were imported together. Several model repos define top-level modules with the same names, especially `utils`, `main`, and related helper packages. After one model is imported, another can resolve against the wrong module tree and fail in non-obvious ways.

That is why `vllm-wm` now uses one orchestrator process plus one worker subprocess per model. The external API still looks unified like `vLLM`; the internal execution remains isolated enough to be reliable.

## Next Refactor Target

The current implementation is intentionally conservative: it runs project-local service runtimes through managed subprocess workers.

The next high-value refactor is:

1. move reusable runtime code out of `services/*` into importable `vllm_wm` runtime modules
2. keep `vendors/*` trimmed to inference-only code paths
3. selectively replace atomic `backend.step()` execution with native step-wise schedulers for diffusion-heavy models
