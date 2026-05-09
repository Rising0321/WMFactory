# vllm-wm

## Project Intent

`vllm-wm` is a standalone source project for interactive world models.

The target shape is:

- one unified backend
- one unified environment
- one unified session API
- one local codebase that can run after users place assets under `checkpoints/*` and `data/*`

At runtime, `vllm-wm` should not need the outer `WMFactory/models/*` or `WMFactory/services/*` trees.

## Core Contract

Every integrated model must fit the same outer loop:

1. `load`
2. `start_session` from an image
3. `step` from an action
4. return the next frame and metadata

The backend is unified at the API level, not at the neural architecture level.

## Layout Rules

- `vllm_wm/`: orchestrator, registry, shared backend logic
- `services/*`: per-model runtime entrypoints launched as managed workers
- `vendors/*`: vendored inference-time upstream code only
- `checkpoints/*`: all model weights and local model assets
- `data/*`: datasets or runtime media assets required by specific models
- `testOutput/*`: regression outputs for real end-to-end validation

Default paths for new integrations must resolve inside this folder.

## Current Model Inventory

The standalone tree currently integrates these models:

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
- `lingbot-world-fast``

## Environment Policy

This project prefers one modern shared environment over many isolated legacy envs.

Current design direction:

- Python `3.12`
- recent `torch`
- recent `transformers`
- recent `diffusers`
- `flash-attn` installed and working

When changing the shared environment, treat it as a backend-wide migration:

- do not optimize for one model and silently break others
- run regression checks after dependency changes
- prefer small compatibility patches over reintroducing per-model env fragmentation
- preserve memory-critical runtime defaults for heavy models unless you revalidate them end to end

## Worker Model

`vllm-wm` uses one orchestrator plus one worker subprocess per model.

This is intentional because upstream repos collide on names like `utils`, `main`, and other top-level modules. Process isolation is the practical replacement for trying to import all models in one Python interpreter.

The design goal is:

- unified serving surface like `vLLM`
- isolated model runtime state like separate apps

## State And Cache Model

Do not assume every world model uses the same KV-cache abstraction.

In this project, the unified backend manages sessions, while each model keeps its own native state:

- `matrixgame`: explicit Transformer KV cache plus VAE cache
- `matrixgame3`: process-resident interactive state inside a persistent subprocess
- `yume`: rolling latent continuation state
- `open-oasis`: latent history plus action history
- `wham`: context image/action/token history
- `vid2world`: video history plus action history
- `infinite-world`: latent history plus cached text-conditioning state
- `diamond`: environment observation/action buffers
- `worldplay`: rolling latent history plus WAN pipeline context
- `mineworld`: frame/action token cache with explicit KV refresh points
- `lingbot-world-fast`: per-call causal KV cache over latent chunks, with session continuity handled by seed-frame rollover plus pose synthesis

The right abstraction here is unified session state management, not a fake one-size-fits-all KV cache class.

## Standalone Asset Policy

Users should be able to:

1. clone or copy `vllm-wm`
2. prepare the documented environment
3. place weights under `checkpoints/*`
4. place required runtime data under `data/*`
5. serve models from this folder alone

For that reason:

- new services must default to `checkpoints/*` and `data/*`
- avoid defaulting to outer repo cache paths
- keep import scripts updated when new models are added

## Vendoring Policy

Do not vendor full upstream repos blindly.

Preferred rule:

- keep only inference-critical code
- exclude training, demos, notebooks, and irrelevant tooling unless runtime import paths require them
- if a model needs a broad subtree for runtime imports, document why

When patching vendored code:

- keep patches small and local
- preserve upstream behavior unless a compatibility or standalone-path fix is required

## Regression Policy

There are two levels of validation:

1. service boot validation
2. real rollout validation

Service boot validation means:

- worker starts
- `/health` is reachable

Real rollout validation means:

- `load`
- feed the correct demo image for the model domain
- run 3 action steps: forward, left, right
- save outputs to `testOutput/<model>/`

Current image rule:

- Minecraft-like models use `demoImage/mc1.png`
- CSGO models use `demoImage/csgo.png`
- bleeding-edge / bleeding domain models use `demoImage/bleeding.png`
- all other models use `demoImage/real.png`

Every successful regression folder should contain:

- `start.png`
- `step1_forward.png`
- `step2_left.png`
- `step3_right.png`
- `meta.json`

## Integration Checklist

When adding a new model:

1. vendor the minimum inference code into `vendors/*`
2. add a local runtime in `services/*`
3. make default paths standalone under `checkpoints/*` and `data/*`
4. register the model in `vllm_wm/registry.py`
5. update `scripts/import_local_assets.py` if local assets already exist in the outer repo
6. update `scripts/verify_demo_outputs.py`
7. run at least health-check validation, then full 3-step rollout validation when GPU capacity allows

## Operational Notes

- GPU availability is often the main blocker, not code correctness.
- Some models prefer or require two visible GPUs; do not collapse those code paths unless you revalidate them.
- If a model only works with project-local compatibility patches, keep the patch in-tree and document it in commit notes or model-specific comments.
- `worldplay` is especially sensitive to decode-time VRAM peaks; its standalone runtime currently relies on VAE tiling plus temporary offload of inactive modules before decode.
- If you change `worldplay` VAE, device placement, or shared `diffusers` behavior, rerun the full 3-step rollout instead of trusting `/health`.
- `lingbot-world-fast` uses a split checkpoint layout: shared base assets (`models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`, `google/umt5-xxl`) plus a separate `lingbot_world_fast/` transformer directory.
