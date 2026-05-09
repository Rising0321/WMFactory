# Matrix-Game-3.0

- Upstream source family: `SkyworkAI/Matrix-Game`
- Expected upstream path: `Matrix-Game-3` / `Matrix-Game-3.0`
- Local integration target: `models/Matrix-Game-3.0`

## Summary

This WMFactory integration keeps the Matrix-Game-3.0 source code required for inference, but excludes the original repository's demo assets, local outputs, downloaded checkpoints, and runtime cache files.

The local WMFactory-facing integration also adds:

- `services/matrixgame3/app.py`
- `frontend/adapters/matrixgame3_adapter.py`
- gateway registration in `frontend/server.py`
- frontend model handling in `frontend/web/app.js`

## Local Matrix-Game-3.0 Source Files Kept

- `LICENSE.txt`
- `generate.py`
- `pipeline/inference_interactive_pipeline.py`
- `pipeline/inference_pipeline.py`
- `pipeline/vae_config.py`
- `pipeline/vae_worker.py`
- `requirements.txt`
- `utils/cam_utils.py`
- `utils/conditions.py`
- `utils/misc.py`
- `utils/transform.py`
- `utils/utils.py`
- `utils/visualize.py`
- `wan/**` source files

## Original Repository Content Intentionally Excluded

- `README.md`
- `assets/`
- `demo_images/`
- `output/`
- `Matrix-Game-3.0/` downloaded checkpoint directory
- `test.sh`
- all `__pycache__/` and local cache files

## Integration-Facing Code Changes

Based on the local import timeline, the main Matrix-Game-3.0 source files touched during integration are:

- `pipeline/inference_pipeline.py`
- `pipeline/inference_interactive_pipeline.py`
- `utils/visualize.py`
- `wan/distributed/ulysses.py`
- `wan/modules/model.py`

These are the files most directly associated with WMFactory-oriented runtime adaptation, interactive inference, and chunked generation behavior.
