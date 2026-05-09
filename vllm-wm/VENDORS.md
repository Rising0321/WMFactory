# Vendored Inference Surface

`vllm-wm` now embeds the minimum code surface needed to run the current priority world models without depending on the outer WMFactory `models/*` and `services/*` directories.

## Vendored Code Roots

- `vendors/matrixgame2`
  - copied: `configs/`, `demo_utils/`, `pipeline/`, `utils/`, `wan/`
- `vendors/matrixgame3`
  - copied: `generate.py`, `pipeline/`, `utils/`, `wan/`
- `vendors/yume`
  - copied: `webapp_single_gpu.py`, `import_shim.py`, `wan/`, `wan23/`, `fastvideo/`
- `vendors/openoasis`
  - copied: root inference `.py` files only
- `vendors/wham`
  - copied: `wham/`, `configs/`
- `vendors/vid2world`
  - copied: `lvdm/`, `main/`, `utils/`, `configs/`, `csgo_utils/`, `nvm_utils/`
- `vendors/infiniteworld`
  - copied: `infworld/`, `configs/`
- `vendors/diamond`
  - copied: `src/`, `config/`
- `vendors/worldplay`
  - copied: `hyvideo/`, `wan/`
- `vendors/mineworld`
  - copied: `configs/`, `diagonal_decoding.py`, `inference.py`, `lvm.py`, `mcdataset.py`, `mineworld.py`, `utils.py`, `vae.py`
- `vendors/lingbotworldfast`
  - copied: fast-only `wan/` subset for single-GPU autoregressive inference
  - kept: `configs/`, `image2video_fast.py`, `modules/{attention,model,model_fast,t5,tokenizers,vae2_1}.py`, `utils/{cam_utils,fm_solvers_unipc,wasd_ijkl_to_c2ws}.py`
- `vendors/lingbotworldfast`
  - copied: `wan/` fast inference subset only

## Local Runtime Roots

- `services/*`
  - internal copies of the model runtime entrypoints now live here
  - each runtime defaults to the vendored code under `vendors/*`
  - checkpoint defaults point into `checkpoints/*`

## Asset Policy

- code is vendored directly into the project
- checkpoints are not committed, but the project now expects them under `checkpoints/*` or the documented env vars
- `scripts/import_local_assets.py` can import already-downloaded local assets from the outer WMFactory tree into this standalone layout
