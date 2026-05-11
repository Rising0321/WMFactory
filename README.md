# WMFactory 0.5

One environment. One command. Eleven interactive world models.

`WMFactory 0.5` is a major update of the WMFactory project. Its main goal is to remove the old per-model environment fragmentation and replace it with a single backend, a single shared runtime environment, and a single serving interface for many different world models.

At the project level, `WMFactory` is the full repository. The current backend implementation lives in `vllm-wm/`.

## Project Goal

This repository is built around one practical promise:

- 1 shared environment
- 1 backend entrypoint
- 1 consistent session API
- 11 different interactive world models

The focus is not to force every model into the same architecture. The focus is to make them usable from one unified system.


## Unified Environment

The backend is designed around one shared Python environment.

Recommended stack:

- Python `3.12`
- PyTorch `2.9.0`
- Transformers `4.57.3`
- Diffusers `0.37.1`

Recommended install(Make sure the python version is 3.12):

```bash
python -m pip install -r requirements.txt
```

`flash-attn` is required. If the normal pip install fails, install the matching Dao-AILab wheel manually, then continue.

## Quick Start

### Start the backend

```bash
cd vllm-wm
python serve.py
```

### Load one model

```bash
curl -X POST http://127.0.0.1:9100/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"matrixgame"}'
```

### Start a session

```bash
curl -X POST http://127.0.0.1:9100/sessions/start \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"matrixgame","init_image_base64":"data:image/png;base64,..."}'
```

### Step the world

```bash
curl -X POST http://127.0.0.1:9100/sessions/step \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<session-id>","action":{"w":true}}'
```

Common action examples:

- forward: `{"w": true}`
- left: `{"a": true}`
- right: `{"d": true}`
- camera up: `{"camera_dy": -1.0}`
- camera right: `{"camera_dx": 1.0}`

The transport format is unified. Exact action semantics remain model-specific.

## Testing

Full rollout regression:

```bash
cd vllm-wm
PYTHONNOUSERSITE=1 python scripts/verify_demo_outputs.py
```

Each successful model rollout writes results into `vllm-wm/testOutput/<model>/`.

## Supported Models

The current backend covers eleven models.

| Model | Upstream Repository | Download Command |
| --- | --- | --- |
| `matrixgame` (Matrix-Game 2.0) | `https://github.com/SkyworkAI/Matrix-Game` | `huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir vllm-wm/checkpoints/matrixgame2` |
| `matrixgame3` (Matrix-Game 3.0) | `https://github.com/SkyworkAI/Matrix-Game-3.0` | `huggingface-cli download Skywork/Matrix-Game-3.0 --local-dir vllm-wm/checkpoints/matrixgame3` |
| `yume` (YUME 1.5) | `https://github.com/stdstu12/YUME` | `huggingface-cli download stdstu123/Yume-5B-720P --local-dir vllm-wm/checkpoints/yume/Yume-5B-720P`<br>`huggingface-cli download OpenGVLab/InternVL3-2B-Instruct --local-dir vllm-wm/checkpoints/yume/InternVL3-2B-Instruct` |
| `diamond` | `https://github.com/eloialonso/diamond` | `huggingface-cli download eloialonso/diamond --include "csgo/*" --local-dir vllm-wm/checkpoints/diamond` |
| `open-oasis` | `https://github.com/etched-ai/open-oasis` | `huggingface-cli download Etched/oasis-500m oasis500m.safetensors --local-dir vllm-wm/checkpoints/openoasis`<br>`huggingface-cli download Etched/oasis-500m vit-l-20.safetensors --local-dir vllm-wm/checkpoints/openoasis` |
| `wham` | `TODO` | `huggingface-cli download microsoft/WHAM models/WHAM_200M.ckpt --local-dir vllm-wm/checkpoints/wham` |
| `vid2world` | `https://github.com/thuml/Vid2World` | `huggingface-cli download thuml/Vid2World-CSGO --local-dir vllm-wm/checkpoints/vid2world/Vid2World-CSGO` |
| `infinite-world` | `TODO` | `huggingface-cli download MeiGen-AI/Infinite-World --local-dir vllm-wm/checkpoints/infiniteworld` |
| `worldplay` (HY-WorldPlay 5B) | `https://github.com/Tencent-Hunyuan/HY-WorldPlay` | `huggingface-cli download tencent/HY-WorldPlay --include "wan_transformer/*" --local-dir vllm-wm/checkpoints/worldplay/HY-WorldPlay`<br>`huggingface-cli download tencent/HY-WorldPlay --include "wan_distilled_model/model.pt" --local-dir vllm-wm/checkpoints/worldplay/HY-WorldPlay`<br>`huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir vllm-wm/checkpoints/worldplay/Wan2.2-TI2V-5B-Diffusers` |
| `mineworld` | `https://github.com/microsoft/mineworld` | `TODO: fill with the public Hugging Face checkpoint command once the release path is stable` |
| `lingbot-world-fast` | `https://github.com/robbyant/lingbot-world` | `huggingface-cli download robbyant/lingbot-world-base-cam --local-dir vllm-wm/checkpoints/lingbotworld/lingbot-world-base-cam`<br>`huggingface-cli download robbyant/lingbot-world-fast --local-dir vllm-wm/checkpoints/lingbotworld/lingbot-world-base-cam/lingbot_world_fast` |

## Acknowledgments

The unified backend design is inspired by [vLLM](https://github.com/vllm-project/vllm), adapted here for interactive world models rather than LLM token serving.

The implementation also builds on [nano-vllm-omni](https://github.com/Rising0321/nano-vllm-omni) and related discussion around unified multimodal runtime design.

The author of [OpenWorldLib](https://github.com/OpenDCAI/OpenWorldLib) for the discussion and inspiration.
