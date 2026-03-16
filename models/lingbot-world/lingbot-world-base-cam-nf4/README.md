---
license: apache-2.0
library_name: pytorch
tags:
  - video-generation
  - image-to-video
  - diffusion
  - quantized
  - nf4
  - bitsandbytes
pipeline_tag: image-to-video
---

# LingBot-World NF4 Quantized

Pre-quantized NF4 weights for LingBot-World video generation model. This is a complete, self-contained package - no additional downloads required.

## Features

- **4-bit NF4 quantization** via bitsandbytes - fits in 32GB VRAM
- **Pre-quantized weights** - no runtime quantization overhead
- **Complete package** - includes T5 encoder, VAE, and diffusion models

## Quick Start

```bash
# Clone the repo
git clone https://huggingface.co/cahlen/lingbot-world-base-cam-nf4
cd lingbot-world-base-cam-nf4

# Install dependencies
pip install -r requirements.txt

# Generate a video
python generate_prequant.py \
    --image your_image.jpg \
    --prompt "A cinematic video of the scene" \
    --frame_num 81 \
    --output output.mp4
```

## Model Contents

| File | Size | Description |
|------|------|-------------|
| `high_noise_model_bnb_nf4/model.safetensors` | ~9.6GB | NF4 quantized diffusion model (high noise) |
| `low_noise_model_bnb_nf4/model.safetensors` | ~9.6GB | NF4 quantized diffusion model (low noise) |
| `models_t5_umt5-xxl-enc-bf16.pth` | ~10.6GB | T5-XXL text encoder |
| `Wan2.1_VAE.pth` | ~485MB | VAE encoder/decoder |

**Total size: ~30GB** (vs ~85GB for full precision models)

## Usage

### Basic Generation

```bash
python generate_prequant.py \
    --image input.jpg \
    --prompt "Your prompt here" \
    --frame_num 81 \
    --size "480*832" \
    --output output.mp4
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | required | Input image path |
| `--prompt` | required | Text prompt describing the video |
| `--frame_num` | 81 | Number of frames (81 = ~5 seconds at 16fps) |
| `--size` | "480*832" | Output resolution (height*width) |
| `--sampling_steps` | 40 | Diffusion sampling steps |
| `--guide_scale` | 5.0 | Classifier-free guidance scale |
| `--seed` | -1 | Random seed (-1 for random) |
| `--output` | "output.mp4" | Output video path |

### With Camera Control

```bash
python generate_prequant.py \
    --image input.jpg \
    --prompt "Your prompt" \
    --action_path /path/to/camera_poses/ \
    --frame_num 81
```

Camera pose directory should contain:
- `poses.npy`: Shape `[num_frames, 4, 4]` - camera transformation matrices
- `intrinsics.npy`: Shape `[num_frames, 4]` - `[fx, fy, cx, cy]`

## Requirements

- Python 3.10+
- CUDA 11.8+ (tested with CUDA 12.x)
- ~32GB VRAM (RTX 4090, RTX 5090, A100, etc.)

## Quantization Details

The diffusion models are quantized using bitsandbytes NF4 with double quantization:

```json
{
  "format": "bnb_nf4",
  "double_quant": true,
  "compute_dtype": "bfloat16",
  "blocksize": 64
}
```

This achieves ~3.9x compression while maintaining generation quality.

## License

This model is based on [LingBot-World](https://github.com/robbyant/lingbot-world) and follows its license terms.

## Citation

```bibtex
@misc{lingbot-world-nf4,
  title={LingBot-World NF4 Quantized},
  year={2025},
  url={https://huggingface.co/cahlen/lingbot-world-base-cam-nf4}
}
```
