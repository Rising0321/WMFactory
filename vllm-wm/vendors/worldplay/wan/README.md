# WAN Pipeline Inference

The WAN pipeline provides a lightweight alternative with distributed inference support. It's optimized for multi-GPU setups and offers faster inference with lower memory footprint.

## Download WAN Models

Download the required WAN models from HuggingFace:

```bash
# Download WAN transformer model
huggingface-cli download tencent/HY-WorldPlay wan_transformer --local-dir /path/to/models/wan_transformer

# Download WAN distilled model checkpoint
huggingface-cli download tencent/HY-WorldPlay wan_distilled_model/model.pt --local-dir /path/to/models/wan_distilled_model
```

## Configuration Parameters

The WAN pipeline accepts the following parameters you need to modified:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Text prompt or path to txt file with prompts | Required |
| `--image_path` | Input image path for I2V generation | None |
| `--num_chunk` | Number of chunks (each chunk = 16 frames) | 4 |
| `--pose` | Camera trajectory (pose string or JSON file) | `w-96` |
| `--ar_model_path` | Path to WAN transformer model | Required |
| `--ckpt_path` | Path to trained checkpoint (model.pt) | Required |
| `--out` | Output directory for generated videos | `outputs` |

## Multi-GPU Distributed Inference

For multi-GPU inference with better performance:

```bash
# Using 4 GPUs
torchrun --nproc_per_node=4 wan/generate.py \
  --input "First-person view walking around ancient Athens, with Greek architecture and marble structures" \
  --image_path /path/to/input/image.jpg \
  --num_chunk 4 \
  --pose "w-96" \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt \
  --out outputs
```

## Batch Processing with Text Files

You can process multiple prompts by providing a text file:

```bash
# Create a prompts.txt file with one prompt per line
echo "First-person view of a medieval castle" > prompts.txt
echo "Walking through a cyberpunk city at night" >> prompts.txt
echo "Exploring an underwater coral reef" >> prompts.txt

# Run inference on all prompts
python wan/generate.py \
  --input prompts.txt \
  --ar_model_path /path/to/models/wan_transformer \
  --ckpt_path /path/to/models/wan_distilled_model/model.pt
```

## Camera Control with WAN

WAN uses the same camera control system as the HunyuanVideo pipeline:

**Pose String Format:**
```bash
# Forward movement for 96 latents (generates 961 frames with num_chunk=4)
--pose "w-96"

# Complex trajectory
--pose "w-20, right-10, d-30, up-36"
```

**Supported Actions:**
- **Movement**: `w` (forward), `s` (backward), `a` (left), `d` (right)
- **Rotation**: `up` (pitch up), `down` (pitch down), `left` (yaw left), `right` (yaw right)
- **Format**: `action-duration` where duration specifies the number of latents corresponding to the given action.
