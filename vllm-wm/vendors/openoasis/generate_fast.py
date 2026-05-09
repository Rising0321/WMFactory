"""
Optimized inference script for Oasis-500M (non-invasive: no model/weight changes).

Key runtime tweaks:
1) Use inference_mode + fp16 autocast + TF32 matmul/cudnn paths.
2) Remove unnecessary x.clone() in DDIM inner loop.
3) Reduce per-step allocations by reusing precomputed constants where possible.
"""

import argparse
import os
from pprint import pprint

import torch
from einops import rearrange
from safetensors.torch import load_model
from torch import autocast
from torchvision.io import write_video
from tqdm import tqdm

from dit import DiT_models
from utils import load_actions, load_prompt, sigmoid_beta_schedule
from vae import VAE_models

assert torch.cuda.is_available()
device = "cuda:0"


def configure_runtime():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_ckpt(model, ckpt_path):
    if ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif ckpt_path.endswith(".safetensors"):
        load_model(model, ckpt_path)
    else:
        raise ValueError(f"unsupported checkpoint format: {ckpt_path}")


def main(args):
    configure_runtime()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = DiT_models["DiT-S/2"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    load_ckpt(model, args.oasis_ckpt)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    load_ckpt(vae, args.vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    # Keep CPU list of scalar noise ids, move to device only when composing t.
    noise_ids = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).long().tolist()
    noise_abs_max = 20
    stabilization_level = 15

    # get prompt image/video
    x = load_prompt(
        args.prompt_path,
        video_offset=args.video_offset,
        n_prompt_frames=n_prompt_frames,
    )
    actions = load_actions(args.actions_path, action_offset=args.video_offset)[:, :total_frames]

    x = x.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)

    with torch.inference_mode():
        # vae encoding
        bsz = x.shape[0]
        h, w = x.shape[-2:]
        scaling_factor = 0.07843137255
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with autocast("cuda", dtype=torch.half):
            x = vae.encode(x * 2 - 1).mean * scaling_factor
        x = rearrange(x, "(b t) (hh ww) c -> b t c hh ww", t=n_prompt_frames, hh=h // vae.patch_size, ww=w // vae.patch_size)
        x = x[:, :n_prompt_frames]

        # diffusion schedule
        betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = rearrange(alphas_cumprod, "t -> t 1 1 1")

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames)):
            chunk = torch.randn((bsz, 1, *x.shape[-3:]), device=device)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)
            # Reuse context timestep for current i.
            t_ctx = torch.full((bsz, i), stabilization_level - 1, dtype=torch.long, device=device)

            for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                t_scalar = noise_ids[noise_idx]
                t_next_scalar = noise_ids[noise_idx - 1]
                if t_next_scalar < 0:
                    t_next_scalar = t_scalar

                t = torch.full((bsz, 1), t_scalar, dtype=torch.long, device=device)
                t_next = torch.full((bsz, 1), t_next_scalar, dtype=torch.long, device=device)
                t = torch.cat([t_ctx, t], dim=1)[:, start_frame:]
                t_next = torch.cat([t_ctx, t_next], dim=1)[:, start_frame:]

                # No clone needed; only final target frame is overwritten after prediction.
                x_curr = x[:, start_frame:]

                with autocast("cuda", dtype=torch.half):
                    v = model(x_curr, t, actions[:, start_frame : i + 1])

                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        # vae decoding
        x = rearrange(x, "b t c h w -> (b t) (h w) c")
        x = (vae.decode(x / scaling_factor) + 1) / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    write_video(args.output_path, x[0].cpu(), fps=args.fps)
    print(f"generation saved to {args.output_path}.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--oasis-ckpt", type=str, default="oasis500m.safetensors", help="Path to Oasis DiT checkpoint.")
    parse.add_argument("--vae-ckpt", type=str, default="vit-l-20.safetensors", help="Path to Oasis ViT-VAE checkpoint.")
    parse.add_argument("--num-frames", type=int, default=32, help="How many frames should the output be?")
    parse.add_argument(
        "--prompt-path",
        type=str,
        default="sample_data/sample_image_0.png",
        help="Path to image/video to condition generation on.",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        default="sample_data/sample_actions_0.one_hot_actions.pt",
        help="File to load actions from (.actions.pt or .one_hot_actions.pt).",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        default=None,
        help="If loading prompt from video, index of frame to start reading from.",
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        default=1,
        help="If the prompt is a video, how many frames to condition on.",
    )
    parse.add_argument("--output-path", type=str, default="video_fast.mp4", help="Path where generated video should be saved.")
    parse.add_argument("--fps", type=int, default=20, help="Framerate for output video.")
    parse.add_argument("--ddim-steps", type=int, default=10, help="How many DDIM steps?")
    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
