from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINGBOTFAST_ROOT = PROJECT_ROOT / "vendors" / "lingbotworldfast"

if str(LINGBOTFAST_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOTFAST_ROOT))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def _str2bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed one-shot LingBot-World-Fast inference wrapper.")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frame_num", type=int, default=13)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload_model", type=_str2bool, default=False)
    parser.add_argument("--t5_cpu", type=_str2bool, default=False)
    parser.add_argument("--convert_model_dtype", type=_str2bool, default=False)
    parser.add_argument("--max_attention_size", type=int, default=None)
    return parser.parse_args()


def _init_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def main() -> None:
    args = _parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    _init_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        init_distributed_group()
    else:
        torch.cuda.set_device(0)

    cfg = WAN_CONFIGS["i2v-A14B"]
    image = Image.open(args.image).convert("RGB")

    logging.info(
        "starting fast inference rank=%s world_size=%s local_rank=%s size=%s frame_num=%s",
        rank,
        world_size,
        local_rank,
        args.size,
        args.frame_num,
    )

    pipeline = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank if world_size > 1 else 0,
        rank=rank,
        t5_fsdp=world_size > 1,
        dit_fsdp=world_size > 1,
        use_sp=world_size > 1,
        t5_cpu=args.t5_cpu if world_size == 1 else False,
        convert_model_dtype=args.convert_model_dtype,
    )

    video = pipeline.generate(
        input_prompt=args.prompt,
        img=image,
        action_path=args.action_path,
        chunk_size=3,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.shift,
        seed=args.seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size,
    )

    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("saving video to %s", output_path)
        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("finished")


if __name__ == "__main__":
    main()
