from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINGBOTFAST_ROOT = PROJECT_ROOT / "vendors" / "lingbotworldfast"

if str(LINGBOTFAST_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOTFAST_ROOT))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent LingBot-World-Fast distributed worker.")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--request_dir", required=True)
    return parser.parse_args()


def _init_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _load_next_command(request_dir: Path) -> dict:
    while True:
        candidates = sorted(request_dir.glob("request_*.json"))
        if candidates:
            path = candidates[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                path.unlink(missing_ok=True)
                return payload
            except Exception as exc:
                request_id = path.stem.removeprefix("request_")
                response_path = request_dir / f"response_{request_id}.json"
                response_path.write_text(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"failed to read request {path}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                path.unlink(missing_ok=True)
        time.sleep(0.2)


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).resolve()
    request_dir.mkdir(parents=True, exist_ok=True)

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
    logging.info(
        "loading persistent LingBot-World-Fast worker rank=%s world_size=%s local_rank=%s",
        rank,
        world_size,
        local_rank,
    )
    pipeline = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank if world_size > 1 else 0,
        rank=rank,
        t5_fsdp=world_size > 1,
        dit_fsdp=world_size > 1,
        use_sp=world_size > 1,
        t5_cpu=False,
        convert_model_dtype=False,
    )

    if rank == 0:
        ready_path = request_dir / "ready.json"
        ready_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "world_size": world_size,
                    "loaded_at": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logging.info("persistent worker ready request_dir=%s", request_dir)

    while True:
        payload_list: list[dict | None] = [None]
        if rank == 0:
            payload_list[0] = _load_next_command(request_dir)
        if dist.is_initialized():
            dist.broadcast_object_list(payload_list, src=0)
        payload = payload_list[0]
        if payload is None:
            continue
        if payload.get("type") == "shutdown":
            break

        request_id = str(payload["request_id"])
        response_path = request_dir / f"response_{request_id}.json"
        try:
            image = Image.open(payload["image_path"]).convert("RGB")
            video = pipeline.generate(
                input_prompt=payload["prompt"],
                img=image,
                action_path=payload["action_path"],
                chunk_size=3,
                max_area=MAX_AREA_CONFIGS[payload["size"]],
                frame_num=int(payload["frame_num"]),
                shift=float(payload["shift"]),
                seed=int(payload["seed"]),
                offload_model=bool(payload["offload_model"]),
                max_attention_size=payload.get("max_attention_size"),
            )
            if rank == 0:
                output_path = Path(payload["output_path"])
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
                response_path.write_text(
                    json.dumps({"ok": True, "output_path": str(output_path)}, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            if rank == 0:
                response_path.write_text(
                    json.dumps(
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        finally:
            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
