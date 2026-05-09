from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class CopyItem:
    name: str
    src: Path
    dst: Path
    optional: bool = False
    is_dir: bool = True


def build_items(include_data: bool) -> list[CopyItem]:
    items = [
        CopyItem(
            name="matrixgame2-checkpoints",
            src=REPO_ROOT / "models" / "Matrix-Game" / "checkpoints" / "Matrix-Game-2.0",
            dst=PROJECT_ROOT / "checkpoints" / "matrixgame2",
        ),
        CopyItem(
            name="matrixgame3-checkpoints",
            src=REPO_ROOT / "models" / "Matrix-Game-3.0" / "Matrix-Game-3.0",
            dst=PROJECT_ROOT / "checkpoints" / "matrixgame3",
        ),
        CopyItem(
            name="yume-main-checkpoint",
            src=REPO_ROOT / "models" / "YUME" / "Yume-5B-720P",
            dst=PROJECT_ROOT / "checkpoints" / "yume" / "Yume-5B-720P",
        ),
        CopyItem(
            name="yume-internvl",
            src=REPO_ROOT / "models" / "YUME" / "InternVL3-2B-Instruct",
            dst=PROJECT_ROOT / "checkpoints" / "yume" / "InternVL3-2B-Instruct",
        ),
        CopyItem(
            name="openoasis-dit",
            src=REPO_ROOT / "models" / "open-oasis" / "oasis500m.safetensors",
            dst=PROJECT_ROOT / "checkpoints" / "openoasis" / "oasis500m.safetensors",
            is_dir=False,
        ),
        CopyItem(
            name="openoasis-vae",
            src=REPO_ROOT / "models" / "open-oasis" / "vit-l-20.safetensors",
            dst=PROJECT_ROOT / "checkpoints" / "openoasis" / "vit-l-20.safetensors",
            is_dir=False,
        ),
        CopyItem(
            name="wham-200m",
            src=REPO_ROOT / "models" / "wham" / "models" / "WHAM_200M.ckpt",
            dst=PROJECT_ROOT / "checkpoints" / "wham" / "WHAM_200M.ckpt",
            is_dir=False,
        ),
        CopyItem(
            name="vid2world-checkpoints",
            src=REPO_ROOT / "models" / "Vid2World" / "checkpoints" / "Vid2World-CSGO",
            dst=PROJECT_ROOT / "checkpoints" / "vid2world" / "Vid2World-CSGO",
        ),
        CopyItem(
            name="infiniteworld-checkpoints",
            src=REPO_ROOT / "models" / "Infinite-World" / "checkpoints",
            dst=PROJECT_ROOT / "checkpoints" / "infiniteworld",
        ),
        CopyItem(
            name="worldplay-distilled-checkpoints",
            src=REPO_ROOT
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--tencent--HY-WorldPlay"
            / "snapshots"
            / "f4c29235647707b571479a69b569e4166f9f5bf8",
            dst=PROJECT_ROOT / "checkpoints" / "worldplay" / "HY-WorldPlay",
            optional=True,
        ),
        CopyItem(
            name="worldplay-base-model",
            src=REPO_ROOT
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
            / "snapshots"
            / "b8fff7315c768468a5333511427288870b2e9635",
            dst=PROJECT_ROOT / "checkpoints" / "worldplay" / "Wan2.2-TI2V-5B-Diffusers",
            optional=True,
        ),
        CopyItem(
            name="mineworld-checkpoints",
            src=REPO_ROOT / "mnt" / "mineworld" / "checkpoints",
            dst=PROJECT_ROOT / "checkpoints" / "mineworld",
            optional=True,
        ),
        CopyItem(
            name="mineworld-gradio-scene",
            src=REPO_ROOT / "mnt" / "mineworld" / "gradio_scene",
            dst=PROJECT_ROOT / "data" / "mineworld" / "gradio_scene",
            optional=True,
        ),
        CopyItem(
            name="diamond-checkpoints",
            src=REPO_ROOT / "models" / "diamond" / "csgo",
            dst=PROJECT_ROOT / "checkpoints" / "diamond" / "csgo",
            optional=True,
        ),
        CopyItem(
            name="lingbotworldfast-t5",
            src=REPO_ROOT / "models" / "lingbot-world" / "lingbot-world-base-act" / "models_t5_umt5-xxl-enc-bf16.pth",
            dst=PROJECT_ROOT / "checkpoints" / "lingbotworld" / "lingbot-world-base-cam" / "models_t5_umt5-xxl-enc-bf16.pth",
            optional=True,
            is_dir=False,
        ),
        CopyItem(
            name="lingbotworldfast-vae",
            src=REPO_ROOT / "models" / "lingbot-world" / "lingbot-world-base-act" / "Wan2.1_VAE.pth",
            dst=PROJECT_ROOT / "checkpoints" / "lingbotworld" / "lingbot-world-base-cam" / "Wan2.1_VAE.pth",
            optional=True,
            is_dir=False,
        ),
        CopyItem(
            name="lingbotworldfast-tokenizer",
            src=REPO_ROOT / "models" / "lingbot-world" / "lingbot-world-base-act" / "google" / "umt5-xxl",
            dst=PROJECT_ROOT / "checkpoints" / "lingbotworld" / "lingbot-world-base-cam" / "google" / "umt5-xxl",
            optional=True,
        ),
        CopyItem(
            name="matrixgame-mouse-icon",
            src=REPO_ROOT / "models" / "Matrix-Game-3.0" / "assets" / "images" / "mouse.png",
            dst=PROJECT_ROOT / "assets" / "images" / "mouse.png",
            optional=True,
            is_dir=False,
        ),
    ]
    if include_data:
        items.append(
            CopyItem(
                name="vid2world-data-csgo",
                src=REPO_ROOT / "data" / "csgo_processed_min" / "full_res",
                dst=PROJECT_ROOT / "data" / "csgo_processed_min" / "full_res",
            )
        )
    return items


def _link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    shutil.copy2(src, dst)


def _link_or_copy_dir(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link, symlinks=False)
        return
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
        return
    shutil.copytree(src, dst, symlinks=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import already-downloaded checkpoints/data into standalone vllm-wm.")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize assets inside vllm-wm. hardlink is space-efficient and still standalone enough for read-only checkpoints.",
    )
    parser.add_argument("--include-data", action="store_true", help="Also import large dataset assets such as Vid2World CSGO history.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be imported.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items = build_items(include_data=args.include_data)

    for item in items:
        exists = item.src.exists()
        status = "missing"
        if exists:
            status = "present"
        elif item.optional:
            status = "optional-missing"
        print(f"[{status}] {item.name}: {item.src} -> {item.dst}")
        if args.dry_run or not exists:
            continue
        if item.is_dir:
            _link_or_copy_dir(item.src, item.dst, args.mode)
        else:
            _link_or_copy_file(item.src, item.dst, args.mode)


if __name__ == "__main__":
    main()
