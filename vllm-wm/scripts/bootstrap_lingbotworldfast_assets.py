from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEST_ROOT = PROJECT_ROOT / "checkpoints" / "lingbotworld" / "lingbot-world-base-cam"
SHARED_SRC_ROOT = REPO_ROOT / "models" / "lingbot-world" / "lingbot-world-base-act"

SHARED_ITEMS = (
    ("models_t5_umt5-xxl-enc-bf16.pth", False),
    ("Wan2.1_VAE.pth", False),
    ("google/umt5-xxl", True),
)


def _link_or_copy(src: Path, dst: Path, mode: str, is_dir: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] {dst}")
        return
    if is_dir:
        if mode == "hardlink":
            shutil.copytree(src, dst, copy_function=os.link, symlinks=False)
        elif mode == "symlink":
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst, symlinks=False)
    else:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
    print(f"[ok] {src} -> {dst}")


def materialize_shared_assets(mode: str) -> None:
    for rel, is_dir in SHARED_ITEMS:
        src = SHARED_SRC_ROOT / rel
        dst = DEST_ROOT / rel
        if not src.exists():
            print(f"[missing] {src}")
            continue
        _link_or_copy(src, dst, mode=mode, is_dir=is_dir)


def _base_env(use_proxy: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_ENDPOINT"] = os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com")
    if use_proxy:
        return env
    env["http_proxy"] = ""
    env["https_proxy"] = ""
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    return env


def download_fast_weights() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    fast_dir = DEST_ROOT / "lingbot_world_fast"
    fast_dir.mkdir(parents=True, exist_ok=True)

    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_bin is None:
        raise RuntimeError("Neither `hf` nor `huggingface-cli` is available in PATH.")

    cmd = [hf_bin, "download", "robbyant/lingbot-world-fast", "--local-dir", str(fast_dir)]
    attempts = [False]
    if os.getenv("http_proxy") or os.getenv("https_proxy") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY"):
        attempts.append(True)

    last_error: Exception | None = None
    for use_proxy in attempts:
        env = _base_env(use_proxy=use_proxy)
        print(f"[download] use_proxy={use_proxy} cmd={' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
            return
        except Exception as exc:  # pragma: no cover - shell failure surface
            last_error = exc
            print(f"[failed] use_proxy={use_proxy}: {exc}")
    raise RuntimeError(f"Failed to download lingbot-world-fast weights: {last_error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap standalone assets for LingBot-World-Fast.")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize shared T5/VAE/tokenizer assets from the local WMFactory tree.",
    )
    parser.add_argument(
        "--skip-shared",
        action="store_true",
        help="Do not materialize shared base assets from the local WMFactory tree.",
    )
    parser.add_argument(
        "--download-fast",
        action="store_true",
        help="Download the official `robbyant/lingbot-world-fast` transformer weights into checkpoints.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.skip_shared:
        materialize_shared_assets(mode=args.mode)
    if args.download_fast:
        download_fast_weights()


if __name__ == "__main__":
    main()
