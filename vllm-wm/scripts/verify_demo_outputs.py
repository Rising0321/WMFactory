from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from vllm_wm.registry import build_backend, get_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent / "demoImage"
OUTPUT_ROOT = PROJECT_ROOT / "testOutput"


MODEL_IMAGE_TYPE = {
    "matrixgame": "real",
    "matrixgame3": "real",
    "yume": "real",
    "diamond": "csgo",
    "open-oasis": "mc",
    "wham": "bleeding",
    "vid2world": "csgo",
    "infinite-world": "real",
    "worldplay": "real",
    "mineworld": "mc",
    "lingbot-world-fast": "real",
}


DEMO_IMAGES = {
    "real": DEMO_ROOT / "real.png",
    "csgo": DEMO_ROOT / "csgo.png",
    "bleeding": DEMO_ROOT / "bleeding.png",
    "mc": DEMO_ROOT / "mc1.png",
}


MODEL_ENVS = {
    "matrixgame": {
        "WM_MATRIXGAME_ENABLE_COMPILE": "0",
        "WM_MATRIXGAME_WARMUP_ON_START": "0",
    },
    "matrixgame3": {
        "WM_MATRIXGAME3_NUM_ITERATIONS": "3",
        "WM_MATRIXGAME3_NUM_INFERENCE_STEPS": "8",
        "WM_MATRIXGAME3_START_TIMEOUT": "3600",
        "WM_MATRIXGAME3_STEP_TIMEOUT": "3600",
    },
    "yume": {
        "WM_YUME_SAMPLE_STEPS": "4",
    },
    "diamond": {},
    "open-oasis": {},
    "wham": {},
    "vid2world": {
        "WM_VID2WORLD_DATA_DIR": str(PROJECT_ROOT / "data" / "csgo_processed_min" / "full_res"),
        "WM_VID2WORLD_DDIM_STEPS": "50",
        "WM_VID2WORLD_TIMESTEP_SPACING": "uniform_trailing",
    },
    "infinite-world": {
        "WM_INFINITEWORLD_WARMUP_ON_START": "0",
        "WM_INFINITEWORLD_NUM_SAMPLING_STEPS": "4",
        "WM_INFINITEWORLD_DECODE_WINDOW_LATENT": "4",
        "WM_INFINITEWORLD_DECODE_STRIDE_LATENT": "3",
        "WM_INFINITEWORLD_MAX_COND_LATENT_FRAMES": "21",
    },
    "worldplay": {
        "WM_WORLDPLAY_NUM_INFERENCE_STEPS": "8",
        "WM_WORLDPLAY_MAX_CHUNKS": "4",
        "WM_WORLDPLAY_LOAD_TIMEOUT": "3600",
        "WM_WORLDPLAY_START_TIMEOUT": "3600",
        "WM_WORLDPLAY_STEP_TIMEOUT": "3600",
        "WM_WORLDPLAY_AUX_DEVICE": "cuda:1",
        "WM_WORLDPLAY_VAE_DEVICE": "cuda:1",
        "WM_WORLDPLAY_DECODE_VAE_DEVICE": "cuda:0",
    },
    "mineworld": {
        "WM_MINEWORLD_LOAD_TIMEOUT": "1800",
        "WM_MINEWORLD_START_TIMEOUT": "300",
        "WM_MINEWORLD_STEP_TIMEOUT": "300",
    },
    "lingbot-world-fast": {
        "WM_LINGBOTWORLDFAST_LOAD_TIMEOUT": "3600",
        "WM_LINGBOTWORLDFAST_START_TIMEOUT": "300",
        "WM_LINGBOTWORLDFAST_STEP_TIMEOUT": "1800",
        "WM_LINGBOTWORLDFAST_SIZE": "480*832",
        "WM_LINGBOTWORLDFAST_FRAME_NUM": "9",
        "WM_LINGBOTWORLDFAST_SHIFT": "10.0",
        "WM_LINGBOTWORLDFAST_NUM_PROCS": "4",
        "WM_LINGBOTWORLDFAST_T5_CPU": "0",
        "WM_LINGBOTWORLDFAST_OFFLOAD_MODEL": "0",
    },
}


STEP_SPECS = (
    ("step1_forward", {"w": True}),
    ("step2_left", {"a": True}),
    ("step3_right", {"d": True}),
)


def encode_image(path: Path) -> str:
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")


def decode_frame(payload: str) -> Image.Image:
    raw = payload.split(",", 1)[1] if "," in payload else payload
    data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert("RGB")


def gpu_state() -> list[str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]


@contextmanager
def patched_env(mapping: dict[str, str]):
    old = {key: os.environ.get(key) for key in mapping}
    try:
        for key, value in mapping.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_model(model_id: str) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / model_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_type = MODEL_IMAGE_TYPE[model_id]
    image_path = DEMO_IMAGES[image_type]
    image_payload = encode_image(image_path)
    seed_meta: dict[str, Any] | None = None

    backend = None
    started_at = time.perf_counter()
    timings: dict[str, float] = {}
    frames: list[dict[str, Any]] = []
    try:
        with patched_env(MODEL_ENVS[model_id]):
            backend = build_backend(model_id)

            t0 = time.perf_counter()
            load_payload = backend.load()
            timings["load_s"] = round(time.perf_counter() - t0, 3)

            spec = get_model_spec(model_id)
            if spec.supports_seed_meta and spec.supports_random_dataset and spec.default_dataset_ids:
                dataset_payload = backend.random_dataset_image(spec.default_dataset_ids[0])
                image_payload = dataset_payload["image_base64"]
                seed_meta = dataset_payload.get("extra", {}).get("seed_meta")
                image_path = Path(str(seed_meta["file"])) if seed_meta is not None and "file" in seed_meta else image_path

            t0 = time.perf_counter()
            start_payload = backend.start_session(image_payload, seed_meta=seed_meta)
            timings["start_s"] = round(time.perf_counter() - t0, 3)

            start_image = decode_frame(start_payload["frame_base64"])
            start_path = output_dir / "start.png"
            start_image.save(start_path)
            frames.append({"name": "start", "path": str(start_path), "size": list(start_image.size)})

            session_id = str(start_payload["session_id"])
            step_results = []
            for step_name, action in STEP_SPECS:
                t0 = time.perf_counter()
                step_payload = backend.step(session_id, action)
                step_elapsed = round(time.perf_counter() - t0, 3)
                timings[f"{step_name}_s"] = step_elapsed

                frame = decode_frame(step_payload["frame_base64"])
                frame_path = output_dir / f"{step_name}.png"
                frame.save(frame_path)
                frames.append({"name": step_name, "path": str(frame_path), "size": list(frame.size)})
                step_results.append(
                    {
                        "name": step_name,
                        "action": action,
                        "ended": bool(step_payload.get("ended", False)),
                        "truncated": bool(step_payload.get("truncated", False)),
                        "extra": step_payload.get("extra"),
                        "output": str(frame_path),
                    }
                )

            result = {
                "model_id": model_id,
                "ok": True,
                "image_type": image_type,
                "image_path": str(image_path),
                "worker_url": load_payload.get("worker_url"),
                "worker_log": load_payload.get("worker_log"),
                "timings": timings,
                "gpu_after": gpu_state(),
                "frames": frames,
                "steps": step_results,
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            }
            save_json(output_dir / "meta.json", result)
            return result
    except Exception as exc:
        result = {
            "model_id": model_id,
            "ok": False,
            "image_type": image_type,
            "image_path": str(image_path),
            "timings": timings,
            "gpu_after": gpu_state(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }
        save_json(output_dir / "meta.json", result)
        (output_dir / "error.txt").write_text(result["traceback"], encoding="utf-8")
        return result
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end rollout verification for vllm-wm models.")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help="If set, only run these model ids (must be keys in MODEL_IMAGE_TYPE).",
    )
    args = parser.parse_args()
    model_ids = list(MODEL_IMAGE_TYPE)
    if args.models:
        unknown = [m for m in args.models if m not in MODEL_IMAGE_TYPE]
        if unknown:
            raise SystemExit(f"Unknown model id(s): {unknown}. Valid: {list(MODEL_IMAGE_TYPE)}")
        model_ids = list(args.models)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for model_id in model_ids:
        print(f"=== verify {model_id} ===", flush=True)
        print("gpu_before", gpu_state(), flush=True)
        result = verify_model(model_id)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    out_name = "summary.json" if not args.models else f"summary_{'_'.join(args.models)}.json"
    summary_path = OUTPUT_ROOT / out_name
    save_json(summary_path, results)
    print("summary_path", summary_path, flush=True)


if __name__ == "__main__":
    main()
