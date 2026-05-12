from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm_wm.registry import build_backend, get_model_spec

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


MODEL_IMAGE_OVERRIDES = {
    "diamond": DEMO_ROOT / "csgo0.png",
}


MODEL_ENVS = {
    "matrixgame": {
        "WM_MATRIXGAME_ENABLE_COMPILE": "0",
        "WM_MATRIXGAME_WARMUP_ON_START": "0",
    },
    "matrixgame3": {
        "WM_MATRIXGAME3_NUM_ITERATIONS": "2",
        "WM_MATRIXGAME3_NUM_INFERENCE_STEPS": "8",
        "WM_MATRIXGAME3_START_TIMEOUT": "3600",
        "WM_MATRIXGAME3_STEP_TIMEOUT": "3600",
        "WM_MATRIXGAME3_STARTUP_TIMEOUT": "3600",
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


DIRECTION_SPECS = {
    "w": {"description": "move forward", "action": {"w": True}},
    "a": {"description": "move left", "action": {"a": True}},
    "s": {"description": "move backward", "action": {"s": True}},
    "d": {"description": "move right", "action": {"d": True}},
    "cu": {"description": "camera up", "action": {"camera_dy": -1.0}},
    "cd": {"description": "camera down", "action": {"camera_dy": 1.0}},
    "cl": {"description": "camera left", "action": {"camera_dx": -1.0}},
    "cr": {"description": "camera right", "action": {"camera_dx": 1.0}},
}


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


def remove_direction_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for direction in DIRECTION_SPECS:
        direction_dir = output_dir / direction
        if direction_dir.exists():
            shutil.rmtree(direction_dir)


def initial_payload_for_model(
    backend: Any,
    model_id: str,
    init_image: Path | None = None,
) -> tuple[str, dict[str, Any] | None, str, str]:
    image_type = MODEL_IMAGE_TYPE[model_id]

    if init_image is not None:
        path = init_image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"init image not found: {path}")
        image_payload = encode_image(path)
        return image_payload, None, image_type, str(path)

    image_path = MODEL_IMAGE_OVERRIDES.get(model_id, DEMO_IMAGES[image_type])
    seed_meta: dict[str, Any] | None = None

    spec = get_model_spec(model_id)
    if spec.supports_seed_meta and spec.supports_random_dataset and spec.default_dataset_ids:
        dataset_id = spec.default_dataset_ids[0]
        dataset_payload = backend.random_dataset_image(dataset_id)
        image_payload = dataset_payload["image_base64"]
        seed_meta = dataset_payload.get("extra", {}).get("seed_meta")
        if seed_meta is not None and "file" in seed_meta:
            image_path = Path(str(seed_meta["file"]))
        return image_payload, seed_meta, image_type, str(image_path)

    image_payload = encode_image(image_path)
    return image_payload, seed_meta, image_type, str(image_path)


def start_or_reset_direction(
    backend: Any,
    previous_session_id: str | None,
    image_payload: str,
    seed_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if previous_session_id is None:
        return backend.start_session(image_payload, seed_meta=copy.deepcopy(seed_meta))
    return backend.reset_session(previous_session_id, image_payload, seed_meta=copy.deepcopy(seed_meta))


def run_direction(
    backend: Any,
    direction_dir: Path,
    direction_name: str,
    action: dict[str, Any],
    image_payload: str,
    seed_meta: dict[str, Any] | None,
    previous_session_id: str | None,
    repeat_steps: int,
) -> tuple[str, dict[str, Any]]:
    direction_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    frames: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    started_at = time.perf_counter()
    t0 = time.perf_counter()
    start_payload = start_or_reset_direction(backend, previous_session_id, image_payload, seed_meta)
    timings["start_or_reset_s"] = round(time.perf_counter() - t0, 3)

    session_id = str(start_payload["session_id"])
    start_image = decode_frame(start_payload["frame_base64"])
    start_path = direction_dir / "start.png"
    start_image.save(start_path)
    frames.append({"name": "start", "path": str(start_path), "size": list(start_image.size)})

    for step_idx in range(1, repeat_steps + 1):
        step_name = f"step{step_idx}"
        t0 = time.perf_counter()
        step_payload = backend.step(session_id, dict(action))
        step_elapsed = round(time.perf_counter() - t0, 3)
        timings[f"{step_name}_s"] = step_elapsed

        frame = decode_frame(step_payload["frame_base64"])
        frame_path = direction_dir / f"{step_name}.png"
        frame.save(frame_path)
        frames.append({"name": step_name, "path": str(frame_path), "size": list(frame.size)})
        steps.append(
            {
                "name": step_name,
                "action": dict(action),
                "ended": bool(step_payload.get("ended", False)),
                "truncated": bool(step_payload.get("truncated", False)),
                "extra": step_payload.get("extra"),
                "output": str(frame_path),
            }
        )

    result = {
        "direction": direction_name,
        "action": dict(action),
        "repeat_steps": repeat_steps,
        "timings": timings,
        "frames": frames,
        "steps": steps,
        "elapsed_s": round(time.perf_counter() - started_at, 3),
    }
    save_json(direction_dir / "meta.json", result)
    return session_id, result


def verify_model(model_id: str, repeat_steps: int, init_image: Path | None = None) -> dict[str, Any]:
    if init_image is not None:
        output_dir = OUTPUT_ROOT / f"{model_id}__{init_image.stem}"
    else:
        output_dir = OUTPUT_ROOT / model_id
    remove_direction_outputs(output_dir)

    backend = None
    started_at = time.perf_counter()
    timings: dict[str, float] = {}
    direction_results: list[dict[str, Any]] = []
    image_type = MODEL_IMAGE_TYPE[model_id]
    image_path = str(DEMO_IMAGES[image_type])
    try:
        with patched_env(MODEL_ENVS[model_id]):
            backend = build_backend(model_id)

            t0 = time.perf_counter()
            load_payload = backend.load()
            timings["load_s"] = round(time.perf_counter() - t0, 3)

            image_payload, seed_meta, image_type, image_path = initial_payload_for_model(
                backend, model_id, init_image=init_image
            )
            session_id: str | None = None
            for direction_name, spec in DIRECTION_SPECS.items():
                print(f"--- {model_id}: {direction_name} ({spec['description']}) ---", flush=True)
                session_id, direction_result = run_direction(
                    backend=backend,
                    direction_dir=output_dir / direction_name,
                    direction_name=direction_name,
                    action=spec["action"],
                    image_payload=image_payload,
                    seed_meta=seed_meta,
                    previous_session_id=session_id,
                    repeat_steps=repeat_steps,
                )
                direction_results.append(direction_result)

            result = {
                "model_id": model_id,
                "ok": True,
                "image_type": image_type,
                "image_path": image_path,
                "worker_url": load_payload.get("worker_url"),
                "worker_log": load_payload.get("worker_log"),
                "timings": timings,
                "directions": direction_results,
                "gpu_after": gpu_state(),
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            }
            save_json(output_dir / "action_sweep_meta.json", result)
            return result
    except Exception as exc:
        result = {
            "model_id": model_id,
            "ok": False,
            "image_type": image_type,
            "image_path": image_path,
            "timings": timings,
            "directions": direction_results,
            "gpu_after": gpu_state(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }
        save_json(output_dir / "action_sweep_meta.json", result)
        (output_dir / "action_sweep_error.txt").write_text(result["traceback"], encoding="utf-8")
        return result
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8-direction action sweep verification for WMBackend models.")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help="If set, only run these model ids (must be keys in MODEL_IMAGE_TYPE).",
    )
    parser.add_argument(
        "--repeat-steps",
        type=int,
        default=3,
        help="How many times to repeat each directional action.",
    )
    parser.add_argument(
        "--init-image",
        type=Path,
        default=None,
        help="Use this image file for every model in the run (sessions start/reset). Output goes to testOutput/<model>__<stem>/.",
    )
    args = parser.parse_args()
    if args.repeat_steps <= 0:
        raise SystemExit("--repeat-steps must be positive")

    model_ids = list(MODEL_IMAGE_TYPE)
    if args.models:
        unknown = [m for m in args.models if m not in MODEL_IMAGE_TYPE]
        if unknown:
            raise SystemExit(f"Unknown model id(s): {unknown}. Valid: {list(MODEL_IMAGE_TYPE)}")
        model_ids = list(args.models)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for model_id in model_ids:
        print(f"=== action sweep {model_id} ===", flush=True)
        print("gpu_before", gpu_state(), flush=True)
        result = verify_model(model_id, repeat_steps=args.repeat_steps, init_image=args.init_image)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if args.init_image is not None:
        stem = args.init_image.stem
        if args.models:
            out_name = f"action_sweep_summary_{'_'.join(args.models)}_{stem}.json"
        else:
            out_name = f"action_sweep_summary_all_{stem}.json"
    elif args.models:
        out_name = f"action_sweep_summary_{'_'.join(args.models)}.json"
    else:
        out_name = "action_sweep_summary.json"
    summary_path = OUTPUT_ROOT / out_name
    save_json(summary_path, results)
    print("summary_path", summary_path, flush=True)


if __name__ == "__main__":
    main()
