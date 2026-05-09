from __future__ import annotations

import argparse
import json

from vllm_wm.config import EngineConfig
from vllm_wm.registry import MODEL_SPECS, build_backend, normalize_model_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start vllm-wm managed workers and probe /health.")
    parser.add_argument("--model", action="append", default=[], help="Probe one or more model ids.")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Do not stop workers after probing. Useful when warming services for manual testing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EngineConfig()
    target_ids = [normalize_model_id(model_id) for model_id in args.model] if args.model else [s.model_id for s in MODEL_SPECS]

    started = []
    summary = []
    for model_id in target_ids:
        backend = build_backend(model_id, config)
        started.append(backend)
        try:
            health = backend.health()
            summary.append(
                {
                    "model_id": model_id,
                    "ok": bool(health.get("ok", False)),
                    "ready": bool(health.get("ready", False)),
                    "worker_url": health.get("worker_url"),
                    "worker_log": health.get("worker_log"),
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "model_id": model_id,
                    "ok": False,
                    "error": str(exc),
                }
            )

    print(json.dumps({"results": summary}, ensure_ascii=False, indent=2))

    if args.keep_alive:
        return
    for backend in reversed(started):
        try:
            backend.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
