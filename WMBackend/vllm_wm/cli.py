from __future__ import annotations

import argparse
import json

from vllm_wm.config import EngineConfig
from vllm_wm.engine.omni_engine import WorldModelEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="WMBackend", description="Unified backend for interactive world models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the unified HTTP server.")
    serve.add_argument("--host", default=EngineConfig().host)
    serve.add_argument("--port", type=int, default=EngineConfig().port)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("list-models", help="Print registered world-model backends.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("vllm_wm.server.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    if args.command == "list-models":
        engine = WorldModelEngine()
        print(json.dumps({"models": engine.list_models()}, ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
