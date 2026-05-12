from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    uvicorn.run(
        "vllm_wm.server.app:app",
        host="0.0.0.0",
        port=9100,
        reload=False,
    )
