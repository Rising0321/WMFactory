import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "vllm_wm.server.app:app",
        host="0.0.0.0",
        port=9100,
        reload=False,
    )
