from vllm_wm.engine.omni_engine import WorldModelEngine


def main() -> None:
    engine = WorldModelEngine()
    models = engine.list_models()
    print(f"registered_models={len(models)}")
    for model in models:
        print(f"{model['id']}: {model['family']}")


if __name__ == "__main__":
    main()
