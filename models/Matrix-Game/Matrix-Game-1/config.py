import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MatrixGame inference script")
    parser.add_argument(
        "--vae-path",
        type=str,
        default="ckpt/vae/",
        help="Name of the VAE model.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        help="Name of the VAE model.",
    )
    parser.add_argument(
        "--vae-precision",
        type=str,
        default="bf16",
        help="Precision mode for the VAE model.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiling for the VAE model to save GPU memory.",
    )
    parser.set_defaults(vae_tiling=True)
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default="ckpt/",
        help="Name of the text encoder model.",
    )
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="bf16",
        help="Precision mode for the text encoder model.",
    )
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=17.0,
        help="Shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )
    parser.add_argument(
        "--num-pre-frames",
        type=int,
        default=5,
        help="num pre frames.",
    )
    parser.add_argument(
        "--dit-path",
        type=str,
        default="ckpts/",
        help="Path to the dit model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--input-image-path",
        type=str,
        default="input/",
        help="input image path.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/",
        help="Path to save output videos.",
    )
    parser.add_argument(
        "--mouse-icon-path",
        type=str,
        default='./ckpt/',
        help="Path to the mouse icon image.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=129,
        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale."
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses degree for xdit parallel args.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ring degree for xdit parallel args.",
    )
    args = parser.parse_args()
    return args
