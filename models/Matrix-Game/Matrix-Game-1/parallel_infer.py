import os
import time
import random
import functools
from typing import Optional
from pathlib import Path

from loguru import logger
import math

import torch
import torch.distributed as dist
from matrixgame.sample.flow_matching_scheduler_matrixgame import FlowMatchDiscreteScheduler
from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline

from config import parse_args
from matrixgame.encoder_variants import get_text_enc
from matrixgame.vae_variants import get_vae
from matrixgame.model_variants.matrixgame_dit_src import MGVideoDiffusionTransformerI2V

from tools.visualize import process_video
from einops import rearrange
import numpy as np
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
        initialize_model_parallel,
        init_distributed_environment
    )
except BaseException:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None
    initialize_model_parallel = None
    init_distributed_environment = None

logger = logger.bind(name=__name__)


def str_to_type(type_str):
    type_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16
    }
    if type_str not in type_map:
        raise ValueError(f"unsupported type {type_str}.")
    return type_map[type_str]


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        guidance: torch.Tensor = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        mouse_condition: torch.Tensor = None,
        keyboard_condition: torch.Tensor = None,
        return_dict: bool = True,
    ):
        seq_parallel_world_size = get_sequence_parallel_world_size()
        seq_parallel_rank = get_sequence_parallel_rank()

        if hidden_states.shape[-2] // 2 % seq_parallel_world_size == 0:
            split_dim = -2
        elif hidden_states.shape[-1] // 2 % seq_parallel_world_size == 0:
            split_dim = -1
        else:
            raise ValueError(
                f"Cannot split video sequence into ulysses_degree x ring_degree ({seq_parallel_world_size}) parts evenly")

        temporal_size, h, w = hidden_states.shape[2], hidden_states.shape[3] // 2, hidden_states.shape[4] // 2
        hidden_states = torch.chunk(
            hidden_states,
            seq_parallel_world_size,
            dim=split_dim)[
            seq_parallel_rank]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos,
                                seq_parallel_world_size,
                                dim=split_dim - 1)[seq_parallel_rank]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin,
                                seq_parallel_world_size,
                                dim=split_dim - 1)[seq_parallel_rank]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)

        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            freqs_cos,
            freqs_sin,
            guidance,
            mouse_condition,
            keyboard_condition,
            return_dict,
        )
        sample = output["x"]
        output["x"] = get_sp_group().all_gather(sample, dim=split_dim)
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward


def gpu_memory_usage():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    return [free, total]


class Inference(object):
    def __init__(
        self,
        args,
        vae,
        text_encoder,
        model,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload
        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args

    @classmethod
    def from_pretrained(cls, pretrained_model_path,
                        args, device=None, **kwargs):

        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        if 1 or args.ulysses_degree > 1 or args.ring_degree > 1:
            assert xfuser is not None, "Ulysses Attention and Ring Attention requires xfuser package."
            local_rank = int(os.environ['LOCAL_RANK'])
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == args.ring_degree * args.ulysses_degree, \
                "number of GPUs should be equal to ring_degree * ulysses_degree."
            init_distributed_environment(rank=rank, world_size=world_size)
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=args.ring_degree,
                ulysses_degree=args.ulysses_degree,
            )
        else:
            rank = 0
            world_size = 1
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        parallel_args = {
            "ulysses_degree": args.ulysses_degree,
            "ring_degree": args.ring_degree}
        torch.set_grad_enabled(False)

        if rank == 0:
            logger.info("Building model...")

            # dit
            model = MGVideoDiffusionTransformerI2V.from_pretrained(
                pretrained_model_path, dtype=torch.bfloat16)
            model = model.to(device)
            model.requires_grad_(False)
            model.eval()

            # vae
            vae = get_vae(vae_name="matrixgame",
                          model_path=args.vae_path,
                          weight_dtype=str_to_type(args.vae_precision))
            if args.use_cpu_offload:
                vae = vae.to("cpu")

            text_encoder = get_text_enc(enc_name='matrixgame',
                                        model_path=args.text_encoder_path,
                                        weight_dtype=str_to_type(
                                            args.text_encoder_precision),
                                        i2v_type='refiner')
            if args.use_cpu_offload:
                text_encoder = text_encoder.to("cpu")

            mem_info = gpu_memory_usage()
            logger.info(
                f"Rank {rank}: GPU Mem, free: {mem_info[0]}, total: {mem_info[1]}")
        else:
            model = None
            vae = None
            text_encoder = None

        if world_size > 1:
            logger.info(f"Rank {rank}: Starting broadcast synchronization")
            dist.barrier()
            if rank != 0:
                logger.info(f"Rank {rank}: Starting Load Model")

                # dit
                model = MGVideoDiffusionTransformerI2V.from_pretrained(
                    pretrained_model_path, dtype=torch.bfloat16)
                model = model.to(device)
                model.requires_grad_(False)
                model.eval()

                # vae
                vae = get_vae(vae_name="matrixgame",
                              model_path=args.vae_path,
                              weight_dtype=str_to_type(args.vae_precision))
                if args.use_cpu_offload:
                    vae = vae.to("cpu")

                text_encoder = get_text_enc(enc_name='matrixgame',
                                            model_path=args.text_encoder_path,
                                            weight_dtype=str_to_type(
                                                args.text_encoder_precision),
                                            i2v_type='refiner')
                if args.use_cpu_offload:
                    text_encoder = text_encoder.to("cpu")

            mem_info = gpu_memory_usage()
            logger.info(
                f"Rank {rank}: GPU Mem, free: {mem_info[0]}, total: {mem_info[1]}")

            logger.info(f"Rank {rank}: Broadcasting model parameters")
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            model.eval()
            logger.info(f"Rank {rank}: Broadcasting transformers")

            if args.use_cpu_offload:
                model = model.to("cpu")

            dist.barrier()

        return cls(
            args=args,
            vae=vae,
            text_encoder=text_encoder,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )


class MatrixGameVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        text_encoder,
        model,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None
    ):
        super().__init__(
            args,
            vae,
            text_encoder,
            model,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            model=self.model,
            device=self.device,
        )

        self.default_negative_prompt = "deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs"
        parallelize_transformer(self.pipeline)

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        model,
        scheduler=None,
        device=None
    ):
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")
        pipeline = MatrixGameVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            transformer=model,
            scheduler=scheduler,
        ).to(torch.bfloat16)
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload(device=device)
        else:
            pipeline = pipeline.to(device)

        return pipeline

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        mouse_condition=None,
        keyboard_condition=None,
        initial_image=None,
        data_type="video",
        vae_ver='884-16c-hy',
        i2v_type=None,
        semantic_images=None,
        enable_tiling=True,
        num_inference_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        args=None,
        **kwargs,
    ):
        out_dict = dict()

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        out_dict["seeds"] = seeds

        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if guidance_scale == 1.0:
            negative_prompt = ""
        if not isinstance(negative_prompt, str) and not isinstance(
                negative_prompt, list):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()] if isinstance(
            negative_prompt, str) else negative_prompt[0]

        message = f"""
                         height: {target_height}
                          width: {target_width}
                   video_length: {target_video_length}
                         prompt: {prompt}
                     neg_prompt: {negative_prompt}
                           seed: {seed}
            num_inference_steps: {num_inference_steps}
          num_videos_per_prompt: {num_videos_per_prompt}
                 guidance_scale: {guidance_scale}
                     flow_shift: {flow_shift}
        embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.info(message)

        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=torch.Generator(device="cuda").manual_seed(42),
            data_type=data_type,
            is_progress_bar=True,
            mouse_condition=mouse_condition,
            keyboard_condition=keyboard_condition,
            initial_image=initial_image,
            negative_prompt=["N.o.n.e."],
            embedded_guidance_scale=4.5,
            vae_ver=vae_ver,
            enable_tiling=enable_tiling,
            i2v_type=i2v_type,
            args=args,
            semantic_images=semantic_images,
        ).videos
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict


def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


def main():
    args = parse_args()
    print(f'args = {args}', flush=True)

    os.makedirs(args.output_path, exist_ok=True)

    video_length = args.video_length
    guidance_scale = args.cfg_scale

    models_root_path = Path(args.dit_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    sampler = MatrixGameVideoSampler.from_pretrained(
        models_root_path, args=args)

    args = sampler.args

    mapping = {
        'forward': [1, 0, 0, 0, 0, 0],
        'back': [0, 1, 0, 0, 0, 0],
        'left': [0, 0, 1, 0, 0, 0],
        'right': [0, 0, 0, 1, 0, 0],
        'jump': [0, 0, 0, 0, 1, 0],
        'attack': [0, 0, 0, 0, 0, 1],
    }
    xs = [0] * video_length
    ys = [0] * video_length
    mouse_list = [[x, y] for x, y in zip(xs, ys)]

    prompt = "N.o.n.e."

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        assert xfuser is not None, "Ulysses Attention and Ring Attention requires xfuser package."
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = "cuda"

    image = load_image(image=args.input_image_path)
    new_width = 1280
    new_height = 720
    initial_image = resizecrop(image, new_height, new_width)
    semantic_image = initial_image
    vae_scale_factor = 2 ** (len(sampler.pipeline.vae.config.block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
    initial_image = video_processor.preprocess(
        initial_image, height=new_height, width=new_width)
    if args.num_pre_frames > 0:
        past_frames = initial_image.repeat(args.num_pre_frames, 1, 1, 1)
        initial_image = torch.cat([initial_image, past_frames], dim=0)

    for idx, (k, v) in enumerate(mapping.items()):
        if k == "attack":
            signal = np.array(mapping[k])
            zero_signal = np.zeros_like(signal)
            sequence = np.array([
                zero_signal if i % 4 == 0 else signal
                for i in range(video_length)
            ])
            keyboard_condition = torch.from_numpy(sequence)[None]
        else:
            keyboard_condition = torch.from_numpy(
                np.array([mapping[k] for _ in range(video_length)]))[None]
        mouse_condition = torch.from_numpy(
            np.array(mouse_list[:video_length]))[None]
        keyboard_condition = keyboard_condition.to(torch.bfloat16).to(device)
        mouse_condition = mouse_condition.to(torch.bfloat16).to(device)
        with torch.no_grad():
            video = sampler.predict(
                prompt=prompt,
                height=new_height,
                width=new_width,
                video_length=video_length,
                mouse_condition=mouse_condition,
                keyboard_condition=keyboard_condition,
                initial_image=initial_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=["N.o.n.e."],
                embedded_guidance_scale=4.5,
                data_type="video",
                vae_ver=args.vae,
                enable_tiling=args.vae_tiling,
                generator=torch.Generator(device="cuda").manual_seed(42),
                i2v_type="refiner",
                args=args,
                semantic_images=semantic_image,
                batch_size=args.batch_size
            )['samples'][0]
        print(f"output video: {video.shape}")

        if 'LOCAL_RANK' not in os.environ or int(
                os.environ['LOCAL_RANK']) == 0:
            img_tensors = rearrange(
                video.permute(
                    1,
                    0,
                    2,
                    3) * 255,
                't c h w -> t h w c').contiguous()
            img_tensors = img_tensors.cpu().numpy().astype(np.uint8)

            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy())
            output_file_name = f"output_{video_length}_{k}_guidance_scale_{guidance_scale}.mp4"
            process_video(img_tensors,
                          os.path.join(args.output_path, output_file_name),
                          config,
                          mouse_icon_path=args.mouse_icon_path,
                          mouse_scale=0.1,
                          mouse_rotation=-20,
                          fps=16)


if __name__ == "__main__":
    main()
