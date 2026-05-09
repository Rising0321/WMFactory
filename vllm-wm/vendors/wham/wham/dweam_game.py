from pathlib import Path
from typing import Literal
import cv2
import numpy as np
import pygame
import torch as th
from tensordict import TensorDict
from huggingface_hub import snapshot_download as hf_snapshot_download

from dweam import Game, Field, get_cache_dir
from .utils import (
    load_model_from_checkpoint,
    POS_BINS_BOUNDARIES,
)

def snapshot_download(**kwargs) -> Path:
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_snapshot_download(cache_dir=str(cache_dir), **kwargs)
    return Path(path)

def be_image_preprocess(image, target_width, target_height):
    # If target_width and target_height are specified, resize the image.
    if target_width is not None and target_height is not None:
        # Make sure we do not try to resize if the image is already the correct size.
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
    return np.transpose(image, (2, 0, 1))


def action_vector_to_be_action_vector(action):
    # Preprocess a BE action vector from 16 numbers with:
    #  12 buttons [0, 1] and 4 stick directions [-1, 1]
    # to values valid for the token model
    #  12 buttons [0, 1] and 4 stick directions [0, 10]
    action = action.copy()  # Don't modify input
    # Scale stick values from [-1, 1] to [0, 10]
    action[-4:] = (action[-4:] + 1) * 5
    return action


class WhamGame(Game):
    class Params(Game.Params):
        checkpoint: Literal["WHAM_200M", "WHAM_1.6B_v1"] = Field(
            default="WHAM_200M",
            description="Model checkpoint to use"
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            # Download model from HuggingFace
            self.log.info("Downloading model from HuggingFace")
            self.path_hf = Path(snapshot_download(repo_id="microsoft/WHAM", allow_patterns=f"models/*.ckpt"))
            checkpoint_path = self.path_hf / "models" / f"{self.params.checkpoint}.ckpt"
            
            # Load model
            self.log.info("Loading model", checkpoint=str(checkpoint_path))
            self.model = load_model_from_checkpoint(checkpoint_path)
            self.log.info("Moving model to CUDA")
            self.model = self.model.cuda()
            self.log.info("Setting model to eval mode")
            self.model.eval()

            # Game state
            self.width = 300  # Changed from 640
            self.height = 180  # Changed from 480
            self.context_images = []
            self.context_actions = []
            self.action_vector = np.zeros(16)  # WHAM expects 16-dim action vector
            
            self.log.info("Initialization complete")
            
        except Exception as e:
            self.log.exception("Failed to initialize game")
            raise

    @th.no_grad()
    def step(self) -> pygame.Surface:
        # Get current frame
        if len(self.context_images) == 0:
            # Initialize with correct dimensions
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            frame = self.context_images[-1]
            # Convert from CHW back to HWC format if needed
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)

        # Process frame and action
        processed_frame = be_image_preprocess(frame, self.width, self.height)
        processed_action = action_vector_to_be_action_vector(self.action_vector)
        
        # Update context
        self.context_images.append(processed_frame)
        self.context_actions.append(processed_action)
        if len(self.context_images) > 10:  # Fixed context length of 10
            self.context_images.pop(0)
            self.context_actions.pop(0)

        # Dream next frame if we have enough context
        if len(self.context_images) == 10:
            # Prepare context tensors
            image_context = np.stack(self.context_images)[None]  # shape: [1, 10, 3, H, W]
            action_context = np.stack(self.context_actions)[None]  # shape: [1, 10, 16]

            context_data = TensorDict({
                "images": th.from_numpy(image_context).cuda(),
                "actions_output": th.from_numpy(action_context).float().cuda()
            }, batch_size=image_context.shape[:2])  # batch_size is [1, 10]

            # Generate next frame
            predicted_step, _ = self.model.predictor.predict_next_step(
                self.model.encode_context(context_data),
                temperature=0.9,
                min_tokens_to_keep=1
            )
            
            # Decode frame
            decoded = self.model.decode_context(predicted_step)
            frame = decoded["images"][0, 0].cpu().numpy()
            frame = frame.transpose(1, 2, 0)

        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def stop(self) -> None:
        """Clean up GPU resources"""
        super().stop()
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
            th.cuda.empty_cache()

    def on_params_update(self, new_params: Params) -> None:
        """Handle parameter updates"""
        if self.params.checkpoint != new_params.checkpoint:
            self.log.info("Loading new checkpoint", checkpoint=new_params.checkpoint)
            checkpoint_path = self.path_hf / "models" / f"{new_params.checkpoint}.ckpt"
            self.model = load_model_from_checkpoint(checkpoint_path).cuda()
            self.model.eval()
            self.context_images = []
            self.context_actions = []

        super().on_params_update(new_params)
