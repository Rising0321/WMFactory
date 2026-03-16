from typing import List
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from abc import ABC, abstractmethod
import os, json, sys
from GameWorld.third_party.droid_slam.droid import Droid

from typing import Union, List, Tuple
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

from GameWorld.utils import load_dimension_info
from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

def open_image(image_location: str) -> Image.Image:
    """
    Opens image with the Python Imaging Library (PIL).
    """
    image: Image.Image
    image = Image.open(image_location)
    return image.convert("RGB")

class BaseMetric(ABC):
    """BaseMetric Class."""

    def __init__(self) -> None:
        self._metric = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _process_image(
        self,
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        rendered_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

        img: torch.Tensor = torch.stack(rendered_images_).to(self._device)

        return img

    def _process_images(
        self,
        rendered_images: List[Union[str, Image.Image]],
        reference_image: Union[str, Image.Image],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        # Handle the reference image input
        if isinstance(reference_image, str):
            reference_image = preprocessing(open_image(reference_image))

        rendered_images_: List[torch.Tensor] = []
        reference_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

            reference_images_.append(reference_image)

        img1: torch.Tensor = torch.stack(rendered_images_).to(self._device)
        img2: torch.Tensor = torch.stack(reference_images_).to(self._device)

        return img1, img2

    def _process_np_to_tensor(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img1 = ToTensor()(rendered_image).unsqueeze(0).to(self._device)
        img2 = ToTensor()(reference_image).unsqueeze(0).to(self._device)
        return img1, img2
    
    @abstractmethod
    def _compute_scores(self, *args):
        pass
def image_stream(video_path, stride, calib):
    """ image generator """

    fx, fy, cx, cy = calib

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # image_list = image_list[::stride]

    # for t, imfile in enumerate(image_list):
    #     image = cv2.imread(imfile)

    #     h0, w0, _ = image.shape
    #     h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
    #     w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

    #     image = cv2.resize(image, (w1, h1))
    #     image = image[:h1-h1%8, :w1-w1%8]
    #     image = torch.as_tensor(image).permute(2, 0, 1)

    #     intrinsics = torch.as_tensor([fx, fy, cx, cy])
    #     intrinsics[0::2] *= (w1 / w0)
    #     intrinsics[1::2] *= (h1 / h0)

    #     yield t, image[None], intrinsics

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % stride == 0:
            h0, w0, _ = frame.shape
            h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

            frame = cv2.resize(frame, (w1, h1))
            frame = frame[:h1 - h1 % 8, :w1 - w1 % 8]
            frame = torch.as_tensor(frame).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield frame_count, frame[None], intrinsics
        frame_count += 1
    cap.release()

class ReprojectionErrorMetric(BaseMetric):
    """
    
    return: Reprojection error
    
    """
    
    def __init__(self, droid_path) -> None:
        super().__init__()
        args = {
            't0': 0,
            'stride': 1,
            'weights': droid_path,
            'buffer': 512,
            'beta': 0.3,
            'filter_thresh': 0.01,
            'warmup': 8,
            'keyframe_thresh': 4.0,
            'frontend_thresh': 16.0,
            'frontend_window': 25,
            'frontend_radius': 2,
            'frontend_nms': 1,
            'backend_thresh': 22.0,
            'backend_radius': 2,
            'backend_nms': 3,
            # need high resolution depths
            'upsample': True,
            'stereo': False,
            'calib': [500., 500., 256., 256.]
        }
        args = argparse.Namespace(**args)
        
        self._args = args
        self.droid = None
        try:
            torch.multiprocessing.set_start_method('spawn')
        except Exception as e:
            print(f"Warning: Error setting start method: {e}")
    
    def _compute_scores(
        self, 
        video_path
    ) -> float:
        
        for (t, image, intrinsics) in tqdm(image_stream(video_path, self._args.stride, self._args.calib)):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        traj_est, valid_errors = self.droid.terminate(image_stream(video_path, self._args.stride, self._args.calib))
        
        if len(valid_errors) > 0:
            mean_error = valid_errors.mean().item()

        self.droid = None
        return mean_error

def ThreeDimensional_consistency(video_list, droid_path):
    video_results = []
    evaluator = ReprojectionErrorMetric(droid_path)
    for i, path in enumerate(video_list):
        score = evaluator._compute_scores(path)
        empirical_max = 2.5
        empirical_min = 0
        normalized_score = (score - empirical_min) / (empirical_max - empirical_min)
        normalized_score = 1 - normalized_score
        video_results.append({'video_path': path, 'video_results': normalized_score, 'score': score})
            
    average_score = sum([d['video_results'] for d in video_results]) / len(video_results)
    return average_score, video_results
    
def compute_object_consistency(json_dir, device, submodules_list, **kwargs):
    droid_path = submodules_list[0]
    video_list, _ = load_dimension_info(json_dir, dimension='object_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = ThreeDimensional_consistency(video_list, droid_path)

    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)

    return all_results, video_results