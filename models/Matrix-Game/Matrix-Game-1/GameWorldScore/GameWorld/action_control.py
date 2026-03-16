import os
import json
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from GameWorld.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm
import cv2
from GameWorld.third_party.IDM.IDM_bench import evaluate_IDM_quality
from argparse import Namespace


from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
    print0
)

def video_frames_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def action_control(video_list):
    # default params
    weights=os.path.expanduser("~/.cache/GameWorld_bench/IDM/4x_idm.weights")
    model=os.path.expanduser("~/.cache/GameWorld_bench/IDM/4x_idm.model")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(root_dir, "third_party/IDM/Bench76")
    infer_demo_num=1

    if not video_list:
        raise ValueError("No video path provided.")
    n_frames = video_frames_count(video_list[0]) - 1 
    args = Namespace(
        weights=weights,
        model=model,
        jsonl_path=jsonl_path,
        infer_demo_num=infer_demo_num,
        n_frames=n_frames,
    )

    (keyboard_precision, camera_precision), video_results = evaluate_IDM_quality(args.model, args.weights, args.jsonl_path, video_list, args.infer_demo_num,args.n_frames, args)

    return (keyboard_precision, camera_precision), video_results

def gather_all_results(local_result):
    tensor = torch.tensor([local_result], dtype=torch.float32, device=torch.device("cuda"))
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [t.item() for t in gathered]

def compute_action_control(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension='action_control', lang='en')
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = action_control(video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        camera_precision = sum([d['camera_precision'] for d in video_results]) / get_world_size()
        keyboard_precision = sum(gather_all_results(all_results[0])) / get_world_size()
        camera_precision_2 = sum(gather_all_results(all_results[0])) / get_world_size()
        print0(f"Camera precision: {camera_precision:.4f}, camera_precision_2: {camera_precision_2:.4f}") # test
        all_results = (keyboard_precision, camera_precision)
    return all_results, video_results

