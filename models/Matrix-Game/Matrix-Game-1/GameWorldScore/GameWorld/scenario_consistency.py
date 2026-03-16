import os
import cv2
import numpy as np
from tqdm import tqdm
from GameWorld.utils import load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

def get_frames(video_path):
    """Decode all frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:                                                     
        raise AssertionError(f"{video_path} has no valid frame.")
    return frames


def resize_frames(frames, target_size=(256, 256)):
    """Resize a list of BGR frames to identical HxW."""
    return [cv2.resize(f, target_size, interpolation=cv2.INTER_AREA) for f in frames]


# --------------------------------------------------------------------------- #
# 对称一致性度量
# --------------------------------------------------------------------------- #
def mse_min_shift(img1, img2, max_shift=4):
    """
    最多允许 |dx|,|dy|≤max_shift 像素的整像素平移，计算两张图的最小 MSE。
    为避免额外 padding，比较两图在重叠区域上的 MSE。
    """
    h, w, _ = img1.shape
    best = np.inf

    for dy in range(-max_shift, max_shift + 1):
        y0a = max(0,  dy)
        y0b = max(0, -dy)
        y1  = min(h, h + dy)              # 重叠下界
        if y1 - y0a <= 0:                 # 无重叠
            continue

        for dx in range(-max_shift, max_shift + 1):
            x0a = max(0,  dx)
            x0b = max(0, -dx)
            x1  = min(w, w + dx)
            if x1 - x0a <= 0:
                continue

            crop1 = img1[y0a:y1, x0a:x1]
            crop2 = img2[y0b:y1 - dy, x0b:x1 - dx]

            mse = np.mean(np.sqrt((crop1.astype(np.float32) - crop2.astype(np.float32)) ** 2))
            best = mse if mse < best else best

    return best


def scenario_consistency_score(frames, max_shift=4):
    """
    计算 2n+1 帧视频的对称一致性分数：
        - 配对方式：第 i 帧 vs 倒数第 i+1 帧 (中心帧忽略)
        - 每对取 min-MSE (允许平移)
        - score = 1 - (mean_mse / 255²) ∈ [0,1]
    """
    n = len(frames) // 2
    mses = []
    for i in range(n):
        mse = mse_min_shift(frames[i], frames[-(i + 1)], max_shift=max_shift)
        mses.append(mse)
    mean_mse = float(np.mean(mses))
    score = 1.0 - mean_mse / (255.0)       # 归一化到 [0,1]
    return max(0.0, min(1.0, score))            # 数值安全


def cal_score(video_path, target_size=(256, 256), max_shift=4):
    """Compute the symmetric-consistency score of **one** video."""
    frames = resize_frames(get_frames(video_path), target_size)
    return scenario_consistency_score(frames, max_shift=max_shift)


def scenario_consistency(video_list, **kwargs):
    """
    Args:
        video_list (List[str]): paths
    Returns:
        avg_score (float)
        video_results (List[Dict]): [{'video_path': p, 'video_results': s, 'score': s}, ...]
                                     ——保留 `score` 字段以保持与旧接口对齐
    """
    sim, video_results = [], []
    for path in tqdm(video_list, disable=get_rank() > 0):
        try:
            score = cal_score(path, **kwargs)
        except AssertionError as e:
            print(f"[Skip] {e}")
            continue
        video_results.append({'video_path': path,
                              'video_results': score})
        sim.append(score)
    avg_score = float(np.mean(sim)) if sim else 0.0
    return avg_score, video_results


def compute_scenario_consistency(json_dir,
                                  device=None,          # 占位，为与 vbench 框架接口一致
                                  submodules_list=None, # ^
                                  target_size=(256, 256),
                                  max_shift=4,
                                  **kwargs):
    """
    入口函数，供 vbench 执行：
        all_results, video_results = compute_scenario_consistency(...)
    """
    video_list, _ = load_dimension_info(json_dir,
                                        dimension='scenario_consistency',
                                        lang='en')
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = scenario_consistency(video_list,
                                                       target_size=target_size,
                                                       max_shift=max_shift)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)

    return all_results, video_results