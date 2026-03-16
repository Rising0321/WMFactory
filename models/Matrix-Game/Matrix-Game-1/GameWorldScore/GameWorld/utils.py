import os
import json
import numpy as np
import logging
import subprocess
import torch
import re
from pathlib import Path
from PIL import Image, ImageSequence
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR


CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'GameWorld_bench')

from .distributed import (
    get_rank,
    barrier,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_transform_Image(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def dino_transform(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# def tag2text_transform(n_px):
#     normalize = Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#     return Compose([ToPILImage(),Resize((n_px, n_px), antialias=False),ToTensor(),normalize])

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.
    
    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.
    
    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.
    
    Raises:
    - NotImplementedError: If the video format is not supported.
    
    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
            num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
        num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def read_frames_decord_by_fps(
        video_path, sample_fps=2, sample='rand', fix_start=None, 
        max_num_frames=-1,  trimmed30=False, num_frames=8
    ):
    import decord
    decord.bridge.set_bridge("torch")
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames
    
# def load_dimension_info(json_dir, dimension, lang):
#     """
#     prompt is not needed
#     Load video list and prompt information based on a specified dimension and language from a JSON file.
    
#     Parameters:
#     - json_dir (str): The directory path where the JSON file is located.
#     - dimension (str): The dimension for evaluation to filter the video prompts.
#     - lang (str): The language key used to retrieve the appropriate prompt text.
    
#     Returns:
#     - video_list (list): A list of video file paths that match the specified dimension.
#     - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding video list.
    
#     The function reads the JSON file to extract video information. It filters the prompts based on the specified
#     dimension and compiles a list of video paths and associated prompts in the specified language.
    
#     Notes:
#     - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
#     - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
#     """
#     video_list = []
#     prompt_dict_ls = []
#     full_prompt_list = load_json(json_dir)
#     for prompt_dict in full_prompt_list:
#         if dimension in prompt_dict['dimension'] and 'video_list' in prompt_dict:
#             prompt = prompt_dict[f'prompt_{lang}']
#             cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
#             video_list += cur_video_list
#             if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
#                 prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
#             else:
#                 prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
#     return video_list, prompt_dict_ls

def load_dimension_info(json_dir, dimension, lang):
    """
    Load video list and prompt information based on a specified dimension and language from a JSON file.
    
    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation to filter the video prompts.
    
    Returns:
    - video_list (list): A list of video file paths that match the specified dimension.
    - None: reserved for future addition 
    
    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension and compiles a list of video paths and associated prompts in the specified language.
    
    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
    - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
    """
    video_list = []
    full_prompt_list = load_json(json_dir)
    for action_dict in full_prompt_list:
        if dimension in action_dict['dimension'] and 'video_list' in action_dict:
            cur_video_list = action_dict['video_list'] if isinstance(action_dict['video_list'], list) else [action_dict['video_list']]
            video_list += cur_video_list
    return video_list, None

def init_submodules(dimension_list, local=False, read_frame=False):
    submodules_dict = {}
    if local:
        logger.info("\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded.")
    for dimension in dimension_list:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if get_rank() > 0:
            barrier()
        if dimension == 'temporal_consistency':
            # read_frame = False
            if local:
                vit_b_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
                if not os.path.isfile(vit_b_path):
                    wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_b_path)]
                    subprocess.run(wget_command, check=True)
            else:
                vit_b_path = 'ViT-B/32'

            submodules_dict[dimension] = [vit_b_path, read_frame]
        elif dimension == 'motion_smoothness':
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            submodules_dict[dimension] = {
                    'config': f'{CUR_DIR}/third_party/amt/cfgs/AMT-S.yaml',
                    'ckpt': f'{CACHE_DIR}/amt_model/amt-s.pth'
                }
            details = submodules_dict[dimension]
            # Check if the file exists, if not, download it with wget
            if not os.path.isfile(details['ckpt']):
                print(f"File {details['ckpt']} does not exist. Downloading...")
                wget_command = ['wget', '-P', os.path.dirname(details['ckpt']),
                                'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth']
                subprocess.run(wget_command, check=True)
        elif dimension == 'action_control':
            submodules_dict[dimension] = {}
        elif dimension == 'object_consistency':
            droid_path = f'{CACHE_DIR}/droid_model/droid.pth'
            submodules_dict[dimension] = [droid_path]
        elif dimension == 'aesthetic_quality':
            aes_path = f'{CACHE_DIR}/aesthetic_model/emb_reader'
            if local:
                vit_l_path = f'{CACHE_DIR}/clip_model/ViT-L-14.pt'
                if not os.path.isfile(vit_l_path):
                    wget_command = ['wget' ,'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', '-P', os.path.dirname(vit_l_path)]
                    subprocess.run(wget_command, check=True)
            else:
                vit_l_path = 'ViT-L/14'
            submodules_dict[dimension] = [vit_l_path, aes_path]
        elif dimension == 'imaging_quality':
            musiq_spaq_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
            if not os.path.isfile(musiq_spaq_path):
                wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(musiq_spaq_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = {'model_path': musiq_spaq_path}
        elif dimension == 'scenario_consistency':
            submodules_dict[dimension] = {}
            
        if get_rank() == 0:
            barrier()
    return submodules_dict


def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    """
    prompt = Path(path).stem
    number_ending = r'-\d+$' # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, '', prompt)
    return prompt

Action_to_Conditions = {
    "forward":     {"keyboard_condition": [1,0,0,0,0,0], "mouse_condition": [0, 0]},
    "back":        {"keyboard_condition": [0,1,0,0,0,0], "mouse_condition": [0, 0]},
    "left":        {"keyboard_condition": [0,0,1,0,0,0], "mouse_condition": [0, 0]},
    "right":       {"keyboard_condition": [0,0,0,1,0,0], "mouse_condition": [0, 0]},
    "jump":        {"keyboard_condition": [0,0,0,0,1,0], "mouse_condition": [0, 0]},
    "attack":      {"keyboard_condition": [0,0,0,0,0,1], "mouse_condition": [0, 0]},
    
    "camera_up":   {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [ 0.15,  0   ]},
    "camera_down": {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [-0.15,  0   ]},
    "camera_l":    {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [ 0,     -0.15]},
    "camera_r":    {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [ 0,    0.15]},
    
    "camera_ur":   {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [ 0.15, 0.15]},
    "camera_ul":   {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [ 0.15, -0.15]},
    "camera_dl":   {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [-0.15, -0.15]},
    "camera_dr":   {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [-0.15, 0.15]},
    
    "empty":       {"keyboard_condition": [0,0,0,0,0,0], "mouse_condition": [0, 0]},
}
def get_actions_from_filepath(path: str):
    """
    0005_test_attack.mp4 -> condition
    0094_test_attack_camera_down.mp4 -> condition
    """
    file_name = os.path.basename(path)
    keyboard_conditions = []
    mouse_conditions = []
    ret_action = {
        "keyboard_condition": [0,0,0,0,0,0],
        "mouse_condition": [0,0]
    }
    for single_act in Action_to_Conditions.keys():
        if single_act in file_name:
            keyboard_conditions.append(Action_to_Conditions[single_act]['keyboard_condition'])
            mouse_conditions.append(Action_to_Conditions[single_act]['mouse_condition'])
    if len(keyboard_conditions) == 0 or len(keyboard_conditions) > 2: # max two action combinations
        raise
    else:
        for i in range(len(keyboard_conditions)):
            ret_action['keyboard_condition'] = [ret_action['keyboard_condition'][j]+keyboard_conditions[i][j] for j in range(len(ret_action['keyboard_condition']))]
            ret_action['mouse_condition'] = [ret_action['mouse_condition'][j]+mouse_conditions[i][j] for j in range(len(ret_action['mouse_condition']))]
        print(f"return action: {ret_action}") # for test
        return ret_action
        
def get_action_name_from_filepath(path: str):
    """
    0005_test_attack.mp4 -> attack
    0094_test_attack_camera_down.mp4 -> attack_camera_down
    """
    filename = os.path.basename(path)                     # e.g. '0094_test_attack_camera_down.mp4'
    name, _ = os.path.splitext(filename)                  # -> '0094_test_attack_camera_down'
    parts = name.split('_')                               # -> ['0094', 'test', 'attack', 'camera', 'down']
    action_parts = ['forward', 'back', 'left', 'right', 'jump', 'attack', 'camera', 'up', 'down', 'l', 'r', 'ur', 'ul', 'dl', 'dr', 'empty']

    # parts = [p for p in parts if not (p.isdigit() or p == 'test')]
    parts = [p for p in parts if p in action_parts]
    action_name = '_'.join(parts)
    # print(f"action_name: {action_name}")
    return action_name

def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_json(path):
    """
    Load a JSON file from the given file path.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
