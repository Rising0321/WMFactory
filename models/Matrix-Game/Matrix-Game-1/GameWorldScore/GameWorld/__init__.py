import os

from .utils import get_prompt_from_filename, init_submodules, save_json, load_json, get_action_name_from_filepath
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0


class GameWorld(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that GameWorld originally provides
        self.output_path = output_path              # output directory to save GameWorld results
        os.makedirs(self.output_path, exist_ok=True)    
    
    def build_GameWorld_dimension_sub_list(self, ):
        return ["temporal_consistency", "aesthetic_quality", "imaging_quality", "action_control", "motion_smoothness", "object_consistency", "scenario_consistency"]        

    def check_dimension_requires_extra_info(self, dimension_list):
        dim_custom_not_supported = set(dimension_list) & set([
            'object_class', 'multiple_objects', 'scene', 'appearance_style', 'color', 'spatial_relationship'
        ])

        assert len(dim_custom_not_supported) == 0, f"dimensions : {dim_custom_not_supported} not supported for custom input"

    def build_custom_image_dict(self, directory):
        '''
        dict {
            image_name: image_path
        }
        '''
        image_dict = {}
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                image_name, extension = os.path.splitext(filename)
                extension = extension.lower()

                if extension in ['.jpg', '.jpeg', '.png']:
                    image_dict[image_name] = file_path
        
        return image_dict
 

    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[],  special_str='', verbose=False, custom_image_folder=None, mode='GameWorld', **kwargs):
        '''
        output: full_info_list, example: [{"prompt_en": "a photo of a cat", "dimension": ["subject_consistency"], "video_list": ["cat.mp4"]}]
        '''
        
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions
        if mode=='custom_input':
            self.check_dimension_requires_extra_info(dimension_list)
            if custom_image_folder:
                custom_image_dict = self.build_custom_image_dict(custom_image_folder)
            
            if os.path.isfile(videos_path):
                if custom_image_folder is None:
                    cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path]}]
                else:
                    cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path], "custom_image_path": custom_image_dict[get_prompt_from_filename(videos_path)]}]
                
                if len(prompt_list) == 1:
                    cur_full_info_list[0]["prompt_en"] = prompt_list[0]
            else:
                video_names = os.listdir(videos_path)

                cur_full_info_list = []

                if custom_image_folder is None:
                    for filename in video_names:
                        postfix = Path(os.path.join(videos_path, filename)).suffix
                        if postfix.lower() not in ['.mp4', '.gif',]: #  '.jpg', '.png'
                            continue
                        cur_full_info_list.append({
                            "prompt_en": get_prompt_from_filename(filename), 
                            "dimension": dimension_list, 
                            "video_list": [os.path.join(videos_path, filename)]
                        })
                else:
                    for filename in video_names:
                        postfix = Path(os.path.join(videos_path, filename)).suffix
                        if postfix.lower() not in ['.mp4', '.gif']: #  '.jpg', '.png'
                            continue
                        cur_full_info_list.append({
                            "prompt_en": get_prompt_from_filename(filename), 
                            "dimension": dimension_list, 
                            "video_list": [os.path.join(videos_path, filename)],
                            "custom_image_path": custom_image_dict[get_prompt_from_filename(filename)]
                        })

                if len(prompt_list) > 0:
                    # 实现 prompt与video_list对应关系
                    prompt_list = {os.path.join(videos_path, path): prompt_list[path] for path in prompt_list}
                    assert len(prompt_list) >= len(cur_full_info_list), """
                        Number of prompts should match with number of videos.\n
                        Got {len(prompt_list)=}, {len(cur_full_info_list)=}\n
                        To read the prompt from filename, delete --prompt_file and --prompt_list
                        """

                    all_video_path = [os.path.abspath(file) for file in list(chain.from_iterable(vid["video_list"] for vid in cur_full_info_list))]
                    backslash = "\n"
                    assert len(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list])) == 0, f"""
                    The prompts for the following videos are not found in the prompt file: \n
                    {backslash.join(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list]))}
                    """

                    video_map = {}
                    for prompt_key in prompt_list:
                        video_map[os.path.abspath(prompt_key)] = prompt_list[prompt_key]

                    for video_info in cur_full_info_list:
                        video_info["prompt_en"] = video_map[os.path.abspath(video_info["video_list"][0])]

        elif mode=='GameWorld':
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)

            for dim_dict in full_info_list:
                # print(dimension_list, dim_dict["dimension"]) # test
                if set(dimension_list) & set(dim_dict["dimension"]):
                    dim_dict['video_list'] = []
                    for video_name in video_names:
                        # action_name = get_action_name_from_filepath(video_name)
                        dim_dict['video_list'].append(os.path.join(videos_path, video_name))
                    cur_full_info_list.append(dim_dict)

        elif mode=='GameWorld_per_scene':
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            if 'scene_index' not in kwargs:
                assert scene is not None, "Please specify the scene to be evaluated with --scene"
            else:
                scene_index = kwargs['scene_index']

            for action_dict in full_info_list:
                if set(dimension_list) & set(action_dict["dimension"]):
                    action_dict['video_list'] = []
                    for video_name in video_names:
                        index = int(video_name.split('_')[0])
                        # print(f"index: {index}") # test
                        index = (index % 32) // 4
                        action_name = get_action_name_from_filepath(video_name)
                        if action_name == action_dict['action'] and index == scene_index:
                            action_dict['video_list'].append(os.path.join(videos_path, video_name))
                    cur_full_info_list.append(action_dict)

        elif mode=='GameWorld_custom':
            full_info_list = [
                                {
                                    "action": "forward_left",
                                    "dimension": [
                                        "temporal_consistency",
                                        "aesthetic_quality",
                                        "imaging_quality",
                                        # "temporal_flickering",
                                        "motion_smoothness"
                                    ]
                                }
                            ]
            video_names = []
            for root, dirs, files in os.walk(videos_path):
                for file in files:
                    if file.endswith(('.mp4')):
                        video_names.append(os.path.join(root, file))
            print(f"length_video_names: {len(video_names)}")
            # video_names = os.listdir(videos_path)
            for action_dict in full_info_list:
                if set(dimension_list) & set(action_dict["dimension"]):
                    action_dict['video_list'] = []
                    for video_name in video_names:
                        action_dict['video_list'].append(os.path.join(videos_path, video_name))
                    cur_full_info_list.append(action_dict)

        else:
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]): 
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    for i in range(5): # video index for the same prompt
                        intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'
                        if intended_video_name in video_names: # if the video exists
                            intended_video_path = os.path.join(videos_path, intended_video_name)
                            prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print0(f'Successfully found video: {intended_video_name}')
                        else:
                            print0(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                    cur_full_info_list.append(prompt_dict)

        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='GameWorld', **kwargs):
        results_dict = {}
        if dimension_list is None:
            # dimension_list = self.build_full_dimension_list()
            dimension_list = self.build_GameWorld_dimension_sub_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'GameWorld.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')

    def evaluate_per_scene(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='GameWorld', **kwargs):
        results_dict = {}
        if dimension_list is None:
            # dimension_list = self.build_full_dimension_list()
            dimension_list = self.build_GameWorld_dimension_sub_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'GameWorld.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        if 'scene_index' in kwargs:
            output_name = os.path.join(self.output_path, f"{name}_{kwargs['scene_index']}_eval_results.json")
        else:
            output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')
    def evaluate_per_action(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='GameWorld', **kwargs):
        results_dict = {}
        if dimension_list is None:
            # dimension_list = self.build_full_dimension_list()
            dimension_list = self.build_GameWorld_dimension_sub_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        action_to_test = ['forward', 'back', 'left', 'right', 'jump', 'attack']
        # action_to_test = ['camera_up', 'camera_down', 'camera_l', 'camera_r', 'camera_ul', 'camera_ur', 'camera_dl', 'camera_dr']

        cur_full_info_list = load_json(cur_full_info_path)

        for action_info in cur_full_info_list:
            # print(f"action_info: {action_info}") # test
            action_name = action_info['action']  # 获取当前 action，比如 "forward_left"
            if action_name not in action_to_test:
                continue
            action_video_list = action_info['video_list']
            print0(f'cur_action: {action_name}, cur_video_list: {len(action_video_list)}') # test

            results_dict = {}  # 每个 action 有自己单独的 results_dict

            for dimension in dimension_list:
                try:
                    dimension_module = importlib.import_module(f'GameWorld.{dimension}')
                    evaluate_func = getattr(dimension_module, f'compute_{dimension}')
                except Exception as e:
                    raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
                
                submodules_list = submodules_dict[dimension]

                
                # 为了统一输入格式，保存一个临时的json 文件
                action_info_path = os.path.join("/mnt/datasets_genie/puyi.wang/GameWorldScore/action_jsons", name+f"_{action_name}_full_info.json")
                action_info_file = [action_info]
                save_json(action_info_file, action_info_path)
                # 注意这里只传当前 action 的视频列表进去
                results = evaluate_func(action_info_path, self.device, submodules_list, **kwargs)

                results_dict[dimension] = results

            output_name = os.path.join(self.output_path, f"{name}_{action_name}_eval_results.json")
            
            if get_rank() == 0:
                save_json(results_dict, output_name)
                print0(f'Evaluation results for action [{action_name}] saved to {output_name}')
