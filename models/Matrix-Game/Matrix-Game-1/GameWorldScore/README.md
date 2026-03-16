# GameWorld: A Unified Benchmark for Minecraft World Models

<a name="Overview"></a>
## :mega: Overview
With the rise of world models, an increasing number of studies have focused on the Minecraft environment, aiming to leverage video generation models to produce videos that not only align with user action inputs, but also adhere to the physical rules inherent in the game. However, existing research lacks a unified evaluation benchmark to consistently measure and compare model performance in the setting with actions input. To address these challenges, we propose **GameWorld**, a unified benchmark that evaluates not only the perceptual quality of generated videos, but also their *controllability* and *physical plausibility*.


<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results
#### Overall Performance
| Model     | Image Quality ↑ | Aesthetic Quality ↑ | Temporal Cons. ↑ | Motion Smooth. ↑ | Keyboard Acc. ↑ | Mouse Acc. ↑ | Object Cons. ↑ | Scenario Cons. ↑ 
|-----------|------------------|-------------|-------------------|-------------------|------------------|---------------|-------------|-------------|
| Oasis     | 0.65             | 0.48        | 0.94              | **0.98**          | 0.77             | 0.56          | 0.56        | 0.86 |
| MineWorld | 0.69             | 0.47        | 0.95              | **0.98**          | 0.86             | 0.64          | 0.51        | 0.92 |
| **Ours**  | **0.72**         | **0.49**    | **0.97**          | **0.98**          | **0.95**         | **0.95**      | **0.76**    |  **0.93**    |


<a name="Installation"></a>
## :hammer: Installation
### Environment Setup <a name="Environment_Setup"></a>
First, create an environment if you want and install required dependencied:
```shell
# Create the environment (example command)
conda create -n GameWorld python=3.10
# Activate the environment
conda activate GameWorld
# Install dependencies
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

To install DROID-SLAM, you should
```shell
cd GameWorld/third_party/DROID-SLAM
python setup.py install
```
`torch_scatter` is needed, for which you should pay attention to your own pytorch and cuda version. In our settings, it is
```shell
# demo pip install torch-scatter -f https://data.pyg.org/whl/<your-torch-version>+<your-cuda-version>.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```
### Dependencies Setup <a name="Environment_Setup"></a>
```shell
# imaging_quality
mkdir -p ~/.cache/GameWorld_bench/pyiqa_model
wget https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth -P ~/.cache/GameWorld_bench/pyiqa_model
# motion_smoothness
mkdir -p ~/.cache/GameWorld_bench/amt_model
wget https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth -P ~/.cache/GameWorld_bench/amt_model
# action_control
mkdir -p "~/.cache/GameWorld_bench/IDM"
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model -P ~/.cache/GameWorld_bench/IDM
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights -P ~/.cache/GameWorld_bench/IDM
# 3d_consistency
mkdir -p ~/.cache/GameWorld_bench/droid_model
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O ~/.cache/GameWorld_bench/droid_model/droid.pth
```


<a name="Usage"></a>
## ✅ Usage
#### World Generation <a name="world-generation"></a>
Before evaluation, you should have the videos prepared. The videos should have a format of {prefix}_{action_name}.mp4. If you want to evaluate per environment, then the prefix should be index. Otherwise, it could be everything (e.g. basename of init_image). 

For "temporal_consistency", "aesthetic_quality", "imaging_quality", "action_control", "motion_smoothness", and "object_consistency", we test 76 actions for 32 init_images, which generates 2432 videos in a format of follows, with _the same action grouped together_:
- data_folder1
  - 0000_attack.mp4
  - 0001_attack.mp4
  - ...
  - 0031_attack.mp4
  - 0032_attack_camera_dl.mp4
  - ...
  - 2431_right_jump.mp4

For "scenario_consistency", the videos to test are generated based on the same 32 init_images as above, controlled by 8 mirror actions in a format of:
- data_folder1
  - 0000_Mirror_0.mp4
  - 0001_Mirror_0.mp4
  - ...
  - 0031_Mirror_0.mp4
  - 0032_Mirror_1.mp4
  - ...
  - 0255_Mirror_7.mp4


#### Evaluation <a name="Evaluation"></a>
For overall metrics calculation,
```shell
bash scripts/evaluate_Matrix.sh
```

If you wants the results of each scene（optional）,
```shell
bash scripts/evaluate_Matrix_per_scene.sh
```

For results of each action（optional）,
```shell
bash scripts/evaluate_Matrix_per_action.sh
```

## 🤗 Acknowledgments
Part of our codes are based on [VBench](https://github.com/Vchitect/VBench) and [VPT](https://github.com/openai/Video-Pre-Training). Thanks for their efforts and innovations. Thank you to everyone who contributed their wisdom and efforts to this project.
