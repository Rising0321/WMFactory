---
license: apache-2.0
language:
- en
pipeline_tag: image-to-video
tags:
- World Model
---


<div align="center">
  <img src="assets/teaser.png">

<h1>LingBot-World: Advancing Open-source World Models</h1>

Robbyant Team

</div>


<div align="center">

[![Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Demo-00bfff)](https://technology.robbyant.com/lingbot-world)
[![Paper](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv)](https://arxiv.org/abs/2601.20540)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow)](https://huggingface.co/robbyant/lingbot-world-base-cam)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://www.modelscope.cn/models/Robbyant/lingbot-world-base-cam)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE.txt)

</div>

-----

We are excited to introduce **LingBot-World**, an open-sourced world simulator stemming from video generation. Positioned
as a top-tier world model, LingBot-World offers the following features. 
- **High-Fidelity & Diverse Environments**: It maintains high fidelity and robust dynamics in a broad spectrum of environments, including realism, scientific contexts, cartoon styles, and beyond. 
- **Long-Term Memory & Consistency**: It enables a minute-level horizon while preserving contextual consistency over time, which is also known as long-term memory. 
- **Real-Time Interactivity & Open Access**: It supports real-time interactivity, achieving a latency of under 1 second when producing 16 frames per second. We provide public access to the code and model in an effort to narrow the divide between open-source and closed-source technologies. We believe our release will empower the community with practical applications across areas like content creation, gaming, and robot learning.

## 🎬 Video Demo

<div align="center">
    <video width="100%" controls>
        <source src="https://gw.alipayobjects.com/v/huamei_u94ywh/afts/video/XQk7Rb44qJwAAAAAgfAAAAgAfoeUAQBr" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>


## 🔥 News
- Jan 29, 2026: 🎉 We release the technical report, code, and models for LingBot-World.

<!-- ## 🔖 Introduction of LingBot-World
We present **LingBot-World**, an **open-sourced** world simulator stemming from video generation. Positioned
as a top-tier world model, LingBot-World offers the following features. 
- It maintains high fidelity and robust dynamics in a broad spectrum of environments, including realism, scientific contexts, cartoon styles, and beyond. 
- It enables a minute-level horizon while preserving contextual consistency over time, which is also known as **long-term memory**. 
- It supports real-time interactivity, achieving a latency of under 1 second when producing 16 frames per second. We provide public access to the code and model in an effort to narrow the divide between open-source and closed-source technologies. We believe our release will empower the community with practical applications across areas like content creation, gaming, and robot learning. -->

## ⚙️ Quick Start
This codebase is built upon [Wan2.2](https://github.com/Wan-Video/Wan2.2). Please refer to their documentation for installation instructions.
### Installation
Clone the repo:
```sh
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world
```
Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```
Install [`flash_attn`](https://github.com/Dao-AILab/flash-attention):
```sh
pip install flash-attn --no-build-isolation
```
### Model Download

| Model | Control Signals | Resolution | Download Links |
| :---  | :--- | :--- | :--- |
| **LingBot-World-Base (Cam)** | Camera Poses | 480P & 720P | 🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-base-cam) 🤖 [ModelScope](https://www.modelscope.cn/models/Robbyant/lingbot-world-base-cam) |
| **LingBot-World-Base (Act)** | Actions | 480P & 720P |  🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-base-act)|
| **LingBot-World-Fast**       |    -    | - | *To be released* |

Download models using huggingface-cli:
```sh
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
```
<!-- Download models using modelscope-cli: -->
<!-- ```sh
pip install modelscope
modelscope download robbyant/lingbot-world-base-cam --local_dir ./lingbot-world-base-cam
``` -->
### Inference
Our model supports video generation at both 480P and 720P resolutions. You can find data samples for inference in the `examples/` directory, which includes the corresponding input images, prompts, and control signals. To enable long video generation, we utilize multi-GPU inference powered by FSDP and DeepSpeed Ulysses.
- 480P:
``` sh
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 --ckpt_dir lingbot-world-base-cam --image examples/00/image.jpg --action_path examples/00 --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 161 --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
```
- 720P:
``` sh
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 720*1280 --ckpt_dir lingbot-world-base-cam --image examples/00/image.jpg --action_path examples/00 --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 161 --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
```
Alternatively, you can run inference without control actions:
``` sh
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 --ckpt_dir lingbot-world-base-cam --image examples/00/image.jpg --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 161 --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
```
Tips:
If you have sufficient CUDA memory, you may increase the `frame_num` parameter to a value such as 961 to generate a one-minute video at 16 FPS.


## 🔗 Links
- **Github**: https://github.com/robbyant/lingbot-world
- **Paper**: https://arxiv.org/abs/2601.20540
- **Project Page**: https://technology.robbyant.com/lingbot-world


## 📚 Related Projects
- [HoloCine](https://holo-cine.github.io/)
- [Ditto](https://editto.net/)
- [WorldCanvas](https://worldcanvas.github.io/)
- [RewardForcing](https://reward-forcing.github.io/)
- [CoDeF](https://qiuyu96.github.io/CoDeF/)

## 📜 License
This project is licensed under the Apache 2.0 License. Please refer to the [LICENSE file](LICENSE.txt) for the full text, including details on rights and restrictions.

## ✨ Acknowledgement
We would like to express our gratitude to the [Wan2.2](https://github.com/Wan-Video/Wan2.2) team for open-sourcing their code and models. Their contributions have been instrumental to the development of this project.

## 📖 Citation
If you find this work useful for your research, please cite our paper:

```
@article{lingbot-world,
      title={Advancing Open-source World Models}, 
      author={Robbyant Team},
      journal={arXiv preprint arXiv:2601.20540},
      year={2026}
}
```