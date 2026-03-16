---






license: apache-2.0
pipeline_tag: image-to-video
tags:
- Text-to-Video
- Image-to-Video
- Diffusion Video Model
- World Model
---

# Yume-1.5: A Text-Controlled Interactive World Generation Model

Yume-1.5 is a framework designed to generate realistic, interactive, and continuous worlds from a single image or text prompt. It supports keyboard-based exploration of the generated environments through a framework that integrates context compression and real-time streaming acceleration.



- [**Paper (Yume-1.5)**](https://huggingface.co/papers/2512.22096)
- [**Paper (Yume-1.0)**](https://huggingface.co/papers/2507.17744)
- [**Project Page**](https://stdstu12.github.io/YUME-Project/)
- [**GitHub Repository**](https://github.com/stdstu12/YUME)

## Features
- **Long-video generation**: Unified context compression with linear attention.
- **Real-time acceleration**: Powered by bidirectional attention distillation.
- **Text-controlled events**: Method for generating specific world events via text prompts.

## Usage

For detailed installation and setup instructions, please refer to the [GitHub repository](https://github.com/stdstu12/YUME). 

### Inference Example
To perform image-to-video generation using the provided scripts:

```bash
# Generate videos from images in the specified directory
bash scripts/inference/sample_jpg.sh --jpg_dir="./jpg" --caption_path="./caption.txt"
```

## Citation

If you use Yume for your research, please cite the following:

```bibtex
@article{mao2025yume,
  title={Yume: An Interactive World Generation Model},
  author={Mao, Xiaofeng and Lin, Shaoheng and Li, Zhen and Li, Chuanhao and Peng, Wenshuo and He, Tong and Pang, Jiangmiao and Chi, Mingmin and Qiao, Yu and Zhang, Kaipeng},
  journal={arXiv preprint arXiv:2507.17744},
  year={2025}
}
@article{mao2025yume,
  title={Yume-1.5: A Text-Controlled Interactive World Generation Model},
  author={Mao, Xiaofeng and Li, Zhen and Li, Chuanhao and Xu, Xiaojie and Ying, Kaining and He, Tong and Pang, Jiangmiao and Qiao, Yu and Zhang, Kaipeng},
  journal={arXiv preprint arXiv:2512.22096},
  year={2025}
}
```