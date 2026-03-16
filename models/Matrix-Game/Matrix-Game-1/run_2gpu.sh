export PYTHONPATH=/path/to/Matrix-Game/:$PYTHONPATH

BASE_PATH=./Matrix-Game
IMAGE_PATH=./input.png
OUTPUT_PATH=./output
ICON_PATH=./mouse.png

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone parallel_infer.py \
  --num-pre-frames 5 \
  --video-length 13  \
  --cfg-scale 6 \
  --denoise-type flow \
  --flow-shift 15.0 \
  --flow-reverse \
  --flow-solver euler \
  --text-encoder-precision bf16 \
  --vae 884-16c-hy \
  --vae-precision bf16 \
  --vae-tiling \
  --vae-path ${BASE_PATH}/vae \
  --text-encoder-path ${BASE_PATH} \
  --dit-path ${BASE_PATH}/dit \
  --use-cpu-offload   \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --num-inference-steps 30 \
  --output-path ${OUTPUT_PATH} \
  --input-image-path ${IMAGE_PATH} \
  --mouse-icon-path ${ICON_PATH}
