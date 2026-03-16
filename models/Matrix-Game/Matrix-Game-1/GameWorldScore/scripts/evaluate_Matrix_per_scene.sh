#!/bin/bash

dimensions=("temporal_consistency" "aesthetic_quality" "imaging_quality" "action_control" "motion_smoothness" "object_consistency")
base_path="data/Matrix_v1"

# Construct the video path
videos_path=$base_path
echo "$dimension $videos_path"

# Run the evaluation script
python evaluate_per_scene.py --videos_path $videos_path --dimension "${dimensions[@]}" --mode "GameWorld_per_scene"
