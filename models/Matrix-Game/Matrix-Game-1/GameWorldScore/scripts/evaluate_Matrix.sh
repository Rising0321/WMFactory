#!/bin/bash

dimensions=("temporal_consistency" "aesthetic_quality" "imaging_quality" "action_control" "motion_smoothness" "object_consistency" "scenario_consistency")
base_path="data/Matrix_v1"

for i in "${!dimensions[@]}"; do
    dimension=${dimensions[i]}

    # Construct the video path
    videos_path=$base_path
    echo "$dimension $videos_path"

    # Run the evaluation script
    python evaluate.py --videos_path $videos_path --dimension $dimension --mode "GameWorld"
done