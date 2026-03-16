#!/bin/bash

dimensions=("temporal_consistency" "aesthetic_quality" "imaging_quality" "action_control" "motion_smoothness" "object_consistency")
base_path="data/Matrix_v1"

videos_path=$base_path
echo "$dimension $videos_path"

python evaluate_per_action.py --videos_path $videos_path --dimension "${dimensions[@]}" --mode "GameWorld"