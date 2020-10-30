#!/bin/sh

set -x

# NOTE: this is a modified version of tensorflow models.
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models_nms:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models_nms/slim:$PYTHONPATH"


mkdir -p "log"

for ((split=0;split<10;++split)); do
  export CUDA_VISIBLE_DEVICES=$((split%5))
  python "tools/extract_shot_boundary.py" \
      --kyle_dir="/afs/cs.pitt.edu/projects/kovashka/ads_group/bmvc_2018/sb_20_30_40" \
      --video_dir="frame_data/" \
      --video_id_path="output/video_id_list.${split}" \
      --output_dir="place_feature" >> missing.txt
#  > "log/extract.${split}.log" 2>&1 &


done
