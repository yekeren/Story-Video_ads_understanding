#!/bin/sh

set -x

# NOTE: this is a modified version of tensorflow models.
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models_nms:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models_nms/slim:$PYTHONPATH"


mkdir -p "log"

for ((split=0;split<10;++split)); do
  export CUDA_VISIBLE_DEVICES=$((split%5))
  python "tools/extract_emotic.py" \
      --model_proto="configs/emotic_detection.pbtxt" \
      --checkpoint_dir="zoo/emotic/train" \
      --label_map="configs/emotic_label_map.pbtxt" \
      --video_dir="frame_data/" \
      --video_id_path="output/video_id_list.${split}" \
      --output_dir="emotic_feature" \
      --output_vocab="configs/emotic_vocab.txt" \
      --min_object_size="0.1" \
      --batch_size="20" \
> "log/extract.${split}.log" 2>&1 &

done
