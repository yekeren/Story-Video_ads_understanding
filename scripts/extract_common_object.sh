#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"


mkdir -p "log"

for ((split=8;split<10;++split)); do
  export CUDA_VISIBLE_DEVICES=$((split%5))
  python "tools/extract_common_object.py" \
      --model_proto="configs/coco_detection.pbtxt" \
      --checkpoint="models/ssd_inception_v2_coco_11_06_2017/model.ckpt" \
      --label_map="configs/mscoco_label_map.pbtxt" \
      --video_dir="frame_data/" \
      --video_id_path="output/video_id_list.${split}" \
      --output_dir="common_object_feature" \
      --output_vocab="configs/mscoco_vocab.txt" \
      --num_classes="90" \
      --min_object_size="0.1" \
      --batch_size="20" \
> "log/extract.${split}.log" 2>&1 &

done
