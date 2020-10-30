#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"


mkdir -p "log"

for ((split=0;split<10;++split)); do
  export CUDA_VISIBLE_DEVICES=$((split%5))
  python "tools/create_tf_record.py" \
      --anno_vocab="data/Sentiments_List.txt" \
      --agreement=1 \
      --clean_topic_anno_json="data/cleaned_result/video_Topics_clean.json" \
      --clean_sentiment_anno_json="data/cleaned_result/video_Sentiments_clean.json" \
      --raw_sentiment_anno_json="data/raw_result/video_Sentiments_raw.json" \
      --video_id_path="output/video_id_list.${split}" \
      --feature_dir="video_feature" \
      --climax_feature_dir="climax_feature_v2" \
      --climax_prediction_dir="climax_predictions" \
      --common_object_feature_dir="common_object_feature" \
      --place_feature_dir="place_feature" \
      --emotic_feature_dir="emotic_feature" \
      --affectnet_feature_dir="affectnet_feature" \
      --shot_boundary_feature_dir="shot_boundary_feature" \
      --optical_flow_feature_dir="optical_flow_feature" \
      --audio_feature_dir="audio_feature" \
      --output_path="output/video_ads_agree1.record.${split}" \
> "log/create.${split}.log" 2>&1 &
done

exit 0
