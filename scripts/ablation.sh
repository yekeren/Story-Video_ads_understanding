#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:${HOME}/work2/video_sentiment"

mkdir -p "log"
number_of_steps="20000"
CONFIG_DIR="configs.bmvc.ablation"
LOGS_DIR="logs.bmvc.ablation"

splits=(0 2 4 6 8)

#  sent_lstm64_topic
#  sent_lstm64_coco
#  sent_lstm64_place
#  sent_lstm64_emotic
#  sent_lstm64_affectnet
#  sent_lstm64_audio
#  sent_lstm64_optical_flow
#  sent_lstm64_shot_boundary
#  sent_lstm64_only_feat
#  sent_lstm64_only_feat_v2
models=(
  sent_lstm64_climax
  sent_lstm64_only_feat_v2
)

for model in ${models[@]}; do
  for split in ${splits[@]}; do
    name="${model}"
  
    export CUDA_VISIBLE_DEVICES=$((split/2))
    #export CUDA_VISIBLE_DEVICES=0
    python "train/train.py" \
        --split="${split}" \
        --pipeline_proto="${CONFIG_DIR}/${model}.pbtxt" \
        --train_log_dir="${LOGS_DIR}/${name}/${split}/train" \
        --number_of_steps="${number_of_steps}" \
      > "log/${name}.${split}.train.log" 2>&1 &
    
    export CUDA_VISIBLE_DEVICES=$((split/2))
    #export CUDA_VISIBLE_DEVICES=1
    python "train/eval.py" \
        --split="${split}" \
        --pipeline_proto="${CONFIG_DIR}/${model}.pbtxt" \
        --train_log_dir="${LOGS_DIR}/${name}/${split}/train" \
        --eval_log_dir="${LOGS_DIR}/${name}/${split}/eval" \
        --saved_ckpts_dir="${LOGS_DIR}/${name}/${split}/saved_ckpts" \
        --number_of_steps="${number_of_steps}" \
        --sentiment_clean_annot_path="data/cleaned_result/video_Sentiments_clean.json" \
        --sentiment_raw_annot_path="data/raw_result/video_Sentiments_raw.json" \
        --sentiment_anno_vocab="data/Sentiments_List.txt" \
        --sentiment_vocab_path="data/sentiments.txt" \
        --topic_vocab_path="data/topics.txt" \
        --topic_clean_annot_path="data/cleaned_result/video_Topics_clean.json" \
        --json_path="results/eval.json" \
        --eval_interval_secs=30 \
        --eval_steps=800 \
        --eval_min_global_steps=500 \
        > "log/${name}.${split}.valid_val.log" 2>&1 &
  done
  wait
done



exit 0
