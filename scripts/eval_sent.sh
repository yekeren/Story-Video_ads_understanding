#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:${HOME}/work2/video_sentiment"

mkdir -p "log" 
number_of_steps="10000"
CONFIG_DIR="configs.bmvc2"
LOGS_DIR="logs.bmvc3"
RESULTS_DIR="saved_results.test"

splits=(0 2 4 6 8)

#  sent_bof
#  sent_bof_feat
#  sent_bof_feat_v2
#  sent_bof_semantic
#  sent_lstm64
#  sent_lstm64_feat
#  sent_lstm64_feat_v2
#  sent_lstm64_semantic
models=(
)

for model in ${models[@]}; do
  name="${model}"
  
  for split in ${splits[@]}; do
    export CUDA_VISIBLE_DEVICES=$((split/2))
    python "train/test.py" \
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
        --json_path="${RESULTS_DIR}/${name}.json.${split}" \
        --eval_interval_secs=30 \
        --eval_steps=800 \
        --eval_min_global_steps=500 \
        &

#> "log/${name}.${split}.valid_val.log" 2>&1 &
  done
  wait
done

python "tools/gather_sent_results.py" || exit -1
cp "${RESULTS_DIR}/final.csv" "${RESULTS_DIR}/sentiment.csv" || exit -1
cp "${RESULTS_DIR}/details.csv" "${RESULTS_DIR}/sentiment_details.csv" || exit -1

cat "${RESULTS_DIR}/sentiment.csv"






exit 0
