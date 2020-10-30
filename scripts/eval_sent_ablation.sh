#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:${HOME}/work2/video_sentiment"



mkdir -p "log" 
number_of_steps="10000"
CONFIG_DIR="configs.bmvc.ablation"
LOGS_DIR="logs.bmvc.ablation"
RESULTS_DIR="saved_results.test"

splits=(0 2 4 6 8)

#  sent_lstm64
#  sent_lstm64_topic
#  sent_lstm64_coco
#  sent_lstm64_place
#  sent_lstm64_emotic
#  sent_lstm64_affectnet
#  sent_lstm64_audio
#  sent_lstm64_optical_flow
#  sent_lstm64_shot_boundary
#  sent_lstm64_only_feat
#  sent_lstm64_feat
#  sent_lstm64_climax
#  sent_lstm64_only_feat_v2
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

python "tools/gather_sent_ablation.py" || exit -1
cp "${RESULTS_DIR}/final.csv" "${RESULTS_DIR}/sentiment_ablation.csv" || exit -1
cp "${RESULTS_DIR}/details.csv" "${RESULTS_DIR}/sentiment_ablation_details.csv" || exit -1

cat "${RESULTS_DIR}/sentiment_ablation.csv"
cat "${RESULTS_DIR}/sentiment_ablation_details.csv"






#exit 0
#
#
#number_of_steps="10000"
#mkdir -p "log"
#CONFIG_DIR="configs.bof"
#CONFIG_DIR="configs"
#CONFIG_DIR="configs.lstm.50d"
#
#RESULTS_DIR="saved_results.valid"
#
#models=(
#  bof_semantic_emotic_comb
#  bof_semantic_emotic_sent
#  bof_semantic_emotic
#  bof_semantic_object_comb
#  bof_semantic_object_sent
#  bof_semantic_object
#)
#models=(
#  inception_kb
#  inception_kb_v2
#)
#models=(
#  inception
#  object
#  emotic
#  semantic_emotic_comb
#  semantic_emotic_sent
#  semantic_emotic
#  semantic_object_comb
#  semantic_object_sent
#  semantic_object
#  semantic_place_comb
#  semantic_place_sent
#  semantic_place
#)
#models=(
#  lstm_50d_semantic_object
#  lstm_50d_semantic_object_sent
#  lstm_50d_semantic_object_comb
#  lstm_50d_semantic_emotic
#  lstm_50d_semantic_emotic_sent
#  lstm_50d_semantic_emotic_comb
#  lstm_50d_semantic_place
#  lstm_50d_semantic_place_sent
#  lstm_50d_semantic_place_comb
#)
#
#models=(
#  inception
#  inception_kb
#  inception_kb_v2
#  object
#  emotic
#  bof_semantic_emotic_comb
#  bof_semantic_emotic_sent
#  bof_semantic_emotic
#  bof_semantic_object_comb
#  bof_semantic_object_sent
#  bof_semantic_object
#  semantic_emotic_comb
#  semantic_emotic_sent
#  semantic_emotic
#  semantic_object_comb
#  semantic_object_sent
#  semantic_object
#  semantic_place_comb
#  semantic_place_sent
#  semantic_place
#  lstm_50d_semantic_emotic_comb
#  lstm_50d_semantic_emotic_sent
#  lstm_50d_semantic_emotic
#  lstm_50d_semantic_object_comb
#  lstm_50d_semantic_object_sent
#  lstm_50d_semantic_object
#  lstm_50d_semantic_place_comb
#  lstm_50d_semantic_place_sent
#  lstm_50d_semantic_place
#)
#
##for model in ${models[@]}; do
##  name="${model}"
##  
##  export CUDA_VISIBLE_DEVICES=0
##  
##  python "train/test.py" \
##      --pipeline_proto="${CONFIG_DIR}/${model}_valid.pbtxt" \
##      --train_log_dir="logs/${name}/train" \
##      --eval_log_dir="logs/${name}/eval" \
##      --saved_ckpts_dir="logs/${name}/saved_ckpts" \
##      --number_of_steps="${number_of_steps}" \
##      --sentiment_clean_annot_path="data/cleaned_result/video_Sentiments_clean.json" \
##      --sentiment_raw_annot_path="data/raw_result/video_Sentiments_raw.json" \
##      --sentiment_anno_vocab="data/Sentiments_List.txt" \
##      --sentiment_vocab_path="data/sentiments.txt" \
##      --topic_vocab_path="data/topics.txt" \
##      --topic_clean_annot_path="data/cleaned_result/video_Topics_clean.json" \
##      --json_path="${RESULTS_DIR}/${model}.json" \
##      --eval_interval_secs=30 \
##      --eval_steps=800 \
##      --eval_min_global_steps=50 &
##done
##exit 0
#
#rm "${RESULTS_DIR}/final.csv"
#for model in ${models[@]}; do
#  name="${model}"
#  mAP=`grep -o -E "\"mAP\": [0-9.]+" "${RESULTS_DIR}/${model}.json" | awk '{print $2}'`
#  accuracy=`grep -o -E "\"accuracy\": [0-9.]+" "${RESULTS_DIR}/${model}.json" | awk '{print $2}'`
#  echo "${name},${accuracy},${mAP}"
#  echo "${name},${accuracy},${mAP}" >> "${RESULTS_DIR}/final.csv"
#done

exit 0

