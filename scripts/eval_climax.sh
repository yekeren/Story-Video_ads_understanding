#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:${HOME}/work2/video_sentiment"



mkdir -p "log" 
number_of_steps="10000"
CONFIG_DIR="configs.bmvc.climax"
LOGS_DIR="logs.bmvc.climax"
RESULTS_DIR="saved_results.climax.test"

splits=(0 2 4 6 8)

models=(
  climax_lr
  climax_lr_feat
  climax_lr_only_feat
  climax_lstm64
  climax_lstm64_feat
  climax_lstm64_only_feat
)
models=(
  climax_lr_only_feat
  climax_lstm64_only_feat
)

for model in ${models[@]}; do
  name="${model}"
  
  for split in ${splits[@]}; do
    export CUDA_VISIBLE_DEVICES=2
    export CUDA_VISIBLE_DEVICES=$((split/2))
    python "train/test_climax.py" \
        --split="${split}" \
        --pipeline_proto="${CONFIG_DIR}/${model}.pbtxt" \
        --train_log_dir="${LOGS_DIR}/${name}/${split}/train" \
        --eval_log_dir="${LOGS_DIR}/${name}/${split}/eval" \
        --saved_ckpts_dir="${LOGS_DIR}/${name}/${split}/saved_ckpts" \
        --number_of_steps="${number_of_steps}" \
        --json_path="${RESULTS_DIR}/${name}.json.${split}" \
        --eval_interval_secs=30 \
        --eval_steps=800 \
        --eval_min_global_steps=50 \
        &
#        > "log/${name}.${split}.valid_val.log" 2>&1 &
  done
  wait
done

python "tools/gather_climax_results.py" || exit -1

cat "${RESULTS_DIR}/final.csv"

exit 0
