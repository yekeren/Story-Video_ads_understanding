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
  sent_lstm64
  sent_lstm64_feat_v2
)


python "tools/gather_sent_results_for_vis.py" || exit -1


exit 0
