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

python "tools/gather_sent_baseline.py" || exit -1
cp "${RESULTS_DIR}/final.csv" "${RESULTS_DIR}/sentiment_baseline.csv" || exit -1

cat "${RESULTS_DIR}/sentiment_baseline.csv"

exit 0
