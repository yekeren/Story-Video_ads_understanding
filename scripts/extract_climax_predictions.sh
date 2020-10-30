#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"


mkdir -p "log"

python "tools/extract_climax_predictions.py" \
    --video_dir="frame_data/" \
    --climax_annotation_path="output/climax_lstm64_feat.json" \
    --output_dir="climax_predictions" \
    || exit -1

exit 0
