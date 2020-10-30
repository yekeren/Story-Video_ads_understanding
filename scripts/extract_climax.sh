#!/bin/sh

set -x

export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models:$PYTHONPATH"
export PYTHONPATH="${HOME}/work2/video_sentiment/tensorflow_models/slim:$PYTHONPATH"


mkdir -p "log"

python "tools/extract_climax_from_raw.py" \
    --video_dir="frame_data/" \
    --climax_annotation_path="data/climax/Batch.csv" \
    --output_path="data/climax/climax_data_v2.csv" \
    || exit -1

python "tools/extract_climax.py" \
    --video_dir="frame_data/" \
    --climax_annotation_path="data/climax/climax_data_v2.csv" \
    --output_dir="climax_feature_v2" \
    || exit -1

exit 0
