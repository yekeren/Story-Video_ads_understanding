#!/bin/sh

python "tools/preprocess_emotic.py" \
    --data_dir="EMOTIC/train" \
    --json_output="EMOTIC/train.json" \
    || exit -1

python "tools/preprocess_emotic.py" \
    --data_dir="EMOTIC/validation" \
    --json_output="EMOTIC/valid.json" \
    || exit -1

python "tools/preprocess_emotic.py" \
    --data_dir="EMOTIC/test" \
    --json_output="EMOTIC/test.json" \
    || exit -1
exit 0
