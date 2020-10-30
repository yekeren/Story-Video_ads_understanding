#!/bin/sh

set -x

#splits=(0 2 4 6 8)
#for split in ${splits[@]}; do
#  python "tools/create_glove_npz.py" \
#      --vocab_path="data/answer_vocab.txt.${split}" \
#      --output_npz_path="data/answer_vocab_200d.npz.${split}" \
#      --data_path="zoo/glove.6B.200d.txt" \
#      --unk_token="false" \
#      || exit -1
#done
#
#exit 0
#
#python "tools/create_glove_npz.py" \
#    --vocab_path="data/densecap_vocab.txt" \
#    --output_npz_path="data/densecap_vocab_200d.npz" \
#    --data_path="zoo/glove.6B.200d.txt" \
#    || exit -1

python "tools/create_glove_npz.py" \
    --vocab_path="data/stmt_vocab.txt" \
    --output_npz_path="data/stmt_vocab_200d.npz" \
    --data_path="zoo/glove.6B.200d.txt" \
    || exit -1

exit 0
