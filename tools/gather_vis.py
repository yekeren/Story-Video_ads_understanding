import os
import re
import sys
import json
from train import eval_utils

import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import average_precision_score

sentiment_anno_vocab = "data/Sentiments_List.txt"
sentiment_raw_annot_path = "data/raw_result/video_Sentiments_raw.json"
sentiment_clean_annot_path = "data/cleaned_result/video_Sentiments_clean.json"

results_dir = './saved_results.test'
names = [
  'vis_sent_lstm64_feat',
]

if __name__ == '__main__':
  name = names[0]
  data = {}
  for split in [0, 2, 4, 6, 8]:
    filename = os.path.join(results_dir, '{}.json.{}'.format(name, split))
    assert os.path.isfile(filename), filename
  
    with open(filename, 'r') as fp:
      data.update(json.loads(fp.read()))
  print >> sys.stderr, 'Load %i records.' % (len(data))

  filename = os.path.join(results_dir, '{}.json'.format(name))
  with open(filename, 'w') as fp:
    fp.write(json.dumps(data))
