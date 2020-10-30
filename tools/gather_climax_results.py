import os
import re
import sys
import json
from train import eval_utils
from tensorflow import logging

import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import average_precision_score

results_dir = './saved_results.climax.test'
names = [
  'climax_lr',
  'climax_lr_feat',
  'climax_lr_only_feat',
  'climax_lstm64',
  'climax_lstm64_feat',
  'climax_lstm64_only_feat',
]

def evaluate(data, name):
  (precision_at_1, precision_at_3, precision_at_5, precision_at_10, diff_ratio
   ) = eval_utils.evaluate_climax_accuracy(data)

  print('=' * 128)
  print(name)
  print('precision@1=%.4lf' % precision_at_1)
  print('precision@3=%.4lf' % precision_at_3)
  print('precision@5=%.4lf' % precision_at_5)
  print('precision@10=%.4lf' % precision_at_10)
  print('diff_ratio=%.4lf' % diff_ratio)

  return {
    'precision@1': precision_at_1,
    'precision@3': precision_at_3,
    'precision@5': precision_at_5,
    'precision@10': precision_at_10,
    'diff_ratio': diff_ratio,
  }

if __name__ == '__main__':
  results = []

  # Evaluate human annotation.
  for name in names:
    # Aggregate results.
    data = {}
    for split in [0, 2, 4, 6, 8]:
      filename = os.path.join(results_dir, '{}.json.{}'.format(name, split))
      assert os.path.isfile(filename), filename
  
      with open(filename, 'r') as fp:
        data.update(json.loads(fp.read()))
  
    metrics = evaluate(data, name)
  
    print >> sys.stderr, 'Loaded {} records.'.format(len(data))
    results.append(metrics)

    filename = os.path.join(results_dir, '{}.json'.format(name))
    with open(filename, 'w') as fp:
      fp.write(json.dumps(data))

  # Save metric data to csv file.
  filename = os.path.join(results_dir, 'final.csv')
  with open(filename, 'w') as fp:
    measures = ['precision@1', 'precision@3', 'precision@5', 'precision@10', 'diff_ratio']
    fp.write('%s\n' % (','.join([''] + measures)))
    for name, metric in zip(names, results):
      items = [name]
      for measure in measures:
        items.append('%.5lf' % (metric[measure]))
      fp.write('%s\n' % (','.join(items)))
