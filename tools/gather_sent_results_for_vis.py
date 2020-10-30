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
  'sent_bof',
  'sent_bof_feat',
  'sent_bof_feat_v2',
  'sent_bof_semantic',
  'sent_lstm64',
  'sent_lstm64_feat',
  'sent_lstm64_feat_v2',
  'sent_lstm64_semantic',
]

def _create_mapping():
  stemmer = PorterStemmer()
  mapping = {}
  with open(sentiment_anno_vocab, 'r') as fp:
    for line in fp.readlines():
      words = re.findall('\w+', line.lower())
      assert words[-1] in eval_utils.sentiments

      for word in words:
        if not word.isdigit() and word != 'abbreviation':
          mapping[stemmer.stem(word)] = words[-1]
  return mapping

def load_raw_annot(filename):
  """Loads the raw sentiments annotation.

  Args:
    filename: path to the raw annotation json file.

  Returns:
    data: a dict mapping from video_id to sentiment id list.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  mapping = _create_mapping()

  for k in data.keys():
    for i in xrange(len(data[k])):
      data[k][i] = eval_utils.revise_sentiment(data[k][i], mapping)
  return data

def evaluate(data, filename, raw_annot, clean_annot, name):
  results =  {}
  results.update(eval_utils.bmvc_evalute_cvpr2016_accuracy(data, clean_annot))
  results.update(eval_utils.bmvc_evalute_vqa_at_k(data, raw_annot, k=1))
  results.update(eval_utils.bmvc_evalute_vqa_at_k(data, raw_annot, k=2))
  results.update(eval_utils.bmvc_evalute_vqa_at_k(data, raw_annot, k=3))
  return results

def analyze(raw_annot, clean_annot):
  count = {}
  for vid, annotators in raw_annot.iteritems():
    for annots in annotators:
      for annot in set(annots):
        count[annot] = count.get(annot, 0) + 1
  print('=' * 128)
  for word, freq in sorted(count.iteritems(), lambda x, y: -cmp(x[1], y[1])):
    print('%s: %i' % (word, freq))
  print('=' * 128)
  hot = set()
  for word, freq in sorted(count.iteritems(), lambda x, y: -cmp(x[1], y[1]))[:10]:
    hot.add(word)

  bingo = 0
  for vid, annot in clean_annot.iteritems():
    if annot == 'amused':
      bingo += 1
  print('Major guess: %.5lf' % (1.0  * bingo / len(clean_annot)))

  export = {}
  for vid, annotators in raw_annot.iteritems():
    vote = {}
    for annots in annotators:
      for annot in set(annots):
        vote[annot] = vote.get(annot, 0) + 1
    for word, freq in sorted(vote.iteritems(), lambda x, y: -cmp(x[1], y[1])):
      if word not in hot and freq >= 3:
        export[vid] = 1
  with open('debug.json', 'w') as fp:
    fp.write(json.dumps(export))

def save_for_vis(filename, data):
  results = {}
  for vid, item in data.iteritems():
    results[vid] = []

    scores = np.array(item['scores'])
    for index in scores.argsort()[::-1][:5]:
      results[vid].append({
          'probability': round(scores[index], 3),
          'sentiment': eval_utils.sentiments[index],
          })
  with open(filename, 'w') as fp:
    fp.write(json.dumps(results))

if __name__ == '__main__':
  raw_annot = load_raw_annot(sentiment_raw_annot_path)
  clean_annot = eval_utils.load_clean_annot(sentiment_clean_annot_path)

  names = [
    'sent_lstm64',
    'sent_lstm64_feat_v2',
  ]
  results = {}

  # Evaluate human annotation.
  for name in names:
    # Aggregate results.
    data = {}
    for split in [0, 2, 4, 6, 8]:
      filename = os.path.join(results_dir, '{}.json.{}'.format(name, split))
      assert os.path.isfile(filename), filename
  
      with open(filename, 'r') as fp:
        data.update(json.loads(fp.read()))
    results[name] = data

  # Look for examples.
  for vid in results['sent_lstm64']:
    vote = {}
    labels = np.zeros((len(eval_utils.sentiments)))
    labels = set()
    for annot_id, annot in enumerate(raw_annot[vid]):
      for ann in set(annot):
        vote[ann] = vote.get(ann, 0) + 1
    for word, count in vote.iteritems():
      if count >= 2:
        labels.add(word)

    pred1 = eval_utils.sentiments[np.array(results['sent_lstm64'][vid]['scores']).argmax()]
    pred2 = eval_utils.sentiments[np.array(results['sent_lstm64_feat_v2'][vid]['scores']).argmax()]

    if pred2 in labels and pred1 not in labels:
      if len(set(labels) & set(['amused', 'cheerful', 'active'])) == 0:
        print >> sys.stderr, '%s\t%s\t%s\t%s' % (vid, labels, pred1, pred2)

  print >> sys.stderr, 'Done'

  exit(0)

  # Save metric data to csv file.
  filename = os.path.join(results_dir, 'final.csv')
  with open(filename, 'w') as fp:
    measures = [
      'Mingda clean annot', '', '',
      '@1 agreement', '', '',
      '@2 agreement', '', '',
      '@3 agreement', '', '',
    ]
    fp.write('%s\n' % (','.join([''] + measures)))
    measures = [
        'map', 'accuracy_at_1', 'accuracy_at_3',
        'vqa@1_map', 'vqa@1_accuracy@1', 'vqa@1_accuracy@3',
        'vqa@2_map', 'vqa@2_accuracy@1', 'vqa@2_accuracy@3',
        'vqa@3_map', 'vqa@3_accuracy@1', 'vqa@3_accuracy@3',
        ]
    fp.write('%s\n' % (','.join([''] + measures)))

    #measures = [
    #  'Mingda clean annot', '', '',
    #  '@1 agreement', '', '', '', '',
    #  '@2 agreement', '', '', '', '',
    #  '@3 agreement', '', '', '', '',
    #]
    #fp.write('%s\n' % (','.join([''] + measures)))
    #measures = [
    #    'map', 'accuracy_at_1', 'accuracy_at_3',
    #    'vqa@1_map', 'vqa@1_accuracy@1', 'vqa@1_accuracy@3', 'vqa@1_recall@1', 'vqa@1_recall@3',
    #    'vqa@2_map', 'vqa@2_accuracy@1', 'vqa@2_accuracy@3', 'vqa@2_recall@1', 'vqa@2_recall@3',
    #    'vqa@3_map', 'vqa@3_accuracy@1', 'vqa@3_accuracy@3', 'vqa@3_recall@1', 'vqa@3_recall@3',
    #    ]
    #fp.write('%s\n' % (','.join([''] + measures)))

    for name, metric in zip(names, results):
      items = [name]
      for measure in measures:
        items.append('%.5lf' % (metric[measure]))
      fp.write('%s\n' % (','.join(items)))


  filename = os.path.join(results_dir, 'details.csv')
  with open(filename, 'w') as fp:
    fp.write('%s\n' % (','.join([''] + eval_utils.eval_sent_strings)))
    for name, metric in zip(names, results):
      items = [name]
      for sent_id, sent in enumerate(eval_utils.eval_sent_strings):
        items.append('%.5lf' % (metric['vqa@2_map_details'][sent_id]))
      fp.write('%s\n' % (','.join(items)))

  for name, metric in zip(names, results):
    filename = os.path.join(results_dir, '{}.thresholds.json'.format(name))
    with open(filename, 'w') as fp:
      fp.write(json.dumps(metric['threshold@2']))
