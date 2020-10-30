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
  'inception_bof',
  'inception_bof_raw',
  'inception_bof_feat',
  'inception_bof_feat_place_emotic',
  'inception_bof_feat_place_emotic_raw',
  'inception_bof_semantic',
  'inception_bof_semantic_place_emotic',
  'inception_bof_semantic_place_emotic_raw',
  'inception_bof_feat_joint',
  'inception_bof_feat_place_emotic_joint',
  'inception_bof_semantic_joint',
  'inception_bof_semantic_place_emotic_joint',
  'inception_lstm64',
  'inception_lstm64_raw',
  'inception_lstm64_feat',
  'inception_lstm64_feat_place_emotic',
  'inception_lstm64_feat_place_emotic_raw',
  'inception_lstm64_semantic',
  'inception_lstm64_semantic_place_emotic',
  'inception_lstm64_semantic_place_emotic_raw',
  'inception_lstm64_feat_joint',
  'inception_lstm64_feat_place_emotic_joint',
  'inception_lstm64_semantic_joint',
  'inception_lstm64_semantic_place_emotic_joint',
  'inception_bof_raw_joint',
  'inception_bof_feat_place_emotic_raw_joint',
  'inception_bof_semantic_place_emotic_raw_joint',
  'inception_lstm64_raw_joint',
  'inception_lstm64_feat_place_emotic_raw_joint',
  'inception_lstm64_semantic_place_emotic_raw_joint',
  'inception_lstm32',
  'inception_lstm32_raw',
  'inception_lstm32_feat_place_emotic',
  'inception_lstm32_feat_place_emotic_raw',
]
names = [
  'sent_bof',
  'sent_bof_feat',
  'sent_lstm64',
  'sent_lstm64_feat',
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
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []
  y_label_at_1 = []
  y_label_at_2 = []

  with open(filename, 'w') as fp:
    for vid in sorted(data.keys()):
      labels = data[vid]['labels']
      scores = np.array(data[vid]['scores'])
      prediction = eval_utils.sentiments[scores.argmax()]

      total_samples += 1

      # Vote.
      vote = {}
      label1 = np.zeros((len(eval_utils.sentiments)))
      label2 = np.zeros((len(eval_utils.sentiments)))
      for annot_id, annot in enumerate(raw_annot[vid]):
        for a in set(annot):
          vote[a] = vote.get(a, 0) + 1
      for word, count in vote.iteritems():
        if count >= 1:
          label1[eval_utils.sentiments.index(word)] = 1.0
        if count >= 2:
          label2[eval_utils.sentiments.index(word)] = 1.0
      y_label_at_1.append(label1)
      y_label_at_2.append(label2)


      # CVPR2016 accuracy.
      if prediction == clean_annot[vid]:
        correct_samples_cvpr2016 += 1

      # VQA accuracy.
      bingo = 0
      for annot in raw_annot[vid]:
        if prediction in annot: bingo += 1
      if bingo >= 1: correct_samples_vqa1 += 1
      if bingo >= 2: correct_samples_vqa2 += 1

      # mAP measurement.
      y_label.append(labels)
      y_score.append(scores)

  y_score = np.stack(y_score, 0)
  y_true = np.zeros_like(y_score, np.float32)
  for i, ind in enumerate(y_label):
    y_true[i, ind] = 1.0

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  average_precision = average_precision_score(y_true, y_score, average=None)
  map_macro = np.mean([x for x in average_precision if not np.isnan(x)])
  map_micro = average_precision_score(y_true, y_score, average='micro')

  # print >> sys.stderr, '=' * 128
  # print >> sys.stderr, name
  # print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  # print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  # print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  # print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  # print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  y_label = np.stack(y_label_at_1, 0)
  _average_precision = average_precision_score(y_label, y_score, average=None)
  map_macro_at_1 = np.mean([x for x in _average_precision if not np.isnan(x)])
  map_micro_at_1 = average_precision_score(y_label, y_score, average='micro')

  y_label = np.stack(y_label_at_2, 0)
  _average_precision = average_precision_score(y_label, y_score, average=None)
  map_macro_at_2 = np.mean([x for x in _average_precision if not np.isnan(x)])
  map_micro_at_2 = average_precision_score(y_label, y_score, average='micro')

  print >> sys.stderr, 'mAP@1 (micro): %.5lf' % (map_micro_at_1)
  print >> sys.stderr, 'mAP@1 (macro): %.5lf' % (map_macro_at_1)
  print >> sys.stderr, 'mAP@2 (micro): %.5lf' % (map_micro_at_2)
  print >> sys.stderr, 'mAP@2 (macro): %.5lf' % (map_macro_at_2)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
    'map_macro_at_1': map_macro_at_1,
    'map_micro_at_1': map_micro_at_1,
    'map_macro_at_2': map_macro_at_2,
    'map_micro_at_2': map_micro_at_2,
  }

def evaluate_annot(raw_annot, clean_annot, name):
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []

  for vid in sorted(clean_annot.keys()):
    label = eval_utils.sentiments.index(clean_annot[vid])
    scores = np.zeros([30], np.float32)
    scores[label] = 1
    
    prediction = eval_utils.sentiments[scores.argmax()]

    total_samples += 1

    # CVPR2016 accuracy.
    if prediction == clean_annot[vid]:
      correct_samples_cvpr2016 += 1

    # VQA accuracy.
    bingo = 0
    for annot in raw_annot[vid]:
      if prediction in annot: bingo += 1
    if bingo >= 1: correct_samples_vqa1 += 1
    if bingo >= 2: correct_samples_vqa2 += 1

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  map_macro = np.nan
  map_micro = np.nan

  print >> sys.stderr, '=' * 128
  print >> sys.stderr, name
  print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
  }

def evaluate_annot_cross(raw_annot, clean_annot, name, annotator):
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []

  for vid in sorted(raw_annot.keys()):
    if len(raw_annot[vid]) <= annotator:
      continue
    annot = raw_annot[vid][annotator]

    if len(annot) == 0:
      continue
    prediction = annot[0]

    total_samples += 1

    # CVPR2016 accuracy.
    if prediction == clean_annot[vid]:
      correct_samples_cvpr2016 += 1

    # VQA accuracy.
    bingo = 0
    for annot_id, annot in enumerate(raw_annot[vid]):
      if annot_id != annotator and prediction in annot: 
        bingo += 1

    if bingo >= 1: correct_samples_vqa1 += 1
    if bingo >= 2: correct_samples_vqa2 += 1

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  map_macro = np.nan
  map_micro = np.nan

  print >> sys.stderr, '=' * 128
  print >> sys.stderr, name
  print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
  }

def evaluate_annot_cross_general(raw_annot, clean_annot, name):
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []

  for annotator in xrange(5):
    for vid in sorted(raw_annot.keys()):
      if len(raw_annot[vid]) <= annotator:
        continue
      annot = raw_annot[vid][annotator]
      if not annot: continue

      bak_annot = annot[:]
      total_samples += 1

      tmp_correct_samples_cvpr2016 = 0
      tmp_correct_samples_vqa1 = 0
      tmp_correct_samples_vqa2 = 0

      for prediction in bak_annot:
        # CVPR2016 accuracy.
        if prediction == clean_annot[vid]:
          tmp_correct_samples_cvpr2016 = 1

        # VQA accuracy.
        bingo = 0
        for annot_id, annot in enumerate(raw_annot[vid]):
          if annot_id != annotator and prediction in annot: 
            bingo += 1

        if bingo >= 1: tmp_correct_samples_vqa1 = 1
        if bingo >= 2: tmp_correct_samples_vqa2 = 1

      correct_samples_cvpr2016 += tmp_correct_samples_cvpr2016
      correct_samples_vqa1 += tmp_correct_samples_vqa1
      correct_samples_vqa2 += tmp_correct_samples_vqa2

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  map_macro = np.nan
  map_micro = np.nan

  print >> sys.stderr, '=' * 128
  print >> sys.stderr, name
  print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
  }

def evaluate_annot_cross_general2(raw_annot, clean_annot, name):
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []

  for annotator in xrange(5):
    for vid in sorted(raw_annot.keys()):
      if len(raw_annot[vid]) <= annotator:
        continue
      annot = raw_annot[vid][annotator]
      if not annot: continue

      bak_annot = annot[:]
      total_samples += 1

      tmp_correct_samples_cvpr2016 = 0
      tmp_correct_samples_vqa1 = 0
      tmp_correct_samples_vqa2 = 0

      for prediction in bak_annot:

        # CVPR2016 accuracy.
        vote = {}
        for annot_id, annot in enumerate(raw_annot[vid]):
          if annot_id != annotator:
            for a in set(annot):
              vote[a] = vote.get(a, 0) + 1
        vote = sorted(vote.iteritems(), lambda x, y: -cmp(x[1], y[1]))
        if not len(vote):
          continue
        majority_vote = vote[0][0]

        if prediction == majority_vote: #clean_annot[vid]:
          tmp_correct_samples_cvpr2016 = 1

        # VQA accuracy.
        bingo = 0
        for annot_id, annot in enumerate(raw_annot[vid]):
          if annot_id != annotator and prediction in annot: 
            bingo += 1

        if bingo >= 1: tmp_correct_samples_vqa1 = 1
        if bingo >= 2: tmp_correct_samples_vqa2 = 1

      correct_samples_cvpr2016 += tmp_correct_samples_cvpr2016
      correct_samples_vqa1 += tmp_correct_samples_vqa1
      correct_samples_vqa2 += tmp_correct_samples_vqa2

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  map_macro = np.nan
  map_micro = np.nan

  print >> sys.stderr, '=' * 128
  print >> sys.stderr, name
  print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
  }

def evaluate_annot_cross_general3(raw_annot, clean_annot, name):
  total_samples = 0
  correct_samples_cvpr2016 = 0
  correct_samples_vqa1 = 0
  correct_samples_vqa2 = 0
  y_label, y_score = [], []

  for annotator in xrange(5):
    for vid in sorted(raw_annot.keys()):
      if len(raw_annot[vid]) <= annotator:
        continue
      annot = raw_annot[vid][annotator]
      if not annot: continue

      bak_annot = annot[:]
      total_samples += 1

      tmp_correct_samples_cvpr2016 = 0
      tmp_correct_samples_vqa1 = 0
      tmp_correct_samples_vqa2 = 0

      for prediction in bak_annot:
        # CVPR2016 accuracy.
        vote = {}
        for annot_id, annot in enumerate(raw_annot[vid]):
          for a in set(annot):
            vote[a] = vote.get(a, 0) + 1
        vote = sorted(vote.iteritems(), lambda x, y: -cmp(x[1], y[1]))
        if not len(vote):
          continue
        majority_vote = vote[0][0]

        if prediction == majority_vote: #clean_annot[vid]:
          tmp_correct_samples_cvpr2016 = 1

        # VQA accuracy.
        bingo = 0
        for annot_id, annot in enumerate(raw_annot[vid]):
          if prediction in annot: 
            bingo += 1

        if bingo >= 1: tmp_correct_samples_vqa1 = 1
        if bingo >= 2: tmp_correct_samples_vqa2 = 1

      correct_samples_cvpr2016 += tmp_correct_samples_cvpr2016
      correct_samples_vqa1 += tmp_correct_samples_vqa1
      correct_samples_vqa2 += tmp_correct_samples_vqa2

  # Accuracy.
  accuracy = 1.0 * correct_samples_cvpr2016 / total_samples
  accuracy_vqa1 = 1.0 * correct_samples_vqa1 / total_samples
  accuracy_vqa2 = 1.0 * correct_samples_vqa2 / total_samples

  # mAP macro.
  map_macro = np.nan
  map_micro = np.nan

  print >> sys.stderr, '=' * 128
  print >> sys.stderr, name
  print >> sys.stderr, 'Accuracy: %.5lf' % (accuracy)
  print >> sys.stderr, 'Accuracy (vqa1): %.5lf' % (accuracy_vqa1)
  print >> sys.stderr, 'Accuracy (vqa2): %.5lf' % (accuracy_vqa2)
  print >> sys.stderr, 'mAP (micro): %.5lf' % (map_micro)
  print >> sys.stderr, 'mAP (macro): %.5lf' % (map_macro)

  return {
    'accuracy': accuracy,
    'accuracy_vqa1': accuracy_vqa1,
    'accuracy_vqa2': accuracy_vqa2,
    'map_macro': map_macro,
    'map_micro': map_micro,
  }

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
  print('Major guess: %.3lf' % (1.0  * bingo / len(clean_annot)))

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

  analyze(raw_annot, clean_annot)

  results = []

  # Evaluate human annotation.
  for name in names:
    # Aggregate results.
    data = {}
    for split in [0, 2, 4]:
      filename = os.path.join(results_dir, '{}.json.{}'.format(name, split))
      assert os.path.isfile(filename), filename
  
      with open(filename, 'r') as fp:
        data.update(json.loads(fp.read()))

    output_filename = os.path.join(results_dir, '{}.json'.format(name))
    save_for_vis(output_filename, data)
  
    output_filename = os.path.join(results_dir, '{}.csv'.format(name))
    metrics = evaluate(
        data, output_filename, raw_annot, clean_annot, name)
  
    print >> sys.stderr, 'Loaded {} records.'.format(len(data))
    results.append(metrics)

#  names.append('human-mingda')
#  metrics = evaluate_annot(raw_annot, clean_annot, 'human')
#  results.append(metrics)
#
#  names.append('human-cross-mingda')
#  metrics = evaluate_annot_cross_general(raw_annot, clean_annot, names[-1])
#  results.append(metrics)
#
#  names.append('human-cross-annotator')
#  metrics = evaluate_annot_cross_general2(raw_annot, clean_annot, names[-1])
#  results.append(metrics)
#
#  names.append('human-cross-annotator-inclusive')
#  metrics = evaluate_annot_cross_general3(raw_annot, clean_annot, names[-1])
#  results.append(metrics)
  
  # Save metric data to csv file.
  filename = os.path.join(results_dir, 'final.csv')
  with open(filename, 'w') as fp:
    measures = [
    'accuracy', 'accuracy_vqa1', 'accuracy_vqa2', 'map_macro', 'map_micro',
    'map_macro_at_1', 'map_micro_at_1', 'map_macro_at_2', 'map_micro_at_2']
    fp.write('%s\n' % (','.join([''] + measures)))
    for name, metric in zip(names, results):
      items = [name]
      for measure in measures:
        items.append('%.5lf' % (metric[measure]))
      fp.write('%s\n' % (','.join(items)))
