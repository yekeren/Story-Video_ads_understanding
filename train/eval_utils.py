
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import numpy as np
from tensorflow import logging
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

sentiments = [
  "active", "afraid", "alarmed", "alert", "amazed",
  "amused", "angry", "calm", "cheerful", "confident",
  "conscious", "creative", "disturbed", "eager", "educated",
  "emotional", "empathetic", "fashionable", "feminine", "grateful",
  "inspired", "jealous", "loving", "manly", "persuaded",
  "pessimistic", "proud", "sad", "thrifty", "youthful" ]


_stemmer = PorterStemmer()

def _default_mapping_fn():
  mapping = {}
  for sent in sentiments:
    mapping[_stemmer.stem(sent)] = sent
  return mapping

_default_mapping = _default_mapping_fn()


def revise_sentiment(annot, mapping=None):
  """Revises sentiment annotations.
  """
  if mapping is None:
    mapping = _default_mapping
    logging.warn('Use default mapping.')

  words = [_stemmer.stem(x.strip(',.')) for x in annot.lower().split()]
  words = [mapping[x] for x in words if x in mapping]
  return words


def load_raw_annot(filename, mapping=None):
  """Loads the clean sentiments annotation.

  Args:
    filename: path to the clean annotation json file.

  Returns:
    data: a dict mapping from video_id to sentiment.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  for k in data.keys():
    for i in xrange(len(data[k])):
      data[k][i] = revise_sentiment(data[k][i], mapping)
  return data


def load_clean_annot(filename):
  """Loads the clean sentiments annotation.

  Args:
    filename: path to the clean annotation json file.

  Returns:
    data: a dict mapping from video_id to sentiment.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  for k in data.keys():
    data[k] = sentiments[data[k] - 1]

  return data


def evaluate_cvpr2016_accuracy(clean_annot, results, key='sentiment'):
  """Evaluates the accuracy mentioned in cvpr2016 ads paper.

  Args:
    clean_annot: a dict mapping from video_id to sentiment.
    results: a dict mapping from video id to top-k predictions.

  Returns:
    accuracy: a float number
  """
  total_samples = 0
  correct_samples = 0
  for vid, predictions in results.iteritems():
    assert vid in clean_annot

    total_samples += 1
    if predictions[0][key] == clean_annot[vid]:
      correct_samples += 1

  accuracy = 1.0 * correct_samples / total_samples
  logging.info('The cvpr2016 accuracy: %.3lf', accuracy)
  return accuracy


def evaluate_vqa_accuracy(raw_annot, results, key='sentiment'):
  """Evaluates the accuracy mentioned in cvpr2016 ads paper.

  Args:
    raw_annot: a dict mapping from video_id to sentiment.
    results: a dict mapping from video id to top-k predictions.

  Returns:
    accuracy: a float number
  """
  total_samples = 0
  correct_samples = 0
  for vid, predictions in results.iteritems():
    assert vid in raw_annot

    total_samples += 1

    bingo = 0
    for annot in raw_annot[vid]:
      if predictions[0][key] in annot:
        bingo += 1
    if bingo >= 2:
      correct_samples += 1

  accuracy = 1.0 * correct_samples / total_samples
  logging.info('The VQA accuracy: %.3lf', accuracy)
  return accuracy


def evaluate_precision_and_recall(raw_annot, results, 
    names=sentiments, key='sentiment'):
  """Evaluates the precision and recall.

  Args:
    raw_annot: a dict mapping from video_id to sentiment.
    results: a dict mapping from video id to top-k predictions.

  Returns:
    precision: a float number.
    recall: a float number.
    per_class_precision: a [num_classes] np array.
    per_class_recall: a [num_classes] np array.
  """
  tp_fp = np.zeros((len(results), len(names)), np.bool)
  tp_fn = np.zeros((len(results), len(names)), np.bool)

  names_r = dict(((name, i) for i, name in enumerate(names)))

  for i, (vid, predictions) in enumerate(results.iteritems()):
    assert vid in raw_annot

    tp_fp_words = set([x[key] for x in predictions])
    tp_fn_words = set(reduce(lambda x, y: x + y, [x for x in raw_annot[vid]]))

    for w in tp_fp_words:
      tp_fp[i, names_r[w]] = True

    for w in tp_fn_words:
      tp_fn[i, names_r[w]] = True

  # Compute precision and recall.
  tp = np.logical_and(tp_fp, tp_fn).sum(0)
  tp_fp, tp_fn = tp_fp.sum(0), tp_fn.sum(0)

  per_class_precision = tp / (tp_fp + 1e-8)
  per_class_recall = tp / (tp_fn + 1e-8)

  precision = tp.sum() / (tp_fp.sum() + 1e-8)
  recall = tp.sum() / (tp_fn.sum() + 1e-8)

  f1_score_fn = lambda x, y: (2 * x * y) / (x + y + 1e-8)

  logging.info('Precision: %.3lf', precision)
  logging.info('Recall: %.3lf', recall)

  for i, sentiment in enumerate(names):
    logging.info('Name=%30s, precision=%.3lf(%4i/%4i), recall=%.3lf(%4i/%4i), f1=%.3lf',
        sentiment, 
        per_class_precision[i], tp[i], tp_fp[i],
        per_class_recall[i], tp[i], tp_fn[i], 
        f1_score_fn(per_class_precision[i], per_class_recall[i]))

  return precision, recall, per_class_precision, per_class_recall

def evaluate_climax_accuracy(results):
  """Evaluates the accuracy of the climax detection.

  Args:
    results: a dict mapping from video_id to predictions.

  Returns:
    top-1 precision: a float number.
    diff_ratio: a float number.
  """
  bingo_at_1 = 0
  bingo_at_3 = 0
  bingo_at_5 = 0
  bingo_at_10 = 0
  diff_ratio = 0.0
  total = 0

  for i, (vid, predictions) in enumerate(results.iteritems()):
    n_frames = predictions['n_frames']
    labels = np.array(predictions['labels'])
    scores = np.array(predictions['scores'])

    # Find the closest label.
    label = -1
    pred = scores.argmax()

    min_v = 1e10
    for l in np.where(labels > 0)[0]:
      if np.abs(l - pred) < min_v:
        min_v = np.abs(l - pred)
        label = l

    if label == -1:  # Evaluate on only the examples that have annotation.
      continue

    total += 1
    if label == pred:
      bingo_at_1 += 1
    if abs(label - pred) < 3:
      bingo_at_3 += 1
    if abs(label - pred) < 5:
      bingo_at_5 += 1
    if abs(label - pred) < 10:
      bingo_at_10 += 1
    
    diff = np.abs(label - pred)
    diff_ratio += 1.0 * diff / n_frames

  precision_at_1 = 1.0 * bingo_at_1 / total
  precision_at_3 = 1.0 * bingo_at_3 / total
  precision_at_5 = 1.0 * bingo_at_5 / total
  precision_at_10 = 1.0 * bingo_at_10 / total
  diff_ratio = diff_ratio / total

  return precision_at_1, precision_at_3, precision_at_5, precision_at_10, diff_ratio

eval_sent_strings = [
  'amused',
  'alert',
  'active',
  'cheerful',
  'amazed',
  'persuaded',
  'eager',
  'inspired',
  'creative',
  'confident',
  'educated',
  'conscious',
  'alarmed',
  'calm',
  'fashionable',
  'emotional',
  'youthful',
  'empathetic',
  'angry',
  'loving',
  'disturbed',
  'feminine',
  'manly',
  'afraid',
  'sad',
  'proud',
  'thrifty',
  'grateful',
  'pessimistic',
  'jealous',
  ]

eval_sentiments = [sentiments.index(s) for s in eval_sent_strings]

def bmvc_evalute_cvpr2016_accuracy(results, clean_annot):
  """Evaluates the cvpr2016 accuracy.
  """
  total_samples = 0
  bingo = bingo3 = 0
  y_label, y_score = [], []

  for vid in sorted(results.keys()):
    scores = np.array(results[vid]['scores'])
    prediction = sentiments[scores.argmax()]
    predictions = [sentiments[x] for x in scores.argsort()[::-1][:3]]

    total_samples += 1
    if prediction == clean_annot[vid]:
      bingo += 1
    if clean_annot[vid] in predictions:
      bingo3 += 1

    labels = np.zeros((len(sentiments)))
    labels[sentiments.index(clean_annot[vid])] = 1.0

    y_label.append(labels)
    y_score.append(scores)

  y_score = np.stack(y_score, 0)
  y_label = np.stack(y_label, 0)
  predictions = y_score.argsort()[:, -1:]

  y_predi = np.zeros((total_samples, len(sentiments)))
  for row, cols in enumerate(predictions):
    y_predi[row, cols] = 1.0

  # Compute recall@1, map.
  ap = average_precision_score(y_label, y_score, average=None)
  avg_ap = np.mean([x for x in ap[eval_sentiments] if not np.isnan(x)])
  accuracy_at_1 = 1.0 * bingo / total_samples
  accuracy_at_3 = 1.0 * bingo3 / total_samples

  return {
    'map': round(avg_ap, 5),
    'accuracy': round(accuracy_at_1, 5),
    'accuracy_at_1': round(accuracy_at_1, 5),
    'accuracy_at_3': round(accuracy_at_3, 5),
  }

def bmvc_evalute_vqa_at_k(results, raw_annot, k=None):
  """Evaluates the cvpr2016 accuracy.
  """
  assert k is not None

  total_samples = 0
  y_label, y_score = [], []

  for vid in sorted(results.keys()):
    scores = np.array(results[vid]['scores'])
    prediction = sentiments[scores.argmax()]

    total_samples += 1
    vote = {}
    labels = np.zeros((len(sentiments)))

    for annot_id, annot in enumerate(raw_annot[vid]):
      for ann in set(annot):
        vote[ann] = vote.get(ann, 0) + 1
    for word, count in vote.iteritems():
      if count >= k:
        labels[sentiments.index(word)] = 1.0

    y_label.append(labels)
    y_score.append(scores)

  y_score = np.stack(y_score, 0)
  y_label = np.stack(y_label, 0)

  # Compute recall@1, recall@3, map.
  ap = average_precision_score(y_label, y_score, average=None)

  y_predi = np.zeros((total_samples, len(sentiments)))
  predictions = y_score.argsort()[:, -1:]
  for row, cols in enumerate(predictions):
    y_predi[row, cols] = 1.0
  #recall_at_1 = recall_score(y_label, y_predi, average='macro')
  accuracy_at_1 = y_label * y_predi
  accuracy_at_1 = accuracy_at_1.sum() / accuracy_at_1.shape[0]

  predictions = y_score.argsort()[:, -3:]
  for row, cols in enumerate(predictions):
    y_predi[row, cols] = 1.0
  #recall_at_3 = recall_score(y_label, y_predi, average='macro')
  accuracy_at_3 = y_label * y_predi
  accuracy_at_3 = (accuracy_at_3.sum(1)>0).astype(np.float).sum() / accuracy_at_3.shape[0]

  avg_ap = np.array([v for v in ap[eval_sentiments] if not np.isnan(v)]).mean()

  ## Find the best threshold.
  #if k == 2:
  #  threshold_list = []
  #  for i in xrange(len(sentiments)):
  #    y_true = y_label[:, i]
  #    y_pred = y_score[:, i]
  #    precision, recall, threshold = 0, 0, 0.005
  #    for t in np.arange(0.005, 0.8, 0.005):
  #      f_v = f1_score(y_true, y_pred > t)
  #      p_v = precision_score(y_true, y_pred > t)
  #      r_v = recall_score(y_true, y_pred > t)
  #      if p_v > precision and r_v > 0.2:
  #        precision, recall, threshold = p_v, r_v, t
  #    print('sent=%s, max precision=%.3lf, recall=%.3lf, threshold=%.3lf' % (
  #          sentiments[i], precision, recall, threshold))
  #    threshold_list.append(round(threshold, 3))

  results = {
    'vqa@{}_map'.format(k): round(avg_ap, 5),
    'vqa@{}_accuracy@1'.format(k): round(accuracy_at_1, 5),
    'vqa@{}_accuracy@3'.format(k): round(accuracy_at_3, 5),
    #'vqa@{}_recall@1'.format(k): round(recall_at_1, 5),
    #'vqa@{}_recall@3'.format(k): round(recall_at_3, 5),
    'vqa@{}_map_details'.format(k): ap[eval_sentiments],
  }
  #if k == 2:
  #  results['threshold@{}'.format(k)] = threshold_list
  return results
