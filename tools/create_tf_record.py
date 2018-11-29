
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from nltk.stem.porter import PorterStemmer

from object_detection.utils import dataset_util
from train import eval_utils

slim = tf.contrib.slim


flags.DEFINE_string('anno_vocab', '', 
                    'Path to the annotation vocab file.')

flags.DEFINE_string('clean_sentiment_anno_json', '', 
                    'Path to the annotation json file.')

flags.DEFINE_string('raw_sentiment_anno_json', '', 
                    'Path to the annotation json file.')

flags.DEFINE_string('clean_topic_anno_json', '', 
                    'Path to the annotation json file.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('feature_dir', '', 
                    'Path to the directory storing video features.')

flags.DEFINE_string('common_object_feature_dir', '', 
                    'Path to the directory storing detection results.')

flags.DEFINE_string('emotic_feature_dir', '', 
                    'Path to the directory storing emotic detection results.')

flags.DEFINE_string('climax_feature_dir', '', 
                    'Path to the directory storing climax annotations.')

flags.DEFINE_string('climax_prediction_dir', '', 
                    'Path to the directory storing climax predictions.')

flags.DEFINE_string('place_feature_dir', '', 
                    'Path to the directory storing place365 results.')

flags.DEFINE_string('affectnet_feature_dir', '', 
                    'Path to the directory storing affectnet results.')

flags.DEFINE_string('shot_boundary_feature_dir', '', 
                    'Path to the directory storing shot boundary features.')

flags.DEFINE_string('optical_flow_feature_dir', '', 
                    'Path to the directory storing optical flow features.')

flags.DEFINE_string('audio_feature_dir', '', 
                    'Path to the directory storing audio features.')

flags.DEFINE_string('output_path', '', 
                    'Path to output file.')

flags.DEFINE_integer('number_of_emotic_classes', 31, 
                    'The number of EMOTIC classes.')

flags.DEFINE_integer('agreement', 1, '')

slim = tf.contrib.slim
FLAGS = flags.FLAGS


def _video_id_iterator(filename):
  with open(filename, 'r') as fp:
    return [line.strip('\n') for line in fp.readlines()]


def load_clean_annot(filename):
  """Loads the clean sentiments annotation.

  Args:
    filename: path to the clean annotation json file.

  Returns:
    data: a dict mapping from video_id to sentiment id [0, num_classes).
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  for k in data.keys():
    data[k] = data[k] - 1
  return data


def _create_mapping():
  stemmer = PorterStemmer()
  mapping = {}
  with open(FLAGS.anno_vocab, 'r') as fp:
    for line in fp.readlines():
      words = re.findall('\w+', line.lower())
      assert words[-1] in eval_utils.sentiments

      for word in words:
        if not word.isdigit() and word != 'abbreviation':
          mapping[stemmer.stem(word)] = words[-1]
  return mapping


def load_sentiment_raw_annot(filename):
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
    sentiments = {}
    num_annotators = 0
    for i in xrange(len(data[k])):
      annots = set(eval_utils.revise_sentiment(data[k][i], mapping))
      for word in annots:
        sentiments[word] = sentiments.get(word, 0) + 1
      if len(annots): 
        num_annotators += 1
    #data[k] = [eval_utils.sentiments.index(x) for x in sentiments if sentiments[x] >= 1]
    #data[k] = [(eval_utils.sentiments.index(x), 1.0 * sentiments[x] / num_annotators) for x in sentiments]

    data[k] = [(eval_utils.sentiments.index(x), 1.0 * sentiments[x] / num_annotators) for x in sentiments if sentiments[x] >= FLAGS.agreement]
  return data

n_climax = 0

def _convert_to_example(video_id, topic_clean_annot, 
    sentiment_clean_annot, sentiment_raw_annot):
  """Convert video data into tf Example.

  Args:
    video_id: the youtube video id.
    topic_clean_annot: an integer indicate the majority vote result.
    sentiment_clean_annot: an integer indicate the majority vote result.
    sentiment_raw_annot: a list of sentiment ids.
  """
  global n_climax

  # Load feature npz file.
  logging.info('Processing on %s', video_id)
  sentiments = [0] * len(eval_utils.sentiments)
  for sent_id, sent_score in sentiment_raw_annot:
    sentiments[sent_id] = sent_score
  sentiments[sentiment_clean_annot] = 1.0

  # The resnet_v2_512 feature.
  filename = os.path.join(FLAGS.feature_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    features = np.load(fp)
  n_frames = len(features)

  # The coco object detection results.
  filename = os.path.join(FLAGS.common_object_feature_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    common_object_features = np.load(fp)
  assert common_object_features.shape[0] == n_frames

  # The place365 classification results.
  filename = os.path.join(FLAGS.place_feature_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    place_features = np.load(fp)
  assert place_features.shape[0] == n_frames

  # The affectet classification results.
  filename = os.path.join(FLAGS.affectnet_feature_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    affectnet_features = np.load(fp)
  assert affectnet_features.shape[0] == n_frames

  # The emotic emotion detection results.
  filename = os.path.join(FLAGS.emotic_feature_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename), filename
  with open(filename, 'rb') as fp:
    emotic_features_list = np.load(fp)

    if len(emotic_features_list.shape) > 1:
      emotic_features_list = [[] for _ in xrange(len(emotic_features_list))]

  emotic_features = np.zeros((len(emotic_features_list), FLAGS.number_of_emotic_classes), np.float32)

  emotic_padding = np.zeros((FLAGS.number_of_emotic_classes), np.float32)
  for frame_id, frame_feature in enumerate(emotic_features_list):
    emotic_features[frame_id] = np.stack(
        [emotic_padding] + emotic_features_list[frame_id]).max(0)
  assert emotic_features.shape[0] == n_frames
    
  # The shot boundary detection results.
  filename = os.path.join(FLAGS.shot_boundary_feature_dir, '%s.npy' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    shot_boundary_features = np.load(fp)
  if shot_boundary_features.shape[0] == n_frames - 1:
    # For Kyle's feature extraction problem: lengths do not agree.
    shot_boundary_features = np.concatenate(
        [shot_boundary_features, [shot_boundary_features[-1]]], axis=0)
  assert shot_boundary_features.shape[0] == n_frames, 'shot boundary shape={}, n_frames={}'.format(shot_boundary_features.shape, n_frames)

  # The optical flow results.
  filename = os.path.join(FLAGS.optical_flow_feature_dir, '%s.npy' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    optical_flow_features = np.load(fp)
  if video_id == 'bB90Vkyqrts':
    while optical_flow_features.shape[0] < n_frames:
      # For Kyle's feature extraction problem: lengths do not agree.
      optical_flow_features = np.concatenate(
          [optical_flow_features, [optical_flow_features[-1]]], axis=0)
  assert optical_flow_features.shape[0] == n_frames, 'vid={}, optical flow shape={}, n_frames={}'.format(video_id, optical_flow_features.shape, n_frames)

  # The audio results.
  filename = os.path.join(FLAGS.audio_feature_dir, '%s.npy' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    audio_features = np.load(fp)

  if audio_features.shape[0] < n_frames:
    # For Kyle's feature extraction problem: lengths do not agree.
    audio_features = np.concatenate(
        [[audio_features[0]], audio_features], axis=0)
  if audio_features.shape[0] < n_frames:
    # For Kyle's feature extraction problem: lengths do not agree.
    audio_features = np.concatenate(
        [audio_features, [audio_features[-1]]], axis=0)
  audio_features = 1.0 * (audio_features - audio_features.min()) / (1e-30 + audio_features.max() - audio_features.min())
  assert audio_features.shape[0] == n_frames, 'audio shape={}, n_frames={}'.format(audio_features.shape, n_frames)

  # The climax prediction results.
  filename = os.path.join(FLAGS.climax_prediction_dir, '%s.npz' % (video_id))
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fp:
    climax_predictions = np.expand_dims(np.load(fp), 1)
  assert climax_predictions.shape[0] == n_frames

  # The climax results.
  filename = os.path.join(FLAGS.climax_feature_dir, '%s.npz' % (video_id))
  if os.path.isfile(filename):
    with open(filename, 'rb') as fp:
      climax_features = np.load(fp)
      n_climax += 1
  else:
    climax_features=  np.zeros((n_frames,))
  assert climax_features.shape[0] == n_frames

  example = tf.train.Example(features=tf.train.Features(feature={
        'video/source_id': dataset_util.bytes_feature(video_id),
        'video/n_frames': dataset_util.int64_feature(n_frames),
        'video/features': dataset_util.float_list_feature(features.flatten().tolist()),
        'video/climax_features': dataset_util.float_list_feature(climax_features.flatten().tolist()),
        'video/common_object_features': dataset_util.float_list_feature(common_object_features.flatten().tolist()),
        'video/emotic_features': dataset_util.float_list_feature(emotic_features.flatten().tolist()),
        'video/place_features': dataset_util.float_list_feature(place_features.flatten().tolist()),
        'video/affectnet_features': dataset_util.float_list_feature(affectnet_features.flatten().tolist()),
        'video/shot_boundary_features': dataset_util.float_list_feature(shot_boundary_features.flatten().tolist()),
        'video/optical_flow_features': dataset_util.float_list_feature(optical_flow_features.flatten().tolist()),
        'video/audio_features': dataset_util.float_list_feature(optical_flow_features.flatten().tolist()),
        'video/climax_predictions': dataset_util.float_list_feature(climax_predictions.flatten().tolist()),
        'anno/topic': dataset_util.int64_feature(topic_clean_annot),
        'anno/sentiment': dataset_util.int64_feature(sentiment_clean_annot),
        'anno/sentiment_list': dataset_util.float_list_feature(sentiments),
        }))

  return example


def main(_):
  logging.set_verbosity(tf.logging.INFO)

  assert os.path.isfile(FLAGS.anno_vocab)
  assert os.path.isfile(FLAGS.clean_topic_anno_json)
  assert os.path.isfile(FLAGS.clean_sentiment_anno_json)
  assert os.path.isfile(FLAGS.raw_sentiment_anno_json)
  assert os.path.isfile(FLAGS.video_id_path)
  assert os.path.isdir(FLAGS.feature_dir)
  assert os.path.isdir(FLAGS.common_object_feature_dir)
  assert os.path.isdir(FLAGS.emotic_feature_dir)
  assert os.path.isdir(FLAGS.place_feature_dir)

  sentiment_raw_annot = load_sentiment_raw_annot(FLAGS.raw_sentiment_anno_json)
  sentiment_clean_annot = load_clean_annot(FLAGS.clean_sentiment_anno_json)

  topic_clean_annot = load_clean_annot(FLAGS.clean_topic_anno_json)

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  for video_id in _video_id_iterator(FLAGS.video_id_path):
    # Convert to tf example.
    example = _convert_to_example(
        video_id, 
        topic_clean_annot[video_id],
        sentiment_clean_annot[video_id], sentiment_raw_annot[video_id])

    if example is not None:
      writer.write(example.SerializeToString())

  writer.close()
  logging.info('with climax data: %i', n_climax)
  logging.info('Done')

if __name__ == '__main__':
  app.run()
