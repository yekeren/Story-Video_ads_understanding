
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
from google.protobuf import text_format

import ads_examples
from protos import ads_examples_pb2

slim = tf.contrib.slim


def _load_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a dict mapping from word to id.
  """
  with open(filename, 'r') as fp:
    words = [line.strip('\n') for line in fp.readlines()]

  return words

vocab_topic = _load_vocab('data/topics.txt')
vocab_sentiment = _load_vocab('data/sentiments.txt')

class AdsExamplesTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      input_path: "output/video_ads_agree1.record.0" 
      batch_size: 32
      feature_dims: 2048
      common_object_feature_dims: 90
      place_feature_dims: 365
      emotic_feature_dims: 31
      sentiment_num_classes: 30
    """
    self.default_config = ads_examples_pb2.AdsExamples()
    text_format.Merge(config_str, self.default_config)

  def test_get_examples(self):
    config = self.default_config

    g = tf.Graph()
    with g.as_default():
      example = ads_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['video_id'].shape, (32,))
      self.assertEqual(example['n_frames'].shape, (32,))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['sentiment'].shape, (32,))
      self.assertEqual(example['sentiment_list'].shape, (32, 30))

      n_frames = example['n_frames'].max()
      self.assertEqual(example['frame_features'].shape, (32, n_frames, 2048))
      self.assertEqual(example['climax_features'].shape, (32, n_frames, 1))
      self.assertEqual(example['climax_predictions'].shape, (32, n_frames, 1))
      self.assertEqual(example['common_object_features'].shape, (32, n_frames, 90))
      self.assertEqual(example['place_features'].shape, (32, n_frames, 365))
      self.assertEqual(example['emotic_features'].shape, (32, n_frames, 31))
      self.assertEqual(example['affectnet_features'].shape, (32, n_frames, 10))
      self.assertEqual(example['shot_boundary_features'].shape, (32, n_frames, 5))
      self.assertEqual(example['optical_flow_features'].shape, (32, n_frames, 1))
      self.assertEqual(example['audio_features'].shape, (32, n_frames, 1))

      for i in xrange(32):
        tf.logging.info('Video [%s]: n_frames=%i, topic=\'%s\', sentiment=\'%s\'',
            example['video_id'][i], 
            example['n_frames'][i], 
            vocab_topic[example['topic'][i]],
            vocab_sentiment[example['sentiment'][i]])

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    tf.test.main()
