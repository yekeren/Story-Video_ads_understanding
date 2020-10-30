
import os
import re
import sys

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory

slim = tf.contrib.slim


flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

flags.DEFINE_string('feature_extractor_name', 'resnet_v2_152', '')

flags.DEFINE_string('feature_extractor_checkpoint', 'models/resnet_v2_152/resnet_v2_152.ckpt', '')

flags.DEFINE_integer('batch_size', 64, '')


FLAGS= flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _video_id_iterator(filename):
  with open(filename, 'r') as fp:
    return [line.strip('\n') for line in fp.readlines()]


def _get_screenshots(dirname):
  filenames = filter(
      lambda x: re.match(r'screenshot\d+.jpg', x), 
      os.listdir(dirname))
  filenames = map(
      lambda x: (re.match(r'screenshot(\d+).jpg', x).group(1), x), 
      filenames)

  filenames.sort(lambda x, y: cmp(int(x[0]), int(y[0])))
  filenames = [filename[1] for filename in filenames]
  return filenames

def _read_image(filename):
  """Reads image data from file.

  Args:
    filename: the path to the image file.
  """
  bgr = cv2.imread(filename)
  rgb = bgr[:, :, ::-1]
  return rgb

def _extract_video_feature(video_id, video_dir, output_dir, extract_fn):
  """Extracts features from video.
  """
  dirname = os.path.join(video_dir, '%s' % video_id)

  filenames = _get_screenshots(dirname)
  n_frames = len(filenames)
  batch_size = FLAGS.batch_size

  # Batch process the frames.
  features, batch = [], []
  for index , filename in enumerate(filenames):
    image_data = _read_image(os.path.join(dirname, filename))
    batch.append(image_data)
    if len(batch) == batch_size:
      features.append(extract_fn(video_id, np.stack(batch, axis=0)))
      batch = []
  if len(batch) > 0:
    features.append(extract_fn(video_id, np.stack(batch, axis=0)))

  features = np.concatenate(features, axis=0)
  assert features.shape[0] == len(filenames)

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, features)
  logging.info('Video features of %s are saved to %s.', video_id, filename)


def main(argv):
  logging.set_verbosity(tf.logging.INFO)

  assert FLAGS.feature_extractor_name == 'resnet_v2_152'

  # Get the function to preprocess and build network for the image.
  net_fn = nets_factory.get_network_fn(
      name=FLAGS.feature_extractor_name, num_classes=None)
  default_image_size = getattr(net_fn, 'default_image_size', 224)

  # Build tensorflow graph.
  g = tf.Graph()
  with g.as_default():
    input_node = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
    images = tf.image.convert_image_dtype(input_node, dtype=tf.float32)
    images = tf.image.resize_bilinear(
        images, size=(default_image_size, default_image_size))
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)

    prediction, end_points = net_fn(images)
    prediction = tf.squeeze(prediction, [1, 2])

    # The init_fn function.
    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.feature_extractor_checkpoint,
        slim.get_model_variables(FLAGS.feature_extractor_name))
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start session to extract video features.
  with tf.Session(graph=g) as sess:
    init_fn(sess)
    warn_names = sess.run(uninitialized_variable_names)
    assert len(warn_names) == 0

    def _extract_feature(video_id, images_data):
      values = sess.run(prediction, feed_dict={ input_node: images_data })
      return values

    # Iterate through video ids.
    for video_id in _video_id_iterator(FLAGS.video_id_path):
      _extract_video_feature(video_id, FLAGS.video_dir, 
          FLAGS.output_dir, _extract_feature)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
