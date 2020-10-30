
import os
import re
import sys
import json

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from google.protobuf import text_format

from object_detection.builders import model_builder
from object_detection.protos import model_pb2
from object_detection.utils import label_map_util

from utils import vis

slim = tf.contrib.slim


flags.DEFINE_string('model_proto', '', 
                    'Path to detection proto file.')

flags.DEFINE_string('checkpoint', '', 
                    'Path to detection model file.')

flags.DEFINE_string('label_map', '', 
                    'Path to the object detection label map file.')

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

flags.DEFINE_string('output_vocab', '', 
                    'Path to the output vocab file.')

flags.DEFINE_integer('batch_size', 64, '')

flags.DEFINE_integer('num_classes', 90, 
                     'Number of classe defined in the detector.')

flags.DEFINE_float('min_object_size', 0.1, 
                   'The minimum size of detected objects.')


FLAGS= flags.FLAGS


def _load_model_proto(filename):
  model_proto = model_pb2.DetectionModel()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)
  return model_proto

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
  features, detections, batch = [], [], []
  for index , filename in enumerate(filenames):
    image_data = _read_image(os.path.join(dirname, filename))
    batch.append(image_data)
    if len(batch) == batch_size:
      predictions, results = extract_fn(video_id, np.stack(batch, axis=0))
      features.append(predictions)
      detections.extend(results)
      batch = []
  if len(batch) > 0:
    predictions, results = extract_fn(video_id, np.stack(batch, axis=0))
    features.append(predictions)
    detections.extend(results)

  features = np.concatenate(features, axis=0)
  assert features.shape[0] == len(filenames)
  assert len(detections) == len(filenames)

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, features)

  filename = os.path.join(output_dir, '%s.json' % video_id)
  with open(filename, 'w') as fp:
    fp.write(json.dumps(detections))

  logging.info('Video features of %s are saved to %s.', video_id, filename)


def main(argv):
  logging.set_verbosity(tf.logging.INFO)

  label_map = label_map_util.load_labelmap(FLAGS.label_map)
  id2name = {}
  for item_id, item in enumerate(label_map.item):
    if item.HasField('display_name'):
      id2name[item.id - 1] = item.display_name
    else:
      id2name[item.id - 1] = item.name

  with open(FLAGS.output_vocab, 'w') as fp:
    for cid, cname in id2name.iteritems():
      fp.write('%i\t%s\n' % (cid, cname))
  logging.info('Dump %s words', len(id2name))

  # Build tensorflow graph.
  g = tf.Graph()
  with g.as_default():
    input_node = tf.placeholder(tf.uint8, shape=(None, None, None, 3))

    # Create detection model.
    model_proto = _load_model_proto(FLAGS.model_proto)
    model = model_builder.build(model_proto, is_training=False)
    predictions = model.predict(
        model.preprocess(tf.cast(input_node, tf.float32)))
    detections = model.postprocess(predictions)

    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.checkpoint, tf.global_variables())
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start session to extract video features.
  with tf.Session(graph=g) as sess:
    init_fn(sess)
    warn_names = sess.run(uninitialized_variable_names)
    assert len(warn_names) == 0

    def _extract_feature(video_id, images_data):
      results = sess.run(detections , feed_dict={ input_node: images_data })
      n_frames = len(images_data)

      features = np.zeros((n_frames, FLAGS.num_classes))
      det_results = []

      for i in xrange(n_frames):
        det_result = []
        for j in xrange(results['num_detections'][i]):
          cid = int(results['detection_classes'][i, j])
          cscore = float(results['detection_scores'][i, j])
          features[i, cid] = max(cscore, features[i, cid])
          y1, x1, y2, x2 = [round(float(x), 4) for x in results['detection_boxes'][i, j]]
          if x2 - x1 >= FLAGS.min_object_size and y2 - y1 > FLAGS.min_object_size:
            det_result.append({
                'cid': cid,
                'cname': id2name.get(cid, 'UNK'),
                'score': round(cscore, 4),
                'bounding_box': { 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 }
                })
            #print(id2name.get(cid, 'UNK'))
            #print(cscore)
            #vis.image_draw_bounding_box(images_data[i], [x1, y1, x2, y2])

        det_results.append(det_result)

        #vis.image_save('tmp/%i.jpg' % (i), images_data[i], True)
      return features, det_results

    # Iterate through video ids.
    for video_id in _video_id_iterator(FLAGS.video_id_path):
      _extract_video_feature(video_id, FLAGS.video_dir, 
          FLAGS.output_dir, _extract_feature)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
