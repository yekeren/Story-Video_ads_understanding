
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

flags.DEFINE_integer('max_videos', -1, 
                     'Maximum number of videos to run.')

flags.DEFINE_string('model_proto', '', 
                    'Path to detection proto file.')

flags.DEFINE_string('checkpoint_dir', '', 
                    'Path to directory storing checkpoint files.')

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

flags.DEFINE_float('min_object_size', 0.1, 
                   'The minimum size of detected objects.')


FLAGS= flags.FLAGS

age_vocab = [ "Adult", "Teenager", "Kid" ]

gender_vocab = [ "Male", "Female" ]

category_vocab = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
  "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy",
  "Doubt/Confusion", "Disconnection", "Fatigue", "Embarrassment", "Yearning",
  "Disapproval", "Aversion", "Annoyance", "Anger", "Sensitivity", "Sadness",
  "Disquietment", "Fear", "Pain", "Suffering"]


def _get_text_label_height():
  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  thickness = 1

  (sx, sy), baseline = cv2.getTextSize(
      'text', font_face, font_scale, thickness)
  return sy + baseline


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
      features.extend(predictions)
      detections.extend(results)
      batch = []
  if len(batch) > 0:
    predictions, results = extract_fn(video_id, np.stack(batch, axis=0))
    features.extend(predictions)
    detections.extend(results)

  assert len(features) == len(filenames)
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

    # Load the latest model checkpoint.
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    init_fn = slim.assign_from_checkpoint_fn(
        model_path, tf.global_variables())
    logging.info('Load variables from %s.', model_path)
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start session to extract video features.
  with tf.Session(graph=g) as sess:
    init_fn(sess)
    warn_names = sess.run(uninitialized_variable_names)
    assert len(warn_names) == 0

    def _extract_feature(video_id, images_data):
      results = sess.run(detections , feed_dict={ input_node: images_data })
      n_frames = len(images_data)

      det_results = []
      num_classes = len(age_vocab) + len(gender_vocab) + len(category_vocab)
      features = []

      vis_dir = os.path.join('tmp', video_id)
      if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)

      for i in xrange(n_frames):
        det_result = []
        person_feature_list = []

        image_disp = cv2.resize(images_data[i], (512, 512))

        for j in xrange(results['num_detections'][i]):
          detection_score = results['detection_scores'][i, j]
          emotic_ages = results['detection_emotic_ages'][i, j]
          emotic_genders = results['detection_emotic_genders'][i, j]
          emotic_categories = results['detection_emotic_categories'][i, j]

          person_feature = np.concatenate(
              [emotic_ages, emotic_genders, emotic_categories], 0)
          person_feature_list.append(person_feature)

          age_ind = emotic_ages.argmax()
          age_pro = int(100 * emotic_ages[age_ind])
          age_str = '{}: {}%'.format(age_vocab[age_ind], age_pro)
          
          gender_ind = emotic_genders.argmax()
          gender_pro = int(100 * emotic_genders[gender_ind])
          gender_str = '{}: {}%'.format(gender_vocab[gender_ind], gender_pro)

          cat_str_list = []
          for cat_ind in emotic_categories.argsort()[::-1]:
            cat_pro = int(100 * emotic_categories[cat_ind])
            if cat_pro < 20: break
            cat_str_list.append('{}: {}%'.format(category_vocab[cat_ind], cat_pro))

          display_str_list = [age_str] + [gender_str] + cat_str_list

          # Process max pooling.
          cat_list = []
          for cat_id, cat_score in enumerate(emotic_categories):
            if cat_score > 0.2:
              cat_list.append({
                  'category': category_vocab[cat_id], 
                  'score': round(float(emotic_categories[cat_id]), 3)
                  })
          emotic = {
            'age': {
              'age': age_vocab[emotic_ages.argmax()],
              'score': round(float(emotic_ages.max()), 3),
            },
            'gender': {
              'gender': gender_vocab[emotic_genders.argmax()],
              'score': round(float(emotic_genders.max()), 3),
            },
            'categories': cat_list
          }

          y1, x1, y2, x2 = [round(float(x), 4) for x in results['detection_boxes'][i, j]]
          if x2 - x1 >= FLAGS.min_object_size and y2 - y1 > FLAGS.min_object_size:
            det_result.append({
                'score': round(float(results['detection_scores'][i, j]), 4),
                'emotic': emotic,
                'bounding_box': { 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 }
                })
            # vis.image_draw_bounding_box(image_disp, [x1, y1, x2, y2])
            # vis.image_draw_text(image_disp, 
            #     [x1, y1], 
            #     '{}%'.format(int(100 * detection_score)),
            #     (0, 0, 0))

            # text_label_height = _get_text_label_height()
            # for disp_ind, display_str in enumerate(display_str_list):
            #   vis.image_draw_text(image_disp, 
            #       [x1, y1 + 1.0 * text_label_height / image_disp.shape[0] * (disp_ind + 1)], 
            #       display_str, 
            #       color=(0, 0, 0),
            #       bkg_color=(0, 0, 0))

        det_results.append(det_result)
        features.append(person_feature_list)
        #vis.image_save('%s/%i.jpg' % (vis_dir, i), image_disp, True)
      return features, det_results

    # Iterate through video ids.
    for video_id in _video_id_iterator(FLAGS.video_id_path):
      _extract_video_feature(video_id, FLAGS.video_dir, 
          FLAGS.output_dir, _extract_feature)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
