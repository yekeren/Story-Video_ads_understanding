
import os
import re
import sys
import json
import traceback

import openface
import cv2
import numpy as np

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from google.protobuf import text_format

import torch
from torch.autograd import Variable
from torchvision import transforms, models
from enum import Enum 
import my_inception
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from utils import vis

flags.DEFINE_integer('max_videos', -1, 
                     'Maximum number of videos to run.')

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

flags.DEFINE_string('output_vocab', '', 
                    'Path to the output vocab file.')

flags.DEFINE_integer('batch_size', 64, '')


FLAGS= flags.FLAGS

vocab_expressions = [
    'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

vocab_valence_arousal = ['pleasant', 'activated']

def _get_text_label_height():
  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  thickness = 1

  (sx, sy), baseline = cv2.getTextSize(
      'text', font_face, font_scale, thickness)
  return sy + baseline


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

  features = np.stack(features, axis=0)

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, features)

  filename = os.path.join(output_dir, '%s.json' % video_id)
  with open(filename, 'w') as fp:
    fp.write(json.dumps(detections))

  logging.info('Video features of %s are saved to %s.', video_id, filename)


def main(argv):
  logging.set_verbosity(logging.INFO)

  # The pytorch model.
  model = my_inception.inception_v3(pretrained=False,num_classes=8,transform_input=True)
  model.cuda().eval()
  expression_results = {}
  transform = transforms.Compose([
              transforms.Resize((299, 299), interpolation=2),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])
  model.load_state_dict(torch.load('trained_expression_inception.pth'))

  dlibFacePredictorPath = 'openface/models/dlib/shape_predictor_68_face_landmarks.dat'
  align = openface.AlignDlib(dlibFacePredictorPath)
  assert align is not None

  def force_range(x, lower, upper):
    if x < lower: x = lower
    if x > upper: x = upper
    assert lower <= x <= upper
    return int(x)

  # Start session to extract video features.
  def _extract_feature(video_id, images_data):
    det_results = []
    features = []

    for i, rgb in enumerate(images_data):
      #bgr_frame = rgb[:, :, ::-1].copy()

      det_result = []
      boxes = align.getAllFaceBoundingBoxes(rgb)
      boxes = [(box.left(), box.top(), box.width(), box.height()) for box in boxes]

      height, width, _ = rgb.shape
      height, width = float(height), float(width)

      face_features = []

      for (x, y, w, h) in boxes:
        #cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x1, y1, x2, y2 = x, y, x + w, y + h

        x1 = force_range(x1, 0, width - 1)
        x2 = force_range(x2, x1 + 1, width - 1)
        y1 = force_range(y1, 0, height - 1)
        y2 = force_range(y2, y1 + 1, height - 1)

        rgb_face = rgb[y1: y2, x1: x2, :]
        face = Image.fromarray(rgb_face, 'RGB')
        
        # Use AffectNet to predict emotions.
        face = transform(face).unsqueeze(0)
        face = Variable(face, volatile=True).cuda()
        expressions, valence_arousal = model(face)
        expressions = F.softmax(expressions, dim=1)
        valence_arousal = F.tanh(valence_arousal)
        expressions = expressions.cpu().data.numpy()[0, :]
        valence_arousal = valence_arousal.cpu().data.numpy()[0, :]

        # Write results.
        face_features.append(np.concatenate([expressions, valence_arousal]))
        det_result.append({
            'score': 1.0,
            'affectnet': 0,
            'bounding_box': { 
                'x1': round(1.0 * x1 / width, 3), 
                'y1': round(1.0 * y1 / height, 3), 
                'x2': round(1.0 * x2 / width, 3), 
                'y2': round(1.0 * y2 / height, 3)
                },
            'expression': vocab_expressions[expressions.argmax()],
            'expression_score': round(expressions.max(), 3),
            'pleasant': round(valence_arousal[0], 3),
            'activated': round(valence_arousal[1], 3),
            })

      if not face_features:
        face_features = np.zeros((10))
      else:
        face_features = np.stack(face_features).mean(0)

      det_results.append(det_result)
      features.append(face_features)

    return features, det_results

  # Iterate through video ids.
  for video_id in _video_id_iterator(FLAGS.video_id_path):
    _extract_video_feature(video_id, FLAGS.video_dir, 
        FLAGS.output_dir, _extract_feature)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
