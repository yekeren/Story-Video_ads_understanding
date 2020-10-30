# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import re
import cv2
import numpy as np
import json
from PIL import Image
from tensorflow import flags
from tensorflow import logging

flags.DEFINE_string('kyle_dir', '', 
                    'Path to the directory storing features.')

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

flags.DEFINE_integer('batch_size', 1, '')

FLAGS= flags.FLAGS

logging.set_verbosity(logging.INFO)

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

def _extract_video_feature(video_id, feature_dir, video_dir, output_dir):
  """Extracts features from video.
  """
  dirname = os.path.join(video_dir, '%s' % video_id)
  filenames = _get_screenshots(dirname)
  n_frames = len(filenames)

  filename = os.path.join(feature_dir, '%s.npz' % video_id)
  if not os.path.isfile(filename):
    print filename

  return

  with open(filename, 'rb') as fp:
    data = np.load(filename)

  logging.info('Video features of %s are saved to %s.', video_id, filename)

# Iterate through video ids.
for video_id in _video_id_iterator(FLAGS.video_id_path):
  _extract_video_feature(
      video_id, FLAGS.kyle_dir, FLAGS.video_dir, FLAGS.output_dir)

logging.info('Done')
