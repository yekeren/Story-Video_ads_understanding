import os
import re
import json
import numpy as np
from tensorflow import app
from tensorflow import flags
from tensorflow import logging

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('climax_annotation_path', '', 
                    'Path to the annotation file.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

FLAGS= flags.FLAGS


def _video_id_iterator(filename):
  """Reads file and yields <video_id, climax> tuple.

  Args:
    filename: path to the original annotation file.

  Yields:
    <video_id, climax> tuple.
  """
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

    for vid, prediction in data.iteritems():
      yield vid, prediction['scores']
    
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


def _extract_video_feature(video_id, prediction, 
    video_dir, output_dir, extract_fn=None):
  """Extracts features from video.
  """
  dirname = os.path.join(video_dir, '%s' % video_id)
  filenames = _get_screenshots(dirname)
  n_frames = len(filenames)

  features = np.array(prediction)
  assert len(features) == n_frames

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, np.array(features))

  logging.info('Video features of %s are saved to %s.', video_id, filename)


def main(argv):
  logging.set_verbosity(logging.INFO)
  
  # Iterate through video ids.
  for video_id, prediction in \
    _video_id_iterator(FLAGS.climax_annotation_path):

    _extract_video_feature(
        video_id, prediction, FLAGS.video_dir, FLAGS.output_dir)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
