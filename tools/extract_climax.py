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
    for line in fp.readlines():
      url, climax = line.strip('\n').split('\t')
      video_id = url.split('?v=')[1]
      climax_list = map(lambda x: int(float(x)), climax.split(','))
      yield video_id, climax_list


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


def _extract_video_feature(video_id, climax_list, 
    video_dir, output_dir, extract_fn=None):
  """Extracts features from video.
  """
  dirname = os.path.join(video_dir, '%s' % video_id)
  filenames = _get_screenshots(dirname)
  n_frames = len(filenames)

  features = []  # Features to be exported to the npz file.
  results = []  # Readable results to be exported to the json file.

  climax_set = set(climax_list)
  for index , filename in enumerate(filenames):
    if index in climax_set:
      features.append(1)
      results.append({'score': 1.0})
    else:
      features.append(0)
      results.append({'score': 0.0})

  assert len(features) == len(results) == n_frames

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, np.array(features))

  filename = os.path.join(output_dir, '%s.json' % video_id)
  with open(filename, 'w') as fp:
    fp.write(json.dumps(results))

  logging.info('Video features of %s are saved to %s.', video_id, filename)


def main(argv):
  logging.set_verbosity(logging.INFO)
  
  # Iterate through video ids.
  data = {}

  for video_id, climax_list in \
    _video_id_iterator(FLAGS.climax_annotation_path):

    _extract_video_feature(
        video_id, climax_list, FLAGS.video_dir, FLAGS.output_dir)

    for climax in climax_list:
      data[climax] = data.get(climax, 0) + 1
  
  data = sorted(data.iteritems(), lambda x, y: cmp(x[1], y[1]))
  for d in data:
    logging.info('Annotation frequencies: %s', json.dumps(d))
  logging.info('Done')

if __name__ == '__main__':
  app.run()
