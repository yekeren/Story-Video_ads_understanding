
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import json

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from google.protobuf import text_format
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import average_precision_score

import eval_utils
from protos import pipeline_pb2
from protos import baseline_model_pb2
from reader import ads_examples
from models import builder
from utils import train_utils

from sklearn.metrics import accuracy_score

flags.DEFINE_float('per_process_gpu_memory_fraction', 0.5, 
                   'GPU usage limitation.')

flags.DEFINE_string('pipeline_proto', '', 
                    'Path to the pipeline proto file.')

flags.DEFINE_string('train_log_dir', '', 
                    'The directory where the graph and checkpoints are saved.')

flags.DEFINE_string('eval_log_dir', '', 
                    'The directory where the graph and checkpoints are saved.')

flags.DEFINE_integer('number_of_steps', 1000000, 
                     'Maximum number of steps.')

flags.DEFINE_string('saved_ckpts_dir', '',
                    'The directory to backup checkpoint')

flags.DEFINE_integer('top_k', 5, 
                     'Export the top_k predictions.')

flags.DEFINE_string('json_path', '',
                    'The path to the output json file.')

flags.DEFINE_integer('split', -1, 
                    'If greater or equal than -1, modify the input_path.')


FLAGS = flags.FLAGS
slim = tf.contrib.slim


def load_pipeline_proto(filename):
  """Load pipeline proto file.

  Args:
    filename: path to the pipeline proto file.

  Returns:
    pipeline_proto: an instance of pipeline_pb2.Pipeline
  """
  def _revise_name(filename, offset):
    filename, postfix = filename.split('.record.')
    filename = '{}.record.{}'.format(filename, (int(postfix) + offset) % 10)
    return filename

  pipeline_proto = pipeline_pb2.Pipeline()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), pipeline_proto)
  if FLAGS.number_of_steps > 0:
    pipeline_proto.train_config.number_of_steps = FLAGS.number_of_steps
    pipeline_proto.eval_config.number_of_steps = FLAGS.number_of_steps

  pipeline_proto.example_reader.batch_size = 1
  pipeline_proto.example_reader.num_epochs = 1

  if FLAGS.split > -1:
    for _ in xrange(4):
      del pipeline_proto.example_reader.input_path[-1]
    n_files = len(pipeline_proto.example_reader.input_path)
    for i in xrange(n_files):
      pipeline_proto.example_reader.input_path[i] = _revise_name(
          pipeline_proto.example_reader.input_path[i], FLAGS.split + 8)
  return pipeline_proto


def load_vocab(filename):
  """Load vocab file.

  Args:
    filename: path to the sentiment vocab file.

  Returns:
    vocab: a list mapping from id to sentiment word.
  """
  with open(filename, 'r') as fp:
    sents = [x.strip('\n') for x in fp.readlines()]
  return sents


def evaluate_once(sess, writer, global_step, video_ids, predictions):
  """Evaluate once for the dataset.

  Returns:
    accuracy: the accuracy of the current model.
  """
  count = 0
  results = {}

  try:
    while True:
      ids, prediction_vals = sess.run([video_ids, predictions])
      vid = ids[0]

      count += 1
      if count % 50 == 0:
        logging.info('On processing video %i.', count)

      n_frames = prediction_vals['n_frames'][0]
      logits = prediction_vals['logits_climax'][0]
      labels = prediction_vals['labels_climax'][0]
      scores = 1.0 / (1 + np.exp(-logits))

      results[vid] = {
        'n_frames': n_frames,
        'labels': [round(x, 1) for x in labels],
        'scores': [round(x, 4) for x in scores],
      }

  except tf.errors.OutOfRangeError:
    logging.info('Done evaluating -- epoch limit reached')

  with open(FLAGS.json_path, 'w') as fp:
    fp.write(json.dumps(results))
  logging.info('Wrote results to %s.', FLAGS.json_path)


def _create_mapping():
  stemmer = PorterStemmer()
  mapping = {}
  with open(FLAGS.sentiment_anno_vocab, 'r') as fp:
    for line in fp.readlines():
      words = re.findall('\w+', line.lower())
      assert words[-1] in eval_utils.sentiments

      for word in words:
        if not word.isdigit() and word != 'abbreviation':
          mapping[stemmer.stem(word)] = words[-1]
  return mapping


def load_raw_annot(filename):
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
    for i in xrange(len(data[k])):
      data[k][i] = eval_utils.revise_sentiment(data[k][i], mapping)
  return data

def load_clean_annot(filename):
  """Loads the clean sentiments annotation.

  Args:
    filename: path to the clean annotation json file.

  Returns:
    data: a dict mapping from video_id to sentiment.
  """
  vocab = load_vocab(FLAGS.topic_vocab_path)

  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  for k in data.keys():
    data[k] = vocab[data[k] - 1]

  return data

def main(_):
  logging.set_verbosity(tf.logging.INFO)

  assert os.path.isfile(FLAGS.pipeline_proto)

  g = tf.Graph()
  with g.as_default():
    pipeline_proto = load_pipeline_proto(FLAGS.pipeline_proto)
    logging.info("Pipeline configure: %s", '=' * 128)
    logging.info(pipeline_proto)

    # Get examples from reader.
    examples = ads_examples.get_examples(pipeline_proto.example_reader)

    # Build model for evaluation.
    global_step = slim.get_or_create_global_step()

    model = builder.build(pipeline_proto.model, is_training=False)
    predictions = model.build_inference(examples)
    loss_dict = model.build_loss(predictions)

    uninitialized_variable_names = tf.report_uninitialized_variables()
    saver = tf.train.Saver()
    
    init_op = tf.group(tf.local_variables_initializer(),
        tf.global_variables_initializer())

  session_config = train_utils.default_session_config( 
      FLAGS.per_process_gpu_memory_fraction)

  # Start session.
  logging.info('=' * 128)
  eval_config = pipeline_proto.eval_config

  model_path = train_utils.get_latest_model(FLAGS.saved_ckpts_dir)
  if model_path is not None:
    with tf.Session(graph=g, config=session_config) as sess:

      # Initialize model.
      sess.run(init_op)
      saver.restore(sess, model_path)
      logging.info('Restore from %s.', model_path)

      warn_names = sess.run(uninitialized_variable_names)
      assert len(warn_names) == 0

      # Evaluation loop.
      step = sess.run(global_step)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      evaluate_once(sess, None, step, 
          examples['video_id'], predictions=predictions)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
