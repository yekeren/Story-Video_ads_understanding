
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from google.protobuf import text_format

from protos import pipeline_pb2
from reader import ads_examples
from models import builder
from utils import train_utils

flags.DEFINE_float('per_process_gpu_memory_fraction', 1.0, 
                   'GPU usage limitation.')

flags.DEFINE_string('pipeline_proto', '', 
                    'Path to the pipeline proto file.')

flags.DEFINE_string('train_log_dir', '', 
                    'The directory where the graph and checkpoints are saved.')

flags.DEFINE_integer('number_of_steps', 0, 
                    'If none zero, use this value instead of that in the config file .')

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

  if FLAGS.split > -1:
    n_files = len(pipeline_proto.example_reader.input_path)
    for i in xrange(n_files):
      pipeline_proto.example_reader.input_path[i] = _revise_name(
          pipeline_proto.example_reader.input_path[i], FLAGS.split)

  return pipeline_proto


def main(_):
  logging.set_verbosity(tf.logging.INFO)

  assert os.path.isfile(FLAGS.pipeline_proto), FLAGS.pipeline_proto

  g = tf.Graph()
  with g.as_default():
    pipeline_proto = load_pipeline_proto(FLAGS.pipeline_proto)
    logging.info("Pipeline configure: %s", '=' * 128)
    logging.info(pipeline_proto)

    train_config = pipeline_proto.train_config

    # Get examples from reader.
    examples = ads_examples.get_examples(pipeline_proto.example_reader)

    # Build model for training.
    global_step = slim.get_or_create_global_step()

    model = builder.build(pipeline_proto.model, is_training=True)
    predictions = model.build_inference(examples)
    loss_dict = model.build_loss(predictions)

    init_fn = model.get_init_fn()
    uninitialized_variable_names = tf.report_uninitialized_variables()

    # Loss and optimizer.
    for loss_name, loss_tensor in loss_dict.iteritems():
      tf.losses.add_loss(loss_tensor)
      tf.summary.scalar('losses/{}'.format(loss_name), loss_tensor)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = train_utils.build_optimizer(train_config)
    if train_config.moving_average:
      optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer,
          average_decay=0.99)

    gradient_multipliers = train_utils.build_multipler(
        train_config.gradient_multiplier)

    variables_to_train = model.get_variables_to_train()
    for var in variables_to_train:
      logging.info(var)

    train_op = slim.learning.create_train_op(total_loss,
        variables_to_train=variables_to_train, 
        clip_gradient_norm=5.0,
        gradient_multipliers=gradient_multipliers,
        summarize_gradients=True,
        optimizer=optimizer)

    saver = None
    if train_config.moving_average:
      saver = optimizer.swapping_saver()

  # Starts training.
  logging.info('Start training.')

  session_config = train_utils.default_session_config( 
      FLAGS.per_process_gpu_memory_fraction)
  slim.learning.train(train_op, 
      logdir=FLAGS.train_log_dir,
      graph=g,
      master='',
      is_chief=True,
      number_of_steps=train_config.number_of_steps,
      log_every_n_steps=train_config.log_every_n_steps,
      save_interval_secs=train_config.save_interval_secs,
      save_summaries_secs=train_config.save_summaries_secs,
      session_config=session_config,
      init_fn=init_fn,
      saver=saver)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
