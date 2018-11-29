
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging

from object_detection.builders import hyperparams_builder

from protos import utils_pb2
from protos import train_config_pb2

slim = tf.contrib.slim


def encode_feature(features, config, is_training=False, reuse=None):
  """Encodes image using the config.

  Args:
    features: a [batch, feature_dimensions] tf.float32 tensor.
    config: an instance of utils_pb2.ImageEncoder.
    is_training: if True, training graph is built.

  Raises:
    ValueError if config is not an instance of ImageEncoder

  Returns:
    features_encoded: a [batch, num_outputs] tf.float32 tensor.
  """
  if not isinstance(config, utils_pb2.FCEncoder):
    raise ValueError('The config has to be an instance of FCEncoder.')

  hp = hyperparams_builder.build(config.fc_hyperparams, is_training=is_training)

  node = features
  node = slim.dropout(node, config.input_dropout_keep_prob, 
      is_training=is_training)
  with slim.arg_scope(hp):
    node = slim.fully_connected(node, config.num_outputs, 
        scope=config.scope, reuse=reuse)
  node = slim.dropout(node, config.output_dropout_keep_prob,
      is_training=is_training)

  return node

def reduce_mean_for_varlen_inputs(features, lengths):
  """Reduces mean on varlen sequence features.

  Args:
    inputs: a [batch, max_seq_len, dims] tf.float32 tensor.
    lengths: a [batch] tf.int64 tensor indicating the length for each feature.

  Returns:
    mean_val: a [batch, dims] tf.float32 tensor.
  """
  sum_val = tf.reduce_sum(features, 1)
  mean_val = tf.div(sum_val,
      tf.expand_dims(tf.maximum(tf.cast(lengths, tf.float32), 1e-8), 1))
  return mean_val

def softmax_for_varlen_logits(logits, lengths):
  """Processes softmax on varlen sequences.

  Args:
    logits: a [batch, max_seq_len] tf.float32 tensor.
    lengths: a [batch] tf.int64 tensor indicating the length for each logit.

  Returns:
    proba: a [batch, max_seq_len] tf.float32 tensor indicating the
      probabilities.
  """
  min_val = -1e9

  max_seq_len = logits.get_shape()[-1].value
  if max_seq_len is None:
    max_seq_len = tf.shape(logits)[1]

  boolean_masks = tf.greater_equal(
      tf.range(max_seq_len, dtype=tf.int32), 
      tf.expand_dims(tf.cast(lengths, tf.int32), 1))
  padding_values = min_val * tf.cast(boolean_masks, tf.float32)

  return tf.nn.softmax(logits + padding_values)

