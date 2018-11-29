
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow import logging
from google.protobuf import text_format

from models import model
from protos import climax_model_pb2

slim = tf.contrib.slim


class Model(model.Model):
  """Baseline model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an instance of climax_model_pb2.ClimaxModel

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, climax_model_pb2.ClimaxModel):
      raise ValueError('The model_proto has to be an instance of ClimaxModel.')

  def _encode_none(self, n_frames, frame_features):
    """Processes identity mapping.

    Args:
      n_frames: a [batch] tf.uint64 tensor.
      frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.

    Returns:
      video_feature: a [batch, video_feature_dims] tf.float32 tensor.
      outputs: a [batch, max_frame_len, output_dims] tf.float32 tensor.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    frame_features = slim.dropout(frame_features,
        model_proto.bof_input_dropout_keep_prob, is_training=is_training)

    return None, frame_features

  def _encode_bilstm(self, n_frames, frame_features):
    """Using bi-directional lstm to encode the frame features.

    Args:
      n_frames: a [batch] tf.uint64 tensor.
      frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.

    Returns:
      video_feature: a [batch, video_feature_dims] tf.float32 tensor.
      outputs: a [batch, max_frame_len, output_dims] tf.float32 tensor.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    # Initialize lstm cell.
    def lstm_cell():
      cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units=model_proto.lstm_hidden_units, 
          forget_bias=1.0)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
            input_keep_prob=model_proto.lstm_input_keep_prob,
            output_keep_prob=model_proto.lstm_output_keep_prob)
      return cell

    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        lstm_cell() for _ in xrange(model_proto.lstm_number_of_layers)])

    outputs, state = tf.nn.bidirectional_dynamic_rnn( 
        cell_fw=rnn_cell, 
        cell_bw=rnn_cell, 
        inputs=frame_features, 
        sequence_length=n_frames, 
        dtype=tf.float32)
    outputs = tf.concat([outputs[0], outputs[1]], axis=2)
    return None, outputs

  def _encode_lstm(self, n_frames, frame_features):
    """Using lstm to encode the frame features.

    Args:
      n_frames: a [batch] tf.uint64 tensor.
      frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.

    Returns:
      video_feature: a [batch, video_feature_dims] tf.float32 tensor.
      outputs: a [batch, max_frame_len, output_dims] tf.float32 tensor.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    # Initialize lstm cell.
    def lstm_cell():
      cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units=model_proto.lstm_hidden_units, 
          forget_bias=1.0)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
            input_keep_prob=model_proto.lstm_input_keep_prob,
            output_keep_prob=model_proto.lstm_output_keep_prob)
      return cell

    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        lstm_cell() for _ in xrange(model_proto.lstm_number_of_layers)])

    outputs, state = tf.nn.dynamic_rnn( 
        cell=rnn_cell, 
        inputs=frame_features, 
        sequence_length=n_frames, 
        dtype=tf.float32)
    return None, outputs

  def _encode_conv(self, n_frames, frame_features):
    """Using lstm to encode the frame features.

    Args:
      n_frames: a [batch] tf.uint64 tensor.
      frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.

    Returns:
      video_feature: a [batch, video_feature_dims] tf.float32 tensor.
      outputs: a [batch, max_frame_len, output_dims] tf.float32 tensor.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    frame_features = slim.dropout(frame_features,
        model_proto.conv_input_dropout_keep_prob, is_training=is_training)

    # Convolution.
    input_dims = frame_features.get_shape()[-1].value
    output_dims = model_proto.conv_output_units
    window_size = model_proto.conv_window_size

    weight_decay = 1e-6
    kernel = tf.get_variable(name='conv',
        shape=[window_size, 1, output_dims],
        regularizer=slim.l2_regularizer(weight_decay))
    kernel = tf.tile(kernel, [1, input_dims, 1])

    outputs = tf.nn.conv1d(
        frame_features, kernel, stride=1, padding='SAME')
    outputs = tf.nn.relu6(outputs)

    outputs = slim.dropout(outputs,
        model_proto.conv_output_dropout_keep_prob, is_training=is_training)

    return None, outputs

  def compute_frame_difference(self, frame_features):
    """Compute the difference of frames as features.

    Args:
      frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.

    Returns:
      frame_diff: a [batch, max_frame_len, feature_dims] tf.float32 tensor, the
        first frame is padding with zero.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    max_frame_len = tf.shape(frame_features)[1]

    frame_features_shifted = tf.concat(
        [frame_features[:, :1, :], frame_features], axis=1)
    frame_features_shifted = frame_features_shifted[:, :max_frame_len, :]

    frame_diff = tf.square(frame_features - frame_features_shifted)
    return frame_diff


  def build_inference(self, examples):
    """Builds tensorflow graph for inference.

    Args:
      examples: a dict mapping from name to input tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    # Manipulate input tensors.
    features_list = []
    if model_proto.use_frame_features:
      features_list.append(examples['frame_features'])
    if model_proto.use_common_object:
      features_list.append(examples['common_object_features'])
    if model_proto.use_place:
      features_list.append(examples['place_features'])
    if model_proto.use_emotic:
      features_list.append(examples['emotic_features'])
    if model_proto.use_affectnet:
      features_list.append(examples['affectnet_features'])
    if model_proto.use_shot_boundary:
      features_list.append(examples['shot_boundary_features'])
    if model_proto.use_optical_flow:
      features_list.append(examples['optical_flow_features'])
    if model_proto.use_audio:
      features_list.append(examples['audio_features'])

    frame_features = tf.concat(features_list, axis=2)
    if model_proto.use_frame_difference:
      diff_features = self.compute_frame_difference(frame_features)
      frame_features = tf.concat([frame_features, diff_features], axis=2)

    n_frames = examples['n_frames']
    batch = n_frames.get_shape()[0]

    # Encoding.
    init_width = model_proto.lstm_init_width
    initializer = tf.random_uniform_initializer(-init_width, init_width)

    encode_methods = {
      climax_model_pb2.ClimaxModel.NONE: self._encode_none,
      climax_model_pb2.ClimaxModel.LSTM: self._encode_lstm,
      climax_model_pb2.ClimaxModel.BILSTM: self._encode_bilstm,
      climax_model_pb2.ClimaxModel.CONV: self._encode_conv,
    }

    with tf.variable_scope('ads_video', initializer=initializer) as scope:
      # LSTM model.
      encode_func = encode_methods[model_proto.encode_method]
      _, outputs = encode_func(n_frames, frame_features)

    # Predict logits.
    outputs_reshaped = tf.reshape(outputs, [-1, outputs.get_shape()[-1]])

    logits_climax_reshaped = slim.fully_connected(
        outputs_reshaped, 
        num_outputs=1, 
        activation_fn=None,
        scope='logits_climax')
    logits_climax = tf.reshape(logits_climax_reshaped, [batch, -1])

    predictions = { 
      'n_frames': examples['n_frames'],
      'logits_climax': logits_climax,
      'labels_climax': tf.squeeze(examples['climax_features'], 2),
    }

    return predictions

  def build_loss(self, predictions):
    """Builds loss tensor for the model.

    Args:
      predictions:

    Returns:
      loss_dict: a dict mapping from name to loss tensor.
    """
    model_proto = self._model_proto

    n_frames = predictions['n_frames']
    max_n_frames = tf.reduce_max(n_frames)
    masks = tf.less(
        tf.range(max_n_frames, dtype=tf.int64), tf.expand_dims(n_frames, 1))

    logits = predictions['logits_climax']
    labels = predictions['labels_climax']
    indices = tf.where(tf.reduce_sum(labels, 1) > 0)
    indices = tf.squeeze(indices, [1])

    tf.summary.scalar('losses/examples', tf.shape(indices)[0])

    # Random negative sampling.
    if model_proto.sample_negatives:
      batch = n_frames.get_shape()[0].value

      # Compute sampling mask.
      max_n_frames = tf.reduce_max(n_frames)
      masks_list = []
      for batch_index in xrange(batch):
        rand_indices = tf.random_uniform(
            shape=[5],
            maxval=n_frames[batch_index],
            dtype=tf.int64)
        masks_list.append(tf.sparse_to_dense(
            rand_indices, output_shape=[max_n_frames], 
            sparse_values=True, default_value=False, validate_indices=False))
      masks = tf.stack(masks_list)
      masks = tf.logical_or(masks, tf.cast(labels, tf.bool))

      # Compute label mask.
      logits = tf.gather(logits, indices)
      labels = tf.gather(labels, indices)
      n_frames = tf.gather(n_frames, indices)
      masks = tf.gather(masks, indices)

      labels = tf.boolean_mask(labels, masks)
      logits = tf.boolean_mask(logits, masks)

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)

    else:
      logits = tf.gather(logits, indices)
      labels = tf.gather(labels, indices)
      n_frames = tf.gather(n_frames, indices)
      masks = tf.gather(masks, indices)

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      losses = tf.reduce_sum(losses * tf.cast(masks, tf.float32), 1)
      losses = tf.div(losses, tf.maximum(tf.cast(n_frames, tf.float32), 1e-30))

    loss_dict = {
      'climax_sigmoid_cross_entropy': tf.reduce_mean(losses)
    }
    return loss_dict
