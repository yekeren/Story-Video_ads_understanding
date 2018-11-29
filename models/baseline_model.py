
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow import logging
from google.protobuf import text_format

from models import model
from models import utils
from protos import baseline_model_pb2

slim = tf.contrib.slim


class Model(model.Model):
  """Baseline model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an instance of baseline_model_pb2.BaselineModel

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, baseline_model_pb2.BaselineModel):
      raise ValueError('The model_proto has to be an instance of BaselineModel.')

  def _encode_bof(self, n_frames, frame_features):
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

    states = []

    frame_features = slim.dropout(frame_features,
        model_proto.bof_input_dropout_keep_prob, 
        is_training=is_training)

    if model_proto.bof_use_avg_pool:
      states.append(
          utils.reduce_mean_for_varlen_inputs(frame_features, n_frames))
    if model_proto.bof_use_max_pool:
      states.append(tf.reduce_max(frame_features, 1))

    state = tf.concat(states, 1)
    return state, None

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
            output_keep_prob=model_proto.lstm_output_keep_prob,
            state_keep_prob=model_proto.lstm_state_keep_prob)
      return cell

    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        lstm_cell() for _ in xrange(model_proto.lstm_number_of_layers)])

    outputs, state = tf.nn.dynamic_rnn( 
        cell=rnn_cell, 
        inputs=frame_features, 
        sequence_length=n_frames, 
        dtype=tf.float32)
    #last_state = tf.concat([state[-1].h, state[-1].c], axis=1)
    #return last_state, outputs
    return state[-1].h, outputs

  def build_step_inference(self, examples):
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
    if model_proto.use_climax:
      features_list.append(examples['climax_predictions'])

    frame_features= tf.concat(features_list, axis=2)

    n_frames = examples['n_frames']
    batch = n_frames.get_shape()[0]

    # RNN encode.
    assert baseline_model_pb2.BaselineModel.LSTM == model_proto.encode_method

    init_width = model_proto.lstm_init_width
    initializer = tf.random_uniform_initializer(-init_width, init_width)

    with tf.variable_scope('ads_video', initializer=initializer) as scope:
      # Initialize lstm cell.
      def lstm_cell():
        cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=model_proto.lstm_hidden_units, 
            forget_bias=1.0)
        if is_training:
          cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
              input_keep_prob=model_proto.lstm_input_keep_prob,
              output_keep_prob=model_proto.lstm_output_keep_prob,
              state_keep_prob=model_proto.lstm_state_keep_prob)
        return cell

      rnn_cell = tf.contrib.rnn.MultiRNNCell([
          lstm_cell() for _ in xrange(model_proto.lstm_number_of_layers)])
      outputs, _ = tf.nn.dynamic_rnn( 
          cell=rnn_cell, 
          inputs=frame_features, 
          sequence_length=n_frames, 
          dtype=tf.float32)  # Ouputs is a [batch, max_frame_len, hidden_units] float tensor.

    # Infer the per-step sentiments and topics.
    batch, _, dims = outputs.get_shape().as_list()
    outputs_reshaped = tf.reshape(outputs, [-1, dims])

    # Model is joint trained with topic objective.
    proba_topic = None
    if model_proto.joint_training_model:
      logits_topic_reshaped = slim.fully_connected(
          outputs_reshaped, 
          num_outputs=model_proto.number_of_topics, 
          activation_fn=None,
          scope='topic_fc')
      proba_topic = tf.reshape(
          tf.nn.softmax(logits_topic_reshaped),
          [batch, -1, model_proto.number_of_topics])
      outputs_reshaped = tf.concat([outputs_reshaped, logits_topic_reshaped], 1)

    logits_sentiment_reshaped = slim.fully_connected(
        outputs_reshaped, 
        num_outputs=model_proto.number_of_sentiments, 
        activation_fn=None,
        scope='sentiment_fc')

    proba_sentiment = tf.reshape(
        tf.nn.softmax(logits_sentiment_reshaped),
        [batch, -1, model_proto.number_of_sentiments])

    predictions = {
      'n_frames': n_frames,
      'sentiments': proba_sentiment,
    }
    if proba_topic is not None:
      predictions['topics'] = proba_topic
    return predictions

    ## Predict logits.
    #if model_proto.joint_training_model:
    #  logits_topic = slim.fully_connected(last_hidden_state, 
    #      num_outputs=model_proto.number_of_topics, 
    #      activation_fn=None,
    #      scope='topic_fc')

    #  logits_sentiment = slim.fully_connected(
    #      tf.concat([last_hidden_state, logits_topic], 1),
    #      num_outputs=model_proto.number_of_sentiments, 
    #      activation_fn=None,
    #      scope='sentiment_fc')

    #  predictions = { 
    #    'logits_topic': logits_topic,
    #    'logits_sentiment': logits_sentiment,
    #  }
    #else:
    #  logits_sentiment = slim.fully_connected(last_hidden_state, 
    #      num_outputs=model_proto.number_of_sentiments, 
    #      activation_fn=None,
    #      scope='sentiment_fc')
    #  predictions = { 'logits_sentiment': logits_sentiment }

    ## Pass labels.
    #predictions['n_frames'] = examples['n_frames']
    #predictions['labels_sentiment'] = examples['sentiment']
    #if 'sentiment_list' in examples:
    #  predictions['labels_sentiment_list'] = examples['sentiment_list']
    #predictions['labels_topic'] = examples['topic']
    #predictions['labels_common_object'] = examples['common_object_features']
    #predictions['labels_emotic'] = examples['emotic_features']

    #return predictions

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
    if model_proto.use_climax:
      features_list.append(examples['climax_predictions'])

    frame_features= tf.concat(features_list, axis=2)

    n_frames=examples['n_frames']
    batch = n_frames.get_shape()[0]

    # RNN encode.
    init_width = model_proto.lstm_init_width
    initializer = tf.random_uniform_initializer(-init_width, init_width)

    encode_methods = {
      baseline_model_pb2.BaselineModel.LSTM: self._encode_lstm,
      baseline_model_pb2.BaselineModel.BOF: self._encode_bof,
    }

    with tf.variable_scope('ads_video', initializer=initializer) as scope:
      # LSTM model.
      encode_func = encode_methods[model_proto.encode_method]
      last_hidden_state, outputs = encode_func(
          n_frames, frame_features)

    # Predict logits.
    if model_proto.joint_training_model:
      logits_topic = slim.fully_connected(last_hidden_state, 
          num_outputs=model_proto.number_of_topics, 
          activation_fn=None,
          scope='topic_fc')

      logits_sentiment = slim.fully_connected(
          tf.concat([last_hidden_state, logits_topic], 1),
          num_outputs=model_proto.number_of_sentiments, 
          activation_fn=None,
          scope='sentiment_fc')

      predictions = { 
        'logits_topic': logits_topic,
        'logits_sentiment': logits_sentiment,
      }
    else:
      logits_sentiment = slim.fully_connected(last_hidden_state, 
          num_outputs=model_proto.number_of_sentiments, 
          activation_fn=None,
          scope='sentiment_fc')
      predictions = { 'logits_sentiment': logits_sentiment }

    # Pass labels.
    predictions['n_frames'] = examples['n_frames']
    predictions['labels_sentiment'] = examples['sentiment']
    if 'sentiment_list' in examples:
      predictions['labels_sentiment_list'] = examples['sentiment_list']
    predictions['labels_topic'] = examples['topic']
    predictions['labels_common_object'] = examples['common_object_features']
    predictions['labels_emotic'] = examples['emotic_features']

    return predictions

  def build_loss(self, predictions):
    """Builds loss tensor for the model.

    Args:
      predictions:

    Returns:
      loss_dict: a dict mapping from name to loss tensor.
    """
    loss_dict = {}
    model_proto = self._model_proto

    n_frames = predictions['n_frames']
    batch = n_frames.get_shape()[0]

    # Loss for sentiment.
    number_of_classes = model_proto.number_of_sentiments
    logits = predictions['logits_sentiment']

    if model_proto.sentiment_loss_function == baseline_model_pb2.BaselineModel.SIGMOID_RAW:
      labels = tf.cast(predictions['labels_sentiment_list'], tf.float32)
    else:
      labels = predictions['labels_sentiment']
      labels = tf.sparse_to_dense(
          tf.stack([tf.range(batch.value, dtype=tf.int64), labels], axis=1),
          output_shape=[batch.value, number_of_classes],
          sparse_values=1.0,
          default_value=0.0,
          validate_indices=False)

    batch_size = batch.value
    if model_proto.sample_negatives:
      multiplier = model_proto.negative_multiplier

      num_positives = tf.reduce_sum(tf.cast(labels > 0, tf.int32), 0)
      num_examples_per_class = tf.minimum(num_positives * multiplier, batch_size)

      masks = []
      for i in xrange(model_proto.number_of_sentiments):
        indices = tf.random_uniform(
            shape=[tf.cast(num_examples_per_class[i], tf.int32)],
            minval=0,
            maxval=batch_size,
            dtype=tf.int32)
        masks.append(tf.sparse_to_dense(
            indices, output_shape=[batch_size], 
            sparse_values=1.0, default_value=0.0, validate_indices=False))
      masks = tf.stack(masks, 1)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      loss_dict['sentiment'] = tf.div(
          tf.reduce_sum(losses * masks),
          tf.reduce_sum(masks) + 1e-30) * model_proto.sentiment_loss_weight

    else:
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      loss_dict['sentiment'] = tf.reduce_mean(losses) * model_proto.sentiment_loss_weight

    # Loss for topic.
    if model_proto.joint_training_model:
      labels = predictions['labels_topic']
      logits = predictions['logits_topic']
      number_of_classes = model_proto.number_of_topics

      labels = tf.sparse_to_dense(
          tf.stack([tf.range(batch, dtype=tf.int64), labels], axis=1),
          output_shape=[batch_size, number_of_classes],
          sparse_values=1.0,
          default_value=0.0,
          validate_indices=False)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      loss_dict['topic'] = tf.reduce_mean(losses) * model_proto.topic_loss_weight

    # # Loss for sentiment.
    # key = 'sentiment'
    # labels = predictions['labels_' + key]
    # logits = predictions['logits_' + key]

    # if model_proto.sentiment_loss_function \
    #   == baseline_model_pb2.BaselineModel.SOFTMAX:
    #   losses = tf.nn.sparse_softmax_cross_entropy_with_logits( 
    #       labels=labels, logits=logits)
    # else:
    #   labels = tf.sparse_to_dense(
    #       tf.stack([tf.range(batch, dtype=tf.int64), labels], axis=1),
    #       output_shape=[batch, number_of_classes],
    #       sparse_values=1.0,
    #       default_value=0.0,
    #       validate_indices=False)
    #   losses = tf.nn.sigmoid_cross_entropy_with_logits( 
    #       labels=labels, logits=logits)

    # loss_dict[key] = tf.reduce_mean(losses) * model_proto.sentiment_loss_weight
    return loss_dict
