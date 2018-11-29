
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
from protos import semantic_model_pb2

slim = tf.contrib.slim


class Model(model.Model):
  """Baseline model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an instance of semantic_model_pb2.SemanticModel

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, semantic_model_pb2.SemanticModel):
      raise ValueError('The model_proto has to be an instance of SemanticModel.')

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

  def build_embedding(self, vocab_size, model_proto):
    is_training = self._is_training

    trainable = model_proto.trainable
    init_width = model_proto.init_width
    weight_decay = model_proto.weight_decay
    embedding_size = model_proto.embedding_size

    embedding_weights = tf.get_variable(
        name='weights',
        shape=[vocab_size, embedding_size],
        trainable=trainable,
        initializer=tf.random_uniform_initializer(-init_width, init_width),
        regularizer=slim.l2_regularizer(weight_decay))
    return embedding_weights

  def embed_feature(self, frame_features, model_proto):
    is_training = self._is_training

    assert isinstance(model_proto, semantic_model_pb2.EmbeddingParam)

    batch, n_frames, feature_dims = frame_features.get_shape().as_list()

    with tf.variable_scope(model_proto.scope) as scope:
      weights = self.build_embedding(feature_dims, model_proto)

      frame_features = tf.reshape(frame_features, [-1, feature_dims])

      #feature_weights = tf.get_variable(
      #    name='feature_weights',
      #    shape=[feature_dims],
      #    trainable=True,
      #    initializer=tf.random_uniform_initializer(-0.08, 0.08),
      #    regularizer=slim.l2_regularizer(0.0))
      #feature_weights = tf.nn.sigmoid(feature_weights)
      #frame_features = slim.batch_norm(frame_features, 
      #    center=False, scale=True, is_training=is_training)
      frame_features = frame_features #* feature_weights
      embeddings = tf.matmul(frame_features, weights)

    embeddings = tf.reshape(embeddings, [batch, -1, model_proto.embedding_size])
    return embeddings, weights

  def build_inference(self, examples):
    """Builds tensorflow graph for inference.

    Args:
      examples: a dict mapping from name to input tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    is_training = self._is_training
    model_proto = self._model_proto

    # Embedding sentiment.
    with tf.variable_scope(model_proto.sent_emb_params.scope) as scope:
      sentiment_weights = self.build_embedding(
          model_proto.number_of_sentiments,
          model_proto.sent_emb_params)

      with open(model_proto.sent_emb_params.init_emb_matrix_path, 'rb') as fp:
        word2vec = np.load(fp)
        init_assign_op, init_feed_dict = slim.assign_from_values({
            sentiment_weights.op.name: word2vec} )

      def _init_fn_sent(sess):
        sess.run(init_assign_op, init_feed_dict)
        logging.info('Initialize coco embedding from %s.',
            model_proto.sent_emb_params.init_emb_matrix_path)

      self._init_fn_list.append(_init_fn_sent)

    # Manipulate input tensors.
    features_list = []
    if model_proto.use_frame_features:
      features_list.append(examples['frame_features'])

    # COCO
    if model_proto.use_common_object:
      embedding, weights = self.embed_feature(
          examples['common_object_features'], 
          model_proto.coco_emb_params)
      features_list.append(embedding)

      with open(model_proto.coco_emb_params.init_emb_matrix_path, 'rb') as fp:
        word2vec = np.load(fp)
        init_assign_op, init_feed_dict = slim.assign_from_values({
            weights.op.name: word2vec} )

      def _init_fn_coco(sess):
        sess.run(init_assign_op, init_feed_dict)
        logging.info('Initialize coco embedding from %s.',
            model_proto.coco_emb_params.init_emb_matrix_path)

      self._init_fn_list.append(_init_fn_coco)

    # PLACE
    if model_proto.use_place:
      embedding, weights = self.embed_feature(
          examples['place_features'], 
          model_proto.place_emb_params)
      features_list.append(embedding)

      with open(model_proto.place_emb_params.init_emb_matrix_path, 'rb') as fp:
        word2vec = np.load(fp)
        init_assign_op, init_feed_dict = slim.assign_from_values({
            weights.op.name: word2vec} )

      def _init_fn_place(sess):
        sess.run(init_assign_op, init_feed_dict)
        logging.info('Initialize place embedding from %s.',
            model_proto.place_emb_params.init_emb_matrix_path)

      self._init_fn_list.append(_init_fn_place)

    # EMOTIC
    if model_proto.use_emotic:
      embedding, weights = self.embed_feature(
          examples['emotic_features'], 
          model_proto.emotic_emb_params)
      features_list.append(embedding)

      with open(model_proto.emotic_emb_params.init_emb_matrix_path, 'rb') as fp:
        word2vec = np.load(fp)
        init_assign_op, init_feed_dict = slim.assign_from_values({
            weights.op.name: word2vec} )

      def _init_fn_emotic(sess):
        sess.run(init_assign_op, init_feed_dict)
        logging.info('Initialize emotic embedding from %s.',
            model_proto.emotic_emb_params.init_emb_matrix_path)

      self._init_fn_list.append(_init_fn_emotic)

    # Other features.
    if model_proto.use_shot_boundary:
      features_list.append(examples['shot_boundary_features'])
    if model_proto.use_optical_flow:
      features_list.append(examples['optical_flow_features'])
    if model_proto.use_audio:
      features_list.append(examples['audio_features'])

    frame_features= tf.concat(features_list, axis=2)

    n_frames=examples['n_frames']
    batch = n_frames.get_shape()[0]

    # RNN encode.
    init_width = model_proto.lstm_init_width
    initializer = tf.random_uniform_initializer(-init_width, init_width)

    encode_methods = {
      semantic_model_pb2.SemanticModel.LSTM: self._encode_lstm,
      semantic_model_pb2.SemanticModel.BOF: self._encode_bof,
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

    if model_proto.sentiment_loss_function == semantic_model_pb2.SemanticModel.SIGMOID_RAW:
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
    #   == semantic_model_pb2.SemanticModel.SOFTMAX:
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
