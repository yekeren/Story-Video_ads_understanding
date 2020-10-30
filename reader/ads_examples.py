
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from protos import ads_examples_pb2

def _handle_frame_features(
    keys_to_tensors, feature_name, feature_dims):
  """Decode frame features.

  Args:
    keys_to_tensors: a dict mapping from tensor names to tensors.  
    feature_name: the name in the keys_to_tensors dict.
    feature_dims: the dimension of the feature

  Returns:
    frame_features: a [batch, max_frame_len, feature_dims] tf.float32 tensor.
  """
  n_frames = tf.cast(keys_to_tensors['video/n_frames'], tf.int32)
  raw_features = tf.sparse_tensor_to_dense(keys_to_tensors[feature_name])

  return tf.reshape(raw_features, [-1, feature_dims])


def _create_tfrecord_dataset(config):
  """Create tfrecord dataset for DatasetDataProvider.

  Args:
    config: an instance of AdsExample proto.

  Returns:
    dataset: a slim.data.dataset.Dataset instance.
  """
  def _handle_frame_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/features', config.feature_dims)

  def _handle_climax_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 'video/climax_features', 1)

  def _handle_climax_predictions_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 'video/climax_predictions', 1)

  def _handle_common_object_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/common_object_features', config.common_object_feature_dims)

  def _handle_place_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/place_features', config.place_feature_dims)

  def _handle_emotic_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/emotic_features', config.emotic_feature_dims)
    
  def _handle_affectnet_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/affectnet_features', config.affectnet_feature_dims)
    
  def _handle_shot_boundary_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/shot_boundary_features', config.shot_boundary_feature_dims)

  def _handle_optical_flow_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/optical_flow_features', config.optical_flow_feature_dims)

  def _handle_audio_features_wrapper(keys_to_tensors):
    return _handle_frame_features(keys_to_tensors, 
        'video/audio_features', config.audio_feature_dims)

  item_handler_frame_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/features'],
      func=_handle_frame_features_wrapper)

  item_handler_climax_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/climax_features'],
      func=_handle_climax_features_wrapper)

  item_handler_climax_predictions = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/climax_predictions'],
      func=_handle_climax_predictions_wrapper)

  item_handler_common_object_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/common_object_features'],
      func=_handle_common_object_features_wrapper)

  item_handler_place_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/place_features'],
      func=_handle_place_features_wrapper)

  item_handler_emotic_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/emotic_features'],
      func=_handle_emotic_features_wrapper)

  item_handler_affectnet_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/affectnet_features'],
      func=_handle_affectnet_features_wrapper)

  item_handler_shot_boundary_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/shot_boundary_features'],
      func=_handle_shot_boundary_features_wrapper)

  item_handler_optical_flow_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/optical_flow_features'],
      func=_handle_optical_flow_features_wrapper)

  item_handler_audio_features = tfexample_decoder.ItemHandlerCallback(
      keys=['video/n_frames', 'video/audio_features'],
      func=_handle_audio_features_wrapper)

  keys_to_features = {
    'video/source_id': tf.FixedLenFeature(
        shape=(), dtype=tf.string, default_value=''),
    'video/n_frames': tf.FixedLenFeature((), tf.int64, default_value=0),
    'video/features': tf.VarLenFeature(tf.float32),
    'video/climax_features': tf.VarLenFeature(tf.float32),
    'video/climax_predictions': tf.VarLenFeature(tf.float32),
    'video/common_object_features': tf.VarLenFeature(tf.float32),
    'video/place_features': tf.VarLenFeature(tf.float32),
    'video/emotic_features': tf.VarLenFeature(tf.float32),
    'video/affectnet_features': tf.VarLenFeature(tf.float32),
    'video/shot_boundary_features': tf.VarLenFeature(tf.float32),
    'video/optical_flow_features': tf.VarLenFeature(tf.float32),
    'video/audio_features': tf.VarLenFeature(tf.float32),
    'anno/topic': tf.FixedLenFeature((), tf.int64),
    'anno/sentiment': tf.FixedLenFeature((), tf.int64),
    'anno/sentiment_list': tf.FixedLenFeature([config.sentiment_num_classes], tf.float32),
  }

  items_to_handlers = {
    'video_id': tfexample_decoder.Tensor('video/source_id'),
    'n_frames': tfexample_decoder.Tensor('video/n_frames'),
    'topic': tfexample_decoder.Tensor('anno/topic'),
    'sentiment': tfexample_decoder.Tensor('anno/sentiment'),
    'frame_features': item_handler_frame_features,
    'climax_features': item_handler_climax_features,
    'climax_predictions': item_handler_climax_predictions,
    'common_object_features': item_handler_common_object_features,
    'place_features': item_handler_place_features,
    'emotic_features': item_handler_emotic_features,
    'affectnet_features': item_handler_affectnet_features,
    'shot_boundary_features': item_handler_shot_boundary_features,
    'optical_flow_features': item_handler_optical_flow_features,
    'audio_features': item_handler_audio_features,
    'sentiment_list': tfexample_decoder.Tensor('anno/sentiment_list'),
  }
  #if config.use_sent_list:
  #  keys_to_features['anno/sentiment_list'] = tf.FixedLenFeature([config.sentiment_num_classes], tf.float32),
  #  items_to_handlers['sentiment_list'] = tfexample_decoder.Tensor('anno/sentiment_list')

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  input_paths = [config.input_path[i] for i in xrange(len(config.input_path))]
  return dataset.Dataset(
      data_sources=input_paths,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config.num_examples,
      items_to_descriptions=None)


def get_examples(config):
  """Get batched tensor of training data.

  Args:
    config: an instance of ads_examples_pb2.AdsExamples.

  Returns:
    tensor_dict: a dictionary mapping data names to tensors.
      'video_id':         tf.string,  [batch]
      'n_frames':         tf.int64,   [batch]
      'sentiment':        tf.int64,   [batch]
      'sentiment_list':   tf.int64,   [batch, 30]
      'frame_features':   tf.float32, [batch, max_frame_len, feature_dims]

  Raises:
    ValueError: if config is not an instance of ads_examples_pb2.AdsExamples.
  """
  if not isinstance(config, ads_examples_pb2.AdsExamples):
    raise ValueError('config has to be an instance of AdsExamples.')

  num_epochs = None
  if config.HasField('num_epochs'):
    num_epochs = config.num_epochs

  dataset = _create_tfrecord_dataset(config)
  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=True,
      num_epochs=num_epochs,
      num_readers=config.data_provider_num_readers,
      common_queue_capacity=config.data_provider_common_queue_capacity,
      common_queue_min=config.data_provider_common_queue_min)

  # Resize image, seperate question and answer.
  items = filter(
      lambda x: x != 'record_key', provider.list_items())

  tensor_dict = {}
  data = provider.get(items)
  for i, item in enumerate(items):
    tensor_dict[item] = data[i]

  # Batch.
  tensor_dict = tf.train.batch(tensor_dict,
      config.batch_size, 
      capacity=config.batch_op_capacity,
      num_threads=config.batch_op_num_threads, 
      enqueue_many=False,
      dynamic_pad=True)

  return tensor_dict
