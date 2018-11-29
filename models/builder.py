
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from models import baseline_model
from models import semantic_model
from models import climax_model


def build(config, is_training=False):
  """Builds a Model based on the config.

  Args:
    config: a model_pb2.Model instance.
    is_training: True if this model is being built for training.

  Returns:
    a Model instance.

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, model_pb2.Model):
    raise ValueError('The config has to be an instance of model_pb2.Model.')

  model = config.WhichOneof('model')

  if 'baseline_model' == model:
    return baseline_model.Model(config.baseline_model, is_training)

  if 'semantic_model' == model:
    return semantic_model.Model(config.semantic_model, is_training)

  if 'climax_model' == model:
    return climax_model.Model(config.climax_model, is_training)

  raise ValueError('Unknown model: {}'.format(model))
