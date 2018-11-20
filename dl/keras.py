from keras import optimizers
from keras.models import Model
from keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Flatten, MaxPooling2D

from dl.categorical import CategoricalParam
from dl.double import DoubleParam
from dl.integer import IntegerParam
from dl.list import ListTunable
from dl.repeated import RepeatedTunable
from dl.tunable import Tunable

ACTIVATIONS = [
  'relu',
  'tanh',
  'sigmoid',
  'linear',
]

OPTIMIZER_MAP = {
  'adam': optimizers.Adam,
  'rmsprop': optimizers.RMSprop,
  'sgd': optimizers.SGD,
}

def activation_param(default, choices=None):
  return CategoricalParam(
    default,
    choices or ACTIVATIONS,
  )

def learning_rate_param(log_default, log_minimum, log_maximum):
  param = DoubleParam(
    default_value=log_default,
    minimum=log_minimum,
    maximum=log_maximum,
  )
  get_value = param.get_value
  param.get_value = lambda context: 10**get_value(context)
  return param

class Layer(Tunable):
  pass

class ActivationLayer(Layer):
  def __init__(self, default, choices=None):
    self.activation_param = activation_param(default, choices)

  def get_tunables(self):
    return {
      'activation': self.activation_param,
    }

  def get_value(self, context):
    return Activation(context['activation'])

class DenseLayer(Layer):
  def __init__(self, nodes_param, activation_param=None):
    self.nodes_param = nodes_param
    self.activation_param = activation_param

  def get_tunables(self):
    tunables = {
      'nodes': self.nodes_param,
    }
    if self.activation_param:
      tunables['activation'] = self.activation_param
    return tunables

  def get_value(self, context):
    if self.activation_param:
      return Dense(context['nodes'], activation=context['activation'])
    else:
      return Dense(context['nodes'])

class ConvLayer(Layer):
  def __init__(self, filters_param, kernel_size_param, activation_param=None):
    self.filters_param = filters_param
    self.kernel_size_param = kernel_size_param
    self.activation_param = activation_param
    self.first = False

  def get_tunables(self):
    tunables = {
      'filters': self.filters_param,
      'kernel_size': self.kernel_size_param,
    }
    if self.activation_param:
      tunables['activation'] = self.activation_param
    return tunables

  def get_value(self, context):
    if self.activation_param:
      return Conv2D(
        filters=context['filters'],
        kernel_size=context['kernel_size'],
        activation=context['activation'],
      )
    else:
      return Conv2D(
        filters=context['filters'],
        kernel_size=context['kernel_size'],
      )

class PoolLayer(Layer):
  pool_type_map = {
    'max': MaxPooling2D,
    'average': AveragePooling2D,
  }

  def __init__(self, pool_size_param=None, pool_type_param=None):
    self.pool_size_param = pool_size_param or IntegerParam(2, 1, 4)
    self.pool_type_param = pool_type_param or CategoricalParam('max', ['max', 'average'])

  def get_tunables(self):
    return {
      'pool_size': self.pool_size_param,
      'pool_type': self.pool_type_param,
    }

  def get_value(self, context):
    pool_class = self.pool_type_map[context['pool_type']]
    return pool_class(
      pool_size=context['pool_size'],
    )

class FlattenLayer(Layer):
  def get_tunables(self):
    return {}

  def get_value(self, context):
    return Flatten()

class ChainLayer(Layer):
  def __init__(self, list_tunable):
    self.layers = list_tunable
    if isinstance(self.layers, list):
      self.layers = ListTunable(list_tunable)
    assert isinstance(self.layers, Tunable)

  def get_tunables(self):
    return {
      'chain': self.layers,
    }

  def get_value(self, context):
    layers = context['chain']

    def apply_layers(input_tensor):
      for layer in layers:
        input_tensor = layer(input_tensor)
      return input_tensor

    return apply_layers

class Compiler(Tunable):
  def __init__(self, inputs, outputs, learning_rate_param, optimizer_param=None, loss='categorical_crossentropy'):
    super().__init__()
    self.inputs = inputs
    self.outputs = outputs
    self.learning_rate_param = learning_rate_param
    self.optimizer_param = optimizer_param or CategoricalParam('adam', list(OPTIMIZER_MAP))
    self.loss = loss

  def get_tunables(self):
    return {
      'outputs': self.outputs,
      'learning_rate': self.learning_rate_param,
      'optimizer': self.optimizer_param,
    }

  def get_value(self, context):
    outputs = context['outputs']
    model = Model(inputs=self.inputs, outputs=outputs(self.inputs))
    model.compile(
      OPTIMIZER_MAP[context['optimizer']](
        lr=context['learning_rate'],
      ),
      loss=self.loss,
      metrics=['accuracy'],
    )
    return model
