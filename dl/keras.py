from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Flatten, MaxPooling2D

from dl.categorical import CategoricalParam
from dl.double import DoubleParam
from dl.integer import IntegerParam
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
  param.get_value = lambda: 10**get_value()
  return param

class Layer(Tunable):
  def add_to_model(self, model):
    print(self.value)
    model.add(self.value)

  def apply(self, input_tensor):
    return self.value(input_tensor)

class ActivationLayer(Layer):
  def __init__(self, default, choices=None):
    self.activation_param = activation_param(default, choices)

  def get_tunables(self):
    return {
      'activation': self.activation_param,
    }

  def get_value(self):
    return Activation(self.activation_param.value)

  def add_to_model(self, model):
    model.add(self.value)

  def apply(self, input_tensor):
    return self.value(input_tensor)

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

  def get_value(self):
    if self.activation_param:
      return Dense(self.nodes_param.value, activation=self.activation_param.value)
    else:
      return Dense(self.nodes_param.value)

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

  def get_value(self):
    if self.activation_param:
      return Conv2D(
        filters=self.filters_param.value,
        kernel_size=self.kernel_size_param.value,
        activation=self.activation_param.value,
      )
    else:
      return Conv2D(
        filters=self.filters_param.value,
        kernel_size=self.kernel_size_param.value,
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

  def get_value(self):
    pool_class = self.pool_type_map[self.pool_type_param.value]
    return pool_class(
      pool_size=self.pool_size_param.value,
    )

class RepeatedLayer(Layer):
  def __init__(self, layer, count_param):
    self.layer = layer
    self.count_param = count_param

  def get_tunables(self):
    return {
      'repeated': self.layer,
      'count': self.count_param,
    }

  def add_to_model(self, model):
    for _ in range(self.count_param.value):
      self.layer.add_to_model(model)

  def apply(self, input_tensor):
    for _ in range(self.count_param.value):
      input_tensor = self.layer.apply(input_tensor)
    return input_tensor

class FlattenLayer(Layer):
  def get_tunables(self):
    return {}

  def get_value(self):
    return Flatten()

class SequentialLayer(Tunable):
  def __init__(self, layers):
    self.layers = layers

  def get_tunables(self):
    return {
      f'layer{i}': self.layers[i]
      for i, layer in enumerate(self.layers)
    }

  def get_value(self):
    return [
      layer.value
      for layer in self.layers
    ]

  def add_to_model(self, model):
    for layer in self.layers:
      layer.add_to_model(model)

  def apply(self, input_tensor):
    for layer in self.layers:
      input_tensor = layer.apply(input_tensor)
    return input_tensor

class SequentialModel(Tunable):
  def __init__(self, inputs, layers):
    self.inputs = inputs
    self.layers = layers

  def get_tunables(self):
    return {
      f'layer{i}': self.layers[i]
      for i, layer in enumerate(self.layers)
    }

  def get_value(self):
    #model = Sequential()
    #for layer in self.layers:
    #  layer.add_to_model(model)
    #return model
    outputs = self.inputs
    for layer in self.layers:
      outputs = layer.apply(outputs)
    model = Model(inputs=self.inputs, outputs=outputs)
    return model

class Compiler(Tunable):
  def __init__(self, model, learning_rate_param, optimizer_param=None, loss='categorical_crossentropy'):
    super().__init__()
    self.model = model
    self.learning_rate_param = learning_rate_param
    self.optimizer_param = optimizer_param or CategoricalParam('adam', list(OPTIMIZER_MAP))
    self.loss = loss

  def get_tunables(self):
    return {
      'model': self.model,
      'learning_rate': self.learning_rate_param,
      'optimizer': self.optimizer_param,
    }

  def get_value(self):
    self.model.value.compile(
      OPTIMIZER_MAP[self.optimizer_param.value](
        lr=self.learning_rate_param.value,
      ),
      loss=self.loss,
      metrics=['accuracy'],
    )
    return self.model.value
