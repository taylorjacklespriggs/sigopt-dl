import json
import keras
from keras.datasets import mnist
from keras.layers import Input

from dl.connection import Connection
from dl.constant import Constant
from dl.keras import activation_param, ChainLayer, Compiler, ConvLayer, DenseLayer, FlattenLayer, learning_rate_param, PoolLayer
from dl.integer import IntegerParam
from dl.list import ListTunable
from dl.repeated import RepeatedTunable

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

if __name__ == '__main__':

  conv_block = ChainLayer([
    ConvLayer(
      filters_param=IntegerParam(64, 8, 128),
      kernel_size_param=IntegerParam(3, 1, 5),
      activation_param=activation_param('relu'),
    ),
    PoolLayer(),
  ])

  convolutions = ChainLayer(
    RepeatedTunable(
      tunable=conv_block,
      count_param=IntegerParam(2, 1, 3),
    )
  )

  fully_connected = ChainLayer(
    RepeatedTunable(
      tunable=DenseLayer(nodes_param=IntegerParam(256, 10, 784), activation_param=activation_param('tanh')),
      count_param=IntegerParam(2, 1, 4),
    )
  )

  model = ChainLayer([
    convolutions,
    FlattenLayer(),
    fully_connected,
    DenseLayer(nodes_param=Constant(10), activation_param=Constant('softmax')),
  ])

  compiled_model = Compiler(
    inputs=Input(shape=(28, 28, 1)),
    outputs=model,
    learning_rate_param=learning_rate_param(log_default=-3, log_minimum=-6, log_maximum=0),
  )

  api_token = '***'
  dev_token = '***'
  connection = Connection(api_token)
  experiment = connection.create_experiment(
    name='mlp',
    budget=100,
    tunables={
      'compiled_model': compiled_model,
      'epochs': IntegerParam(4, 1, 12),
      'batch_size': IntegerParam(100, 10, 1000),
    },
  )
  def evaluate(components):
    model = components['compiled_model']
    epochs = components['epochs']
    batch_size = components['batch_size']
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model.evaluate(x_test, y_test)[1]
  experiment.loop(evaluate)
