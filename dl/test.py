import json
import keras
#from keras.datasets import mnist as dataset
from keras.datasets import cifar10 as dataset
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from dl.connection import Connection
from dl.constant import Constant
from dl.keras import activation_param, learning_rate_param
from dl.integer import IntegerParam
from dl.list import ListTunable
from dl.repeated import repeat
from dl.tune import tune

(x_train, y_train), (x_test, y_test) = dataset.load_data()
num_classes = 10

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

@tune(
  filters=IntegerParam(64, 8, 128),
  kernel_size=IntegerParam(3, 1, 5),
  activation=activation_param('relu'),
  pool_size=IntegerParam(2, 1, 4),
)
def conv_block(inputs, filters, kernel_size, activation, pool_size):
  conv_part = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(inputs)
  return MaxPooling2D(pool_size=pool_size)(conv_part)

@tune(
  units=IntegerParam(64, 16, 1024),
  activation=activation_param('tanh'),
)
def dense_layer(inputs, units, activation):
  return Dense(units=units, activation=activation)(inputs)

def chain(funcs, inputs):
  for func in funcs:
    inputs = func(inputs)
  return inputs

@tune(
  conv_layers=repeat(conv_block, IntegerParam(2, 0, 4)),
  dense_layers=repeat(dense_layer, IntegerParam(2, 0, 4)),
)
def complete_net(inputs, conv_layers, dense_layers):
  conv_output = chain(conv_layers, inputs)

  flat = Flatten()(conv_output)

  dense_output = chain(dense_layers, flat)

  return Dense(units=10, activation='softmax')(dense_output)

@tune(
  net=complete_net,
  learning_rate=learning_rate_param(log_default=-3, log_minimum=-6, log_maximum=0),
  batch_size=IntegerParam(100, 10, 1000),
  epochs=IntegerParam(4, 1, 16),
)
def train_model(net, learning_rate, batch_size, epochs):
  inputs = Input(shape=(x_train.shape[1:]))
  outputs = net(inputs)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
  return model

def evaluate_model(model):
  return model.evaluate(x_test, y_test)[1]

if __name__ == '__main__':
  api_token = '***'
  dev_token = '***'
  connection = Connection(api_token)
  experiment = connection.create_experiment(
    name=f'alexnet:{dataset.__name__}',
    budget=100,
    tunable=train_model,
    evaluator=evaluate_model,
  )
  experiment.loop()
