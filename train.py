'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import argparse

import mlflow
import mlflow.keras
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env

import cloudpickle
import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


batch_size = 256
epochs = 4
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets

# TODO : use mnist.load_data to create x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# TODO : reshape your data according to keras setting and define an
# input_shape argument you will feed the model
if K.image_data_format() == 'channels_first':
    x_train = 
    x_test = 
    input_shape = 
else:
    x_train = 
    x_test = 
    input_shape = 

# TODO : convert X_train and X_test to float and normalise them

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# TODO : convert class vectors to binary class matrices using
# keras.utils.to_categorical

# TODO : Define a sequential model with layers :
# Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)
# MaxPooling2D(pool_size=(2, 2)
# Conv2D(64, kernel_size=(3, 3), activation='relu')
# MaxPooling2D(pool_size=(2, 2))
# Dropout(0.25)
# Flatten()
# Dense(128, activation='relu')
# Dense(num_classes, activation='softmax')


# TODO : Compile model with categorical cross entropy loss, optimizer Adam and 
# accuracy metric
model.compile()

# TODO : fit the model
model.fit(, ,
          batch_size=,
          epochs=,
          verbose=1,
          validation_data=(, ))

# TODO : store the result of model. evaluate in a variable score
score = 

# TODO : log the loss and score as metric to mlflow

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# TODO : log the model to mlflow
