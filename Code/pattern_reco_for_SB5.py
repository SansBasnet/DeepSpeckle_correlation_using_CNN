import os
import glob
import csv
import sys
import keras
import tensorflow as tf
from skimage import io
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline


// Data preprocessing one-hot-encoding


def unpickle(file):
  import pickle
  with open (file, 'rb') as fo:
    speckle_dict = pickle.load(fo, encoding='bytes')
  return speckle_dict

dirs = ['batches.meta','data_batch1','data_batch2','data_batch3']
all_data = [0,1,2,3,4]


speckle = csv.reader(open('SB5.csv'))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_train, axis=1)

x_train, x_test = x_train/255.0, x_test/255.0


%--------------------------------
% Beginning of the Convolution - first attempt for evaluation
%--------------------------------

def generate_model():
  model = tf.keras.Sequential([
    #first convolutional layer
    tf.keras.layers.Conv2D(32, filter_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    #second convolutional layer   
    tf.keras.layers.Conv2D(64, filter_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ), 

    #fullt connected classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation = 'softmax') # 10 outputs         
  ])

  return model

%-----------------------------------
% Model evaluation 
%-----------------------------------


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)
print(predictions)

%%%%%%%%%%%%

import numpy as np
print(np.argmax(predictions[8]))

import matplotlib.pyplot as plt
plt.imshow(x_test[8])
plt.show()

import matplotlib.pyplot as plt
print(x_train[0])

plt.imshow(x_train[0])
plt.show()

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
