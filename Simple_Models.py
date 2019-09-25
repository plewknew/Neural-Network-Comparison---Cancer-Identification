import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, add, InputLayer
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
from google.colab import files
import zipfile

def Simple_Shallow_Dense(num_classes, input_shape_input,regweight,final_activation):

  mlp_basemodel = Sequential([
      Dense(hidden_size, input_shape=input_shape_input, activation='relu',kernel_regularizer=regularizers.l2(regweight)),
      Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight)),
      Dense(num_classes, activation=final_activation)])
  
  mlp_basemodel.compile(optimizer, "categorical_crossentropy", metrics=['accuracy'])
  
  return mlp_basemodel



def Simple_CNN(num_classes, input_shape_input, kernel_size_input, final_activation):

    cnn = Sequential()

    cnn.add(Conv2D(32, input_shape=input_shape_input, data_format="channels_last", kernel_size=kernel_size_input))
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(32, (3, 3)))
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(num_classes, activation=final_activation))

    cnn.compile("adam", "binary_crossentropy", metrics=['accuracy'])

    return cnn