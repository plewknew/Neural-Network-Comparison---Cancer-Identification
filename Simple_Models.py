import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, add
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
from google.colab import files
import zipfile










def Simple_CNN(num_classes, input_shape_input, kernel_size_input)

    cnn = Sequential()

    cnn.add(Conv2D(32, input_shape=input_shape_input data_format="channels_last", kernel_size=kernel_size_input)
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(32, (3, 3)))
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(num_classes, activation='sigmoid'))

    cnn.compile("adam", "binary_crossentropy", metrics=['accuracy'])

    history_cnn = cnn.fit_generator(train_generator,
            steps_per_epoch=2000,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=800)