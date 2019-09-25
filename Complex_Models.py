import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, add, InputLayer
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization
from keras import regularizers
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
from google.colab import files
import zipfile
import warnings

def deep_CNN_without_Res(num_classes, input_shape_input, kernel_size_input, final_activation):

    warnings.filterwarnings("ignore")

    inputs = Input(shape=input_shape_input)

    conv1_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(inputs)
    conv1_1 = Activation("relu")(conv1_1)
    conv1_1 = batch2 = BatchNormalization()(conv1_1)


    conv1_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv1_1)
    conv1_2 = Activation("relu")(conv1_2)
    conv1_2 = batch2 = BatchNormalization()(conv1_2)

    conv1_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv1_2)
    conv1_3 = Activation("relu")(conv1_3)
    conv1_3 = batch2 = BatchNormalization()(conv1_3)


    #skip1 = add([conv1_1, conv1_3])
    conv1_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv1_3)
    conv1_4 = Activation("relu")(conv1_4)
    conv1_4 = batch2 = BatchNormalization()(conv1_4)

    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1_4)

    conv2_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool1)
    conv2_1 = Activation("relu")(conv2_1)
    conv2_1 = batch2 = BatchNormalization()(conv2_1)


    conv2_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv2_1)
    conv2_2 = Activation("relu")(conv2_2)
    conv2_2 = batch2 = BatchNormalization()(conv2_2)

    #skip2 = add([maxpool1, conv2_2])
    conv2_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv2_2)
    conv2_3 = Activation("relu")(conv2_3)
    conv2_3 = batch2 = BatchNormalization()(conv2_3)

    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2_3)

    conv3_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool1)
    conv3_1 = Activation("relu")(conv3_1)
    conv3_1 = batch2 = BatchNormalization()(conv3_1)


    conv3_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv3_1)
    conv3_2 = Activation("relu")(conv3_2)
    conv3_2 = batch2 = BatchNormalization()(conv3_2)

    #skip3 = add([maxpool2, conv3_2])
    conv3_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv3_2)
    conv3_3 = Activation("relu")(conv3_3)
    conv3_3 = batch2 = BatchNormalization()(conv3_3)

    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

    conv4_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool3)
    conv4_1 = Activation("relu")(conv4_1)
    conv4_1 = batch2 = BatchNormalization()(conv4_1)

    conv4_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv4_1)
    conv4_2 = Activation("relu")(conv4_2)
    conv4_2 = batch2 = BatchNormalization()(conv4_2)

    conv4_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv4_2)
    conv4_3 = Activation("relu")(conv4_3)
    conv4_3 = batch2 = BatchNormalization()(conv4_3)

    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)

    conv5_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool4)
    conv5_1 = Activation("relu")(conv5_1)
    conv5_1 = batch2 = BatchNormalization()(conv5_1)

    conv5_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv5_1)
    conv5_2 = Activation("relu")(conv5_2)
    conv5_2 = batch2 = BatchNormalization()(conv5_2)

    #skip5 = add([maxpool4, conv5_2])
    conv5_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv5_2)
    conv5_3 = Activation("relu")(conv5_3)
    conv5_3 = batch2 = BatchNormalization()(conv5_3)

    maxpool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)

    conv6_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool5)
    conv6_1 = Activation("relu")(conv6_1)
    conv6_1 = batch2 = BatchNormalization()(conv6_1)


    conv6_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv6_1)
    conv6_2 = Activation("relu")(conv6_2)
    conv6_2 = batch2 = BatchNormalization()(conv6_2)

    #skip6 = add([maxpool5, conv6_2])
    conv6_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv6_2)
    conv6_3 = Activation("relu")(conv6_3)
    conv6_3 = batch2 = BatchNormalization()(conv6_3)

    maxpool6 = MaxPooling2D(pool_size=(2, 2))(conv6_3)

    flat = Flatten()(maxpool6)
    dense = Dense(64, activation='relu')(flat)
    predictions = Dense(num_classes, activation=final_activation)(dense)

    model_CNN_nores = Model(inputs=inputs, outputs=predictions)

    model_CNN_nores.compile("adam", "binary_crossentropy", metrics=['accuracy'])

    warnings.filterwarnings("default")

    return model_CNN_nores

def deep_CNN_with_Res(num_classes, input_shape_input, kernel_size_input, final_activation):
    
    warnings.filterwarnings("ignore")

    inputs = Input(input_shape_input)

    conv1_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(inputs)
    conv1_1 = Activation("relu")(conv1_1)

    conv1_1 = BatchNormalization()(conv1_1)


    conv1_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv1_1)
    conv1_2 = Activation("relu")(conv1_2)
    conv1_2 = batch2 = BatchNormalization()(conv1_2)

    conv1_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv1_2)
    conv1_3 = Activation("relu")(conv1_3)
    conv1_3 = batch2 = BatchNormalization()(conv1_3)


    skip1 = add([conv1_1, conv1_3])
    conv1_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip1)
    conv1_4 = Activation("relu")(conv1_4)
    conv1_4 = BatchNormalization()(conv1_4)

    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1_4)


    conv2_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool1)
    conv2_1 = Activation("relu")(conv2_1)
    conv2_1 = batch2 = BatchNormalization()(conv2_1)


    conv2_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv2_1)
    conv2_2 = Activation("relu")(conv2_2)
    conv2_2 = batch2 = BatchNormalization()(conv2_2)

    conv2_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv2_2)
    conv2_3 = Activation("relu")(conv2_3)
    conv2_3 = batch2 = BatchNormalization()(conv2_3)


    skip2 = add([conv2_1, conv2_3])
    conv2_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip2)
    conv2_4 = Activation("relu")(conv2_4)
    conv2_4 = BatchNormalization()(conv2_4)



    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2_4)


    conv3_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool1)
    conv3_1 = Activation("relu")(conv3_1)
    conv3_1 = batch2 = BatchNormalization()(conv3_1)


    conv3_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv3_1)
    conv3_2 = Activation("relu")(conv3_2)
    conv3_2 = batch2 = BatchNormalization()(conv3_2)

    conv3_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv3_2)
    conv3_3 = Activation("relu")(conv3_3)
    conv3_3 = batch2 = BatchNormalization()(conv3_3)


    skip3 = add([conv3_1, conv3_3])
    conv3_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip3)
    conv3_4 = Activation("relu")(conv3_4)
    conv3_4 = BatchNormalization()(conv3_4)



    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3_4)


    conv4_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool3)
    conv4_1 = Activation("relu")(conv4_1)
    conv4_1 = batch2 = BatchNormalization()(conv4_1)


    conv4_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv4_1)
    conv4_2 = Activation("relu")(conv4_2)
    conv4_2 = batch2 = BatchNormalization()(conv4_2)

    conv4_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv4_2)
    conv4_3 = Activation("relu")(conv4_3)
    conv4_3 = batch2 = BatchNormalization()(conv4_3)


    skip4 = add([conv4_1, conv4_3])
    conv4_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip4)
    conv4_4 = Activation("relu")(conv4_4)
    conv4_4 = BatchNormalization()(conv4_4)



    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4_4)



    conv5_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool4)
    conv5_1 = Activation("relu")(conv5_1)
    conv5_1 = batch2 = BatchNormalization()(conv5_1)


    conv5_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv5_1)
    conv5_2 = Activation("relu")(conv5_2)
    conv5_2 = batch2 = BatchNormalization()(conv5_2)

    conv5_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv5_2)
    conv5_3 = Activation("relu")(conv5_3)
    conv5_3 = batch2 = BatchNormalization()(conv5_3)


    skip5 = add([conv5_1, conv5_3])
    conv5_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip5)
    conv5_4 = Activation("relu")(conv5_4)
    conv5_4 = BatchNormalization()(conv5_4)



    maxpool5 = MaxPooling2D(pool_size=(2, 2))(conv5_4)


    conv6_1 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(maxpool5)
    conv6_1 = Activation("relu")(conv6_1)
    conv6_1 = batch2 = BatchNormalization()(conv6_1)


    conv6_2 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv6_1)
    conv6_2 = Activation("relu")(conv6_2)
    conv6_2 = batch2 = BatchNormalization()(conv6_2)

    conv6_3 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(conv6_2)
    conv6_3 = Activation("relu")(conv6_3)
    conv6_3 = batch2 = BatchNormalization()(conv6_3)


    skip6 = add([conv6_1, conv6_3])
    conv6_4 = Conv2D(32, kernel_size=kernel_size_input, padding='same')(skip6)
    conv6_4 = Activation("relu")(conv6_4)
    conv6_4 = BatchNormalization()(conv6_4)



    maxpool6 = MaxPooling2D(pool_size=(2, 2))(conv6_4)


    flat = Flatten()(maxpool6)
    dense = Dense(64, activation='relu')(flat)
    predictions = Dense(num_classes, activation=final_activation)(dense)

    model_CNN_res = Model(inputs=inputs, outputs=predictions)
    model_CNN_res.compile("adam", "binary_crossentropy", metrics=['accuracy'])
    
    warnings.filterwarnings("default")

    return model_CNN_res


def dense_Res_noCNN(num_classes, input_shape_input,regweight,final_activation,hidden_size):
    
    warnings.filterwarnings("ignore")
    
    inputs = Input(shape=input_shape_input)
    dense_1 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(inputs)
    batch_dense_1 = BatchNormalization()(dense_1)

    dense_2 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(batch_dense_1)
    batch_dense_2 = BatchNormalization()(dense_2)
    act_2 = Activation('relu')(batch_dense_2)
    skip2 = add([act_2,batch_dense_2])


    dense_3 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(skip2)
    batch_dense_3 = BatchNormalization()(dense_3)

    dense_4 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(batch_dense_3)
    batch_dense_4 = BatchNormalization()(dense_4)
    act_4 = Activation('relu')(batch_dense_4)
    skip4 = add([act_4,batch_dense_4])

    dense_5 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(skip4)
    batch_dense_5 = BatchNormalization()(dense_5)

    dense_6 = Dense(hidden_size, activation='relu',kernel_regularizer=regularizers.l2(regweight))(batch_dense_5)
    batch_dense_6 = BatchNormalization()(dense_6)
    predictions = Dense(num_classes, activation=final_activation)(batch_dense_6)


    resnet_batch_model = Model(inputs=inputs, outputs=predictions)


    resnet_batch_model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

    warnings.filterwarnings("default")

    return resnet_batch_model