# Importing all the libraries needed
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import pandas as pd
import os, requests, cv2, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# loading the pictures of tomatoes
train_data_dir = '/home/aadi/Desktop/tomato/archivekaggle/tomato/train'
test_data_dir = '/home/aadi/Desktop/tomato/archivekaggle/tomato/val'  # this folder will be used for evaluating model's performance

# for this challenge we are using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)  # specifying the validation split inside the function

test_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='validation')

test_gen = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # shuffle will not affect the accuracy of the model but will affect the computation of some metrics that depend on the order of the samples

# CNN model layers
cnn = models.Sequential()

cnn.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[224, 224, 3])),
cnn.add(layers.MaxPooling2D(pool_size=(2, 2))),

cnn.add(layers.Conv2D(64, (3, 3), activation='relu')),
cnn.add(layers.MaxPooling2D((2, 2))),

cnn.add(layers.Conv2D(64, (3, 3), activation='relu')),
cnn.add(layers.MaxPooling2D((2, 2))),

cnn.add(layers.Conv2D(64, (3, 3), activation='relu')),
cnn.add(layers.MaxPooling2D((2, 2))),

cnn.add(layers.Conv2D(64, (3, 3), activation='relu')),
cnn.add(layers.MaxPooling2D((2, 2))),

cnn.add(layers.Conv2D(64, (3, 3), activation='relu')),
cnn.add(layers.MaxPooling2D((2, 2))),

cnn.add(layers.Flatten()),

cnn.add(layers.Dense(64, activation='relu'))
# output layer
cnn.add(layers.Dense(10, activation='softmax'))

cnn.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)

cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True)

history = cnn.fit(x=train_gen,
                  callbacks=[es],
                  steps_per_epoch=7000/32,
                  epochs=100,
                  validation_steps=3000/32,
                  validation_data=val_gen)

scores = cnn.evaluate(test_gen)
