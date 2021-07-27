import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam

class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class CNN_Model(object):
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 1e-3
        # Building model
        self._build_model()
        self.model.compile(optimizer= Adam(learning_rate=self.learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

    def _build_model(self):
        # CNN model
        self.model =  tf.keras.Sequential([
        layers.Conv2D(64, (7,7), use_bias=False, input_shape=(32,32,1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(3,3),padding='same'),
        layers.Conv2D(128, (3,3), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(36, activation='softmax')
])

    