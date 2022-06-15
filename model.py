import tensorflow as tf
import keras
import numpy as np
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
# from keras.optimizers import Adam
from sklearn.metrics import f1_score 
class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class CNN_Model(object):
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 1e-3
        # Building model
        self._build_model()
        # self.model.compile(optimizer= Adam(learning_rate=self.learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[self.custom_f1score])
    def f1score(y, y_pred):
        return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')
    def custom_f1score(self,y, y_pred):
        return tf.py_function(self.f1score, (y, y_pred), tf.double)
    def _build_model(self):
        # CNN model
        
        model = Sequential()
        model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(36, activation='softmax'))
        self.model = model



    