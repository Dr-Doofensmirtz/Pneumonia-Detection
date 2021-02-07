from config import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


with tf.device(DEVICE):
    MODEL = Sequential([
                        Conv2D(16, (3,3), activation = 'relu', input_shape = (IMG_H,IMG_W,3),padding='same'),
                        Conv2D(16, (3,3), activation = 'relu',padding='same'),
                        MaxPooling2D((2,2)),
                                     
                        Conv2D(16, (3,3), activation = 'relu',padding='same'),
                        Conv2D(16, (3,3), activation = 'relu',padding='same'),
                        MaxPooling2D((2,2)),
                                     
                        Conv2D(32, (3,3), activation = 'relu',padding='same'),
                        Conv2D(32, (3,3), activation = 'relu',padding='same'),
                        MaxPooling2D((2,2)),

                        Conv2D(64, (3,3), activation = 'relu',padding='same'),
                        Conv2D(64, (3,3), activation = 'relu',padding='same'),
                        MaxPooling2D((2,2)),

                        Flatten(),
                        Dense(512, activation='relu'),
                        Dense(1, activation='sigmoid')
])