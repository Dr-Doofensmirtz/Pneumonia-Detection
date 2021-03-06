from config import *
from model import MODEL
from checkpoint import get_checkpoints
from datasets import get_datasets

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img


train_data, validation_data, test_data = get_datasets()

classification_mod = MODEL

classification_mod.compile(optimizer= tf.keras.optimizers.Adam(lr = 1e-5),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

check_p = get_checkpoints()

with tf.device(DEVICE):
    classification_mod.fit(train_data,
                            epochs = EPOCHS,
                            validation_data = validation_data,
                            callbacks = check_p)

