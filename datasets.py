import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *

# image data generators
def get_generators():
    train_img_generator = ImageDataGenerator(rescale= 1.0/255,
                                            shear_range=0.2,
                                            zoom_range=0.2,)

    val_img_generator = ImageDataGenerator(rescale = 1.0/255)

    test_img_generator = ImageDataGenerator(rescale = 1.0/255)

    return train_img_generator, val_img_generator, test_img_generator

# datasets
def get_datasets():
    tr, va , te = get_generators()

    train_data_gen = tr.flow_from_directory(batch_size = BATCH_SIZE,
                                                            directory = TRAIN_DATA,
                                                            shuffle=True,
                                                            target_size = (IMG_H,IMG_W),
                                                            class_mode = 'binary')

    val_data_gen = va.flow_from_directory(batch_size = BATCH_SIZE,
                                                        directory = VALIDATION_DATA,
                                                        shuffle = True,
                                                        target_size = (IMG_H,IMG_W),
                                                        class_mode = 'binary')
                                                        
    test_data_gen = te.flow_from_directory(batch_size = BATCH_SIZE,
                                                        directory = TEST_DATA,
                                                        shuffle = False,
                                                        target_size = (IMG_H,IMG_W),
                                                        class_mode = 'binary')

    return train_data_gen, val_data_gen, test_data_gen