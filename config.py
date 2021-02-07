import os

DATA_PATH = "./data"
TRAIN_DATA = os.path.join(DATA_PATH, "train")
VALIDATION_DATA = os.path.join(DATA_PATH, 'val')
TEST_DATA = os.path.join(DATA_PATH, 'test')

DEVICE = "/GPU:0"
BATCH_SIZE = 32
EPOCHS = 30
IMG_H = 200
IMG_W = 200
