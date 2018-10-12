import sys
import os
import time
import pylidc as pl
import numpy as np
from sklearn.utils import shuffle
import better_exceptions
import pickle
from numba import jit
from sklearn.feature_extraction import image
import keras
import time
import random
from tqdm import tqdm
import pickle

import unet_3d
from model import get_model, get_unet_model
import datetime

# turn off futurn warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('../')
sys.path.append('../..')
sys.path.append('/home/toosyou/projects/LungTumor')
import data_util
from preprocessing import scan_index_split
from keras_retinanet.callbacks import RedirectModel

LIDC_IDRI_BATCHES_PREFIX = '/mnt/ext/lidc_idri_batches'

def batch_generator(set, batch_size):
    X_path = os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'X')
    y_path = os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'y')

    all_files = os.listdir(X_path)

    while True:
        filename = np.random.choice(all_files)
        X_fileanme = os.path.join(X_path, filename)
        y_filename = os.path.join(y_path, filename)

        X = np.load(X_fileanme)
        y = np.load(y_filename)

        for i in range(X.shape[0] // batch_size):
            indexes = np.random.randint(X.shape[0], size=batch_size)
            yield X[indexes], y[indexes]

        del X
        del y

if __name__ == '__main__':
    model, training_model = get_unet_model()
    model.summary()

    callbacks=[
        RedirectModel(keras.callbacks.ModelCheckpoint(
            os.path.join(
                './model_checkpoints',
                '{epoch:02d}.h5'
            ),
            verbose=1,
        ), model),
        keras.callbacks.TensorBoard(
            log_dir='./logs/' + datetime.datetime.now().strftime('%Y%m%d%H%M')
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        )
    ]

    train_generator = batch_generator('train', 64)
    valid_generator = batch_generator('valid', 64)
    training_model.fit_generator(train_generator,
                        steps_per_epoch=1024,
                        epochs=100,
                        validation_data=valid_generator,
                        validation_steps=10,
                        use_multiprocessing=False,
                        # workers=2,
                        callbacks=callbacks)
