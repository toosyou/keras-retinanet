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
import cProfile

import unet_3d
from model import get_model, get_unet_model
import datetime
from resnet3d import Resnet3DBuilder
from keras.optimizers import Adam

import multiprocessing as mp
import ctypes
from functools import partial
from contextlib import closing
from multiprocessing import Pool

# turn off futurn warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('../')
sys.path.append('../..')
sys.path.append('/home/toosyou/projects/LungTumor')
import data_util
from preprocessing import scan_index_split
from generate_batch import get_scan, get_patches, extract_patch
from keras_retinanet.callbacks import RedirectModel

LIDC_IDRI_NP_PREFIX = '/mnt/ext/lidc_idri_np'
LIDC_IDRI_BATCHES_PREFIX = '/mnt/ext3/lidc_idri_batches'

def batch_generatorV3(set, batch_size, negative_ratio=0.9, n_batch_per_scan=10, n_scan_bundle=5):
    # load positive patches
    positive_patches = np.load(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'positive.npy'), mmap_mode='r')
    # load all detections from fast_detection_model
    all_detections = pickle.load(open(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'all_detections.pl'), 'rb'))
    indexes = scan_index_split(1018)[{'train': 0, 'valid': 1, 'test': 2}[set]]

    while True:
        # load random scan
        negative_Xs, negative_ys = list(), list()
        for i in range(n_scan_bundle):
            index_scan = np.random.choice(indexes)
            volume, lung_mask, nodule_mask, layer_probability = get_scan(index_scan)
            if volume is None:
                continue

            for index_detection in np.random.randint(len(all_detections[index_scan]), size=int(n_batch_per_scan*batch_size*negative_ratio)):
                d = all_detections[index_scan][index_detection] # [x0, y0, x1, y1, z, score]
                x, y, z = int((d[0]+d[2])/2), int((d[1]+d[3])/2), int(d[4])
                patch, label = extract_patch(volume, nodule_mask, x, y, z)
                if patch is None:
                    continue

                # normalize
                negative_Xs.append(patch)
                negative_ys.append([label, 1-label])

        negative_Xs, negative_ys = np.array(negative_Xs), np.array(negative_ys)

        for i in range(n_batch_per_scan*n_scan_bundle):
            # randomly choose positive patches
            positive_X = positive_patches[np.random.randint(positive_patches.shape[0], size=int(batch_size*(1.-negative_ratio))), ...]
            positive_y = np.array([[1, 0]]*positive_X.shape[0])

            negative_indexes = np.random.randint(negative_Xs.shape[0], size=int(batch_size*negative_ratio))
            negative_X = negative_Xs[negative_indexes, ...]
            negative_y = negative_ys[negative_indexes, ...]

            # generate batch
            X = np.append(negative_X, positive_X, axis=0)
            y = np.append(negative_y, positive_y, axis=0)
            X = (X - 418.) / 414. # normalize
            yield shuffle(X, y)

def batch_generatorV2(set, batch_size, negative_ratio=0.9, n_batch_per_scan=10, n_scan_bundle=5):
    # load positive patches
    positive_patches = np.load(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'positive.npy'))
    indexes = scan_index_split(1018)[{'train': 0, 'valid': 1, 'test': 2}[set]]

    while True:
        # load random scan
        negative_Xs, negative_ys = np.ndarray((0, 64, 64, 16, 1), dtype=np.float), np.ndarray((0, 2), dtype=np.float)

        for i in range(n_scan_bundle):
            volume, lung_mask, nodule_mask, layer_probability = get_scan(np.random.choice(indexes))
            if volume is None:
                continue

            tmp_Xs, tmp_ys = get_patches(volume=volume,
                                            size=int(batch_size*negative_ratio*n_batch_per_scan),
                                            is_positive=False,
                                            lung_mask=lung_mask,
                                            nodule_mask=nodule_mask,
                                            layer_probability=layer_probability,
                                            patch_size=(64, 64, 16))
            negative_Xs = np.append(tmp_Xs, negative_Xs, axis=0)
            negative_ys = np.append(tmp_ys, negative_ys, axis=0)

        for i in range(n_batch_per_scan*n_scan_bundle):
            # randomly choose positive patches
            positive_X = positive_patches[np.random.randint(positive_patches.shape[0], size=int(batch_size*(1.-negative_ratio))), ...]
            positive_y = np.array([[1, 0]]*positive_X.shape[0])

            negative_indexes = np.random.randint(negative_Xs.shape[0], size=int(batch_size*negative_ratio))
            negative_X = negative_Xs[negative_indexes, ...]
            negative_y = negative_ys[negative_indexes, ...]

            # generate batch
            X = np.append(negative_X, positive_X, axis=0)
            y = np.append(negative_y, positive_y, axis=0)
            X = (X - 418.) / 414. # normalize
            yield shuffle(X, y)

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
    # model, training_model = get_unet_model()
    # model, training_model = get_model()
    model = Resnet3DBuilder.build_resnet_50((64, 64, 16, 1), 2)
    training_model = keras.utils.multi_gpu_model(model)
    training_model.compile(optimizer=Adam(amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
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
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.1,
        #     patience=3
        # )
    ]

    train_generator = batch_generatorV2('train', 128, n_batch_per_scan=20, negative_ratio=0.8)
    valid_generator = batch_generatorV2('valid', 128, n_batch_per_scan=20, negative_ratio=0.8)
    training_model.fit_generator(train_generator,
                        steps_per_epoch=1024,
                        epochs=100,
                        validation_data=valid_generator,
                        validation_steps=100,
                        use_multiprocessing=False,
                        # workers=4,
                        callbacks=callbacks)
