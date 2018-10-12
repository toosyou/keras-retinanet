import os
import sys
import pylidc as pl
from tqdm import tqdm
import numpy as np
import pickle
import random
from sklearn.utils import shuffle
import time
import better_exceptions
from contextlib import closing
from multiprocess import Pool

# turn off futurn warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('../')
from preprocessing import scan_index_split

sys.path.append('/home/toosyou/projects/LungTumor')
import data_util

LIDC_IDRI_NP_PREFIX = '/mnt/ext3/lidc_idri_np'
LIDC_IDRI_BATCHES_PREFIX = '/mnt/ext/lidc_idri_batches'

def generate_volume_mask():
    scans = pl.query(pl.Scan).filter()
    for index_scan, scan in enumerate(tqdm(scans, total=scans.count(), desc='generate_volume_mask')):

        volume = scan.to_volume(verbose=False)
        lung_mask = data_util.lung_mask(volume, times_dilation=20, times_erosion=15, verbose=False)
        nodule_mask = get_nodule_mask(scan, volume)

        if (not lung_mask.any()) or (not nodule_mask.any()):
            print(index_scan, 'HAS SOMETHING WRONG!')
            continue

        os.makedirs(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan)), exist_ok=True)
        np.save(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'volume.npy'), volume)
        np.save(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'lung_mask.npy'), lung_mask)
        np.save(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'nodule_mask.npy'), nodule_mask)

def generate_layer_probability():
    scans = pl.query(pl.Scan).filter()
    for index_scan, scan in enumerate(tqdm(scans, total=scans.count(), desc='generate_layer_probability')):
        volume, lung_mask, nodule_mask = get_scan(index_scan)

        # validate
        if volume is None:
            continue

        # calculate layer probability
        layer_probability = {
            'negative': np.zeros((lung_mask.shape[2], )),
            'positive': np.zeros((nodule_mask.shape[2], ))
        }
        for z in range(lung_mask.shape[2]):
            layer_probability['negative'][z] = lung_mask[:,:,z].sum()
            layer_probability['positive'][z] = nodule_mask[:,:,z].astype(np.float).sum()
        for set in ['negative', 'positive']: # normalize
            layer_probability[set] = layer_probability[set] / layer_probability[set].sum()

        with open(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'layer_probability.pl'), 'wb') as out_file:
            pickle.dump(layer_probability, out_file)

def get_scan(index_scan):
    if os.path.isfile(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'volume.npy')):
        volume = np.load(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'volume.npy'))
        lung_mask = np.load(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'lung_mask.npy'))
        nodule_mask = np.load(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'nodule_mask.npy'))

        with open(os.path.join(LIDC_IDRI_NP_PREFIX, str(index_scan), 'layer_probability.pl'), 'rb') as in_file:
            layer_probability = pickle.load(in_file)

        return volume, lung_mask, nodule_mask, layer_probability
    else:
        return None, None, None, None

def get_nodule_mask(scan, volume):
    bbox=np.array([[0, 511], [0, 511], [0, volume.shape[2]-1]])
    mask = np.zeros_like(volume)
    for annotation in scan.annotations:
        tmp_mask = annotation.boolean_mask(bbox=bbox)
        mask = np.logical_or(mask, tmp_mask)
    mask = mask.astype(np.bool)
    return mask

def random_flip(patch):
    flip_number = np.random.randint(8) # 0 - 7
    for i in range(3):
        if flip_number & (1 << i):
            patch = np.flip(patch, i)
    return patch

def patch_generator(set='train', batch_size=32, batch_per_scan=50, patch_size=(64, 64, 16)):
    def get_patches(volume, size, is_positive, lung_mask, nodule_mask, layer_probability, patch_size=patch_size):
        def extract_patch(volume, mask, x, y, z, patch_size=patch_size):
            xs = np.arange(-patch_size[0]//2, patch_size[0]//2, dtype=np.int) + x
            ys = np.arange(-patch_size[1]//2, patch_size[1]//2, dtype=np.int) + y
            zs = np.arange(-patch_size[2]//2, patch_size[2]//2, dtype=np.int) + z

            patch = volume.take( xs, mode='wrap', axis=0).take(
                                ys, mode='wrap', axis=1).take(
                                zs, mode='warp', axis=2)

            # center
            xs = np.arange(-patch_size[0]//4, patch_size[0]//4, dtype=np.int) + x
            ys = np.arange(-patch_size[1]//4, patch_size[1]//4, dtype=np.int) + y
            zs = np.arange(-patch_size[2]//4, patch_size[2]//4, dtype=np.int) + z

            label = mask.take( xs, mode='wrap', axis=0).take(
                                ys, mode='wrap', axis=1).take(
                                zs, mode='warp', axis=2).any()

            patch = random_flip(patch)[..., np.newaxis]
            return patch, label

        # randomly choose one layer
        mask = nodule_mask.astype(np.bool) if is_positive else lung_mask.astype(np.bool)
        layer_probability = layer_probability['positive'] if is_positive\
                                    else layer_probability['negative']

        indexes_z = np.random.choice(volume.shape[2], size=size, replace=True, p=layer_probability)
        indexes_z_times = dict()
        for z in indexes_z: # accumulate
            indexes_z_times[z] = indexes_z_times.get(z, 0) + 1

        X = list()
        y = list()
        for z, times in indexes_z_times.items():
            # get size patches
            coor_x, coor_y = np.where(mask[:,:,z])
            for _ in range(times):
                i = np.random.randint(len(coor_x))
                patch, label = extract_patch(volume, nodule_mask, coor_x[i], coor_y[i], z)
                X.append(patch)
                y.append([label, 1-label])
        return X, y

    # scans = pl.query(pl.Scan).filter()
    train, valid, test = scan_index_split(1018)
    if set == 'train':
        indexes = train
    elif set == 'valid':
        indexes = valid
    else:
        indexes = test

    while True:
        scan_index = random.choice(indexes) # randomly choose one scan
        volume, lung_mask, nodule_mask, layer_probability = get_scan(scan_index)

        # validate
        if volume is None:
            continue

        # yield batch_per_scan number of batches
        for bi in range(batch_per_scan):
            # get batch_size // 2 negative patches
            negative_patches = get_patches(volume, batch_size//2, False, lung_mask, nodule_mask, layer_probability)
            positive_patches = get_patches(volume, batch_size//2, True, lung_mask, nodule_mask, layer_probability)
            X = negative_patches[0] + positive_patches[0]
            y = negative_patches[1] + positive_patches[1]
            X, y = shuffle(np.array(X), np.array(y))
            X = (X - 418.) / 414.
            yield X, y

def next_train_generator(set, batch_size=64):
    return next(patch_generator(set, batch_size=batch_size, batch_per_scan=1))

def generate_batch(set, number_batch):
    os.makedirs(LIDC_IDRI_BATCHES_PREFIX, exist_ok=True)
    os.makedirs(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set), exist_ok=True)
    os.makedirs(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'X'), exist_ok=True)
    os.makedirs(os.path.join(LIDC_IDRI_BATCHES_PREFIX, set, 'y'), exist_ok=True)

    X, y = np.ndarray((0, 64, 64, 16, 1), dtype=np.float), np.ndarray((0, 2), dtype=np.float)
    for j in tqdm(range(number_batch), desc='generate_batch'):
        with closing(Pool(processes=2)) as workers:
            for i, (xi, yi) in enumerate(tqdm(workers.imap(next_train_generator, [set]*512), desc='mini batch', total=512)):
                X = np.append(X, xi, axis=0)
                y = np.append(y, yi, axis=0)

        j = j+3
        x_file_name = os.path.join('/mnt/ext/lidc_idri_batches/', set, 'X', str(j)+'.npy')
        y_file_name = os.path.join('/mnt/ext/lidc_idri_batches/', set, 'y', str(j)+'.npy')
        np.save(x_file_name, X)
        np.save(y_file_name, y)

        X, y = np.ndarray((0, 64, 64, 16, 1), dtype=np.float), np.ndarray((0, 2), dtype=np.float)

if __name__ == '__main__':
    train, valid, test = scan_index_split(1018)
    print(len(train), len(valid), len(test))
    generate_batch('train', 14)
    generate_batch('valid', 1)
