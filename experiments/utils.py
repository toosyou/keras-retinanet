import sys
import os
import configparser

sys.path.append('/home/toosyou/projects/LungTumor')
import data_util

import pylidc as pl
import numpy as np
import better_exceptions
from sklearn.utils import shuffle
import glob
import dicom

config = configparser.ConfigParser()
config.read('./configs.ini')

def images2volume(images):
    # transform to numpy array format
    volume = np.zeros((512,512,len(images)))
    for j in range(len(images)):
        volume[:,:,j] = (images[j].pixel_array)*(images[j].RescaleSlope) + images[j].RescaleIntercept
    return volume

def get_dicom_volume(folder_path):
    addresses = glob.glob(folder_path+'/*.dcm') # get all dicom addresses
    images = []
    for fname in addresses:
        with open(fname, 'rb') as f:
            image = dicom.read_file(f)
            images.append(image)

    # ##############################################
    # Clean multiple z scans.
    #
    # Some scans contain multiple slices with the same `z` coordinate
    # from the `ImagePositionPatient` tag.
    # The arbitrary choice to take the slice with lesser
    # `InstanceNumber` tag is made.
    # This takes some work to accomplish...
    zs    = [float(img.ImagePositionPatient[-1]) for img in images]
    inums = [float(img.InstanceNumber) for img in images]
    inds = range(len(zs))
    while np.unique(zs).shape[0] != len(inds):
        for i in inds:
            for j in inds:
                if i!=j and zs[i] == zs[j]:
                    k = i if inums[i] > inums[j] else j
                    inds.pop(inds.index(k))

    # Prune the duplicates found in the loops above.
    zs     = [zs[i]     for i in range(len(zs))     if i in inds]
    images = [images[i] for i in range(len(images)) if i in inds]

    # Sort everything by (now unique) ImagePositionPatient z coordinate.
    sort_inds = np.argsort(zs)
    images    = [images[s] for s in sort_inds]
    # End multiple z clean.
    # ##############################################

    return images2volume(images) # convert images to volume

def get_patches(index_scan, include_negative=False, randomly_select=False, z_patch=16, verbose=False):
    def bboxes_contain_z(z, bboxes):
        selected = list()
        for bbox in bboxes:
            if z >= bbox[4] and z <= bbox[5]:
                selected.append(bbox)
        return selected

    def generate_labels(bboxes):
        labels = list()
        for bbox in bboxes:
            labels.append((bbox[0], bbox[2], bbox[1], bbox[3], 0)) # x1, y1, x2, y2, 1
        return labels

    X, y = list(), list()
    if os.path.isfile(os.path.join('/mnt/ext/lidc_idri_np', str(index_scan), 'volume.npy')):
        if randomly_select:
            volume = np.load(os.path.join('/mnt/ext/lidc_idri_np', str(index_scan), 'volume.npy'), mmap_mode='r')
        else:
            volume = np.load(os.path.join('/mnt/ext/lidc_idri_np', str(index_scan), 'volume.npy'))
        bboxes = np.load(os.path.join('/mnt/ext/lidc_idri_np', str(index_scan), 'bboxes.npy'))
    else:
        return None, None
    # lung_mask = data_util.lung_mask(volume, times_dilation=20, times_erosion=15, verbose=verbose)
    # volume[lung_mask < 0.5] = volume.min()

    if randomly_select:
        if include_negative:
            zs = [np.random.randint(z_patch//2, volume.shape[2]-z_patch//2)]
        else:
            positive_zs = list()
            for z in range(z_patch//2, volume.shape[2]-z_patch//2):
                selected_bboxes = bboxes_contain_z(z, bboxes)
                if len(selected_bboxes): positive_zs.append(z)
            zs = [np.random.choice(positive_zs)]
    else:
        zs = list(range(z_patch//2, volume.shape[2]-z_patch//2))

    for z in zs:
        selected_bboxes = bboxes_contain_z(z, bboxes)
        if include_negative or len(selected_bboxes):
            X.append(volume[:,:, z-z_patch//2: z+z_patch//2])
            y.append(generate_labels(selected_bboxes))

    X, y = np.array(X), np.array(y)
    return X, y

def save_patch(image, annotations, index, which_set):
    if which_set not in ['train', 'valid', 'test']:
        raise ValueError('which_set must be train/valid/test.')

    data_dir        = config['Lung']['DataDirectory']
    directory_size  = int(config['Lung']['DirectorySize'])
    image_dir       = os.path.join(data_dir, which_set, 'X', str(index//directory_size*directory_size))
    annotations_dir = os.path.join(data_dir, which_set, 'y', str(index//directory_size*directory_size))

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    image, annotations = np.array(image), np.array(annotations)

    if os.path.isfile(os.path.join(image_dir, str(index))) or\
        os.path.isfile(os.path.join(annotations_dir, str(index))):
        print('Warning: patch-{} in {} set exists'.format(index, which_set))

    np.save(os.path.join(image_dir, str(index)+'.npy'), image)
    np.save(os.path.join(annotations_dir, str(index)+'.npy'), annotations)

def load_image(index, which_set):
    if which_set not in ['train', 'valid', 'test']:
        raise ValueError('which_set must be train/valid/test.')

    data_dir        = config['Lung']['DataDirectory']
    directory_size  = int(config['Lung']['DirectorySize'])
    image_dir       = os.path.join(data_dir, which_set, 'X', str(index//directory_size*directory_size))
    image           = np.load(os.path.join(image_dir, str(index)+'.npy'))
    return image

def load_annotations(index, which_set):
    if which_set not in ['train', 'valid', 'test']:
        raise ValueError('which_set must be train/valid/test.')

    data_dir        = config['Lung']['DataDirectory']
    directory_size  = int(config['Lung']['DirectorySize'])
    annotations_dir = os.path.join(data_dir, which_set, 'y', str(index//directory_size*directory_size))
    annotations     = np.load(os.path.join(annotations_dir, str(index)+'.npy'))
    return annotations
