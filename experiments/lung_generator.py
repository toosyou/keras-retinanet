import pylidc as pl

import itertools
import numpy as np
from numba import jit
import better_exceptions
from sklearn.utils import shuffle

import os
import sys
import math
sys.path.append('../../LungTumor/')
import data_util

sys.path.append('..')
import keras_retinanet
from keras_retinanet.preprocessing.generator import Generator

import utils
import configparser
import pickle
from preprocessing import scan_index_split
from utils import get_patches
from scipy.ndimage.interpolation import rotate

def random_transform(
    image,
    annotations,
    flip_x_chance=0.5,
    flip_y_chance=0.5,
    flip_z_chance=0.5
):
    def fix_annotaions(annotations):
        fix_annotaions = np.zeros((annotations.shape[0], 5))
        fix_annotaions[:, 0] = np.minimum(annotations[:, 0], annotations[:, 2])
        fix_annotaions[:, 1] = np.minimum(annotations[:, 1], annotations[:, 3])
        fix_annotaions[:, 2] = np.maximum(annotations[:, 0], annotations[:, 2])
        fix_annotaions[:, 3] = np.maximum(annotations[:, 1], annotations[:, 3])
        return fix_annotaions

    image, annotations = image.copy(), annotations.copy()
    for axis, chance in enumerate([flip_x_chance, flip_y_chance, flip_z_chance]):
        if np.random.random() < chance:
            image = np.flip(image, axis)
            if axis < 2:
                axis = 1 - axis
                annotations[:, axis] = 512. - annotations[:, axis]
                annotations[:, axis+2] = 512. - annotations[:, axis+2]

    annotations = fix_annotaions(annotations)

    image = np.array(image)
    annotations = np.array(annotations)
    return image, annotations


class LungGenerator(Generator):
    def __init__(self, set_name, **kwargs):
        def preprocess_image(image):
            """ Preprocess image and its annotations.
            """
            MEAN, STD = 174., 825.
            # image = (image - image.mean()) / image.std()
            image = (image - MEAN) / STD
            return image

        self.set_name = set_name
        self.config = configparser.ConfigParser()
        self.config.read('./configs.ini')
        if self.set_name == 'valid':
            data_dir    = self.config['Lung']['DataDirectory']
            infos       = pickle.load(open(os.path.join(data_dir, 'infos.pl'), 'rb'))
            self.valid_size = min(200, infos['valid_size'])
            self.random_index = np.random.choice(int(infos['valid_size']), size=self.valid_size, replace=False)

        super(LungGenerator, self).__init__(**dict(kwargs, group_method='random', preprocess_image=preprocess_image))

    def size(self):
        if self.set_name == 'valid':
            return self.valid_size
        else:
            data_dir    = self.config['Lung']['DataDirectory']
            infos       = pickle.load(open(os.path.join(data_dir, 'infos.pl'), 'rb'))
            return int(infos[self.set_name+'_size'])

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return 0

    def label_to_name(self, label):
        """ Map label to name.
        """
        return 'nodule'

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        return 1

    def load_image(self, image_index, repeat=False):
        """ Load an image at the image_index.
        """
        if self.set_name == 'valid':
            image_index = self.random_index[image_index]
        image = utils.load_image(image_index, self.set_name)
        image = image.reshape((512, 512, 16, 1))
        # image = image[:,:,6:9]
        if repeat:
            image = np.repeat(image, 3, axis=2) # to rgb
            if image.shape != (512, 512, 3, 16):
                raise ValueError('image size error!', image.shape)
        # image = np.random.rand(*image.shape)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        if self.set_name == 'valid':
            image_index = self.random_index[image_index]
        annotations = utils.load_annotations(image_index, self.set_name) # (x1, y1, x2, y2, label)
        permuted_annotations = np.zeros((annotations.shape[0], 5))
        if annotations.shape[0] > 0:
            permuted_annotations[:, 0] = annotations[:, 1]
            permuted_annotations[:, 1] = annotations[:, 0]
            permuted_annotations[:, 2] = annotations[:, 3]
            permuted_annotations[:, 3] = annotations[:, 2]
            permuted_annotations[:, 4] = annotations[:, 4]
        return permuted_annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        image = self.preprocess_image(image)
        image, annotations = random_transform(image, annotations)
        return image, annotations

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        return np.array(image_group)

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return image, 1.

class LungScanGenerator(Generator):
    def __init__(self, set_name, index, **kwargs):
        def preprocess_image(image):
            """ Preprocess image and its annotations.
            """
            MEAN, STD = 174., 825.
            # image = (image - image.mean()) / image.std()
            image = (image - MEAN) / STD
            return image

        self.set_name = set_name
        self.scan_index = scan_index_split(1018)[{'train': 0, 'valid': 1, 'test': 2}[set_name]][index]
        self.scan = pl.query(pl.Scan).filter()[self.scan_index]
        self.X, self.y = get_patches(self.scan, negative_ratio=2.)
        super(LungScanGenerator, self).__init__(**dict(kwargs, group_method='none', preprocess_image=preprocess_image))

    def size(self):
        return self.X.shape[0]

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return 0

    def label_to_name(self, label):
        """ Map label to name.
        """
        return 'nodule'

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        return 1

    def load_image(self, image_index, repeat=False):
        """ Load an image at the image_index.
        """
        image = self.X[image_index]
        image = image.reshape((512, 512, 16, 1))
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = np.array(self.y[image_index])
        permuted_annotations = np.zeros((annotations.shape[0], 5))
        if annotations.shape[0]:
            permuted_annotations[:, 0] = annotations[:, 1]
            permuted_annotations[:, 1] = annotations[:, 0]
            permuted_annotations[:, 2] = annotations[:, 3]
            permuted_annotations[:, 3] = annotations[:, 2]
            permuted_annotations[:, 4] = annotations[:, 4]
        return permuted_annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        image = self.preprocess_image(image)
        return image, annotations

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        return np.array(image_group)

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return image, 1.

if __name__ == '__main__':
    gen = LungGenerator(
        'train',
        **{
            'batch_size'       : 1,
            'image_min_side'   : 800,
            'image_max_side'   : 1333,
            'preprocess_image' : lambda x: x,
        }
    )
    next(gen)
