import pylidc as pl
from pylidc.utils import consensus

import itertools
import numpy as np
from numba import jit
import better_exceptions
from sklearn.utils import shuffle

import os
import sys
sys.path.append('../../LungTumor/')
import data_util

sys.path.append('..')
import keras_retinanet
from keras_retinanet.preprocessing.generator import Generator

import utils
import configparser
import pickle

class LungGenerator(Generator):
    def __init__(self, set_name, **kwargs):
        self.set_name = set_name
        self.config = configparser.ConfigParser()
        self.config.read('./configs.ini')
        if self.set_name == 'valid':
            data_dir    = self.config['Lung']['DataDirectory']
            infos       = pickle.load(open(os.path.join(data_dir, 'infos.pl'), 'rb'))
            self.valid_size = min(200, infos['valid_size'])
            self.random_index = np.random.choice(int(infos['valid_size']), size=self.valid_size, replace=False)

        super(LungGenerator, self).__init__(**dict(kwargs, group_method='random'))

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
        permuted_annotations = np.zeros_like(annotations)
        permuted_annotations[:, 0] = annotations[:, 1]
        permuted_annotations[:, 1] = annotations[:, 0]
        permuted_annotations[:, 2] = annotations[:, 3]
        permuted_annotations[:, 3] = annotations[:, 2]
        permuted_annotations[:, 4] = annotations[:, 4]
        return permuted_annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        MEAN, STD = 175., 825.
        # image = (image - image.mean()) / image.std()
        image = (image - MEAN) / STD
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
            'batch_size'       : 32,
            'image_min_side'   : 800,
            'image_max_side'   : 1333,
            'preprocess_image' : lambda x: x,
        }
    )
    mean = 0.
    std = 0.
    for i in np.random.choice(gen.size(), 100):
        img = gen.load_image(i)
        mean += img.mean()
        std += img.std()

    print(mean/100., std/100.)
