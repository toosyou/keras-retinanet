import pylidc as pl
from pylidc.utils import consensus

import itertools
import numpy as np
from numba import jit
import better_exceptions
from sklearn.utils import shuffle

import sys
sys.path.append('../../LungTumor/')
import data_util

sys.path.append('..')
import keras_retinanet
from keras_retinanet.preprocessing.generator import Generator

def get_patch(scan, z_patch=16, verbose=False):
    def get_max_bboxes(nods):
        max_bboxes = list()
        for nod in nods:
            max_bbox = [np.Inf, -np.Inf, np.Inf, -np.Inf, np.Inf, -np.Inf]
            for annotation in nod:
                bbox = annotation.bbox()
                # update max_bbox
                for i in range(3):
                    max_bbox[i*2] = min(max_bbox[i*2], int(bbox[i].start))
                    max_bbox[i*2+1] = max(max_bbox[i*2+1], int(bbox[i].stop))
            max_bboxes.append(max_bbox)
        return max_bboxes

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

    X_STD, X_MEAN = 579.5747354211392, -1776.2860315918922

    X, y = list(), list()
    volume = scan.to_volume(verbose=False)
    lung_mask = data_util.lung_mask(volume, times_dilation=20, times_erosion=15, verbose=verbose)
    volume[lung_mask < 0.5] = volume.min()
    nods = scan.cluster_annotations()
    bboxes = get_max_bboxes(nods)

    for z in range(z_patch//2, volume.shape[2]-z_patch//2):
        selected_bboxes = bboxes_contain_z(z, bboxes)
        if len(selected_bboxes):
            X.append(np.repeat(volume[:,:, z-z_patch//2: z+z_patch//2].reshape((512, 512, 1, z_patch)), 3, axis=2)) # to rgb
            y.append(generate_labels(selected_bboxes))

    X, y = shuffle(np.array(X), np.array(y))
    X = (X - X_MEAN) / X_STD
    return X, y

class LungGenerator(Generator):
    def __init__(self, **kwargs):
        scans = pl.query(pl.Scan).filter()
        self.X, self.y = get_patch(scans[0])
        super(LungGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.y)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        return 1

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return self.X[image_index]

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        return np.array(self.y[image_index])

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        return image, annotations

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        return np.array(image_group)

if __name__ == '__main__':
    gen = LungGenerator(
        **{
            'batch_size'       : 32,
            'image_min_side'   : 800,
            'image_max_side'   : 1333,
            'preprocess_image' : lambda x: x,
        }
    )
    print(next(gen))
    pass
