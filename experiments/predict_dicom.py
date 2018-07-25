#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import keras
import tensorflow as tf

import numpy as np
import cv2

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
sys.path.append('../')
from keras_retinanet import models
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.visualization import draw_detections

from lung_generator import LungGenerator, LungScanGenerator
import utils

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('dicom_dir',         help='Path to the directory of dicoms.')
    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).')

    return parser.parse_args(args)

class DicomGenerator:
    def __init__(self, dicom_dir):
        def preprocess_image(image):
            """ Preprocess image and its annotations.
            """
            MEAN, STD = 174., 825.
            # image = (image - image.mean()) / image.std()
            image = (image - MEAN) / STD
            return image

        self.volume = utils.get_dicom_volume(dicom_dir)
        self.z_indices = list(range(16//2, self.volume.shape[2]-16//2))
        self.preprocess_image = preprocess_image

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

    def size(self):
        return len(self.z_indices)

    def load_image(self, image_index, repeat=False):
        """ Load an image at the image_index.
        """
        z = self.z_indices[image_index]
        return self.volume[:,:,z-16//2: z+16//2].reshape(512, 512, 16, 1)

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return image, 1.


def create_generator(args):
    return DicomGenerator(args.dicom_dir)

def get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    def normalized_image(raw_image):
        raw_image = raw_image[...,raw_image.shape[-2]//2, 0]
        raw_image = (raw_image - raw_image.min())
        raw_image = raw_image / raw_image.max() * 255.0
        raw_image = np.repeat(raw_image.reshape((512, 512, 1)), 3, axis=2) # to rgb
        raw_image = raw_image.astype(np.uint8).copy()
        return raw_image

    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            raw_image = normalized_image(raw_image)
            # draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        # print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model)

    # start evaluation
    all_detections = get_detections(
        generator,
        model,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
