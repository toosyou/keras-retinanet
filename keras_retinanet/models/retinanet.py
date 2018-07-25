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

import tensorflow as tf
import keras
from keras.layers import TimeDistributed, Reshape, Permute

from .. import initializers
from .. import layers

import numpy as np

PYRAMID_FEATURE_SIZE = 32
CLASSIFICATION_FEATURE_SIZE = 32
REGRESSION_FEATURE_SIZE = 32

def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=PYRAMID_FEATURE_SIZE,
    prior_probability=0.002,
    classification_feature_size=CLASSIFICATION_FEATURE_SIZE,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=PYRAMID_FEATURE_SIZE, regression_feature_size=REGRESSION_FEATURE_SIZE, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features3D(C1, C2, C3, C4, feature_size=PYRAMID_FEATURE_SIZE):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C1           : Feature stage C1 from the backbone. (256, 256, 8, ?)
        C2           : Feature stage C2 from the backbone. (128, 128, 4, ?)
        C3           : Feature stage C3 from the backbone. (64, 64, 2, ?)
        C4           : Feature stage C4 from the backbone. (32, 32, 1, ?)
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P1, P2, P3, P4].
    """
    def ConvP3D(filters):
        def f(x):
            x = keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=(3, 3, 1),
                            padding='same',
                            activation='relu'
                            )(x)
            # x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=(1, 1, 3),
                            padding='same',
                            activation='relu'
                            )(x)
            # x = keras.layers.BatchNormalization()(x)
            return x
        return f

    # upsample C4 to get P4 from the FPN paper
    P4           = keras.layers.Conv3D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4_upsampled = keras.layers.UpSampling3D(name='P4_upsampled')(P4)
    P4           = ConvP3D(feature_size)(P4) # (32, 32, 1, ?)

    # upsample C3 to get P3 from the FPN paper
    P3           = keras.layers.Conv3D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = keras.layers.UpSampling3D(name='P3_upsampled')(P3)
    P3           = ConvP3D(feature_size)(P3) # (64, 64, 2 ?)

    # add P3 elementwise to C2
    P2           = keras.layers.Conv3D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2           = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2_upsampled = keras.layers.UpSampling3D(name='P2_upsampled')(P2)
    P2           = ConvP3D(feature_size)(P2) # (128, 128, 4, ?)

    # add P2 elementwise to C1
    P1           = keras.layers.Conv3D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced')(C1)
    P1           = keras.layers.Add(name='P1_merged')([P2_upsampled, P1])
    P1           = ConvP3D(feature_size)(P1) # (256, 256, 8, ?)

    # "P4 is obtained via a 3x3 stride-2 conv on C3"
    '''
    C3_pooled    = keras.layers.MaxPool3D(pool_size=(1, 1, 2), padding='same', name='C3_maxpooled')(C3) # (64, 64, 1, ?)
    C3_pooled    = keras.layers.Reshape((64, 64, -1))(C3_pooled)
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P4')(C3_pooled) # (32, 32, ?)
    '''

    # "P6 is computed by applying ReLU followed by a 3x3 stride-2 conv on P5"
    # P6 = keras.layers.Activation('relu', name='C6_relu')(P5)
    # P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(P6)

    P1 = keras.layers.MaxPool3D(pool_size=(1, 1, 8), padding='same', name='P1_maxpooled')(P1)
    P2 = keras.layers.MaxPool3D(pool_size=(1, 1, 4), padding='same', name='P2_maxpooled')(P2)
    P3 = keras.layers.MaxPool3D(pool_size=(1, 1, 2), padding='same', name='P3_maxpooled')(P3)

    P1 = keras.layers.Reshape((256, 256, -1), name='P1')(P1)
    P2 = keras.layers.Reshape((128, 128, -1), name='P2')(P2)
    P3 = keras.layers.Reshape((64, 64, -1), name='P3')(P3)
    P4 = keras.layers.Reshape((32, 32, -1), name='P4')(P4)

    return [P1, P2, P3, P4]

def __create_pyramid_features(C2, C3, C4, feature_size=64):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C2           : Feature stage C2 from the backbone.
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P2, P3, P4, P5, P6].
    """
    # upsample C4 to get P4 from the FPN paper
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C2])
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # add P3 elementwise to C2
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    # "P5 is obtained via a 3x3 stride-2 conv on C4"
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P5')(C4)

    # "P6 is computed by applying ReLU followed by a 3x3 stride-2 conv on P5"
    # P6 = keras.layers.Activation('relu', name='C6_relu')(P5)
    # P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(P6)

    return [P2, P3, P4, P5]# , P6]


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    # sizes   = [32, 64, 128, 256, 512],
    sizes   = [6, 12, 24, 48],
    # strides = [8, 16, 32, 64, 128],
    strides = [2, 4, 8, 16],
    ratios  = np.array([1], keras.backend.floatx()),
    scales  = np.array([2 ** (-2.0 / 3.0), 2 ** 0, 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = 3,
    create_pyramid_features = __create_pyramid_features3D,
    submodels               = None,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C1, C2, C3, C4 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C1, C2, C3, C4)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    anchor_parameters     = AnchorParameters.default,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters     : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    # compute the anchors
    # features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    features = [model.get_layer(p_name).output for p_name in ['P1', 'P2', 'P3', 'P4']]
    anchors  = __build_anchors(anchor_parameters, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    outputs = detections

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)
