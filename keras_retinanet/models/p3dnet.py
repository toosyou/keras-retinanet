import keras

from . import retinanet
from . import Backbone


class P3DNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(P3DNetBackbone, self).__init__(backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return p3dnet_retinanet(*args, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        return None

def custom_model(inputs): # (?, ?, 16, 1)
    def ConvP3D(filters):
        def f(x):
            x = keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=(3, 3, 1),
                            padding='same',
                            activation='relu'
                            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=(1, 1, 3),
                            padding='same',
                            activation='relu'
                            )(x)
            x = keras.layers.BatchNormalization()(x)
            return x
        return f

    # pool 3 times
    outputs = list()

    x = inputs # (512, 512, 16, 1)
    x = ConvP3D(8)(x)
    x = ConvP3D(8)(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (256, 256, 8, 256)

    x = ConvP3D(16)(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (128, 128, 4, 256)

    outputs.append(x) # C2

    x = ConvP3D(32)(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (64, 64, 2, 256)

    outputs.append(x) # C3

    x = ConvP3D(64)(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (32, 32, 1, 256)

    outputs.append(x) # C4
    return outputs

def p3dnet_retinanet(num_classes, inputs=None, modifier=None, weights=None, **kwargs):
    """ Constructs a retinanet model using a p3dnet backbone.

    Args
        num_classes: Number of classes to predict.
        inputs: The inputs to the network (defaults to a Tensor of shape (512, 512, 16, 1)).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(512, 512, 16, 1))

    outputs = custom_model(inputs)

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=outputs, **kwargs)
