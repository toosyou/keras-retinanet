import keras

from . import retinanet
from . import Backbone
from ..layers.group_norm import GroupNormalization
from ..layers.convp3d import ConvP3D

# DON'T USE GROUP NORM EVER AGAIN!
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

def p3d_resnet(inputs): # (?, ?, 16, 1)
    def block(filters, stage=0, block=0):
        if stage != 0 and block == 0:
            stride = 2
        else:
            stride = 1

        def f(x):
            y = keras.layers.Conv3D(filters, kernel_size=3, strides=stride, padding='same', activation='relu', use_bias=True)(x)
            y = keras.layers.Conv3D(filters, kernel_size=3, padding='same', use_bias=True)(y)

            if block == 0:
                shortcut = keras.layers.Conv3D(filters, kernel_size=1, strides=stride, padding='same', use_bias=True)(x)
                # shortcut = keras.layers.BatchNormalization()(shortcut)
            else:
                shortcut = x

            y = keras.layers.Add()([y, shortcut])
            y = keras.layers.Activation('relu')(y)
            return y
        return f

    blocks = [1, 3, 3, 3]
    # pool 3 times
    outputs = list()
    filters = 8

    x = inputs # (512, 512, 16, 1)
    x = keras.layers.Conv3D(filters, kernel_size=3, padding='same', activation='relu', use_bias=True)(x)
    x = keras.layers.Conv3D(filters, kernel_size=3, padding='same', activation='relu', use_bias=True)(x)
    x = keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same')(x)

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(filters, stage_id, block_id)(x)

        filters *= 2
        outputs.append(x)

    return outputs


def custom_model(inputs): # (?, ?, 16, 1)
    # pool 3 times
    outputs = list()

    # batch_norm = False
    base_features = 16

    x = inputs # (512, 512, 16, 1)
    x = keras.layers.Conv3D(base_features, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv3D(base_features, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (256, 256, 8, ?)

    outputs.append(x) # C1

    # x = ConvP3D(base_features*2, batch_norm=batch_norm, padding='same', activation='relu')(x)
    x = keras.layers.Conv3D(base_features*2, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (128, 128, 4, ?)

    outputs.append(x) # C2

    # x = ConvP3D(base_features*4, batch_norm=batch_norm, padding='same', activation='relu')(x)
    x = keras.layers.Conv3D(base_features*4, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (64, 64, 2, ?)

    outputs.append(x) # C3


    # x = ConvP3D(base_features*4, batch_norm=batch_norm, padding='same', activation='relu')(x)
    x = keras.layers.Conv3D(base_features*4, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool3D(
                        pool_size=(2, 2, 2),
                        padding='same'
                        )(x) # (32, 32, 1, ?)

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

    # outputs = custom_model(inputs)
    outputs = p3d_resnet(inputs)

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=outputs, **kwargs)
