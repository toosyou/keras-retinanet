import keras
from .group_norm import GroupNormalization

def ConvP3D(filters, kernel_size=3, stride=1, batch_norm=False, **kwargs):
    if isinstance(kernel_size, int):
        first_kernel_size   = (kernel_size, kernel_size, 1)
        second_kernel_size  = (1, 1, kernel_size)
    else:
        first_kernel_size   = (kernel_size[0], kernel_size[1], 1)
        second_kernel_size  = (1, 1, kernel_size[2])

    if isinstance(stride, int):
        first_stride   = (stride, stride, 1)
        second_stride  = (1, 1, stride)
    else:
        first_stride   = (stride[0], stride[1], 1)
        second_stride  = (1, 1, stride[2])

    def f(x):
        x = keras.layers.Conv3D(
                        filters=filters,
                        kernel_size=first_kernel_size,
                        strides=first_stride,
                        **kwargs
                        )(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(
                        filters=filters,
                        kernel_size=second_kernel_size,
                        strides=second_stride,
                        **kwargs
                        )(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
        return x
    return f
