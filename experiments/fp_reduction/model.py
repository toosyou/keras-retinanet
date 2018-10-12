import sys
import keras
import tensorflow as tf
from keras import backend
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, GlobalAvgPool3D, GlobalMaxPooling3D
from keras.layers import concatenate, Reshape, Activation, Permute, Softmax, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

def focal_loss(alpha=.25, gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

        return  K.sum(cls_loss) / K.sum(y_true)
    return focal_loss_fixed

def get_unet_model(number_filter_base = 16):
    def downsample_block(input, concated, filters):
        net = Conv3D(filters, 3, activation='relu', padding='same')(input)
        net = Conv3D(filters, 3, activation='relu', padding='same')(net)
        net = BatchNormalization()(net)
        net = MaxPooling3D(2, padding='same')(net)
        if concated is not None:
            net = concatenate([net, concated])
        return net

    def upsample_block(input, concated, filters):
        net = concatenate([Conv3DTranspose(filters, 3, strides=2, padding='same')(input), concated])
        net = Conv3D(filters, 3, activation='relu', padding='same')(net)
        net = Conv3D(filters, 3, activation='relu', padding='same')(net)
        net = BatchNormalization()(net)
        return net

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # building unet-like model
    downsample_outputs = [0] * 4
    upsample_outputs = [0] * 4
    inputs = Input((64, 64, 16, 1))
    net = inputs
    for i in range(4):
        net = downsample_block(net, None, number_filter_base*(2**i))
        downsample_outputs[i] = net

    upsample_outputs[0] = net
    for i in range(3):
        net = upsample_block(net, downsample_outputs[2-i], number_filter_base*(2**(2-i)))
        upsample_outputs[i+1] = net

    for i in range(3):
        net = downsample_block(net, upsample_outputs[2-i], number_filter_base*(2**(i+1)))

    net = Conv3D(number_filter_base*8, 3, activation='relu', padding='same')(net)
    net = BatchNormalization()(net)
    net = Conv3D(number_filter_base*8, 3, activation='relu', padding='same')(net)
    net = GlobalMaxPooling3D()(net)
    net = BatchNormalization()(net)
    net = Dense(2, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=net)
    training_model = keras.utils.multi_gpu_model(model, gpus=2)
    training_model.compile(optimizer=Adam(amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    return model, training_model

def get_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    number_filter_base = 32
    model = Sequential([
        Conv3D(number_filter_base, 3, padding='same', activation='relu', input_shape=(64, 64, 16, 1)),
        BatchNormalization(),
        Conv3D(number_filter_base, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling3D(2, padding='same'), # 32, 32, 8, ?

        Conv3D(number_filter_base*2, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv3D(number_filter_base*2, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling3D(2, padding='same'), # 16, 16, 4, ?

        Conv3D(number_filter_base*4, 3, padding='same', activation='relu'),
        BatchNormalization(),
        # Conv3D(number_filter_base*4, 3, padding='same', activation='relu'),
        # BatchNormalization(),
        MaxPooling3D(2, padding='same'), # 8, 8, 2, ?

        Conv3D(number_filter_base*8, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv3D(number_filter_base*8, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling3D(2, padding='same'), # 4, 4, 1, ?

        # Conv3D(number_filter_base*16, 3, padding='same', activation='relu'),
        # BatchNormalization(),
        # Conv3D(number_filter_base*16, 3, padding='same', activation='relu'),
        # BatchNormalization(),
        # MaxPooling3D((2, 2, 1), padding='same'), # 2, 2, 1, ?

        GlobalMaxPooling3D(), # number_filter_base*16
        # Flatten(),
        # Dense(512, activation='relu'),
        # Dropout(rate=0.2),
        # BatchNormalization(),
        # Dense(256, activation='relu'),
        # Dropout(rate=0.2),
        # BatchNormalization(),
        # Dense(128, activation='relu'),
        # Dropout(rate=0.2),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])
    training_model = keras.utils.multi_gpu_model(model, gpus=2)
    training_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model, training_model

if __name__ == '__main__':
    model, training_model = get_unet_model()
    model.summary()
