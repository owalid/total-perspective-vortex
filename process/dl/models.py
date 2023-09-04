from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Conv2D, AveragePooling2D, Activation

def cnn2d_classic(input_shape, n_channels):
    
    model = Sequential()
    # Block 1: Temporal Convolution
    model.add(Conv2D(8, (1, n_channels), strides=(1, 1), padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Block 2: Spacial Convolution
    model.add(Conv2D(int(n_channels/2), (1, n_channels), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    # model.add(AveragePooling2D(pool_size=(1, 4), strides=(1, 4)))
    model.add(Dropout(0.25))

    # Block 3: Separable Convolution
    model.add(Conv2D(8, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(Conv2D(16, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    # model.add(AveragePooling2D(pool_size=(1, 4), strides=(1, 4)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # Classifier
    model.add(Dense(4, activation='softmax'))
    return model


def gcn_classic(input_shape, n_channels):
    model = Sequential()
    model.add(Conv2D(8, (1, n_channels), strides=(1, 1), padding='same', use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Conv2D(8, (1, n_channels), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Activation('elu'))
    model.add(Conv2D(8, (1, n_channels), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Activation('elu'))
    model.add(Conv2D(8, (1, n_channels), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Activation('elu'))

    # Block 2: Spacial Convolution
    model.add(Conv2D(16, (n_channels, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, 4), strides=(1, 4)))
    model.add(Dropout(0.25))

    # Block 3: Separable Convolution
    model.add(Conv2D(16, (1, 16), strides=(1, 1), padding='same', use_bias=False))
    model.add(Conv2D(8, (1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(momentum=0.01, epsilon=1e-3))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, 8), strides=(1, 8)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(4, activation='softmax'))

    return model



# https://arxiv.org/pdf/2004.00077.pdf