from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Conv2D, AveragePooling2D, Activation

def cnn2d_classic(input_shape, n_channels, input_window_size, n_classes):
    
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
    if n_classes == 1:
        model.add(Dense(n_classes, activation='sigmoid'))
    else:
        model.add(Dense(n_classes, activation='softmax'))
    return model


def cnn2d_advanced(input_shape, n_channels, input_window_size, n_classes):
    model = Sequential()

    model.add(Conv2D(25, (15, 1), strides=(1, 1), padding='valid', activation='elu', input_shape=input_shape))
    #conv_pool_block_1
    model.add(Conv2D(filters=25, kernel_size=(15,1),dilation_rate=(2, 1),strides=(1, 1), padding='valid', activation='elu', input_shape=(input_window_size,n_channels,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=25, kernel_size=(1,n_channels),strides=(1, 1), padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(3,1)))

    #conv_pool_block_2
    model.add(Conv2D(filters=50, kernel_size=(10,1),strides=(1, 1), padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(3,1)))

    #conv_pool_block_3
    model.add(Conv2D(filters=100, kernel_size=(10,1),strides=(1, 1), padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(3,1)))

    #conv_pool_block_4
    model.add(Conv2D(filters=200, kernel_size=(10,1),strides=(1, 1), padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(3,1)))

    #classification Layer
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    if n_classes == 1:
        model.add(Dense(n_classes, activation='sigmoid'))
    else:
        model.add(Dense(n_classes, activation='softmax'))

    return model


def gcn_classic(input_shape, n_channels, input_window_size, n_classes):
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
    
    if n_classes == 1:
        model.add(Dense(n_classes, activation='sigmoid'))
    else:
        model.add(Dense(n_classes, activation='softmax'))

    return model



# https://arxiv.org/pdf/2004.00077.pdf