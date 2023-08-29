


'''
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
'''