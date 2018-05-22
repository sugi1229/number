from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Reshape
from keras.engine.topology import Layer
from keras.models import Input, Model
import keras.backend as K
import numpy as np
from keras.activations import softmax

from keras import optimizers

OUTPUT=41

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

def softMaxAxis1(x):
    return softmax(x,axis=1)

def model(mode='train'):
    img_width = 128
    img_height = 64

    i = Input(shape=(img_width,img_height,1))
    conv1 = Conv2D(48, kernel_size=(5, 5), padding='same', activation='relu')(i)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.2)(pool1)
    conv2 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(drop1)
    pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
    drop2 = Dropout(0.2)(pool2)
    conv3 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu')(drop2)
    pool3 = MaxPooling2D(pool_size=(1, 2))(conv3)
    drop3 = Dropout(0.2)(pool3)
    flatt = Flatten()(drop3)
    hidden1 = Dense(2048, activation='relu')(flatt)
    o = Dense(OUTPUT, activation='sigmoid')(hidden1)
    model = Model(inputs=i, outputs=o)

    # hidden2 = Dense(OUTPUT)(hidden1)
    # reshape1 = Reshape((10,4))(hidden2)
    # mysoftmax = Activation(softMaxAxis1)(reshape1)
    # o = Reshape((OUTPUT,))(mysoftmax)
    # model = Model(inputs=i, outputs=o)

    model.summary()

    if mode=='train':
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mean_squared_error', metrics=[dice_coef])
        return model
    else:
        return model
