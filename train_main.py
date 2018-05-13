from PIL import Image, ImageFilter
import glob
import numpy as np
from numpy.random import *
import re
import model

from keras import backend as K
import tensorflow as tf
from keras import optimizers

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

import os

train_filepath = './sample'
WIDHT_LEN = 128
HIGH_LEN = 64
OUTPUT = 41
f_model = './fmodel'

def calc_output(filename):
    output = np.zeros((OUTPUT))

    number = int(re.search('[0-9]{4}',filename)[0])
    st = 10**3
    cnt = 0
    mod = number

    while st>=10:
        q, mod = divmod(mod, int(st))
        idx = cnt*10 + q
        output[idx] = 1
        st/=10
        cnt+=1

    idx = cnt*10 + mod
    output[idx] = 1

    return output

def generate_arrays_from_file(filepath, batch_size=50):
    files = glob.glob(filepath+'/*')
    num = len(files)

    images = np.empty((batch_size, 128, 64))
    y = np.empty((batch_size, OUTPUT))
    count = 0

    while True:
        ran_dir = randint(num)
        file_path = files[ran_dir]

        img = Image.open(file_path)
        img = img.convert("L")
        img = img.filter(ImageFilter.SMOOTH)
        img = img.filter(ImageFilter.SHARPEN)

        img = img.resize((128, 64))

        #name = 't_'+str(count)+'.png'
        #img.save(name)

        numimg = np.array(img).T
        numimg = numimg/255.

        images[count,:,:] = numimg[np.newaxis,:,:]

        #TODO: いいかんじに変える
        if len(file_path)==19:
            output = np.zeros((OUTPUT))
            output[OUTPUT-1] = 1
            y[count,:] = output
        else:
            y[count,:] = calc_output(file_path)

        count+=1

        if count==batch_size:
            in_put = images
            tar = y
            images = np.empty((batch_size, 128, 64))
            y = np.empty((batch_size, OUTPUT))
            count = 0
            yield (in_put[:,:,:,np.newaxis], tar)

old_session = K.get_session()
session = tf.Session()
K.set_session(session)
K.set_learning_phase(1)

callbacks = []
callbacks.append(CSVLogger("history_test_v2.csv"))
callbacks.append(ModelCheckpoint(filepath="model_v2.ep{epoch:02d}.h5", period=10))

model = model.model()
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

res = model.fit_generator(
    generate_arrays_from_file(train_filepath),
    samples_per_epoch=10,
    nb_epoch=200,
    initial_epoch=0,
    # validation_data=generate_arrays_from_file('/var/recommend/img_data/gerbool/test/', labels),
    # validation_steps=1,
    verbose=1,
    callbacks=callbacks
)
