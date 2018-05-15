from PIL import Image
Image.LOAD_TRUNCARTED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000

from keras.models import load_model
import numpy as np
import sys
import os
from keras.backend import tensorflow_backend as backend
from numpy.random import *
import model

R = 0.5

def read_image(path, j):
    out = []
    name = []
    i=0

    for x in os.listdir(path):
        img = Image.open(path + x)
        img = img.convert("L")
        img = img.resize((128, 64))
        numimg = np.array(img).T
        numimg = numimg/255.
        numimg = numimg[np.newaxis,:,:,np.newaxis]
        output = m.predict_on_batch(numimg)
        out.append(output.flatten())
        name.append(x)
        #print(x)
        #img.save(str(j)+'_'+str(i)+'.png')
        i+=1
    return out, name

def output_num(out, num):
    ind = int(out.shape[0]/10)
    n = []
    for d in range(0, ind):
        test = out[(d*10):((d+1)*10-1)]
        max_num = np.argmax(test)
        n.append(max_num)
    return n

# argvs = sys.argv
# a = argvs[1]
# b = argvs[2]
m = model.model(mode='sample')
m = load_model('./model_v2.ep200.h5', custom_objects={'softMaxAxis1':model.softMaxAxis1, 'dice_coef':model.dice_coef})
#model = load_model('./model.test5_ep10.h5')

#filename = argvs[3]
path = './0422/'
#path = '/var/recommend/img_data/gerbool/test/'+filename+'/'
f = open('res.txt','w')

for j in range(0,1):
    out, name = read_image(path, j)
    name_nparr = np.array(name)
    nparr = np.array(out)

    range_name = nparr.shape[0]
    for k in range(0, range_name):
        n = output_num(nparr[k,:], k)
        print(name_nparr[k])
        #print(n)
        #print(nparr[k,40])

        #閾値 0.5
        if nparr[k,40]<=R:
            f.write(name_nparr[k])
            f.write(': '+str(n))
            f.write("\n")

f.close()
backend.clear_session()
