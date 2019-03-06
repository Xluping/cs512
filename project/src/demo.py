# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:24:17 2018

@author: gyang
"""

import cv2
from keras.models import load_model
from keras.models import model_from_json
import keras.backend as K
import numpy as np
from scipy import stats
import tensorflow as tf
import os
import matplotlib.pyplot as plt
#%matplotlib inline

#path = 'D:\Masters\CV\Project'
#os.chdir(path)
path = os.getcwd()

j_son=['\\data\\new\\adadelta_mean_squared_error_model.json']
h5=['\\data\\new\\adadelta_mean_squared_error_model.h5']
c=1
i=0
# load json and create model
json_file = open(path+j_son[i], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(path+h5[i])
#    print("Loaded model from disk")
#D:\\Masters\\CV\\Project\\0032_NOISY_SRGB\\NOISY_SRGB_001.PNG
img_rows, img_cols, ch = 256, 342, 3
#filename = path+'\\0032_NOISY_SRGB\\NOISY_SRGB_001.PNG'
while(c!=ord('q') and c!=27): 
    filename = input("Input file name (Input Format - .\\\\folder\\\\filename.extension) : ")
#    img = cv2.imread(filename)
    image = cv2.imread(filename,-1)
    cv2.imshow('Noisy Image',cv2.resize(image,(800,600)))
    image = cv2.resize(image,(img_rows,img_cols))
    im_bw = image[:256,:256]
#    im_bw.shape
#    im_bw = cv2.resize(im_bw,(0,0),fx=4,fy=4)
    im_bw = im_bw.astype('float32')
    im_bw /= 255
    if K.image_data_format() == 'channels_first':
        im_bw = im_bw.reshape(1, ch, 256, 256)
    else:
        im_bw = im_bw.reshape(1, 256, 256, ch)


    prob = model.predict(im_bw)
    cv2.imshow('Denoised Image',prob[0])
#    prob.shape
#    prob = prob.reshape(256,256,3)
#    prob = cv2.resize(prob,(0,0),fx=1/4, fy=1/4)
#    plt.imshow(prob[0])
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    plt.show()
    
    c = cv2.waitKey()