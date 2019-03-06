# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 00:45:05 2018

@author: gyang
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
from keras import optimizers
import cv2
import os
import glob
from keras import backend as K
from keras.models import model_from_json
import matplotlib
matplotlib.use('Agg')

path = 'D:\Masters\CV\Project'

def new_model(input_shape,opt,opt_spec,lossfn):
    print("New model")
    autoencoder = Sequential()
    
    # Encoder Layers
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # Decoder Layers
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(3, (3, 3), padding='same'))

    autoencoder.compile(optimizer=opt_spec, loss=lossfn)
    model_save_loc(autoencoder,opt,lossfn)
    print("compiled")
    return autoencoder


def model_save_loc(autoencoder,opt,lossfn):
    autoencoder_json = autoencoder.to_json()
    with open(path+'\\data\\new\\'+ opt + '_' + lossfn + '_model.json', 'w') as json_file:
        json_file.write(autoencoder_json)
    autoencoder.save_weights(path+'\\data\\new\\'+ opt + '_' + lossfn + '_model.h5')
    print("model saved")
    
def load_model_loc(opt,lossfn):
    modelpath_json = path + '\\data\\new\\' + opt + '_' + lossfn + '_model' + '.json'
    model_wt_path = path + '\\data\\new\\' + opt + '_' + lossfn + '_model' + '.h5'
    with open(modelpath_json, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_wt_path)
    
    return model

def train_model(opt,opt_spec,lossfn,x_train,y_train,x_test,y_test):
    autoencoder = load_model_loc(opt,lossfn)
    print(opt,lossfn)
    autoencoder.compile(optimizer=opt_spec, loss=lossfn)
    history = autoencoder.fit(x_train, y_train,
                    epochs=25,
                    batch_size=10,
                    validation_data=(x_test, y_test))

    model_save_loc(autoencoder,opt,lossfn)

def get_150():
    ctr=0
    image_list=[]
    gt_list=[]
    folder_n = '\\data\\noisy_set\\'
    file_path_n = path + folder_n

    folder_g = '\\data\\gt_set\\'
    file_path_g = path + folder_g

    optimizers_list = ['adadelta']
    loss_functions_list = ['mean_squared_error']
#    adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)


    for opt,lossfn in zip(optimizers_list,loss_functions_list):
        for filename_n,filename_g in zip(os.listdir(file_path_n),os.listdir(file_path_g)): #assuming png
            img_rows, img_cols, ch = 64, 64, 3
#            print (filename_n)
            with open(file_path_n + filename_n,'rb') as infile_n:
                buf = infile_n.read()
            
            x = np.fromstring(buf, dtype = 'uint8')
            img = cv2.imdecode(x,1)
            img = cv2.resize(img,(img_rows,img_cols))
            img = cv2.resize(img,(0,0),fx=4,fy=4)
            image_list.append(img)
            
            with open(file_path_g + filename_g,'rb') as infile_g:
                buf = infile_g.read()
            x = np.fromstring(buf, dtype = 'uint8')
            img = cv2.imdecode(x,cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img,(img_rows,img_cols))
            img = cv2.resize(img,(0,0),fx=4,fy=4)
            gt_list.append(img)
            
            ctr = ctr + 1
            if ctr%150 == 0:
                print(ctr)
                images_ip = np.asarray(image_list,dtype='float32')
                train_size=int(0.8*len(images_ip))
                x_train = images_ip[:train_size]
                x_test = images_ip[train_size:]
                del(images_ip)
                x_train /= 255
                x_test /= 255
    
                images_ip_gt = np.asarray(gt_list,dtype='float32')
                y_train = images_ip_gt[:train_size]
                y_test = images_ip_gt[train_size:]
                del(images_ip_gt)
                y_train /= 255
                y_test /= 255
                image_list=[]
                gt_list=[]
    
                img_rows, img_cols, ch = 256, 256, 3
                
                if K.image_data_format() == 'channels_first':
                    x_train = x_train.reshape(x_train.shape[0], ch, img_rows, img_cols)
                    x_test = x_test.reshape(x_test.shape[0], ch, img_rows, img_cols)
                    y_train = y_train.reshape(y_train.shape[0], ch, img_rows, img_cols)
                    y_test = y_test.reshape(y_test.shape[0], ch, img_rows, img_cols)
                    input_shape = (ch, img_rows, img_cols)
                else:
                    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, ch)
                    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, ch)
                    y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, ch)
                    y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, ch)
                    input_shape = (img_rows, img_cols, ch)

#                if opt == 'adagrad':
#                    opt_spec = adagrad
#                if opt == 'adam':
#                    opt_spec = adam
                if opt == 'adadelta':
                    opt_spec = adadelta
                    
                    
                if ctr == 150:
                    print(ctr)
                    autoencoder = new_model(input_shape,opt,opt_spec,lossfn)
                    print(opt,lossfn)

                
                train_model(opt,opt_spec,lossfn,x_train,y_train,x_test,y_test)
                

if __name__ == '__main__':
    get_150()

