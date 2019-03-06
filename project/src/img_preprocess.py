# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:44:44 2018

@author: gyang
"""

import cv2
import os
import glob
import numpy as np

path = 'D:\Masters\CV\Project'
os.chdir(path)
os.mkdir(path+'\\noisy_set')
os.mkdir(path+'\\gt_set')

img_rows = 192
img_cols = 256

#image_list=[]

path_list_noise = ['\\0022_NOISY_SRGB\\','\\0025_NOISY_SRGB\\']
#x_test = np.array()
phone = ['N6','G4']
for pl_noise,ph in zip(path_list_noise,phone):
    for filename in glob.glob(path+pl_noise+'*.png'): #assuming png
        img = cv2.imread(filename,-1)
        img = cv2.resize(img,(img_cols,img_rows))
    #    print(img.shape)
    #    cv2.imwrite('img.png',img)
        print(filename)
    #   format img[rows,cols]
        a00=img[:64,:64]
        a01=img[:64,65:128]
        a02=img[:64,129:192]
        a03=img[:64,193:]
        a10=img[65:128,:64]
        a11=img[65:128,65:128]
        a12=img[65:128,129:192]
        a13=img[65:128,193:]
        a20=img[129:,:64]
        a21=img[129:,65:128]
        a22=img[129:,129:192]
        a23=img[129:,193:]
        
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_00_'+filename[-18:],a00)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_01_'+filename[-18:],a01)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_02_'+filename[-18:],a02)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_03_'+filename[-18:],a03)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_10_'+filename[-18:],a10)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_11_'+filename[-18:],a11)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_12_'+filename[-18:],a12)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_13_'+filename[-18:],a13)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_20_'+filename[-18:],a20)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_21_'+filename[-18:],a21)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_22_'+filename[-18:],a22)
        cv2.imwrite(path+'\\noisy_set\\'+ph+'_23_'+filename[-18:],a23)

path_list_gt = ['\\0022_GT_SRGB\\','\\0022_GT_SRGB\\']
phone = ['N6','G4']
for pl_gt,ph in zip(path_list_gt,phone):
    for filename in glob.glob(path+pl_gt+'*.png'): #assuming png
        img = cv2.imread(filename,-1)
    #    print(img.shape)
        img = cv2.resize(img,(img_cols,img_rows))
    #    print(filename)
    #   format img[rows,cols]
    #    print(img.shape)
        a00=img[:64,:64]
        a01=img[:64,65:128]
        a02=img[:64,129:192]
        a03=img[:64,193:]
        a10=img[65:128,:64]
        a11=img[65:128,65:128]
        a12=img[65:128,129:192]
        a13=img[65:128,193:]
        a20=img[129:,:64]
        a21=img[129:,65:128]
        a22=img[129:,129:192]
        a23=img[129:,193:]
        
        cv2.imwrite(path+'\\gt_set\\'+ph+'_00_'+filename[-15:],a00)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_01_'+filename[-15:],a01)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_02_'+filename[-15:],a02)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_03_'+filename[-15:],a03)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_10_'+filename[-15:],a10)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_11_'+filename[-15:],a11)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_12_'+filename[-15:],a12)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_13_'+filename[-15:],a13)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_20_'+filename[-15:],a20)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_21_'+filename[-15:],a21)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_22_'+filename[-15:],a22)
        cv2.imwrite(path+'\\gt_set\\'+ph+'_23_'+filename[-15:],a23)
    
    
    
    
    
    
    
    
