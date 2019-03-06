# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:20:22 2018

@author: russi
"""

import cv2
import numpy as np
#import os
import sys
import matplotlib.pyplot as plt

#os.chdir('D:\\IIT semester 3 - fall 2018\\CS 512 - Computer Vision\\assignments\\2')
#print image.shape

#cv2.imshow('image', image)
#cv2.waitKey()
#cv2.destroyAllWindows()
def showImage(win_name, image):
    cv2.destroyAllWindows()
    global curr_image
    curr_image = image
    cv2.imshow(win_name, image)
    global curr_window
    curr_window = win_name

def showImage_over(win_name, image):
    global curr_image
    curr_image = image
    cv2.imshow(win_name, image)
    global curr_window
    curr_window = win_name

def convolve(image,kernel):
    iH, iW = image.shape[:2]
    kH, kW = kernel.shape[:2]
    pad = (kW - 1) /2

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype = "float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            area = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            k = (area * kernel).sum()
            output[y - pad, x - pad] = k

    return output

def smoothingSlider(x):
	global img
	n = cv2.getTrackbarPos('smooth',winName)
	dst = cv2.blur(img,(n+1,n+1))
	showImage_over(winName, dst)    

def smoothingSlider2(X):
    global img
    n = cv2.getTrackbarPos('smooth2',winName2)
    n = n * 2+1
    kernel = np.ones((n,n))/((n)*(n))
    result = convolve(img,kernel)
    showImage_over(winName2,result)

def rgb2grey(image):
    grey = np.zeros((image.shape[0], image.shape[1]))
    for row in range(0,image.shape[0]):
        for col in range(0,image.shape[1]):
            b = image[row,col,0] 
            g = image[row,col,1]
            r = image[row,col,2]
            grey[row][col] = 0.177*b + 0.813*g + 0.011*r
    return cv2.normalize(grey,0,255)

def magnitude(image):
    magnitude = np.zeros((image.shape[0], image.shape[1]))
    sobelX = np.array((
            [1, 2, 0, -2, -1],
            [4, 8, 0, -8, -4],
            [6, 12, 0, -12, -6],
            [4, 8, 0, -8, -4],
            [1, 2, 0, -2, -1]))
    sobelY = np.array((
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]))
    x = cv2.filter2D(image, -1 ,sobelX)
    y = cv2.filter2D(image, -1 ,sobelY)
    dst = np.zeros((image.shape[0], image.shape[1]))
    dst2 = np.zeros((image.shape[0], image.shape[1]))
    x = cv2.normalize(x, dst, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F) 
    y = cv2.normalize(y, dst2, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F) 

    for row in range(0,image.shape[0]):
        for col in range(0,image.shape[1]):
            a = x[row][col]
            b = y[row][col]
            magnitude[row][col] = np.sqrt(sum(a*a + b*b))
    
    return cv2.normalize(magnitude, 0 , 255, norm_type=cv2.NORM_MINMAX)

def rotationSlider(x):
    global img
    rows,cols = img.shape
    theta = cv2.getTrackbarPos('rotate',winName3)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst2 = cv2.warpAffine(img,M,(cols,rows))
    showImage_over(winName3,dst2)

if (len(sys.argv) < 2):
    print "Input file name"
else:
    file = sys.argv[1]
    orig_image = cv2.imread(file)
    global curr_image, prev_ch, curr_window
    curr_image = orig_image
    prev_ch = ''
    curr_window = ''

    showImage('Image', orig_image)
    
    while (True):
        key = cv2.waitKey(10) & 255
        
        if key == ord('i'):
            showImage('Image', orig_image)
            
        if key == ord('w'):
            cv2.imwrite('out.jpg', curr_image)
            
        if key == ord('g'):
            image_bw = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            showImage('Image Grayscale OpenCV', image_bw)
        
        if key == ord('G'):
            img = rgb2grey(orig_image)
            showImage('Image Grayscale Custom.jpg', img)
        
        if key == ord('c'):
            r,g,b = cv2.split(orig_image)
            if prev_ch == 'blue' or prev_ch == '':
                showImage('red channel', r)
                prev_ch = 'red'
            elif prev_ch == 'red':
                showImage('green channel', g)
                prev_ch = 'green'
            elif prev_ch == 'green':
                showImage('blue channel', b)
                prev_ch = 'blue'
                
        if key == ord('s'):
            winName = "Smoothing OpenCV"
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            showImage(winName, img)
            cv2.createTrackbar('smooth',winName,0,10,smoothingSlider)
        
        if key == ord('S'):
            winName2 = "Smoothing Custom"
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            showImage(winName2, img)
            cv2.createTrackbar('smooth2',winName2,0,2,smoothingSlider2)
        
        if key == ord('d'):
            img = orig_image[::2,::2]
            showImage('Downsample without Smoothing', img)
        
        if key == ord('D'):
            img = cv2.pyrDown(orig_image)
            showImage('Downsample with Smoothing', img)
        
        if key == ord('x'):
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            sobelX = np.array((
                    [1, 2, 0, -2, -1],
                    [4, 8, 0, -8, -4],
                    [6, 12, 0, -12, -6],
                    [4, 8, 0, -8, -4],
                    [1, 2, 0, -2, -1]))
            img = cv2.filter2D(img, -1 ,sobelX)
            dst = np.zeros((img.shape[0], img.shape[1]))
            img = cv2.normalize(img, dst, 0, 255, norm_type=cv2.NORM_MINMAX) 
            showImage('X-derivative', img)
        
        if key == ord('y'):
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            sobelY = np.array((
                    [-1, -4, -6, -4, -1],
                    [-2, -8, -12, -8, -2],
                    [0, 0, 0, 0, 0],
                    [2, 8, 12, 8, 2],
                    [1, 4, 6, 4, 1]))
            img = cv2.filter2D(img, -1 ,sobelY)
            dst = np.zeros((img.shape[0], img.shape[1]))
            img = cv2.normalize(img, dst, 0, 255, norm_type=cv2.NORM_MINMAX) 
            showImage('Y-derivative', img)
        
        if key == ord('m'):
            img = magnitude(orig_image)
            showImage('Magnitude of Gradient', img)
        
        if key == ord('p'):
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
            dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
            img = plt.quiver(dx, -dy)
            plt.show(img)	
        
        if key == ord('r'):
            winName3 = "Rotation"
            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            showImage(winName3, img)
            cv2.createTrackbar('rotate', winName3, 0, 360, rotationSlider)
        
        if key == ord('h'):
            print 'Help menu:'
            print 'i - Reload the original image i.e cancel any processing'
            print 'w - Save the current image as "out.jpg"'
            print 'g - Convert image into Grayscale using OpenCV'
            print 'G - Convert image into Grayscale using Conversion Function'
            print 'c - Cycle through the different color channels of the image'
            print 's - Convert image into Grayscale and smooth using OpenCV'
            print 'S - Convert image into Grayscale and smooth using Smoothing Function'
            print 'd - Downsample the image by a factor of 2 without Smoothing'
            print 'D - Downsample the image by a factor of 2 with Smoothing'
            print 'x - Convert image into Grayscale and take X-derivatives'
            print 'y - Convert image into Grayscale and take Y-derivatives'
            print 'm - Display the magnitude of the image gradient'
            print 'p - Convert image into Grayscale and plot gradient vectors'
            print 'r - Convert image into Grayscale and rotate the image'
            print 'h - Display the Help Menu'

        if key == 27:
            cv2.destroyAllWindows()
            break
            
            
            
            