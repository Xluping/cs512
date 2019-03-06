import keras
#from keras.preprocessing import image
import numpy as np
#from keras.models import load_model
from keras.models import model_from_json
import os
import cv2
#import tensorflow as tf
#import keras_metrics

save_dir = os.path.join(os.getcwd(), 'saved_models')
cnn_model_path = os.path.join(save_dir, 'AS4_cnn_model.h5')
cnn_model_path_json = os.path.join(save_dir, 'AS4_cnn_model.json')
cnn_model_wt_path = os.path.join(save_dir, 'AS4_cnn_model_wt.h5')
img_x, img_y = 28,28


#def as_keras_metric(method):
#    import functools
#    from keras import backend as K
#    import tensorflow as tf
#    @functools.wraps(method)
#    def wrapper(self, args, **kwargs):
#        """ Wrapper for turning tensorflow metrics into keras metrics """
#        value, update_op = method(self, args, **kwargs)
#        K.get_session().run(tf.local_variables_initializer())
#        with tf.control_dependencies([update_op]):
#            value = tf.identity(value)
#        return value
#    return wrapper
#
#precision = as_keras_metric(tf.metrics.precision)
#recall = as_keras_metric(tf.metrics.recall)


#cnn_model = load_model(cnn_model_path, custom_objects={'precision': keras_metrics.precision(), 'recall': keras_metrics.recall()})
with open(cnn_model_path_json, 'r') as f:
    cnn_model = model_from_json(f.read())
cnn_model.load_weights(cnn_model_wt_path)

img_path = 'D:\\IIT semester 3 - fall 2018\\CS 512 - Computer Vision\\assignments\\4\\test\\5.jpg'


while(True):
    print ("Enter image path:")
    filepath = input()
    
    img = cv2.imread(filepath)
    
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.resize(img_bw, (img_x, img_y))
    
    #img3 = cv2.GaussianBlur(img2, (5, 5), 0)
    (thresh, img3) = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img3 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    cv2.imshow('Original image', img)
    cv2.imshow('Binary image', img3)
    
    img3 = img3.astype('float32')
    img3 /= 255
    
    if keras.backend.image_data_format() == 'channels_first':
        img3 = img3.reshape(1, 1, img_x, img_y)
    else:
        img3 = img3.reshape(1, img_x, img_y, 1)
    
    result = cnn_model.predict(img3)
    
    if np.argmax(result) == 0:
        print ("Even number")
    else:
        print ("Odd number")
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    key1 = cv2.waitKey(10) & 255
    
    if key1 == ord('q') or key1 == 27:
        cv2.destroyAllWindows()
        break


    


