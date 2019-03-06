#import numpy as np
#import scipy
#import h5py
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
import os
import keras_metrics

num_classes=2
batch_size=128
epochs=5
img_x, img_y = 28,28
save_dir = os.path.join(os.getcwd(), 'saved_models')

#def load_data(path='../input/mnist/mnist.npz'):
##     path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
#    f = np.load('../input/mnist/mnist.npz')
#    x_train = f['x_train']
#    y_train = f['y_train']
#    x_test = f['x_test']
#    y_test = f['y_test']
#    f.close()
#    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
for i in range(0, len(y_train)):
    if (y_train[i]%2 == 0):
        y_train[i] = 0
    else:
        y_train[i] = 1

for i in range(0, len(y_test)):
    if (y_test[i]%2 == 0):
        y_test[i] = 0
    else:
        y_test[i] = 1


if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_x, img_y)
    x_test = x_test.reshape(x_test.shape[0], 1, img_x, img_y)
    input_shape = (1, img_x, img_y)
else:
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes = 10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes = 10)


cnn_model = Sequential()

cnn_model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, (7, 7), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.4))
cnn_model.add(Flatten())
cnn_model.add(Dense(num_classes, activation='softmax'))


#model.load_weights(model_wt_path, by_name=True)
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

sgd = SGD(lr=0.001)

cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

#class AccuracyHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.acc = []
#
#    def on_epoch_end(self, batch, logs={}):
#        self.acc.append(logs.get('acc'))
#
#history = AccuracyHistory()

history = cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#score = cnn_model.evaluate(x_test, y_test, verbose=1)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
cnn_model_path = os.path.join(save_dir, 'AS4_cnn_model.h5')
cnn_model_path_json = os.path.join(save_dir, 'AS4_cnn_model.json')
cnn_model_wt_path = os.path.join(save_dir, 'AS4_cnn_model_wt.h5')
#model.save(model_path)
cnn_model.save_weights(cnn_model_wt_path)

with open(cnn_model_path_json, 'w') as f:
    f.write(cnn_model.to_json())
    

import matplotlib.pyplot as plt
# Plot training accuracy values
plt.plot(history.history['acc'], marker='o')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training loss values
plt.plot(history.history['loss'], marker='o')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot testing accuracy values
plt.plot(history.history['val_acc'], marker='o')
plt.title('Evaluation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()

# Plot testing loss values
plt.plot(history.history['val_loss'], marker='o')
plt.title('Evaluation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()

# Plot testing precision values
plt.plot(history.history['val_precision'], marker='o')
plt.title('Evaluation Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()

# Plot testing recall values
plt.plot(history.history['val_recall'], marker='o')
plt.title('Evaluation Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()




