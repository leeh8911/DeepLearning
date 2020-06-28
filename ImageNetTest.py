import matplotlib.pylab as plt
import numpy as np
import math
import json

import tensorflow as tf

from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getModel(img_rows, img_cols, img_chl):
    
    inputs = keras.Input(shape = (img_rows, img_cols, img_chl))

    x1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', input_shape = (img_rows, img_cols, img_chl))(inputs)
    y1 = keras.layers.MaxPool2D(pool_size = (2,2))(x1)

    x2 = keras.layers.Conv2D(16, 5, activation = 'relu', padding = 'same', input_shape = (img_rows, img_cols, img_chl))(inputs)
    y2 = keras.layers.MaxPool2D(pool_size = (2,2))(x2)

    x3 = keras.layers.Conv2D(16, 7, activation = 'relu', padding = 'same', input_shape = (img_rows, img_cols, img_chl))(inputs)
    y3 = keras.layers.MaxPool2D(pool_size = (2,2))(x3)

    x23 = keras.layers.Conv2D(16, 5,activation = 'relu', padding = 'same')(tf.add(x2, x3))
    x123 = keras.layers.Conv2D(16, 5,activation = 'relu', padding = 'same')(tf.add(x1, x23))
    y2 =  keras.layers.MaxPool2D(pool_size = (2,2))(x123)

    x1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same')(x123)
    y1 = keras.layers.MaxPool2D(pool_size = (2,2))(x1)

    x2 = keras.layers.Conv2D(8, 5, activation = 'relu', padding = 'same')(x123)
    y2 = keras.layers.MaxPool2D(pool_size = (2,2))(x2)

    x3 = keras.layers.Conv2D(8, 7, activation = 'relu', padding = 'same')(x123)
    y3 = keras.layers.MaxPool2D(pool_size = (2,2))(x3)

    x3 = keras.layers.Conv2D(8, 9, activation = 'relu', padding = 'same')(x123)
    y3 = keras.layers.MaxPool2D(pool_size = (2,2))(x3)

    x4 = keras.layers.Conv2D(8, 11, activation = 'relu', padding = 'same')(x123)
    y3 = keras.layers.MaxPool2D(pool_size = (2,2))(x4)

    x34 = keras.layers.Conv2D(8, 3,activation = 'relu', padding = 'same')(tf.add(x3, x4))
    x234 = keras.layers.Conv2D(8, 3,activation = 'relu', padding = 'same')(tf.add(x2, x34))
    x1234 = keras.layers.Conv2D(8, 3,activation = 'relu', padding = 'same')(tf.add(x1, x234))
    y2 =  keras.layers.MaxPool2D(pool_size = (2,2))(x1234)

    flat_1 = keras.layers.Flatten()(y2)
    outputs = keras.layers.Dense(100, activation = 'softmax')(flat_1)
    outputs = keras.layers.Dense(10, activation = 'softmax')(outputs)
    model = keras.models.Model(inputs = inputs, outputs = outputs)

    model.summary()

    return model

# load datasets
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
img_chl = x_train.shape[3]
# x_train, x_test : [N, 32, 32, 3], uint8(0 ~ 255)
# y_train, y_test : N, (1 ~ 10)

# data normalization
nx_train = np.zeros(x_train.shape)
nx_test = np.zeros(x_test.shape)
x_train_mean = np.zeros(x_train.shape[3])
x_train_std = np.zeros(x_train.shape[3])

for i in range(3):
    x_train_mean[i] = np.mean(x_train[:,:,:,i])
    x_train_std[i] = np.std(x_train[:,:,:,i])
    
for i in range(3):
    nx_train[:,:,:,i] = x_train[:,:,:,i] - x_train_mean[i]
    nx_train[:,:,:,i] = nx_train[:,:,:,i] / x_train_std[i]
    nx_test[:,:,:,i] = x_test[:,:,:,i] - x_train_mean[i]
    nx_test[:,:,:,i] = nx_test[:,:,:,i] / x_train_std[i]

x_train = nx_train
x_test = nx_test

model = getModel(img_rows, img_cols, img_chl)

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

callback_checkpoint = keras.callbacks.ModelCheckpoint("./check_points", monitor = 'val_loss', verbose = 0, save_best_only = False, mode='auto', save_freq=1)
callback_tfboard = keras.callbacks.TensorBoard(log_dir='./logs', profile_batch = 100000000)
MyCallbacks = [
callback_checkpoint, 
callback_tfboard
]
hist = model.fit(x_train, y_train, batch_size = 100, epochs = 100, validation_data=(x_test, y_test), callbacks = MyCallbacks)

fig, ax_loss = plt.subplots()
ax_acc = ax_loss.twinx()

ax_loss.plot(hist.history['loss'], 'y', label = 'train loss')
ax_loss.plot(hist.history['val_loss'], 'r', label = 'valid loss')

ax_acc.plot(hist.history['accuracy'], 'b', label = 'train acc')
ax_acc.plot(hist.history['val_accuracy'], 'g', label = 'valid acc')

plt.show()


model.predict(x_test, y_test)


