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

    x = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', input_shape = (img_rows, img_cols, img_chl))(inputs)
    y = keras.layers.MaxPool2D(pool_size = (2,2))(x)
    shortcut = y

    x = keras.layers.Conv2D(64, 5, activation = 'relu', padding = 'same')(y)
    y = keras.layers.MaxPool2D(pool_size = (2,2))(x)

    x = keras.layers.Conv2D(64, 7, activation = 'relu', padding = 'same')(y)
    y = keras.layers.MaxPool2D(pool_size = (2,2))(x)

    shortcut = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(shortcut)
    shortcut = keras.layers.MaxPool2D(pool_size = (4,4))(shortcut)

    y = keras.layers.Add()([y, shortcut])

    x = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(y)
    y = keras.layers.MaxPool2D(pool_size = (2,2))(x)

    flat_1 = keras.layers.Flatten()(y)
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
opt = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = opt)

callback_checkpoint = keras.callbacks.ModelCheckpoint("./check_points", monitor = 'val_loss', verbose = 1, save_best_only = False, mode='auto', save_freq=10)
callback_tfboard = keras.callbacks.TensorBoard(log_dir='./logs', profile_batch = 100000000, histogram_freq=0, write_graph=True, write_images=True)
callback_reduceLR =  keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # 검증 손실을 기준으로 callback이 호출됩니다
    factor=0.5,          # callback 호출시 학습률을 1/2로 줄입니다
    patience=10,         # epoch 10 동안 개선되지 않으면 callback이 호출됩니다
)
MyCallbacks = [
callback_checkpoint, 
callback_tfboard,
#callback_reduceLR
]
hist = model.fit(x_train, y_train, batch_size = 1000, epochs = 1000, validation_data=(x_test, y_test), callbacks = MyCallbacks)

fig, ax_loss = plt.subplots()
ax_acc = ax_loss.twinx()

ax_loss.plot(hist.history['loss'], 'y', label = 'train loss')
ax_loss.plot(hist.history['val_loss'], 'r', label = 'valid loss')

ax_acc.plot(hist.history['accuracy'], 'b', label = 'train acc')
ax_acc.plot(hist.history['val_accuracy'], 'g', label = 'valid acc')

plt.show()


model.predict(x_test, y_test)


