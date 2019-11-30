import numpy as np
import os

from readData import *

import keras

if __name__ == "__main__":
    os.makedirs("../model", exist_ok=True)

    data = readDataSet("../dataset")
    data = normalize(data)
    data = splitColor(data)
    data_train, data_test = splitData(data)
    data_train = cropData(data_train)
    data_test = cropData(data_test)
    
    x_train, y_train, index2color = permuteData(data_train)
    x_test, y_test, _ = permuteData(data_test)
    classNum = len(index2color)

    x_train = x_train[..., 1:]
    x_test = x_test[..., 1:]

    input_ = keras.layers.Input(shape=(512, 4))
    output = input_

    output = keras.layers.Conv1D(64, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(64, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    
    output = keras.layers.Conv1D(128, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(128, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)

    output = keras.layers.Conv1D(256, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(256, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    output = keras.layers.TimeDistributed(keras.layers.Dense(5, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))(output)


    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dense(50)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dense(classNum, activation = "softmax")(output)

    
    model = keras.models.Model(input_, output)
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint("../model/best.hdf5", monitor='val_loss', save_best_only=True, mode='min')
    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
    
    pretrain_y = np.ones((y_train.shape[0], classNum))/classNum
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["mse"])
    model.fit(x_train,pretrain_y,validation_split=0.1, epochs = 10)


    model.compile(optimizer="adam", loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    model.fit(x_train,y_train,validation_split=0.5, callbacks=[checkpoint], epochs = 20000)
