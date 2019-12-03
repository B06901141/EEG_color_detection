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
    data_train, data_val = splitData(data_train)

    data_train = cropData(data_train, cropSize = 512, step = 50)
    data_val = cropData(data_val, cropSize = 512, step = 50)
    data_test = cropData(data_test, cropSize = 512, step = 50)

    x_train, y_train, index2color = permuteData(data_train)
    x_val, y_val, _ = permuteData(data_val)
    x_test, y_test, _ = permuteData(data_test)
    classNum = len(index2color)

    x_train = x_train[..., 1:]
    x_val = x_val[..., 1:]
    x_test = x_test[..., 1:]

    x_train = eigenSplit(x_train)
    x_val = eigenSplit(x_val)
    x_test = eigenSplit(x_test)

    input_ = keras.layers.Input(shape=(512, 4))
    output = input_
    """
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    #output = keras.layers.Dropout(0.8)(output)
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(3, 10, strides=1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    #output = keras.layers.TimeDistributed(keras.layers.Dense(100))(output)
    """
    #output = keras.layers.Conv1D(50, 10, strides=1)(output)
    output = keras.layers.CuDNNLSTM(50, return_sequences=True)(output)
    output = keras.layers.CuDNNLSTM(50, return_sequences=True)(output)
    output = keras.layers.CuDNNLSTM(50, return_sequences=True)(output)
    output = keras.layers.CuDNNLSTM(50, return_sequences=False)(output)
    
    #output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(50)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dense(50)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dense(classNum, activation = "softmax")(output)

    model = keras.models.Model(input_, output)
    model.summary()
    #input()
    
    checkpoint = keras.callbacks.ModelCheckpoint("../model/best.hdf5", monitor='val_loss', save_best_only=True, mode='auto')
    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
    
    pretrain_y = np.ones((y_train.shape[0], classNum))/classNum
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["mse"])
    model.fit(x_train,pretrain_y, epochs = 5)
    
    opt = keras.optimizers.RMSprop(0.001, clipvalue=0.5)
    model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    model.fit(x_train,y_train,validation_data = (x_val,y_val), callbacks=[checkpoint], epochs = 300)
