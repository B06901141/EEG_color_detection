import numpy as np
import os

from readData import *

import keras

def gen_train_test(dataPath):
    data = readDataSet("../dataset")
    data = normalize(data)
    data = splitColor(data)

    data_train, data_test = splitData(data, testNum = 1)

    data_train = cropData(data_train, cropSize = 128, step = 128)
    data_test = cropData(data_test, cropSize = 128, step = 128)

    x_train, y_train, index2color = permuteData(data_train)
    x_test, y_test, _ = permuteData(data_test)
    classNum = len(index2color)
    return (x_train, y_train), (x_test, y_test), classNum
def save_history(history, filename):
    with open(filename,"w") as file1:
        file1.write("loss,acc,val_loss,val_acc\n")
        loss = history.history["loss"]
        acc = history.history["acc"]
        val_loss = history.history["val_loss"]
        val_acc = history.history["val_acc"]
        for l,a,vl,va in zip(loss,acc,val_loss,val_acc):
            file1.write("%f,%f,%f,%f\n"%(l,a,vl,va))

if __name__ == "__main__":
    os.makedirs("../model", exist_ok=True)
    (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset")
    

    #x_train = x_train[..., 1:]
    #x_val = x_val[..., 1:]
    #x_test = x_test[..., 1:]

    #x_train = eigenSplit(x_train)
    #x_val = eigenSplit(x_val)
    #x_test = eigenSplit(x_test)

    input_ = keras.layers.Input(shape=(128, 5))
    output = input_
    """
    output = keras.layers.Conv1D(32, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(32, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)

    output = keras.layers.Conv1D(64, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(64, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    
    output = keras.layers.Conv1D(128, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Conv1D(128, 10, strides=1)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.MaxPooling1D()(output)
    """
    output = keras.layers.Bidirectional(
        keras.layers.CuDNNGRU(50, return_sequences=True))(output)
    output = keras.layers.Activation("relu")(output)
    ooutput = keras.layers.Bidirectional(
        keras.layers.CuDNNGRU(50, return_sequences=True))(output)
    output = keras.layers.Activation("relu")(output)
    output = keras.layers.Bidirectional(
        keras.layers.CuDNNGRU(50, return_sequences=True))(output)
    output = keras.layers.Activation("relu")(output)
    output = keras.layers.Bidirectional(
        keras.layers.CuDNNGRU(50, return_sequences=False))(output)
    output = keras.layers.Activation("relu")(output)
    
    #output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(100)(output)
    output = keras.layers.LeakyReLU(alpha=0.0)(output)
    output = keras.layers.Dense(100)(output)
    output = keras.layers.LeakyReLU(alpha=0.0)(output)
    output = keras.layers.Dense(classNum, activation = "softmax", kernel_regularizer=keras.regularizers.l1_l2(l1 = 0.00, l2 = 0.01))(output)

    model = keras.models.Model(input_, output)
    model.summary()
    #input()
    
    checkpoint = keras.callbacks.ModelCheckpoint("../model/best.hdf5", monitor='val_loss', save_best_only=True, mode='auto')
    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
    
    pretrain_y = np.ones((y_train.shape[0], classNum))/classNum
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["mse"])
    model.fit(x_train,pretrain_y, epochs = 1)
    
    opt = keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    history = model.fit(x_train,y_train,validation_split = 0.1, callbacks=[checkpoint], epochs = 50)
    save_history(history,"history.csv")
