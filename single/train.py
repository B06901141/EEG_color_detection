import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from readData import *

import scipy
from scipy import signal

import keras

def gen_train_test(dataPath, index = 0):
    data = readDataSet(dataPath, index = 0)
    data = normalize(data)
    data = splitColor(data)
    filt = [1] + [-1/10] * 10
    prepro = lambda arr, filt: np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=1, arr=arr)
    data = {i:prepro(j, filt) for i, j in data.items()}

    data_train, data_test = splitData(data, testNum = data["r"].shape[0] - 1)



    data_train = cropData(data_train, cropSize = 128, step = 128)
    data_test = cropData(data_test, cropSize = 128, step = 128)

    x_train, y_train, index2color = permuteData(data_train)
    x_test, y_test, _ = permuteData(data_test)

    x_train = np.apply_along_axis(lambda x: signal.periodogram(x, 200)[1], axis=1, arr=x_train)
    x_test = np.apply_along_axis(lambda x: signal.periodogram(x, 200)[1], axis=1, arr=x_test)

    classNum = len(index2color)

    #x_train = x_train[..., 1:]
    #x_test = x_test[..., 1:]

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

    for index in range(0,7):
        for i in range(10):
            (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset", index = index)


            input_ = keras.layers.Input(shape=(65, 5))
            output = input_
            """
            output = keras.layers.Bidirectional(
                keras.layers.GRU(10, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))(output)
            output = keras.layers.Bidirectional(
                keras.layers.GRU(10, return_sequences=False,
                    kernel_regularizer=keras.regularizers.l1_l2(l1 = 0.01, l2 = 0.01)))(output)
            """
            output = keras.layers.Flatten()(output)
            output = keras.layers.Dense(20)(output)
            output = keras.layers.LeakyReLU(alpha=0.3)(output)
            output = keras.layers.Dense(20)(output)
            output = keras.layers.LeakyReLU(alpha=0.3)(output)
            output = keras.layers.Dense(classNum, activation = "softmax")(output)

            model = keras.models.Model(input_, output)
            model.summary()
            #input()
            
            checkpoint = keras.callbacks.ModelCheckpoint("../model/best.hdf5", monitor='val_loss', save_best_only=True, mode='auto')
            #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
            
            pretrain_y = np.ones((y_train.shape[0], classNum))/classNum
            model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["mse"])
            model.fit(x_train,pretrain_y, epochs = 1)
            
            opt = keras.optimizers.RMSprop(0.001, clipvalue=0.5)
            model.compile(optimizer=opt, loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
            model.fit(x_train,y_train, callbacks=[checkpoint], epochs = 100, validation_split = 0.5)
            if i == 0:
                with open("value_%d.txt"%index,"w") as file1:
                    file1.write("")
            with open("value_%d.txt"%index,"a") as file1:
                file1.write(str(model.evaluate(x_test, y_test)[1])+"\n")
            keras.backend.clear_session()
            exit()

