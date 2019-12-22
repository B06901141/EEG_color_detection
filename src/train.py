import os
import numpy as np

from readData import *

import pickle

import tqdm

import keras

def oneHot(x,classNum):
    result = np.zeros([*x.shape] + [classNum])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i][j][x[i,j]] = 1
    return np.swapaxes(result, 1, 2)
def getGradientOutput(x, modelPath = "../model"):
    result = []
    print("getting gradient boosting result...")
    for i in tqdm.tqdm(range(1,61), ncols = 70):
        fileName = os.path.join(modelPath, "model%d.pkl"%i)
        with open(fileName, "rb") as file1:
            model = pickle.load(file1)
        result.append(model.predict(x))
    return np.array(result, dtype=np.int32).T

def genModel():
    input_ = keras.layers.Input(shape = (3, 60))
    output = input_
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(3, kernel_initializer = "ones")(output)
    output = keras.layers.Activation("softmax")(output)

    model = keras.models.Model(input_, output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset")
    x_train = getGradientOutput(x_train, modelPath = "../model")
    x_test = getGradientOutput(x_test, modelPath = "../model")

    x_train = oneHot(x_train, classNum)
    x_test = oneHot(x_test, classNum)

    model = genModel()
    model.summary()

    model.fit(x_train, y_train, epochs = 50)
    print(model.evaluate(x_test, y_test))

    model.save("../model/final.hdf5")
