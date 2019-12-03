import numpy as np
import os

from readData import *

import keras

if __name__ == "__main__":
    data = readDataSet("../dataset")
    data = normalize(data)
    data = splitColor(data)
    data_train, data_test = splitData(data)
    data_train = cropData(data_train, cropSize = 512, step = 50)
    data_test = cropData(data_test, cropSize = 512, step = 50)

    x_train, y_train, index2color = permuteData(data_train)
    x_test, y_test, _ = permuteData(data_test)
    classNum = len(index2color)

    x_train = x_train[..., 1:]
    x_test = x_test[..., 1:]
    model = keras.models.load_model("../model/best.hdf5")
    model.summary()

    print(model.evaluate(x_test, y_test))