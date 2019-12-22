import numpy as np
import os

from readData import *
from train import *

import keras

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset")
    
    model = keras.models.load_model("../model/best.hdf5")
    model.summary()

    print(model.evaluate(x_test, y_test))