import os
import numpy as np
import pickle

from readData import *
import tqdm

from sklearn.ensemble import GradientBoostingClassifier

def gen_train_test(dataPath):
    data = readDataSet("../dataset")
    data = splitColor(data)
    data = filterData(data, filt = [1] + [-1/10] * 10)

    data_train, data_test = splitData(data, testNum = 1)

    data_train = cropData(data_train, cropSize = 128, step = 128)
    data_test = cropData(data_test, cropSize = 128, step = 128)

    x_train, y_train, index2color = permuteData(data_train)
    x_test, y_test, _ = permuteData(data_test)

    x_train = x_train[...,0]
    x_test = x_test[...,0]

    x_train = psd(x_train)
    x_test = psd(x_test)

    classNum = len(index2color)
    return (x_train, y_train), (x_test, y_test), classNum

if __name__ == '__main__':
    modelPath = "../model"
    os.makedirs(modelPath, exist_ok = True)
    (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset")

    print("Using gradient boosting to train model...")
    for i in tqdm.tqdm(range(1,61), ncols = 70):
        fileName = os.path.join(modelPath, "model%d.pkl"%i)
        model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt')
        model.fit(x_train,y_train)
        with open(fileName, "wb") as file1:
            pickle.dump(model, file1)
