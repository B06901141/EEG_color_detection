import os
import numpy as np
import pickle

from readData import *
import tqdm

from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    modelPath = "../model"
    os.makedirs(modelPath, exist_ok = True)
    (x_train, y_train), (x_test, y_test), classNum = gen_train_test("../dataset")

    data_num = x_train.shape[0]
    val = int(data_num * 0.1)
    x_train = x_train[:-val]
    y_train = y_train[:-val]

    print("Using gradient boosting to train model...")
    for i in tqdm.tqdm(range(1,61), ncols = 70):
        fileName = os.path.join(modelPath, "model%d.pkl"%i)
        model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt')
        model.fit(x_train,y_train)
        with open(fileName, "wb") as file1:
            pickle.dump(model, file1)
