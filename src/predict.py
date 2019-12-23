import os
import numpy as np

import pickle

import scipy.io as sio
from scipy import signal

import tqdm

import argparse

psd = lambda x: signal.periodogram(x, 200)[1]
filt = lambda arr, filt: np.convolve(arr, filt, mode='same')
def getGradientOutput(x, modelPath = "../model"):
    result = []
    print("getting gradient boosting result...")
    for i in tqdm.tqdm(range(1,61), ncols = 70):
        fileName = os.path.join(modelPath, "model%d.pkl"%i)
        with open(fileName, "rb") as file1:
            model = pickle.load(file1)
        result.append(model.predict(x))
    return np.array(result, dtype=np.int32).T
def oneHot(x,classNum = 3):
    result = np.zeros([*x.shape] + [classNum])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i][j][x[i,j]] = 1
    return np.swapaxes(result, 1, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inMat", help = "input path of the matFile")
    parser.add_argument("outNpy", help = "output path of the npyFile")
    args = parser.parse_args()

    import keras
    model = keras.models.load_model("../model/final.hdf5")
    model.summary()

    waveForm = sio.loadmat(args.inMat)["data"][...,0]
    timeStep = len(waveForm)
    waveForm = [waveForm[0]]*128+[*filt(waveForm, [1]+[-1/10]*10)]
    waveForm = [waveForm[i-128:i] for i in range(128, timeStep+128)]
    PSD = [psd(i) for i in waveForm]
    PSD = np.array(PSD, dtype=np.float32)
    gradientOutput = getGradientOutput(PSD)
    oneHot = oneHot(gradientOutput)

    rgb = model.predict(oneHot)
    with open(args.outNpy, "wb") as file1:
        np.save(file1, rgb)
