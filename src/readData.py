import numpy as np
import os
from itertools import permutations

import tqdm
import scipy.io as sio

"""
mat format
array(['uV', 'uV', 'uV', 'uV', 'uV']
array(['EEG', 'alpha', 'beta', 'delta', 'theta']
5 ms
200 Hz
"""
"""
0-10 rest
10-20 color1
20-25 rest
25-35 color2
35-40 rest
40-50 color3
"""

time2index = lambda s,f=200: int(s*f)
def readDataSet(dataPath, cmap ="rgb"):
    color = ["".join(i) for i in permutations(cmap)]
    data = {i:[] for i in color}

    dataFolder = [os.path.join(dataPath, i) for i in os.listdir(dataPath) if i != ".git"]
    dataFolder = [i for i in dataFolder if os.path.isdir(i)]
    matFiles = [[j for j in os.listdir(i) if j.split(".")[-1] == "mat"] for i in dataFolder]

    print("Reading data...")
    for folder, matList in zip(tqdm.tqdm(dataFolder, ncols = 70), matFiles):
        for mat in matList:
            fileName = os.path.join(folder, mat)
            colorOrder = mat.split(".")[0].split("_")[-1]
            if colorOrder not in color:
                print("Warning: File %s doesn't end with either \"%s\", skipping..."%(fileName,"\" or \"".join(color)))
                continue
            data[colorOrder].append(sio.loadmat(fileName)["data"])
    return data
def normalize(data):
    print("Normalizing data...")
    for waveform in tqdm.tqdm(data.values(), ncols = 70):
        for i in range(len(waveform)):
            base = waveform[i][:time2index(9.5)]
            mean = base.mean(axis = 0)
            std = base.std(axis = 0)
            waveform[i] = (waveform[i] - mean) / std
    return data
def splitColor(data, cmap = "rgb"):
    d = {i:[] for i in cmap}
    d["dummy"] = []
    time_start = np.array([10,25,40,0]) + 1
    time_end = np.array([20,35,50,10]) - 1
    index_start = [time2index(i) for i in time_start]
    index_end = [time2index(i) for i in time_end]
    for color, waveform in data.items():
        for i in range(3):
            d[color[i]].extend([wave[index_start[i]:index_end[i]] for wave in waveform])
        i = -1
        d["dummy"].extend([wave[index_start[i]:index_end[i]] for wave in waveform])
    for i in d:
        d[i] = np.array(d[i], dtype=np.float32)
    return d
def splitData(data, testNum = 6):
    test_data = {i:j[-testNum:, ...] for i,j in data.items()}
    train_data = {i:j[:-testNum, ...] for i,j in data.items()}
    return train_data, test_data
def cropData(data, cropSize = 512, step = 256):
    for color in data:
        totalTime = data[color].shape[1]
        cropped = []
        for time in range(0,totalTime-cropSize,step):
            cropped.extend(data[color][:,time:time+cropSize,:])
        data[color] = np.array(cropped, dtype=np.float32)
    return data
def permuteData(data, color = "rgb", useDummy = False):
    color = [*color]
    if useDummy:
        color.append("dummy")
    index2color = {i:j for i,j in enumerate(color)}
    color2index = {j:i for i,j in index2color.items()}
    x = []
    y = []
    for c in color:
        index = color2index[c]
        x.extend(data[c])
        y.extend([index]*data[c].shape[0])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]

    return x, y , index2color

if __name__ == '__main__':
    data = readDataSet("../dataset")
    data = normalize(data)
    data = splitColor(data)
    data_train, data_test = splitData(data)
    data_train = cropData(data_train)
    data_test = cropData(data_test)
    x_train, y_train, index2color = permuteData(data_train)
    print(data_train.keys())
    for wave in data_train.values():
        print(wave.shape) #(datanum, time, channel)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
