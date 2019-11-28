import numpy as np
import os
from itertools import permutations

#import scipy.io as sio

def readWave(fileName):
    #TODO
    #data = sio.loadmat("108_1_G3_EEGR2-L03.mat")
    pass
def readDataSet(dataPath):
    color = ["".join(i) for i in permutations("rgb")]
    data = {i:[] for i in color}

    dataFolder = [os.path.join(dataPath, i) for i in os.listdir(dataPath) if i != ".git"]
    dataFolder = [i for i in dataFolder if os.path.isdir(i)]
    matFiles = [[j for j in os.listdir(i) if j.split(".")[-1] == "mat"] for i in dataFolder]

    for folder, matList in zip(dataFolder, matFiles):
        for mat in matList:
            fileName = os.path.join(folder, mat)
            if mat.split("_")[0] not in color:
                print("Warning: File %s doesn't start with either \"%s\", skipping..."%(fileName,"\" or \"".join(color)))
                continue
            data[mat.split("_")[0]].append(readWave(fileName))
    return data

if __name__ == '__main__':
    data = readDataSet("../dataset")
    print(data)
