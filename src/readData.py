import numpy as np
import os
from itertools import permutations

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
            data[mat.split("_")[0]].append(sio.loadmat(fileName)["data"])
    return data
def normalize(data):
    #TODO
    pass

if __name__ == '__main__':
    data = readDataSet("../dataset")
    print(data)
