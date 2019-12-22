import numpy as np
import os
import matplotlib.pyplot as plt

from readData import *

def fft(x):
    freq = np.fft.fftfreq(x.shape[0])
    p = np.argsort(freq)
    freq = freq[p]
    X = np.fft.fft(x)[p]
    return freq, np.abs(X)

def read_history(fileName):
    with open(fileName, 'r') as file1:
        s = file1.read().split("\n")
    label = s[0].split(",")
    value = [i.split(",") for i in s[1:] if "," in i]
    value = np.array(value, dtype=np.float32)
    return {i:j for i,j in zip(label,value.T)}
def plot_history(history, filename):
    plt.subplot(2,1,1)
    plt.plot(history["loss"],label = "loss")
    plt.plot(history["val_loss"],label = "val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend(loc = "best")
    

    plt.subplot(2,1,2)
    plt.plot(history["acc"],label = "acc")
    plt.plot(history["val_acc"],label = "val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("acc")
    plt.axhline(y=0.333,color='g')
    plt.legend(loc = "best")

    plt.tight_layout()
    plt.suptitle("Training history")
    plt.gcf().set_size_inches((16,8))
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    history = read_history("history.csv")
    plot_history(history, "multi.jpg")
    """
    path = "../graphic"
    os.makedirs(path, exist_ok=True)

    data = readDataSet("../dataset")
    data = normalize(data)
    data = splitColor(data)
    data_train, data_test = splitData(data)
    data_train, data_val = splitData(data_train)

    data_train = cropData(data_train, cropSize = 512, step = 50)
    data_val = cropData(data_val, cropSize = 512, step = 50)
    data_test = cropData(data_test, cropSize = 512, step = 50)

    x_train, y_train, index2color = permuteData(data_train)
    x_val, y_val, _ = permuteData(data_val)
    x_test, y_test, _ = permuteData(data_test)


    print(y_train[:10])
    print(y_val[:10])
    print(y_test[:10])

    find_index = lambda y,num, j = 0: np.argwhere(y==num)[j][0]


    for i in range(3):
        for k, (name, x, y) in enumerate(zip(["train", "val", "test"],
                                [x_train, x_val, x_test], [y_train, y_val, y_test])):
            x = x[...,0]
            for j in range(3):
                index = find_index(y,i,j)
                plt.subplot(3,3,1+k*3+j)
                plt.title(name)
                freq, X = fft(x[index])
                plt.plot(freq, X)
                plt.xlabel("freq")
                plt.ylabel("value")
        plt.tight_layout()
        plt.suptitle(index2color[i])
        plt.gcf().set_size_inches((16,8))
        plt.savefig(os.path.join(path,"%s.jpg"%index2color[i]))
        plt.clf()
    """
