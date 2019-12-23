import os
import numpy as np

import turtle

import scipy.io as sio

import tqdm

def sqeeze(color, waveForm, speed):
    result_color = []
    result_waveform = []
    for i in range(0,color.shape[0]-speed,speed):
        result_color.append(np.mean(color[i:i+speed],axis = 0))
        result_waveform.append(np.mean(waveForm[i:i+speed]))
    color = np.array(result_color, dtype=np.float32)
    waveform = np.array(result_waveform, dtype=np.float32)
    return color, waveform
def generalizeColor(color):
    for i in range(color.shape[0]):
        color[i] = np.mean(color[max(0,i-5):i+5],axis = 0)
        color[i] = color[i] / (np.sum(color[i])+1e-7) * 255
    color[color < 140] = 0
    return color
if __name__ == '__main__':
    speed = 20

    with open("../npy/168_rgb.npy", "rb") as file1:
        color = np.load(file1)
    waveForm = sio.loadmat("../dataset/168/168_rgb.mat")["data"][...,0]

    color, waveForm = sqeeze(color, waveForm, speed)
    color[...,1] *= 0.9
    color = np.array(color*255,dtype=np.uint8)
    color[color < 150] = 0
    color = generalizeColor(color)

    time = 0
    totalTime = waveForm.shape[0] * 0.005 * speed
    turtle.setworldcoordinates(llx = -100,urx = 10, lly = -500, ury = 500)
    turtle.speed(5)
    turtle.colormode(255)
    turtle.penup()
    for i, (wave, c) in enumerate(zip(waveForm, color)):
        if(i % 50 == 0):
            turtle.setpos(i,wave*100)
            turtle.penup()
            turtle.sety(-400)
            turtle.pencolor(0,0,0)
            turtle.write("%.4f"%(time))
            turtle.sety(wave*100)
            turtle.pendown()
        time += 0.005 * speed
        turtle.setpos(i, wave*100)
        turtle.setworldcoordinates(llx = i-100,urx = i+10, lly = -500, ury = 500)
        turtle.pencolor(c)
