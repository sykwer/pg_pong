import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pre_process_img(img):
    img = img[35:195] # crop
    img = img[::2, ::2, 0] # downsample
    img[img == 144] = 0 # erase background
    img[img == 109] = 0 # erase background
    img[img != 0] = 1 # paddles and ball set to 1
    return img.astype(np.float).ravel()
