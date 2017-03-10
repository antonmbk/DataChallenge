import numpy as np
from keras.preprocessing import image
import sys
from os import path
sys.path.append(path.join('deep-learning-models-master'))
from imagenet_utils import preprocess_input


def prepare(img_path, preprocess=True):
    ''' Code to convert a filename to useable 4 dim image tensor'''
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    except:
        x = np.zeros((1, 3, 224, 224), dtype='float32')
    if preprocess:
        x = preprocess_input(x)
    return x


def colormap(x):
    ''' Normalizes colors to avoid image intensity wraparound'''
    y = x - np.min(x, axis=(-3, -2, -1), keepdims=True)
    y = y / np.max(y, axis=(-3, -2, -1), keepdims=True).astype('float32')
    return y


def getpredfun(modelname, include_top=True):
    ''' Grabs the prediction function from a deep network'''
    model = modelname(include_top=include_top, weights='imagenet')
    if not include_top:
        return lambda x: model.predict(x)[:, :, 0, 0]
    else:
        return model.predict


def border(x, pix=1):
    ''' Calculates variance of border pixels (pix = dist from image edge)'''
    top = np.var(x[:, :, :pix], axis=(-2, -1))
    bot = np.var(x[:, :, -pix:], axis=(-2, -1))
    left = np.var(x[:, :, pix:-pix][:, :, :, :pix], axis=(-2, -1))
    right = np.var(x[:, :, pix:-pix][:, :, :, -pix:], axis=(-2, -1))
    return np.sum((top + bot + left + right).reshape((x.shape[0], -1)),
                  axis=1, keepdims=True)


def blackandwhite(x):
    ''' Calculates if color levels are equal or varied to detect b&w images'''
    return np.mean(np.var(x, axis=1), axis=(-2, -1))[:, None]
