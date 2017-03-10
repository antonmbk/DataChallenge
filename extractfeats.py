import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from glob import glob
import sys

if not path.isdir('deep-learning-models-0.4'):
    warnings.warn("Cannot find deep-learning-models-0.4 in this directory")
    raise SystemExit
sys.path.append(path.join('deep-learning-models-0.4'))
from imagenet_utils import decode_predictions, preprocess_input
from resnet50 import ResNet50
from vgg19 import VGG19

from challenge_utils import *

# Create a working subdirectory
workdir = 'features'
if not os.path.exists(workdir):
    os.makedirs(workdir)

# The feature dict that will determine what features to extract
featuredict = [('ResNet50_hidden', getpredfun(ResNet50, include_top=False)),
               ('ResNet50_cls', getpredfun(ResNet50, include_top=True)),
               ('VGG19_hidden', getpredfun(VGG19, include_top=False)),
               ('VGG19_cls', getpredfun(VGG19, include_top=True)),
               ('border', border),
               ('border_pix5', lambda x:border(x, pix=5)),
               ('border_pix10', lambda x:border(x, pix=10)),
               ('b&w', blackandwhite),
               ('b&w2', blackandwhite2)]
          
featuredict = dict(featuredict)

batch_size = 1000

files = glob(path.join('..', 'data', 'images', '*'))
files = np.array(files)

np.save(path.join(workdir, 'files.npy'), files)

nimgs = len(files)

img0 = prepare(files[0])

# Loop through each feature
for k, v in featuredict.items():
    feats0 = v(img0)
    feats = np.zeros((nimgs, feats0.shape[1]))
    # Loop through all the images
    for i in range(0, nimgs, batch_size):
        print i, nimgs
        nimgs_i = np.minimum(feats[i:i + batch_size].shape[0], batch_size)
        imgs_batch = np.nan * np.ones((nimgs_i, ) + img0.shape[1:], dtype='float32')
        # Accumulate a batch of images
        for j in range(0, nimgs_i):
            try:
                imgs_batch[[j]] = prepare(files[i + j])
            except:
                pass
        # Calculate features
        feats[i:i + batch_size] = v(imgs_batch)
        feats[np.any(np.isnan(imgs_batch), axis=(-3, -2, -1))] = np.nan
    np.save(path.join(workdir, 'features_' + k + '.npy'), feats)
