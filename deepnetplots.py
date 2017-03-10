import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
from os import path
from glob import glob
import sys

from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

sys.path.append(path.join('deep-learning-models-master'))
from imagenet_utils import decode_predictions, preprocess_input
from resnet50 import ResNet50
from vgg19 import VGG19

from challenge_utils import *

# The working directory where the data was dumped
workdir = 'features'

# Templates are 5 random images and n is the number of neighbors
ntemplates = 10
n = 10

# Load the filenames
files = np.load(path.join(workdir, 'files.npy'))

# Randomly pick some images (some are corrupted so use while loop)
template_inds = np.random.choice(len(files), size=len(files))
j = 0
filled = False
templates = []
template_ind_mem = []
while not filled:
    img = colormap(prepare(files[template_inds[j]], preprocess=False))
    if ~np.any(np.isnan(img)):
        templates.append(img)
        template_ind_mem.append(template_inds[j])
    if len(templates) == n ** 2:
        filled = True
    j += 1

# The features we have calculated are stored as features*.npy files
features_files = glob(path.join(workdir, 'features_*.npy'))

# Nearest neighbor plots
# Go through each feature collection and find template nearest neighbors
for featfile in features_files:
    feature_name = path.splitext(path.split(featfile)[1])[0]
    if feature_name.split('_')[1] in ['ResNet50', 'VGG19']:
        F = np.load(featfile)
        imgs_plot = []
        for i in range(ntemplates):
            template_ind = template_ind_mem[i]
            template = templates[i]
            topn = np.argsort(np.sum((F[files[template_ind] == files] - F) ** 2,
                                     axis=1))[1:n + 1]
            imgs_topn = []
            for x in files[topn]:
                try:
                    imgs_topn.append(prepare(x, preprocess=False))
                except:
                    imgs_topn.append(np.zeros((1, 3, 224, 224), dtype='float32'))
            imgs_topn = np.concatenate(imgs_topn, axis=0)
            imgs_plot_i = colormap(np.concatenate((templates[i], imgs_topn),
                                                  axis=0))
            imgs_plot.append(np.hstack(imgs_plot_i.transpose(0, 2, 3, 1)))
        imgs_plot = np.vstack(imgs_plot)
        imgfnameout = path.join(workdir, feature_name + '.png')
        misc.imsave(imgfnameout, (imgs_plot * 255.).astype('uint8'))
