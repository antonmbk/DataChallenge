import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from os import path
from glob import glob

from challenge_utils import *

# The working directory where the data was dumped
workdir = 'features'
# n is the number of images along each image dimension
n = 10
featurelist = ['border', 'border_pix5', 'border_pix10', 'b&w', 'b&w2']

# Load the filenames
files = np.load(path.join(workdir, 'files.npy'))

# The features we have calculated are stored as features*.npy files
features_files = glob(path.join(workdir, 'features_*.npy'))

# Plots of images sorted by feature
for featfile in features_files:
    feature_name = path.splitext(path.split(featfile)[1])[0]
    if feature_name.split('_')[1] in featurelist:
        F = np.load(featfile)
        F_notnan = F[~np.isnan(F.ravel())].ravel()
        files_notnan = files[~np.isnan(F.ravel())].ravel()
        filesforimg = files_notnan[np.argsort(F_notnan)]
        j = 0
        filled = False
        imgs_plot = []
        while not filled:
            img = colormap(prepare(filesforimg[j], preprocess=False))
            if ~np.any(np.isnan(img)):
                imgs_plot.append(img)
            if len(imgs_plot) == n ** 2:
                filled = True
            j += 1                
        imgs_plot = np.concatenate(imgs_plot, axis=0)
        imgs_plot = imgs_plot.reshape((n, n, ) + imgs_plot.shape[-3:])
        imgs_plot = np.hstack(np.hstack(imgs_plot.transpose(0, 1, 3, 4, 2)))
        imgfnameout = path.join(workdir, feature_name + '.png')
        misc.imsave(imgfnameout, (imgs_plot * 255.).astype('uint8'))
