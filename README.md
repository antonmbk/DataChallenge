# DataChallenge
Skeleton of a simple image feature extraction framework.

This code was developed on Windows 10 with Anaconda 4.3.1 + Keras 1.2.2.

Should work out of box, however deep-learning-models version .4 must be in the directory.
Get it here:
https://github.com/fchollet/deep-learning-models/archive/v0.4.zip

Instructions:
1) Run extractfeats.py.  It will create a features folder and put collections of features there.

2) Run deepnetplots.py to generate Nearest Neighbor plots for deep networks.  "hidden" in the output image means the model has not top, "cls" means model has top.

3) Run sortbyfeatureplots.py.  This will create a plot of the top ranked images for each non-deep network feature.