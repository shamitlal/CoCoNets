# import tensorflow as tf
# import torch
# import torchvision.transforms
import cv2
# import os
import numpy as np
# from matplotlib import cm
# import hyperparams as hyp
# import utils_geom
# import matplotlib
import imageio
import scipy.misc
# from itertools import combinations
# from tensorboardX import SummaryWriter

# from utils_basic import *
# import utils_basic
# from sklearn.decomposition import PCA

# name = "CameraRGB11"
# name = "CameraRGB10"
name = "CameraRGB14"

os = []
for step in list(range(80, 280)):
    fn = '%s/%06d.npy' % (name, step)
    o = np.load(fn)
    os.append(o)
os = np.stack(os, axis=0)
print(os.shape)

B, C, H, W = list(os.shape)

for ind, im in enumerate(os):
    out_fn = '%s_vis_%06d.png' % (name, ind)
    # i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_NEAREST)
    # i = numpy.array(Image.fromarray(i).resize())
    # i = scipy.misc.imresize(i, 
    imageio.imwrite(out_fn, im)
    print('saved %s' % out_fn)

# out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
