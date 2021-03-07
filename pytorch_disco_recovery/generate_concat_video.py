# import tensorflow as tf
# import torch
# import torchvision.transforms
import cv2
import sys
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

name1 = "01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load09"
name2 = "01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load10"
thing = 'emb3D_g'

os = []
for step in list(range(1, 301)):
    fn1 = '%s_%s_vis_%06d.png' % (name1, thing, step)
    fn2 = '%s_%s_vis_%06d.png' % (name2, thing, step)
    fn3 = '%s_rgb_%06d.png' % (name2, step)
    out_fn = '%s_concat_%06d.png' % (thing, step)
    im1 = imageio.imread(fn1)
    im2 = imageio.imread(fn2)
    im3 = imageio.imread(fn3)
    # print(im1.shape)
    # print(im2.shape)
    # print(im3.shape)
    im3 = cv2.resize(im3, (512, 256))#, interpolation=cv2.INTER_BILINAR)
    cat = np.concatenate([im1, im2], axis=1)
    cat = np.concatenate([im3, cat], axis=0)
    imageio.imwrite(out_fn, cat)
    sys.stdout.write('.')
    sys.stdout.flush()

    # scipy.misc.imshow(im1)
#     fn = '%s_%s_%06d.npy' % (name, thing, step)
#     o = np.load(fn)
#     os.append(o)
# os = np.concatenate(os, axis=0)
# print(os.shape)

# B, C, Z, Y, X = list(os.shape)

# os = np.mean(os, axis=3)
# print(os.shape)
# # os is B x C x Z x X


# vec = np.transpose(os, axes=[0, 2, 3, 1])
# vec = np.reshape(vec, (-1, C))
# P = PCA(3)
# P.fit(vec)
# vec = P.transform(vec)
# print(vec.shape)

# im = vec.reshape(B, Z, X, 3)
# for ind, i in enumerate(im):
#     i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_NEAREST)
#     # i = numpy.array(Image.fromarray(i).resize())
#     # i = scipy.misc.imresize(i, 
#     imageio.imwrite(out_fn, i)
#     print('saved %s' % out_fn)

# # out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
