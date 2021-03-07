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
from sklearn.decomposition import PCA

def pca_embed(emb, keep):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + EPS
    #emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

    emb_reduced = list()

    B, H, W, C = np.shape(emb)
    for img in emb:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        pixelskd = np.reshape(img, (H*W, C))
        P = PCA(keep)
        P.fit(pixelskd)
        pixels3d = P.transform(pixelskd)
        out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

# name = "01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load09"
name = "01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load10"
thing = 'emb3D_g'

os = []
for step in list(range(1, 301)):
    fn = '%s_%s_%06d.npy' % (name, thing, step)
    o = np.load(fn)
    os.append(o)
os = np.concatenate(os, axis=0)
print(os.shape)

B, C, Z, Y, X = list(os.shape)

os = np.mean(os, axis=3)
print(os.shape)
# os is B x C x Z x X


vec = np.transpose(os, axes=[0, 2, 3, 1])
vec = np.reshape(vec, (-1, C))
P = PCA(3)
P.fit(vec)
vec = P.transform(vec)
print(vec.shape)

im = vec.reshape(B, Z, X, 3)
for ind, i in enumerate(im):
    out_fn = '%s_%s_vis_%06d.png' % (name, thing, ind+1)
    i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_NEAREST)
    # i = numpy.array(Image.fromarray(i).resize())
    # i = scipy.misc.imresize(i, 
    imageio.imwrite(out_fn, i)
    print('saved %s' % out_fn)

# out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
