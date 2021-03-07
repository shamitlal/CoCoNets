from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
# current = 'stage2_builder'
# current = 'stage2_trainer'
# # current = 'stage2_radial_builder'
# current = 'stage2_radial_trainer'

mod = '"00"' # nothing
mod = '"01"' # add a vqrgb net
mod = '"02"' # train a bit
mod = '"03"' # return the right loss
mod = '"04"' # get recon error too
mod = '"05"' # faster logging
mod = '"06"' # bigger B
mod = '"07"' # clamp the output
mod = '"08"' # plot perplexity too
mod = '"09"' # slightly different hyps; print the shape of z; 
mod = '"10"' # deeper encoding (downsamp 8 this time) (slow gpu)
mod = '"11"' # one more 3x3 conv before the residual stack (fast gpu)
mod = '"12"' # run val iters
mod = '"13"' # compute number of used dictionary elements in past pool of 1k
mod = '"14"' # use pool of 10k
mod = '"15"' # cleaned up a bit

mod = '"pr00"' # load pret 04_m64x16x64_1e-3_Vr_r1_l1_mabs7i3t_11 and fire it a bit
mod = '"pr01"' # get ind image
mod = '"pr02"' # same
mod = '"pr03"' # run some test iters; print shapes
mod = '"pr04"' # not volatile
mod = '"pr05"' # don't cast long
mod = '"pr06"' # sample and decode
mod = '"pr07"' # again; don't test so much
mod = '"pr08"' # log less
mod = '"pr09"' # again (faster gpu; fixed a small bug)
mod = '"pr10"' # log less
mod = '"pr11"' # 3 more layers in pixelcnn
mod = '"pr12"' # filter5 instead of 7

mod = '"rad00"' # radial builder
mod = '"rad01"' # install sigen net, as copy of gen2d
mod = '"rad02"' # use sigen net
mod = '"rad03"' # SimpleCNN
mod = '"rad04"' # SingleresCNN; summ_feat(emb)
mod = '"rad05"' # output dict_dim
mod = '"rad06"' # train a bit, just to see if it goes < hm, optimizes slower than gen2dvq 
mod = '"rad07"' # simplecnn arch trying to match gen2dvq
mod = '"rad08"' # relu instead of lrelu < not all params are showing properly
mod = '"rad09"' # modulelist < still no
mod = '"rad10"' # dcaa < ok holy smokes, and this thing can overfit like insanity
mod = '"rad11"' # fewer prints 
mod = '"rad12"' # dropout mask
mod = '"rad13"' # simpleblock; num_layers=10; layer list
mod = '"rad14"' # simpleblock; num_layers=10; layer dict
mod = '"rad15"' # simpleblock; num_layers=10; layer dict but also declare each
mod = '"rad16"' # show mask input and output
mod = '"rad17"' # train and val
mod = '"rad18"' # show confidence; faster logging
mod = '"rad19"' # 20% mask
mod = '"rad20"' # 1 layer
mod = '"rad21"' # proper 20%
mod = '"rad22"' # proper 50%
mod = '"rad23"' # proper 90%
mod = '"rad24"' # proper 50% (coeff 0.5); 6-layer conf; on val iters, don't drop
mod = '"rad25"' # proper 20% (coeff 0.8); 6-layer conf; on val iters, don't drop
mod = '"rad26"' # 20%; run test iters in raster scan
mod = '"rad27"' # 50%
mod = '"rad28"' # 50%; 100k; log less
mod = '"rad29"' # conditional sampling; 50%; log more
mod = '"rad30"' # conditional sampling; 20%; log less
mod = '"rad31"' # plot cond recon error
mod = '"rad32"' # show cond mask
mod = '"rad33"' # cond with 50%; plot cond mask
mod = '"rad34"' # cond with 50%; also show complete recon
mod = '"rad35"' # drop coeff 0.1 in train
mod = '"rad36"' # more logging; figure out hwo to show that mask
mod = '"rad37"' # sample twice
mod = '"rad38"' # drop coeff 0.8 in train
mod = '"rad40"' # cond on less (using coeff 0.9 instead of 0.1); show conds, to see it radiating
mod = '"rad41"' # use a while loop instead of a "for"
mod = '"rad42"' # show gif of sample masks
mod = '"rad43"' # show gif of sample masks
mod = '"rad44"' # show gif of sample masks
mod = '"rad46"' # put x = x * mc ; assert; show first and last masks
mod = '"rad47"' # 300k; 
mod = '"rad48"' # freeze the vq (finally)
mod = '"rad49"' # don't mult by mc


mod = '"rad50"' # slightly cleaner DCAA arch (use conv weights)
mod = '"rad51"' # slightly cleaner DCAA arch (use conv weights[0:1,0:1])
mod = '"rad52"' # for "uncond" use a real topleft 
mod = '"rad53"' # use kernel_size=5
mod = '"rad54"' # concat grid
mod = '"rad55"' # chans=64 (instead of 10!!!!)
mod = '"rad56"' # choice pixels (one per batch el)
mod = '"rad57"' # log less
mod = '"rad58"' # 32 choices
mod = '"rad59"' # 64 choices
mod = '"rad60"' # no additional dropout
mod = '"rad61"' # go up to layer10 (instead of layer6) < bug; no additional layers were used
mod = '"rad62"' # 3x3 instead of 5x5 convs, for slower dilation < lower training error!?
mod = '"rad63"' # saconv, batchnorm, relu
mod = '"rad64"' # 128 choices; 5x5 convs; slower logging
mod = '"rad65"' # 128 choices; 5x5 convs; slower logging; no batchnorm
mod = '"rad66"' # 256 choices; 5x5 convs; slower logging; no batchnorm; actually use all 10 layers finally
mod = '"rad67"' # uncond sample in raster scan order
mod = '"rad68"' # bias=False in each layer
mod = '"rad69"' # no grid
mod = '"rad70"' # yes batchnorm < much worse 
mod = '"rad71"' # 3x3 convs instead of 5x5, for narrower fov
mod = '"rad72"' # fixed a confusing summ (duplicate cond sample); yes grid again, since sometimes i think we end up in the water
mod = '"rad73"' # additional dropout 0.5
mod = '"rad74"' # no drop; last four convs standard
mod = '"rad75"' # heavier (3x3) input mask, so that bigger context is req
mod = '"rad76"' # 64 choices
mod = '"rad77"' # kernel size 5
mod = '"rad78"' # take over two more layers (converting to dense)



mod = '"c00"' # start on training the classifier a bit

        
############## define experiment ##############

exps['builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_vqrgb',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_val_data', # dataset
    # # 'carla_multiview_train10_val10_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '100k_iters',
    'lr3',
    'B4',
    'train_vqrgb',
    'faster_logging',
]
exps['stage2_builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'train_vqrgb',
    'train_gen2dvq',
    'pretrained_vqrgb',
    'fastest_logging',
]
exps['stage2_trainer'] = [
    'carla_vqrgb', # mode
    # 'carla_multiview_train_val_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B4',
    'pretrained_vqrgb',
    'train_gen2dvq',
    'slow_logging',
]
exps['stage2_radial_builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'pretrained_vqrgb',
    'train_sigen2d',
    'fastest_logging',
]
exps['stage2_radial_trainer'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_val_test_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B8',
    'pretrained_vqrgb',
    'frozen_vqrgb', 
    'train_sigen2d',
    'slower_logging',
]

############## net configs ##############

groups['train_vqrgb'] = [
    'do_vqrgb = True',
    'vqrgb_recon_coeff = 1.0',
    'vqrgb_latent_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]
groups['train_gen2dvq'] = [
    'do_gen2dvq = True',
    'gen2dvq_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]
groups['train_sigen2d'] = [
    'do_sigen2d = True',
    'sigen2d_coeff = 1.0',
]

############## datasets ##############

# dims for mem
SIZE = 16
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 1
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_val10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3ten"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "mabs7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)
