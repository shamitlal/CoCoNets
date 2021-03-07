from exp_base import *


############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"res00"' # start
mod = '"res01"' # resolvenet
mod = '"res02"' # again
mod = '"res03"' # extract low,mid,high feats, and print shapes
mod = '"res04"' # show low/mid/high/full
mod = '"res05"' # mo
mod = '"res06"' # mo
mod = '"res07"' # pad and upsample each
mod = '"res08"' # small bugfix

# let's first see if i can double the resolution effectively
mod = '"res09"' # uncertainty = abs(logits)
mod = '"res10"' # again
mod = '"res11"' # pret
mod = '"res12"' # -abs
mod = '"res13"' # add min, so that 0 is min
mod = '"res14"' # sub min, so that 0 is min
mod = '"res15"' # radius=1
mod = '"res16"' # samp feat
mod = '"res17"' # feed things to resolvenet
mod = '"res18"' # resolve_coeff=1.0
mod = '"res19"' # scatter out zeros
mod = '"res20"' # scatter out new logits
mod = '"res21"' # crop before vis
mod = '"res22"' # crop*2
mod = '"res23"' # train a while on 10
mod = '"res24"' # log50; train1k
mod = '"res25"' # K = 512
mod = '"res26"' # 1, no shuf
mod = '"res27"' # solid centroid
mod = '"res28"' # show sup
mod = '"res29"' # just put some zeros
mod = '"res30"' # show new logit
mod = '"res31"' # do not crop it < ah ok, everything sits at the top. why is that?
mod = '"res32"' # do not crop it
mod = '"res33"' # show scat directly; fix bug in sizes
mod = '"res34"' # again
mod = '"res35"' # scale then pad
mod = '"res36"' # train
mod = '"res37"' # crop new_logit
mod = '"res38"' # K=1024
mod = '"res39"' # K=2048
mod = '"res40"' # K=4096
mod = '"res41"' # ten
mod = '"res42"' # allow everyone to train
mod = '"res43"' # 2048
mod = '"res44"' # occ loss

# wouldn't it be cool to try a point-based model
# or at least, real sparse convs
# and really use those query masks
# and
# maybe i only need to query at locations along rays

mod = '"res45"' # sample at random locs too; 1024 top, 1024 rand
mod = '"res46"' # fixed the topk
mod = '"res47"' # same

# only query at locations where you have labels

mod = '"res48"' # only query at locations where you have labels
mod = '"res49"' # use 1024 random, and 512 uncertain
mod = '"res50"' # do not pret

# if you go the sparse conv route, you can have proper high-res local patches


# try to make sure the labels are right

# get kpconv working, at least for fun
# the motivation is: i want to query this type of model at arbitrary locations
# i'll start at coarse grid locations, then i'll go to certain subpixels, etc
# i do feel like this is maybe more efficient, as long as i can do the computation just at the locations that matter

# a conter-arg to this effort is:
# you already have spconv working
# is there a reason you can't do spconv here?
# spconv requires that we specify the locations we want to convolve, and then,


# ok, suppose we do submanifold conv at the sparse locations
# is this better than an MLP?
# the benefit here is: you would be able to use a 3d receptive field with size bigger than 1x1x1
# that's a nice benefit
# and then we say what? do a few layers with 3x3x3x256 filters, and thereby underfit less
# pretty strong argument

# let's first see what's going on with this 1x1x1 method, since it should work quite well
# experiments so far seem to show zero benefit of this method

# how about this:
# if i fit on a single example, i should be able to outperform bilinear upsampling  

mod = '"grow00"' # measure super resolve acc
mod = '"grow01"' # show high-res accuracy before and after
mod = '"grow02"' # frozen feat3d
mod = '"grow03"' # K = 2048
mod = '"grow04"' # pretrained and frozen!

# ok, the accuracy boost is there, but it's extremely tiny

# what i now want is:
# generative model
# apply it at multiple scales, at the critical points, 
# conditioning on the stuff generated so far, within a limited fov
# i don't know if i need a different model at each scale; probably not

mod = '"sig00"' # build sigen3d
mod = '"sig01"' # do-nothing; just get output
mod = '"sig02"' # get visibility separatley
mod = '"sig03"' # clamp outer vis
mod = '"sig04"' # no shuf
mod = '"sig05"' # again
mod = '"sig06"' # compute loss
mod = '"sig07"' # compute loss
mod = '"sig08"' # train a while
mod = '"sig09"' # halfres inputs and outputs
mod = '"sig10"' # halfres, bugfixed 
mod = '"sig11"' # quarter
mod = '"sig12"' # same
mod = '"sig13"' # test
mod = '"sig14"' # no dense layers
mod = '"sig15"' # input and output are just occ, so that it's truly autoregressive
# once i add rgb outputs, i can resolve this
mod = '"sig16"' # summ_occ instead of summ_feat
mod = '"sig17"' # log50
mod = '"sig18"' # more efficient version, using only occ
mod = '"sig19"' # same
mod = '"sig20"' # supervise everywhere at training time
mod = '"sig21"' # same
mod = '"sig22"' # log100
# ok, although things look fine at training time, the testing still seems broken
mod = '"sig23"' # bernoulli sample from logits
mod = '"sig24"' # measure accuracy at test time
mod = '"sig25"' # S_test = 5
mod = '"sig26"' # random visibility mask at training time
# ok nice, now, acc_bal at least rises convincingly at test time
mod = '"sig27"' # 5k iters
mod = '"sig28"' # train10,test10
mod = '"sig29"' # same
mod = '"sig30"' # speed=8; 60/40 split on loss
mod = '"sig31"' # show gt also
mod = '"sig32"' # snap5k
mod = '"sig33"' # full data
mod = '"sig34"' # log500
mod = '"sig35"' # shift acc scope to sigen3d; measure accuracy at training time too; on val iters, test train data
mod = '"sig36"' # speed=16
mod = '"sig37"' # actually have a val set please

# do you underfit?

# this apparently failed on iter 3808: assert(B==B_)
# maybe this is something else
mod = '"sig38"' # set S_val=5; set proper sizes; chans=128
mod = '"sig39"' # chans=64
mod = '"sig40"' # speeed=4
mod = '"sig41"' # drop_last=True < ok, this fixed the 3808 bug
mod = '"sig42"' # chans=128 < almost no effect
mod = '"sig43"' # yes biases, no batchnorms < seems worse
mod = '"sig44"' # rand centroid
mod = '"sig45"' # batchnorms, no biases

# on a walk, i figured out:
# it's ok to use the whole set of answers delivered on the forward pass, as long as we only do it on the frontier; this will probably deliver much cleaner results
# i want to condition on a 67-dim input at each scale: featprior, occprior, occ_g, free_g
# for finer scales, i can create a computation mask by doing conv3d the right number of times with ones -- just clone the net but give it ones
# but be careful to only use the answers at the narrower band

# let's first see single-pass test outputs at this resolution
mod = '"sig46"' # single-pass
mod = '"sig47"' # same
mod = '"sig48"' # go up to layer9 instead of just layer7
mod = '"sig49"' # log50; round
mod = '"sig50"' # just layer7
mod = '"sig51"' # log500
mod = '"sig52"' # S = 5
mod = '"sig53"' # just compute the stats inside sigen3d
mod = '"sig54"' # kernel 3
mod = '"sig55"' # show free_g
mod = '"sig56"' # 70/30 instead of 60/40, to get more free
mod = '"sig57"' # compute labels at 2x res, and take the ones with 0.6 conf
mod = '"sig58"' # compute loss on test iters too (but no bprop); wider centroid range
# there is some huge train/val gap
# the only difference between sets right now should be the input dropout, which makes train harder
# also S is different; it gives more labels on val/test
mod = '"sig59"' # debugged inputs/masks in both train and test < this fixes the gap
mod = '"sig60"' # S = 5 on train/val/test
mod = '"sig61"' # drop=False on train
mod = '"sig62"' # drop=True; generate labels directly at Z4 res, instead of Z2 with 0.6
# somehow all of these have a border artifact, where the top (front) part of occ is ones
mod = '"sig63"' # padding=0 on all convs; cube; Z2 resolution; crop 8
# nice, border effect gone
mod = '"sig64"' # diff size
# okokokokok
# this is good, but i am doing SAConv ops right now, not computation-saving sparse conv
mod = '"sig65"' # custom3d, with 4 res blocks, standard padding, and dilation
mod = '"sig66"' # sharp3d arch, with upsample at the end
mod = '"sig67"' # simple3d arch
# ok i like this
# no. this is still not saving computation!
mod = '"sig68"' # sp
mod = '"sig69"' # installed spconv
mod = '"sig70"' # do not downsamp
mod = '"sig71"' # no labels at the edge
mod = '"sig72"' # fourth res block
mod = '"sig73"' # compute-1-14 with new wheel
mod = '"sig74"' # use a computation mask dilated 4 times
mod = '"sig75"' # similar but only sparsify/densify once, and no resnet < indeed much faster, but this has smaller layers
mod = '"sig76"' # 4 layers
mod = '"sig77"' # Z4 
mod = '"sig78"' # do not zero border

mod = '"unc00"' # show uncertainty and scat
mod = '"unc01"' # get N1, N2
mod = '"unc02"' # N = 1024
mod = '"unc03"' # train a while longer


mod = '"unc04"' # replace at scat; pret 5k 02_s2_m64x64x64_1e-4_S3i_c1_mags7i3t_unc03
mod = '"unc05"' # mult output by computation mask
mod = '"unc06"' # show g; use summ_writer for second guy
mod = '"unc07"' # show computation mask
mod = '"unc08"' # mask loss by computation_mask
mod = '"unc09"' # train1, to see super-resolution
mod = '"unc10"' # uncertainty = closeness to 0.5; torch.exp(-torch.abs(occ_e2 - 0.5))
mod = '"unc11"' # linear uncertainty model; N = 2048; show occ_e4
mod = '"unc12"' # detach quarter-res things
mod = '"unc13"' # just take random samples
mod = '"unc14"' # mean along y
mod = '"unc15"' # print sum of scat, because i do not believe it
mod = '"unc16"' # different sigen3dnets
mod = '"unc17"' # use vis_memX0 on the second pass < ok looks fine again
mod = '"unc18"' # use scat, but just 512 occupied points < ah. scat looks real bad. all at the top.
mod = '"unc19"' # do that again, but with occ_sup points < 
mod = '"unc20"' # show scat_rand and scat_sort
mod = '"unc21"' # N = 128
mod = '"unc22"' # .gather
mod = '"unc23"' # hacky; K*N manual gather
mod = '"unc24"' # hack the scatter too
mod = '"unc25"' # gather into indlist
mod = '"unc26"' # solid centroid
mod = '"unc27"' # proper scat; no_shuf
mod = '"unc28"' # N = 1024
mod = '"unc29"' # use gather
mod = '"unc30"' # take free and occ
mod = '"unc31"' # 0.01 reg loss, so that we prefer empty
mod = '"unc32"' # 0.1 reg loss
mod = '"unc33"' # N = 2048
mod = '"unc34"' # use answers at scat2
mod = '"unc35"' # comute z4 labels at z2 resolution and downsample



# why does the upsampled occ look so bad?
mod = '"fun00"' # just train at z2 resolution
mod = '"fun01"' # summ
mod = '"fun02"' # vis=ones
mod = '"fun03"' # reg=0.0
mod = '"fun04"' # one more layer < nice, this helps
mod = '"fun05"' # installed reg coeff 
mod = '"fun06"' # one more layer
mod = '"fun07"' # zero border
mod = '"fun08"' # zero border 6 (to match net)
mod = '"fun09"' # 8 layers; zero 8
mod = '"fun10"' # Z resolution
mod = '"fun11"' # 2-stage net; output at half res. 
mod = '"fun12"' # do not zero border
mod = '"fun14"' # skipcon, to produce full-res outs
mod = '"fun15"' # zero5 < ok looks nice
# many of these models sometimes jump to a lower acc; maybe i am not zeroing enough labels?
mod = '"fun16"' # zero6
# < no, the issue is just that my learning rate is too high

# i do wonder if there's something a bit complicated but efficient i can do with explicitly cropping patches
# and running them through some encoder-decoder
# supposedly i am doing the convolutional version of this already
# but it's not that great
mod = '"fun17"' # zero7
mod = '"fun18"' # zero8
mod = '"fun19"' # zero9
mod = '"fun20"' # zero10
mod = '"fun21"' # downsampler has kernel 2, stride 2, pad 0
mod = '"fun22"' # show vertical view too
mod = '"fun23"' # show vert AND bev, by fixing bug
mod = '"fun24"' # single-scale net again please; four 3x3x3 convs
mod = '"fun25"' # padding=0; do not share indices < ah, submconv pads for me
mod = '"fun26"' # do share indices
mod = '"fun27"' #


# i think what i need here is:
# 



############## exps ##############

exps['builder'] = [
    'carla_resolve', # mode
    'carla_multiview_train10_data', # dataset
    'carla_16-8-16_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    # 'no_shuf',
    'no_backprop',
    # 'pretrained_feat3d',
    # 'train_feat3d',
    # 'train_occ',
    # 'train_resolve',
    'train_sigen3d',
    # 'train_match',
    'log1',
]
exps['trainer'] = [
    'carla_resolve', # mode
    'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train1_test1_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    # 'carla_multiview_test10_data', # dataset
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_val_data', # dataset
    # 'carla_multiview_test_data', # dataset
    'carla_32-16-32_bounds_train',
    'carla_32-16-32_bounds_val',
    'carla_32-16-32_bounds_test',
    # 'pretrained_sigen3d',
    # 'carla_32-32-32_bounds_train',
    # 'carla_32-32-32_bounds_val',
    # 'carla_32-32-32_bounds_test',
    '20k_iters',
    'lr5',
    'snap5k',
    'no_shuf',
    'B1',
    'train_sigen3d',
    'log50',
]

############## groups ##############

groups['carla_resolve'] = ['do_carla_resolve = True']
groups['do_test'] = ['do_test = True']
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 4',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_rgb_coeff = 10.0',
    # 'render_depth_coeff = 0.1',
    # 'render_smooth_coeff = 0.01',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
]
groups['train_resolve'] = [
    'do_resolve = True',
    'resolve_stages = 1',
    'resolve_coeff = 2.0',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    # 'rgb_smooth_coeff = 0.1',
]
groups['train_sigen3d'] = [
    'do_sigen3d = True',
    'sigen3d_coeff = 1.0',
    # 'sigen3d_reg_coeff = 0.01',
]


############## datasets ##############

# # dims for mem
# SIZE = 32
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 4
S_val = 5
S_test = 5
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

# SIZE = 32
# SIZE_val = 32
# SIZE_test = 32
# SIZE_zoom = 32

# SIZE = 24
# SIZE_val = 24
# SIZE_test = 24
# SIZE_zoom = 24

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

# SIZE = 16
# SIZE_val = 16
# SIZE_test = 16
# SIZE_zoom = 16

# SIZE = 12
# SIZE_val = 12
# SIZE_test = 12
# SIZE_zoom = 12

# SIZE = 10
# SIZE_val = 10
# SIZE_test = 10
# SIZE_zoom = 10

SIZE = 8
SIZE_val = 8
SIZE_test = 8
SIZE_zoom = 8

# SIZE = 4
# SIZE_val = 4
# SIZE_test = 4
# SIZE_zoom = 4

dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla_odometry/processed"

groups['carla_16-8-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-8-16_bounds_val'] = [
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_16-8-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_32-16-32_bounds_train'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_32-16-32_bounds_val'] = [
    'XMIN_val = -32.0', # right (neg is left)
    'XMAX_val = 32.0', # right
    'YMIN_val = -16.0', # down (neg is up)
    'YMAX_val = 16.0', # down
    'ZMIN_val = -32.0', # forward
    'ZMAX_val = 32.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_32-16-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_32-32-32_bounds_train'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -32.0', # down (neg is up)
    'YMAX = 32.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_32-32-32_bounds_val'] = [
    'XMIN_val = -32.0', # right (neg is left)
    'XMAX_val = 32.0', # right
    'YMIN_val = -32.0', # down (neg is up)
    'YMAX_val = 32.0', # down
    'ZMIN_val = -32.0', # forward
    'ZMAX_val = 32.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*8)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_32-32-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train1_test1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "mags7i3one"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3ten"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "mags7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
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
