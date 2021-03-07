from exp_base import *

############## choose an experiment ##############

# current = 'builder'
current = 'rigid_trainer'
# current = 'rigid_tester'

mod = '"vs00"' # ok
mod = '"vs01"' # S_test = 30
mod = '"vs02"' # log50
mod = '"vs03"' # pret > crash
mod = '"vs04"' # print centers pix 
mod = '"vs05"' # try the other trainer
mod = '"vs06"' # diff sizes
mod = '"vs07"' # restore some old files
mod = '"vs08"' # rigid trainer
mod = '"vs09"' # check out vox util < error!
mod = '"vs10"' # old vox util again
mod = '"vs11"' # delete the bounds arg
mod = '"vs12"' # mostly working vox
mod = '"vs13"' # shift by half a pix
mod = '"vs14"' # VOX_SIZE= self.vox_size < error
mod = '"vs15"' # print the vox size before and after the assignment
mod = '"vs16"' # compute vox size on the fly
mod = '"vs17"' # add iscube assert
mod = '"vs18"' # use iou pools
mod = '"vs19"' # use iou pools; print info
mod = '"vs20"' # no interrupts < ok looks great
mod = '"vs21"' # updated hyp < somehow, now it's all junk
mod = '"vs22"' # pret red54
mod = '"vs23"' # fak data for training
mod = '"vs24"' # pret rigid too
mod = '"vs25"' # 0-36; scratch data
mod = '"vs26"' # 0-16; scratch
mod = '"vs27"' # retry mix
mod = '"vs28"' # return nan if we don't have at least half a pool
mod = '"vs29"' # pool size 20 for the ious < a bit jumpy, but seems accurate
mod = '"vs30"' # trainvaltest, to see if i have a trainval gap in the loss
mod = '"vs31"' # builder; show the inputs to the rigid thing
mod = '"vs32"' # 
mod = '"vs33"' # again but higher res

mod = '"rule00"' # sample some corners, show them
mod = '"rule01"' # show multiple views of the boxes
mod = '"rule02"' # print lengths, since some seem too low
mod = '"rule03"' # sample until all inbounds
mod = '"rule04"' # sample until all in mask
mod = '"rule05"' # really do that
mod = '"rule06"' # print less
mod = '"rule07"' # more examples
mod = '"rule08"' # different size, to see if it gets much harder
mod = '"rule09"' # plot the number of tries
mod = '"rule10"' # rigid trainer from scratch, just for a baseline
mod = '"rule11"' # use cubes, print shapes
mod = '"rule12"' # retry trainer, iwth bug fixed
mod = '"rule13"' # builder withcubes
mod = '"rule14"' # trainer with cubes
mod = '"rule15"' # off gpu
mod = '"rule16"' # same but log50, and fix test bug, and pool=4
mod = '"rule17"' # another bugfix
mod = '"rule18"' # shifted centroid by 1px; ZZ=32; set med=3456 due to the higher size. < bad for some reason

# maybe i need to change this so that the search region and zoom region are not so closely tied
# or maybe that's wrong
# the main thing is: i need to use the hyps properly when defining self.med, or take some kind of spatial average

mod = '"rule19"' # take a spatial avg after last layer, and restore the 1024 < yes, worse
mod = '"rule20"' # clean repeat of rule17 i think (but with the shifted centroid), but pool=20 and log500 < this guy eventually failed. maybe for high res i need a wider training search?
mod = '"rule21"' # add rigid_repeats param
mod = '"rule22"' # ZZ=32, so 3456 for first layer of pred2
mod = '"rule23"' # ZZ=32, so 3456 for first layer of pred2

# things i want soon/now:
# when encoding the template, don't zoom in so far. use ZZ*2 let's say. or, just increase ZZ and let the search region be the same size. for these cube things, i don't need anything special to do this

# when training at high res, use the 0.5 window, rather than 0.25, for training


# i have a big train/val gap! this is great news.

mod = '"rule24"' # like rule23 but use 0.5 window
mod = '"rule25"' # eliminate the 1px sub
mod = '"rule26"' # higher ZZ, but zoom==search
mod = '"rule27"' # fix bug with rule26; use proper SZ
mod = '"rule28"' # halfpix adjustment for zoom 
mod = '"rule29"' # hidden_dim=512


# katerina and i agree:
# we need to predict and check
# predict where the object will be at time t+1
# then, when searching for it, only use the parts of it that you expect will be visible.
# version1 of this should just determine visibility based on image bounds
# version2 should consider occlusions too
# version3 should say: if not enough is visible, just use const velocity
# note this is mostly a test-time thing, but it may improve training also, by making the problem more fair


# to do this, first i need an OK pret model, which i can test
# > done

mod = '"vis00"' # tester; pret 06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27
mod = '"vis01"' # same but log5
# OK,
# let's start with a zero-velocity estimate
# my 8-corner thing is pretty damn convenient right now
# i need to project the points of the object mask into the image, and use this to trim the mask
mod = '"vis02"' # print some shapes as i gather the xyz out
mod = '"vis03"' # nothing really, but loaded the 20k
# mean_ious [0.77 0.74 0.73 0.72 0.72 0.72 0.71 0.71 0.7  0.7  0.7  0.69 0.69 0.68
#             0.68 0.68 0.67 0.66 0.66 0.66 0.65 0.65 0.65 0.64 0.64 0.63 0.62 0.63
#             0.61 0.61]
mod = '"vis04"' # show the gathered points and their moved versions < somehow it's not a coherent box
mod = '"vis05"' # gather more carefully
mod = '"vis06"' # gather yet more carefully

mod = '"vis07"' # more careufl transfomration < ok looks great
mod = '"vis08"' # get to camX coords and show depth
mod = '"vis09"' # get to camX coords and show depth
mod = '"vis10"' # prove to me it's a subset
mod = '"vis11"' # get back to template coords; show the masks
mod = '"vis12"' # use xyz_camR0
mod = '"vis13"' # print less
mod = '"vis14"' # transform all the way back to oldR0
mod = '"vis15"' # 100 iters
mod = '"vis16"' # use the new template
mod = '"vis17"' # dilate by 1, so that we do not have infinte loops
# somehow we still have the infinity problem. is it that the mask was empty to start with?
# probably yes, though the template sum said 416
# very soon i need the const-velocity decision, instead of firing the tracker

# anyway: nope: this is not improving results yet. not sure why.
# maybe because this throws away too much
# i may need an alternate route. something might be lost in discretization. 

mod = '"vis18"' # don't let length be 1.0

# hey, note that if you allowed offsets of the centroid then sampling would be a lot easier, and would let us sample from the back


mod = '"vis19"' # use const vel when sum <20
# a better version of this might try to choose 8corners first

mod = '"vis20"' # sample 8 corners outside
mod = '"vis21"' # proper update of _prev
# mean_ious [0.8  0.75 0.74 0.72 0.73 0.72 0.72 0.71 0.7  0.71 0.7  0.69 0.69 0.68
#             0.67 0.66 0.66 0.65 0.66 0.64 0.64 0.64 0.65 0.64 0.64 0.63 0.62 0.62
#             0.62 0.6 ]
# not a clear improvement

mod = '"vis22"' # use proper vel update
mod = '"vis23"' # use ref_next when creating that mask
# mean_ious [0.8  0.75 0.74 0.73 0.73 0.71 0.72 0.71 0.71 0.71 0.71 0.69 0.7  0.69
#             0.68 0.68 0.68 0.66 0.65 0.66 0.66 0.64 0.65 0.65 0.64 0.63 0.63 0.63
#             0.62 0.61]


# previously:
# mean_ious [0.77 0.74 0.73 0.72 0.72 0.72 0.71 0.71 0.7  0.7  0.7  0.69 0.69 0.68
#             0.68 0.68 0.67 0.66 0.66 0.66 0.65 0.65 0.65 0.64 0.64 0.63 0.62 0.63
#             0.61 0.61]


# manage occlusions a bit

mod = '"vis24"' # dilate the mask once, if halfway through we still fail
# mean_ious [0.78 0.74 0.74 0.72 0.72 0.72 0.72 0.7  0.71 0.71 0.71 0.69 0.68 0.68
#             0.68 0.67 0.68 0.67 0.66 0.66 0.66 0.66 0.65 0.65 0.64 0.65 0.63 0.63
#             0.63 0.62]
mod = '"vis25"' # center the search on _next
# mean_ious [0.78 0.74 0.74 0.74 0.73 0.74 0.72 0.72 0.72 0.71 0.7  0.7  0.7  0.68
#             0.68 0.68 0.67 0.67 0.66 0.67 0.66 0.64 0.65 0.64 0.64 0.62 0.62 0.62
#             0.61 0.6 ]
# no clear help. actually seems a bit worse.
# undo this
mod = '"vis26"' # check if obj depth matches real depth
# bad
mod = '"vis27"' # check if obj depth matches real depth, thresh 2.0
# mean_ious [0.8  0.76 0.75 0.73 0.74 0.72 0.72 0.71 0.7  0.7  0.7  0.68 0.68 0.67
#             0.65 0.64 0.64 0.64 0.65 0.64 0.64 0.63 0.62 0.63 0.62 0.61 0.6  0.59
#             0.59 0.58]
# not a clear win
mod = '"vis28"' # ten steps with log1
mod = '"vis29"' # pool1
mod = '"vis30"' # mean of two prevs < some iters are better, others are worse
mod = '"vis31"' # allow 2k tries, instead of 1k

# you need to fix the sampling centroid offset, so that you can track the car from only its tip

mod = '"cen00"' # allow random centroid, and plot this
mod = '"cen01"' # center-1
mod = '"cen02"' # random centroid, and account for it
mod = '"cen03"' # offset the other way


mod = '"cen04"' # train a while, with rigid[:,:,:3] = rigid[:,:,:3] + (sampled_centers.reshape(B, R, 3) - normal_centers.reshape(B, R, 3))
mod = '"cen05"' # allow 4k tries
mod = '"cen06"' # allow 2k tries only; else return early 
mod = '"cen07"' # allow 1k tries only; else return early 
mod = '"cen08"' # allow 2k tries; with rigid[:,:,:3] = rigid[:,:,:3] + (normal_centers.reshape(B, R, 3) - sampled_centers.reshape(B, R, 3))



mod = '"pep00"' # pret 06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27; solid center; tester
mod = '"pep01"' # 2k tries, solid center
# mean_ious [0.72 0.67 0.66 0.65 0.64 0.64 0.64 0.64 0.64 0.64 0.63 0.62 0.62 0.62
#             0.62 0.62 0.61 0.61 0.61 0.6  0.6  0.6  0.59 0.59 0.58 0.57 0.57 0.56
#             0.55 0.55]
mod = '"pep02"' # tester on train set
# mean_ious [0.73 0.69 0.68 0.68 0.68 0.67 0.67 0.67 0.67 0.67 0.67 0.66 0.66 0.65
#             0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.64 0.64 0.64 0.63 0.63
#             0.62 0.62]
mod = '"pep03"' # debug exporting of visX_e,g
mod = '"pep04"' # show ious and not tids
mod = '"pep05"' # show gt box alongside estim
mod = '"pep06"' # show bev vis too
mod = '"pep07"' # higher res, partly so that things match
mod = '"pep08"' # summ_rgb
mod = '"pep09"' # 10 iters; return the right thing; eliminate duplicate prep
mod = '"pep10"' # cat bev with the other thing
mod = '"pep11"' # bigger text
mod = '"pep12"' # test on train
mod = '"pep13"' # test on train again, but SIZE=48
# mean_ious[0.73,0.69,0.68,0.68,0.68,0.67,0.67,0.67,0.67,0.67,0.67,0.65,0.65,0.65,0.65,0.64,0.65,0.65,0.64,0.65,0.65,0.64,0.64,0.64,0.64,0.64,0.64,0.63,0.62,0.61]
mod = '"pep14"' # proper test, but SIZE=48
# mean_ious[0.72,0.67,0.65,0.65,0.64,0.64,0.64,0.64,0.63,0.64,0.64,0.62,0.62,0.62,0.62,0.62,0.61,0.61,0.6,0.6,0.59,0.59,0.58,0.59,0.58,0.57,0.56,0.56,0.55,0.54]
mod = '"pep15"' # no export vis; just go
# mean_ious [0.73 0.67 0.66 0.65 0.65 0.65 0.64 0.65 0.64 0.63 0.63 0.62 0.61 0.61
#             0.61 0.6  0.6  0.6  0.59 0.59 0.58 0.58 0.58 0.58 0.57 0.57 0.57 0.56
#             0.55 0.55]
mod = '"pep16"' # pret 20k intead of 300k
mod = '"pep17"' # actually pret 20k intead of 300k
# mean_ious [0.72 0.67 0.65 0.65 0.64 0.64 0.64 0.64 0.63 0.63 0.63 0.62 0.62 0.61
#             0.61 0.61 0.6  0.6  0.6  0.59 0.59 0.59 0.59 0.59 0.58 0.58 0.57 0.57
#             0.57 0.56]
# i think this is the wrong ckpt
mod = '"pep18"' # pret 20k 06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27
# train:
# mean_ious [0.82 0.79 0.79 0.78 0.78 0.78 0.77 0.77 0.77 0.76 0.77 0.76 0.76 0.75
#             0.76 0.75 0.75 0.76 0.75 0.75 0.75 0.74 0.74 0.74 0.74 0.73 0.71 0.72
#             0.7  0.7 ]
# test:
# mean_ious [0.8  0.77 0.76 0.74 0.73 0.72 0.72 0.72 0.71 0.71 0.71 0.69 0.68 0.69
#             0.67 0.67 0.66 0.65 0.65 0.64 0.63 0.62 0.63 0.62 0.61 0.6  0.6  0.59
#             0.58 0.57]
mod = '"pep19"' # do export
# train:
# mean_ious [0.81 0.79 0.79 0.79 0.79 0.78 0.79 0.78 0.79 0.77 0.78 0.78 0.77 0.77
#             0.77 0.77 0.77 0.77 0.76 0.76 0.76 0.76 0.75 0.76 0.75 0.74 0.74 0.74
#             0.72 0.72]
# test:
# mean_ious [0.8  0.77 0.77 0.74 0.73 0.72 0.72 0.72 0.71 0.7  0.7  0.69 0.68 0.67
#             0.66 0.66 0.65 0.65 0.64 0.63 0.63 0.62 0.61 0.62 0.61 0.6  0.59 0.58
#             0.58 0.56]

mod = '"rot00"' # basic model with solid centroid, hopefully similar to 06_s2_m256x128x256_z48x48x48_1e-4_F3_d64_Ric2_r.1_t1_faks30i1t_faks30i1v_faks30i1v_rule27

mod = '"rot01"' # builder; paste in some noyaw code and see what it produces
mod = '"rot02"' # new noyaw, with exact rotation cancellation
mod = '"rot03"' # norot (some renames)
mod = '"rot04"' # use fancy new func and show occ_0
mod = '"rot05"' # randomize the rotation a bit
mod = '"rot06"' # return the lrt also, and adjust that mask
mod = '"rot07"' # alt version, inside vox util
mod = '"rot08"' # make occ_1 too
mod = '"rot09"' # print dt_1; make dr_1 = dr_0
mod = '"rot10"' # show mask_1 too < ok no, the lrt is correct but the occ/transformation is not
mod = '"rot11"' # retry, using the placer
mod = '"rot12"' # show mask_1
mod = '"rot13"' # retry the other guy but use camX0s < ok good, simple bug
mod = '"rot14"' # make delta rot < no, very bad
mod = '"rot15"' # use the placer
mod = '"rot16"' # feed this new data for training
mod = '"rot17"' # make dr_1 = dr_0, to focus on dt_1
mod = '"rot18"' # make dr_0 = 0
mod = '"rot19"' # more prints
mod = '"rot20"' # feed mid_xyz
mod = '"rot21"' # allow dr
mod = '"rot22"' # train a while
mod = '"rot23"' # fewer prints, log50 
mod = '"rot24"' # fewer prints, log500
mod = '"rot25"' # halve the rotation deltas at training time
mod = '"rot26"' # ZERO rotation deltas at training time
mod = '"rot27"' # d_1 = d_0; fixed the bev summ scope; feed proper delta to the net
mod = '"rot28"' # d_1 = d_0 + rand_r
mod = '"rot29"' # d_1 = d_0 + rand_r*2

# i need to wait a bit and see that my estimated deltas are in the right direction to test nicely

# i should sub 1 from the center
# i should avoid voxelizing

# mod = '"rot30"' # d_0 = 0 < no, that doesn't make any sense, unless you handle this at test time too. 
mod = '"rot31"' # higher lr, higher batch size
mod = '"rot32"' # only voxelize at test time




############## define experiments ##############

exps['builder'] = [
    'carla_vsiamese', # mode
    # 'carla_train_data', # dataset
    # 'carla_traintest_data', # dataset
    # 'carla_traintest1_data', # dataset
    # 'carla_train1_data', # dataset
    'carla_train_data', # dataset
    'nearcube_traintest_bounds', 
    '10_iters',
    # 'pretrained_feat3D', 
    'train_feat3D', 
    'train_rigid', 
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    'fastest_logging',
]
exps['rigid_trainer'] = [
    'carla_vsiamese', # mode
    # 'mix_carla_trainvaltest_data', # dataset
    # 'old_carla_traintest_data', # dataset
    # 'carla_traintest_data', # dataset
    'carla_trainvaltest_data', # dataset
    'nearcube_traintest_bounds', 
    '300k_iters',
    # 'pretrained_feat3D', 
    # 'pretrained_rigid', 
    'train_feat3D', 
    'train_rigid', 
    'B12',
    'lr3', 
    'log500',
]
exps['rigid_tester'] = [
    'carla_vsiamese', # mode
    'carla_trainvaltest_data', # dataset
    # 'carla_test_on_train_data', # dataset
    'nearcube_traintest_bounds', 
    '100_iters',
    # '10_iters',
    'no_shuf',
    'do_test', 
    'do_export_vis',
    'B1',
    'pretrained_feat3D',
    'pretrained_rigid',
    'train_feat3D', 
    'train_rigid', 
    'no_backprop',
    # 'log5',
    'log1',
]

############## net configs ##############

groups['carla_vsiamese'] = ['do_carla_vsiamese = True']
groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']

groups['include_summs'] = [
    'do_include_summs = True',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    # 'feat3D_dim = 32',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.01',
    # 'feat_do_sparse_invar = True', 
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    # 'emb_3D_ml_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    # 'emb_3D_mindist = 16.0',
    # 'emb_3D_num_samples = 2',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
    'emb_3D_ce_coeff = 1.0',
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
]
groups['train_translation'] = [
    'do_translation = True',
    'translation_coeff = 1.0', 
]
groups['train_rigid'] = [
    'do_rigid = True',
    'rigid_use_cubes = True', 
    'rigid_repeats = 2', 
    'rigid_r_coeff = 0.1', 
    'rigid_t_coeff = 1.0', 
]
groups['train_robust'] = [
    'do_robust = True',
    # 'robust_corner_coeff = 1.0', 
    # 'robust_r_coeff = 0.1', 
    # 'robust_t_coeff = 1.0', 
]

############## datasets ##############

# DHW for mem stuff
SIZE = 64
SIZE_test = 64
# SIZE = 48
# SIZE_test = 48
# SIZE = 32
# SIZE_test = 32

# Z = SIZE*4
# Y = SIZE*1
# X = SIZE*4

# zoom stuff

# ZZ = int(SIZE/2)
# ZY = int(SIZE/2)
# ZX = int(SIZE/2)
# ZZ = 24
# ZY = 24
# ZX = 24
# ZZ = 32
# ZY = 32
# ZX = 32
# ZZ = 40
# ZY = 40
# ZX = 40
ZZ = 48
ZY = 48
ZX = 48
# ZZ = 64
# ZY = 64
# ZX = 64

K = 8 # how many objects to consider
N = 8

# S = 10
S = 2
S_test = 30

H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

dataset_location = "/data4/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/scratch"


groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['nearcube_traintest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['cube_traintest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['narrow_traintest_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]

groups['carla_flowtrack_mini_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "fags16i3ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3ten"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    # 'testset_seqlen = 2',
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "fags16i3t"',
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_trainvaltest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1t"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['old_carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['mix_carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintest100_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1hun"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1hun"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1ten"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest1_simple_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_trainvaltest1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'backprop_on_val = True',
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
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
