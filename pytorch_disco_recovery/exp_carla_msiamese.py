from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'trainer'
# current = 'tester'

mod = '"mul00"' # start multiview builder
mod = '"mul01"' # on val iters, run multiview trainer < OOM. why?
mod = '"mul02"' # JUST do the val iters. (still OOM!). disable smooth loss
mod = '"mul03"' # run ml too
mod = '"mul04"' # JUST a train iter; see the vox size
mod = '"mul05"' # train and val iters, but with diff resolutions
mod = '"mul06"' # proper train and val iters, but with diff resolutions
mod = '"mul07"' # proper trainvaltest
mod = '"mul08"' # allow random centroid on multiview iters
mod = '"mul09"' # noshuf
mod = '"mul10"' # one more time, to see if centroids are changing
mod = '"mul11"' # rand centroid this time
mod = '"mul12"' # diff SIZE_val 
mod = '"mul13"' # diff SIZE_val  proper
mod = '"mul14"' # test too
mod = '"mul15"' # train a while; curved_extreme (ce) on test
mod = '"mul16"' # pret 04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1v_faks30i1v_cig09
mod = '"mul17"' # like mul16 but no ml (basline)
mod = '"mul18"' # like mul16 but wider bounds on multiview iters (despite lower res)
mod = '"mul19"' # train = t, val = tce, test = vce
mod = '"mul20"' # SIZE_val = 64 as proper
mod = '"mul21"' # new split; pret 10k 04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul20


mod = '"tul00"' # tester 04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul20
# on train:
# mean_ious [0.96 0.93 0.91 0.89 0.86 0.84 0.82 0.8  0.78 0.77 0.75 0.75 0.74 0.72
#             0.71 0.7  0.69 0.68 0.68 0.67 0.67 0.66 0.65 0.63 0.61 0.59 0.57 0.56
#             0.54 0.52]
# on test:
# mean_ious [0.96 0.92 0.9  0.87 0.85 0.84 0.82 0.8  0.78 0.76 0.75 0.74 0.72 0.71
#             0.69 0.68 0.67 0.65 0.64 0.63 0.62 0.61 0.61 0.6  0.59 0.58 0.57 0.56
#             0.55 0.55]
mod = '"tul01"' # tester; pret cig09
# on train:
# mean_ious [0.97 0.93 0.91 0.88 0.86 0.83 0.8  0.78 0.75 0.74 0.73 0.72 0.71 0.7
#             0.69 0.68 0.67 0.66 0.66 0.65 0.64 0.63 0.61 0.6  0.58 0.56 0.54 0.52
#             0.5  0.48]
# on test:
# mean_ious [0.96 0.92 0.9  0.88 0.85 0.83 0.82 0.8  0.78 0.76 0.74 0.73 0.71 0.69
#             0.68 0.67 0.65 0.64 0.63 0.62 0.61 0.59 0.59 0.58 0.57 0.56 0.55 0.54
#             0.53 0.53]


# i want to find out if the corr heat can be decoded into a reliable certainty metric

mod = '"tul02"' # 10 iters tester mul20; sample the xyzr location from the corr map, print its value
mod = '"tul03"' # poolsize 3; pool the confs too
mod = '"tul04"' # take softmax first < error due to shape. this means my argmax is wrong
mod = '"tul05"' # sample xyz_e-offset; return this < looks pretty great now.
mod = '"tul06"' # save all confs and ious offline
mod = '"tul07"' # make 100 iters of that
mod = '"tul08"' # just 20 iters
mod = '"tul09"' # softmax the corr before getting conf

# it seems the answer is yes. if i use conf>=9, iou should typically be >= 0.7

# next i want to use that certainty metric to do some hypothesis management
# step1 here, really, is to put the forecaster into the mix

# on the phone i've said: just use forecasts from the classifier.
# the bigger goal here is to track through occlusions


# i agree that you need the classifiernet in here...
# and probably if you train this classifier on your own, performance will go up, since you are awesome
#

# i don't really agree with the idea of using that classifier though, since it cannot share any info, or provide conf of its own. should i ignore that?
## actually it can share info, if i give it a featnet3D backbone. i just don't think it will use it very well, since it is collapsing space

# a fun but maybe unproductive thing to do is: make a BEV PriNet, and turn that into a forecaster by just evaluating sampled trajectories on it
#


# let's say i use the classifier
# then: i might fire it on each timestep, and get the distr over trajectories
# it would be cool if i could train that classifier to be invariant to the number of timesteps it has as input


# maybe: an early step here is:
## 


mod = '"tul10"' # tester pret 10k mul20; obj_r = obj_r - obj_dr
# mean_ious [0.96 0.92 0.89 0.86 0.82 0.79 0.76 0.75 0.73 0.71 0.7  0.67 0.66 0.65
#             0.63 0.63 0.61 0.6  0.59 0.57 0.57 0.56 0.55 0.55 0.54 0.54 0.53 0.53
#             0.52 0.52]
mod = '"tul11"' # 100 iters
mod = '"tul12"' # 20 iters again; obj_r = obj_r + obj_dr
# mean_ious [0.96 0.92 0.91 0.88 0.86 0.85 0.83 0.81 0.8  0.78 0.77 0.76 0.74 0.73
#             0.72 0.71 0.7  0.69 0.68 0.66 0.65 0.64 0.63 0.62 0.62 0.61 0.61 0.6
#             0.59 0.59]
# this is clearly better

mod = '"tul13"' # check -12,-6,0,6,12 instead of -6,0,6
# mean_ious [0.96 0.92 0.9  0.87 0.84 0.82 0.79 0.78 0.76 0.76 0.74 0.73 0.73 0.71
#             0.7  0.69 0.68 0.67 0.66 0.64 0.64 0.63 0.62 0.61 0.6  0.59 0.59 0.58
#             0.58 0.57]
# ok, slightly worse
mod = '"tul14"' # check -12,0,12
# mean_ious [0.96 0.92 0.91 0.88 0.86 0.85 0.82 0.81 0.79 0.78 0.76 0.75 0.74 0.72
#             0.71 0.7  0.68 0.67 0.65 0.64 0.63 0.62 0.61 0.59 0.59 0.58 0.58 0.58
#             0.58 0.57]
mod = '"tul15"' # check -6,0,6; pret 20k 04_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul21
nmod = '"tul16"' # redo
# mean_ious [0.96 0.92 0.91 0.88 0.86 0.85 0.83 0.81 0.8  0.78 0.77 0.76 0.74 0.73
#             0.72 0.71 0.7  0.69 0.68 0.66 0.65 0.64 0.63 0.62 0.62 0.61 0.61 0.61
#             0.6  0.59]
# i'm not sure i really loaded it. i think not
mod = '"tul17"' # check -6,0,6; pret 10k 04_s2_m256x128x256_z64x64x64_1e-5_F3_d64_M_c1_r.1_faks30i1t_faks30i1tce_faks30i1vce_mul21
# mean_ious [0.96 0.93 0.91 0.88 0.86 0.84 0.82 0.81 0.79 0.78 0.76 0.75 0.74 0.73
#             0.71 0.7  0.69 0.68 0.67 0.66 0.65 0.65 0.64 0.64 0.64 0.63 0.63 0.63
#             0.62 0.61]
# good job net
mod = '"tul18"' # 100 iters
# mean_ious [0.96 0.92 0.9  0.88 0.86 0.84 0.82 0.8  0.78 0.77 0.75 0.74 0.72 0.7
#             0.69 0.68 0.67 0.66 0.64 0.63 0.63 0.62 0.61 0.61 0.6  0.59 0.58 0.58
#             0.57 0.56]
# compare with [... 0.55] on test. so it's still going up very very very slowly

mod = '"mul22"' # pret 10k mul20; dr_1 = self.obj_rlist_camR0s[:,1] - self.obj_rlist_camR0s[:,0]
mod = '"mul23"' # pret 10k mul20; dr_1_ = dr_0 + (self.obj_rlist_camR0s[:,1] - self.obj_rlist_camR0s[:,0]); dr_1 = (dr_1 + dr_1_)/2.0



mod = '"mul24"' # vis data



############## define experiments ##############

exps['builder'] = [
    'carla_msiamese', # mode
    'carla_vis_data', # dataset
    # 'carla_traintraintest_data', # dataset
    # 'carla_traintraintest_data', # dataset
    # 'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '10_iters',
    'train_feat3D', 
    'train_emb3D', 
    'train_match', 
    'B1',
    'no_shuf',
    # 'no_backprop',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_msiamese', # mode
    # 'carla_traintraintestce_data', # dataset
    'carla_t_tce_vce_data', # dataset
    # 'carla_traintestce_data', # dataset
    'train_on_trainval',
    # 'nearcube_traintest_bounds_narrow_val', 
    'nearcube_trainvaltest_bounds', 
    '300k_iters',
    'pretrained_feat3D',
    'pretrained_match',
    'train_feat3D', 
    # 'train_emb3D', 
    'train_match', 
    'B4',
    'vB4', # valset batchsize 4 
    'lr4', 
    'log500',
]
exps['tester'] = [
    'carla_msiamese', # mode
    # 'carla_trainvaltest_data', # dataset
    # 'carla_test_on_train_data', # dataset
    # 'carla_test_on_curved_train_data', # dataset
    # 'carla_test_on_straight_train_data', # dataset
    # 'carla_test_on_curved_data', # dataset
    # 'carla_test_on_curved_train_data', # dataset
    'carla_test_on_curved_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '100_iters',
    # '20_iters',
    # '10_iters',
    'no_shuf',
    'do_test', 
    'do_export_vis',
    'do_export_stats',
    'B1',
    'pretrained_feat3D',
    'pretrained_match',
    'train_feat3D', 
    'train_match', 
    'no_backprop',
    # 'log5',
    'log1',
]

############## net configs ##############

groups['carla_msiamese'] = ['do_carla_msiamese = True']
groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']
groups['do_export_stats'] = ['do_export_stats = True']

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
    'match_r_coeff = 0.1', 
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
SIZE_val = 64
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
# ZZ = 48
# ZY = 48
# ZX = 48
ZZ = 64
ZY = 64
ZX = 64

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
groups['nearcube_traintest_bounds_narrow_val'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_val = -8.0', # right (neg is left)
    'XMAX_val = 8.0', # right
    'YMIN_val = -4.0', # down (neg is up)
    'YMAX_val = 4.0', # down
    'ZMIN_val = -8.0', # forward
    'ZMAX_val = 8.0', # forward
    'Z_val = %d' % (int(SIZE_val*4)),
    'Y_val = %d' % (int(SIZE_val*2)),
    'X_val = %d' % (int(SIZE_val*4)),
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
groups['nearcube_trainvaltest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*4)),
    'Y_val = %d' % (int(SIZE_val*2)),
    'X_val = %d' % (int(SIZE_val*4)),
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
groups['carla_train1_test1_data'] = [
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
groups['carla_train10_test10_data'] = [
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
groups['carla_train10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1ten"',
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
groups['carla_traintraintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintraintestce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_t_tce_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "faks30i1tce"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vis_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vaas16i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "vaas16i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintestce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintrain_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
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
groups['carla_test_on_curved_data'] = [
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
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_curved_train_data'] = [
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
    'testset = "faks30i1tce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_straight_train_data'] = [
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
    'testset = "faks30i1t_straight"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_straight_test_data'] = [
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
    'testset = "faks30i1v_straight"',
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
