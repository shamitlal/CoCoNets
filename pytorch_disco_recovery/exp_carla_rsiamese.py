from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
current = 'tester'

mod = '"cig00"' # builder; feat3D
mod = '"cig01"' # delete everything except feat and match
mod = '"cig02"' # match
mod = '"cig03"' # train a while
mod = '"cig04"' # uncomment matchnet
mod = '"cig05"' # train only; ten; show obj_loc_e; 
mod = '"cig06"' # no rot at all
mod = '"cig07"' # print corr.shape; i worry it's just 1x1x1 < yup
mod = '"cig08"' # *2 for occ_1 < gt is wrong
mod = '"cig09"' # /2 for occ_0
mod = '"cig10"' # undo that trim
mod = '"cig11"' # bring back trim, but regular _0
mod = '"cig12"' # trim 6
mod = '"cig13"' # print shapes and xyz offset itself
mod = '"cig14"' # train to regress a soft argmax
mod = '"cig15"' # print summ
mod = '"cig16"' # summ properly
mod = '"cig17"' # fewer prints
mod = '"cig18"' # proper /2
mod = '"cig19"' # show trim
mod = '"cig20"' # allow dr_0 rot
mod = '"cig21"' # bigger dat
mod = '"cig22"' # B4
mod = '"cig23"' # some renames
mod = '"cig24"' # use proper estimate at test time
mod = '"cig25"' # print matchnet output in test
mod = '"cig26"' # trim for test
mod = '"cig27"' # trim for test properly
mod = '"cig28"' # use search_lrt instead of that other bs < nice; this outperforms cig27, at least at the start
mod = '"cig29"' # set matchnet output to be 15.5, to see if it drifts < not anymore anyway; looks good
mod = '"cig30"' # cig28 but larger B
mod = '"cig31"' # fewer prints
mod = '"cig32"' # show pca of matchnet inputs
mod = '"cig33"' # train with dr_0 = 0, to see worse perf < actually it's better, but maybe only bc of bias
mod = '"cig34"' # dr_0 = anything again; tester; pret 01_s2_m256x128x256_z64x64x64_1e-3_F3_d64_M_c1_faks30i1t_faks30i1v_faks30i1v_cig28
# mean_ious [0.92 0.85 0.83 0.82 0.8  0.79 0.78 0.78 0.78 0.78 0.77 0.77 0.76 0.75
#             0.74 0.74 0.72 0.72 0.71 0.7  0.7  0.69 0.69 0.69 0.68 0.67 0.66 0.66
#             0.66 0.64]
# this is better than i got with rigidnet!
mod = '"cig35"' # deglist = [-5, 0, 5] < bad
mod = '"cig36"' # no test
mod = '"cig37"' # deglist = [0]
mod = '"cig38"' # dr_1 = dr_0
mod = '"cig39"' # don't penalize rad part
mod = '"cig40"' # old code < ok works again
mod = '"cig41"' # halfway: use stacking/whatever, but old argmax < ok looks fine
mod = '"cig42"' # xyzr argmax, use xyz_offsetlist[0] < bad
mod = '"cig43"' # fix mixup between z and r
mod = '"cig44"' # degs = [-5, 0, 5]
mod = '"cig45"' # train rot thing too
mod = '"cig46"' # bigger dat < no, test not implemented
mod = '"cig47"' # train again, with self.place
mod = '"cig48"' # traintest; pretty sophisticated (make it straight at the start, use the lrts in a smart way)
mod = '"cig49"' # tester; pret; pretty sophis
mod = '"cig50"' # same but voxelize the right data
mod = '"cig51"' # 100 iters < no rotation being used here; this just benefits from straightening
# mean_ious [0.92 0.86 0.86 0.84 0.83 0.82 0.82 0.81 0.81 0.8  0.79 0.79 0.78 0.77
#             0.75 0.75 0.74 0.74 0.73 0.73 0.73 0.72 0.71 0.71 0.7  0.69 0.69 0.68
#             0.68 0.67]
# holy smokes this is high
mod = '"cig52"' # 100 iters; do export; 
mod = '"cig53"' # 100 iters; do export; use the estimated rotation as obj_r = obj_r + rad_e 
mod = '"cig54"' # use deglist = [-10, 0, 10] < OK: on norot trajs the rot drifts, on rot trajs i can see it following sometimes. not 100% sure.
# mean_ious [0.92 0.86 0.86 0.84 0.82 0.82 0.81 0.8  0.79 0.78 0.77 0.76 0.75 0.74
#             0.72 0.71 0.7  0.69 0.68 0.68 0.67 0.66 0.65 0.64 0.64 0.62 0.62 0.61
#             0.6  0.59]
mod = '"cig55"' # train with dr_1 in -pi/16,pi/16, which i think is -11.25,11.25
mod = '"cig56"' # tester; obj_r = obj_r - obj_dr < on norot trajs not so much rot drift, but on rot trajs it's almost always opposite
# mean_ious [0.92 0.85 0.83 0.82 0.8  0.79 0.78 0.78 0.78 0.78 0.77 0.77 0.76 0.75
#             0.74 0.74 0.72 0.72 0.71 0.7  0.7  0.69 0.69 0.69 0.68 0.67 0.66 0.66
#             0.66 0.64]
# ok, this is better than when the rad direction was the right way, but hm. 
mod = '"cig57"' # obj_r = obj_r + obj_dr; do indeed train please
mod = '"cig58"' # deglist = [-5, 0, 5]
mod = '"cig59"' # deglist = [-5, 0, 5]; log50 instead of log5
mod = '"cig60"' # same but eliminate that corr = 0.001 * corr < wow, a lot worse
mod = '"cig61"' # half those rot deltas at training time
mod = '"cig62"' # parallelize the featnet call at training time; at B4, this was even slower than the non-parallel!
mod = '"cig63"' # self.trim = 8 instead of 6
mod = '"cig64"' # undo that parallelize < still oddly slow. maybe the machine needs a reboot? or maybe trim=8 makes it slow, since there are more locations to conv. hey that makes some sense.
mod = '"cig65"' # self.trim = 5
mod = '"cig66"' # bring back the 0.001
mod = '"cig67"' # train with -dr_1
mod = '"cig68"' # dt_1 = dt_0
mod = '"cig69"' # solve for rot in a separate softmax
mod = '"cig70"' # positive dr_1
mod = '"cig71"' # -4,-2,0,2,4
mod = '"cig72"' # gt rots /3., so they are within 4 deg
mod = '"cig73"' # place object at rad_ instead of rad
mod = '"cig74"' # print rad_e and rad_g < wow, rad_g is huge! 
mod = '"cig75"' # use rad_g = dr_1 - dr_0
mod = '"cig76"' # use rad_g = dr_1 - dr_0 proper
mod = '"cig77"' # eliminate the /3
mod = '"cig78"' # deglist is -5,0,5
mod = '"cig79"' # allow rots in -11,11; deglist -10,-5,0,5,10
mod = '"cig80"' # apply loss in deg space instad of rad
mod = '"cig81"' # eliminate that *2 from nowhere
mod = '"cig82"' # allow some translation, apply translation loss; put coeff on rot to 0
mod = '"cig83"' # bins: -8,-6,...
mod = '"cig84"' # no translation; penalize rot; 1 ex
mod = '"cig85"' # show corrlist as a gif
mod = '"cig86"' # dr_0 *= 0.01
mod = '"cig87"' # dr_1 ONLY has a y part
mod = '"cig88"' # one_shot = True < pretty good actually, except when B=2
mod = '"cig89"' # B2; print rand_r
mod = '"cig90"' # proper stacking
mod = '"cig91"' # deglist -5,0,5 < faster but worse
mod = '"cig92"' # deglist -5,-2.5,0,2.5,5
mod = '"cig93"' # allow dr_0 any
mod = '"cig94"' # deglist -8,-4,0,4,8
mod = '"cig95"' # open up dr_1
mod = '"cig96"' # open up dt_1 < some shape bug
mod = '"cig97"' # reshape for the rad_
mod = '"cig98"' # some test
mod = '"cig99"' # bigger dat (self.deglist = [-8, -4, 0, 4, 8])
mod = '"cig00"' # self.deglist = [-8, -6, -4, -2, 0, 2, 4, 6, 8] < killed
mod = '"cig01"' # fewer prints
mod = '"cig02"' # self.deglist = [-8, -4, 0, 4, 8]; dt_1 *= 0.5 < on test, this seems to rotate the WRONG WAY
mod = '"cig03"' # dr_0 *= 0.1, so that the object is near-straight
mod = '"cig04"' # seprate coeff (0.1) for rot part
mod = '"cig05"' # at test time, obj_r = obj_r - obj_dr (instead of +) < ok, works, but why? < actually i'm not sure this worked.
mod = '"cig06"' # at test time, obj_r = obj_r + obj_dr, but train it with negative gt, which i expect to fail < OK; on training, rot error does not descend, so this is definitely wrong
mod = '"cig07"' # restore sanity: train it with pos gt, and test with obj_r = obj_r - obj_dr because we are rotating the TEMPLATE not the target; deglist [8,0,8] < again, not sure anymore
mod = '"cig08"' # -12,0,12
mod = '"cig09"' # -6,0,6, and use angles within 180/32 (rand_r *= 0.5)

# after these train a lot, i need to just test with both directions, and see what happens on the curved set
# note that this does not affect the training; only test time

mod = '"tig00"' # pret 04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_faks30i1t_faks30i1v_faks30i1v_cig09
mod = '"tig01"' # test on curved
# mean_ious [0.96 0.92 0.9  0.86 0.82 0.79 0.77 0.75 0.73 0.71 0.7  0.68 0.67 0.66
#             0.64 0.63 0.62 0.6  0.59 0.58 0.58 0.57 0.57 0.56 0.56 0.55 0.54 0.53
#             0.52 0.51]
mod = '"tig02"' # reverse on test < higher perf. ok. so the proper update is obj_r = obj_r + obj_dr
# mean_ious [0.96 0.92 0.9  0.87 0.85 0.83 0.81 0.79 0.78 0.76 0.75 0.73 0.72 0.7
#             0.69 0.68 0.67 0.65 0.64 0.63 0.63 0.62 0.61 0.61 0.6  0.59 0.58 0.57
#             0.56 0.56]
mod = '"tig03"' # deglist -8,-4,0,4,8 instead of -6,0,6
# mean_ious [0.96 0.92 0.9  0.87 0.84 0.82 0.81 0.79 0.78 0.76 0.75 0.73 0.72 0.71
#             0.69 0.68 0.67 0.66 0.65 0.64 0.63 0.63 0.62 0.62 0.61 0.6  0.59 0.58
#             0.57 0.56]
# ok same. so those extra bins don't help a lot.
mod = '"tig04"' # deglist -6,0,6, more eval (trainset, straight, curved, etc.)

# 01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1t_curved_ns_tig04
# mean_ious [0.96 0.93 0.91 0.89 0.86 0.84 0.81 0.79 0.77 0.76 0.75 0.74 0.73 0.72
#             0.71 0.71 0.7  0.7  0.69 0.69 0.68 0.67 0.65 0.64 0.62 0.61 0.59 0.57
#             0.55 0.53]
# so even on the trainset, curved perf is low

# 01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1t_straight_ns_tig04
# mean_ious [0.97 0.94 0.93 0.92 0.9  0.88 0.85 0.82 0.81 0.8  0.8  0.79 0.79 0.78
#             0.78 0.78 0.78 0.78 0.78 0.77 0.77 0.77 0.77 0.77 0.77 0.76 0.76 0.76
#             0.75 0.75]
# (on straight, trainest perf is fine)

# 01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1v_straight_ns_tig04
# mean_ious [0.97 0.94 0.92 0.91 0.89 0.87 0.85 0.84 0.82 0.81 0.81 0.8  0.78 0.78
#             0.77 0.77 0.77 0.77 0.77 0.77 0.77 0.77 0.76 0.76 0.76 0.76 0.75 0.75
#             0.75 0.74]
# (on straight, testset perf is fine)


# EVEN WHEN the angle/directions are right, rotations are hard because the appearance of the object changes drastically
# i may be able to partially mitigate this with a smarter corr op, which uses visibility
# in general i also want to search with a prediction of the full object, rather than just what i have


############## define experiments ##############

exps['builder'] = [
    'carla_rsiamese', # mode
    'carla_train_data', # dataset
    'nearcube_traintest_bounds', 
    '10_iters',
    'train_feat3D', 
    'train_match', 
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_rsiamese', # mode
    # 'carla_train10_data', # dataset
    # 'carla_train10_data', # dataset
    # 'carla_train10_test10_data', # dataset
    # 'carla_train1_test1_data', # dataset
    # 'carla_traintest_data', # dataset
    # 'carla_train_data', # dataset
    'carla_trainvaltest_data', # dataset
    'nearcube_traintest_bounds', 
    '300k_iters',
    'train_feat3D', 
    'train_match', 
    'B4',
    # 'B8',
    'lr4', 
    'log50',
]
exps['tester'] = [
    'carla_rsiamese', # mode
    # 'carla_trainvaltest_data', # dataset
    # 'carla_test_on_train_data', # dataset
    # 'carla_test_on_curved_train_data', # dataset
    # 'carla_test_on_straight_train_data', # dataset
    'carla_test_on_straight_test_data', # dataset
    'nearcube_traintest_bounds', 
    '100_iters',
    # '10_iters',
    'no_shuf',
    'do_test', 
    'do_export_vis',
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

groups['carla_rsiamese'] = ['do_carla_rsiamese = True']
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
    'testset = "faks30i1v_curved"',
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
    'testset = "faks30i1t_curved"',
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
