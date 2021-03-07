from exp_base import *

############## choose an experiment ##############

# current = 'builder'
# current = 'trainer'
# current = 'match_trainer'
current = 'tester'

mod = '"play00"' # nothing; copy from carla 
mod = '"play01"' # new kodo data
mod = '"play02"' # 100; test every 10 iters
mod = '"play03"' # run feat/ego across the seq, a bit inefficiently, but without summs
mod = '"play04"' # pret 1k 01_s2_m128x32x128_1e-4_F3_d32_G_2x11x2x1x2_r4_t1_d1_kafs8i1t_kafs64i1v_play02
mod = '"play05"' # use _gN
mod = '"play06"' # use s < _e looks ok, but xyz_g looks too jumpy
mod = '"play07"' # show rgbs < indeed, rgbs seem shuffled
mod = '"play08"' # no shuf < ok resolved, but i want to noshuf 
mod = '"play09"' # shuf, but put kodo into the if case
mod = '"play10"' # summ _e and _g traj < hm, i cannot see either one
mod = '"play11"' # centroid 0,0,0 < ok, they seem to be going backwards
mod = '"play12"' # get cam0s_T_camIs and plot the clist
mod = '"play13"' # noshuf
mod = '"play14"' # show occs
mod = '"play15"' # alt vis/computation method
mod = '"play16"' # 5 iters; 6k ckpt
mod = '"play17"' # measure and plot epe
mod = '"play18"' # better scope
mod = '"play19"' # show traj_e/g in same plot
mod = '"play20"' # train a while
mod = '"play21"' # more complete data

mod = '"mat00"' # start on matchnet; builder; show occ_rs
mod = '"mat01"' # egonet again, but with total synth
mod = '"mat02"' # trainer with egonet and synth on every step
mod = '"mat03"' # builder; matchnet
mod = '"mat04"' # builder; matchnet; 0-16
mod = '"mat05"' # get and show rgb_rs too
mod = '"mat06"' # feats too
mod = '"mat07"' # pret 01_s2_m128x32x128_1e-4_F3_d32_G_2x11x2x1x2_r4_t1_d1_kafs8i1t_kafs64i1v_play21 < not that pretty
mod = '"mat08"' # trim
mod = '"mat09"' # random rt is only translation; degs is [0]
mod = '"mat10"' # get corr  
mod = '"mat11"' # relu
mod = '"mat12"' # zero mot < ok nice, centroid is hot. and note that if there are no border cheats, i can just train this centroid directly
mod = '"mat13"' # use degs, get corrs
mod = '"mat14"' # get xyz_e, rad_e
mod = '"mat15"' # deglist [0-]
mod = '"mat16"' # allow some t
mod = '"mat17"' # compute backwap
mod = '"mat18"' # .round()
mod = '"mat19"' # show unaligned
mod = '"mat20"' # output inverse
mod = '"mat21"' # output camX_T_cam0; train a bit
mod = '"mat22"' # log10; do not test
mod = '"mat23"' # train10
mod = '"mat24"' # show occ_mems_0
mod = '"mat25"' # include rot
mod = '"mat26"' # alt way to extract r and t parts
mod = '"mat27"' # deglist [-6,0,6]
mod = '"mat28"' # print rad_g too
mod = '"mat29"' # no relu
mod = '"mat30"' # mse
mod = '"mat31"' # test sometimes
mod = '"mat32"' # pret 1k 01_s2_m128x32x128_z32x32x32_1e-5_M_c1_r.1_F3_d32_kafs8i1ten_mat30
mod = '"mat33"' # total_loss += mean_epe 
mod = '"mat34"' # log500
mod = '"mat35"' # full data
mod = '"mat36"' # do not show zoom res, since match is part of ego now; lr4
mod = '"mat37"' # size 10
mod = '"mat38"' # size 6
mod = '"mat39"' # size 8
# ok interesting; not all of these succeeded. there seems to be some dependence on initialization
mod = '"mat40"' # t_amount = 2.0
mod = '"mat41"' # tester; pret 54k 01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kafs8i1t_kafs64i1v_mat39
# testing on 500 gave some error while loading the data; 100 seems to work
mod = '"mat42"' # use vox_util_wide, which has an extra 64m on each side
mod = '"mat43"' # keep the voxels cubes
mod = '"mat44"' # bigger
mod = '"mat45"' # use mean of xyz_g as centroid ([-6, 0, 6])
mod = '"mat46"' # deglist [-6, -3, 0, 3, 6] < much better!
mod = '"mat47"' # deglist [-4, -2, 0, 2, 4] < slightly better still
mod = '"mat48"' # deglist [-4, -2, -1, 0, 1, 2, 4] < worse; too stiff now
mod = '"mat49"' # deglist [-4, -2, 0, 2, 4]; ai data
mod = '"mat50"' # more dat; 10 iters
mod = '"mat51"' # shift things into matchnet; apply rot loss in degs
mod = '"mat52"' # pret 
mod = '"mat53"' # log10, since i don't believe it < somehow, corrs are empty
mod = '"mat54"' # again, but no test iters
mod = '"mat55"' # show template and rots within match scope
mod = '"mat56"' # -6,0,6
mod = '"mat57"' # old method
mod = '"mat58"' # new method; feed trimmed
mod = '"mat59"' # pret 54k mat39; train with -4,-2,0,2,4 ; log500; test 100

mod = '"mat60"' # test with matchnet
mod = '"mat61"' # pret 1k 01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kafs8i1t_kais100i1v_mat59
mod = '"mat62"' # again but sigma=1

mod = '"mat63"' # pret 54k mat39; train; log500; test 100; use new ai data (with 100k pts and no phantom centroids)
mod = '"mat64"' # pret 54k mat39; train; log500; test 100; really use new ai data (with 100k pts and no phantom centroids)



mod = '"map00"' # test 10 iters; no map yet; pret 1k 01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kais8i1t_kais100i1v_mat64
mod = '"map01"' # slightly cleaner impl of the chaining
mod = '"map02"' # aggregate the entire map
mod = '"map03"' # really do that < nice. this improves results in all but one case, which had a hard turn. i think this is a good reason to add a coarse search
mod = '"map04"' # add occ stable vis (in 0 coords)
mod = '"map05"' # show _e stab; use moving g centroid; fps=16
mod = '"map06"' # show _g stab too
mod = '"map07"' # show those stabs at higher res
mod = '"map08"' # show diffs
mod = '"map09"' # do not darken bkg for traj vis; show better diff vis; use *2 res, for speed
mod = '"map10"' # make the wide vis 818
mod = '"map11"' # narrower Y for wide vis
mod = '"map12"' # two step!
mod = '"map13"' # print the partial-to-1 mat
mod = '"map14"' # 100 iters; log5
mod = '"map15"' # 10 iters; log1; slightly higher res
mod = '"map16"' # add feat vis of mapped area 
mod = '"map17"' # share the wide vis coords across both; use a more stable feat vis, by doing pca of all timesteps at once
mod = '"map18"' # pret 12k 01_s2_m128x32x128_1e-4_M_c1_r.1_F3_d8_kais8i1t_kais100i1v_mat64 (instead of 1k)
mod = '"map19"' # try those two tests < we seem to get 09 twice
mod = '"map20"' # print filename and wait < indeed
mod = '"map21"' # num_workers=1
mod = '"map22"' # drop_last=False
mod = '"map23"' # run orbslam 
mod = '"map24"' # run our model
mod = '"map25"' # measure bev epe of ours
mod = '"map26"' # measure bev epe of orb
mod = '"map27"' # ours; no map
mod = '"map28"' # ours; no map; no second stage
mod = '"map29"' # load _e from disk, representing 2.5d sfmnet
mod = '"map30"' # slightly diff loading strategy
mod = '"map31"' # read sfm results
mod = '"map32"' # use cam_T_velo
mod = '"map34"' # use 1:4
mod = '"map35"' # do not use cams_T_velos
mod = '"map36"' # 
mod = '"map37"' # load gt < matches!
mod = '"map38"' # use assoc and such
mod = '"map39"' # get e and g
mod = '"map40"' # optimize scaling


############## exps ##############

exps['builder'] = [
    'kitti_ego', # mode
    'kitti_odo_trainset_data', # dataset
    'kitti_16-4-16_bounds_train',
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3d',
    'train_feat3d',
    # 'train_ego',
    'train_match',
    'log1',
]
exps['trainer'] = [
    'kitti_ego', # mode
    'kitti_odo_trainset_data', # dataset
    'kitti_odo_testset_data', # dataset
    'kitti_16-4-16_bounds_train', 
    'kitti_16-4-16_bounds_test', 
    '100k_iters',
    'lr4',
    'B1',
    'snap1k',
    'train_feat3d',
    'train_ego',
    'log50',
]
exps['match_trainer'] = [
    'kitti_ego', # mode
    'kitti_odo_trainset_data', # dataset
    'kitti_odo_testset_data', # dataset
    'kitti_16-4-16_bounds_train', 
    'kitti_16-4-16_bounds_test', 
    '100k_iters',
    'lr4',
    'B1',
    'snap1k',
    'pretrained_feat3d',
    'train_feat3d',
    'train_match',
    'log500',
    # 'log10',
]
exps['tester'] = [
    'kitti_ego', # mode
    # 'kitti_odo_testset_data', # dataset
    'kitti_odo_testset_two_data', # dataset
    'kitti_16-4-16_bounds_test', 
    # '100_iters',
    # '10_iters',
    '3_iters',
    'lr4',
    'B1',
    'pretrained_feat3d',
    # 'pretrained_ego',
    'train_feat3d',
    'train_match',
    'no_shuf',
    'do_test',
    'log1',
    # 'log5',
]
# exps['tester'] = [
#     'kitti_ego', # mode
#     'kitti_odo_testset_data', # dataset
#     'kitti_16-4-16_bounds_test', 
#     '100_iters',
#     'lr4',
#     'B1',
#     'train_feat3d',
#     'train_ego',
#     'do_test',
#     'log50',
# ]

############## groups ##############


groups['kitti_ego'] = ['do_kitti_ego = True']
groups['do_test'] = ['do_test = True']

groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 8',
    # 'feat3d_smooth_coeff = 0.01',
]
groups['train_ego'] = [
    'do_ego = True',
    'ego_t_l2_coeff = 1.0',
    'ego_deg_l2_coeff = 0.1',
    'ego_num_scales = 2',
    'ego_num_rots = 11',
    'ego_max_deg = 4.0',
    'ego_max_disp_z = 2',
    'ego_max_disp_y = 1',
    'ego_max_disp_x = 2',
    'ego_synth_prob = 0.0',
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
    'match_r_coeff = 0.1', 
]

############## datasets ##############

# # dims for mem
# SIZE = 32
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 100
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

SIZE = 8
SIZE_val = 8
SIZE_test = 8
SIZE_zoom = 8

# dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"
dataset_location = "/projects/katefgroup/datasets/kitti_odometry/processed"

groups['kitti_16-4-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*16)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*16)),
]
groups['kitti_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*16)),
    'Y = %d' % (int(SIZE*16)),
    'X = %d' % (int(SIZE*16)),
]
groups['kitti_16-4-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*16)),
]
groups['kitti_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*16)),
    'X_test = %d' % (int(SIZE_test*16)),
]
groups['kitti_odo_train10_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kais8i1ten"',
    'trainset_format = "kodo"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_trainset_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kais8i1t"',
    'trainset_format = "kodo"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_testset_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    # 'testset = "kafs64i1v"',
    'testset = "kais100i1v"',
    'testset_format = "kodo"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_testset_two_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "kais100i1two"', # this is seqs 09, 10, starting from frame 0
    'testset_format = "kodo"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
