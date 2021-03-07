from exp_base import *

############## choose an experiment ##############

# current = 'builder'
# current = 'trainer'
current = 'tester'
# current = 'error_finder'

mod = '"siam00"' # start 
mod = '"siam01"' # at least load tracktrack and summ it
mod = '"siam02"' # get boxes too
mod = '"siam03"' # voxelize near object
mod = '"siam04"' # on train iters, train_over_pair
mod = '"siam05"' # on test iters, track_over_seq
mod = '"siam06"' # fastest, to see the test
mod = '"siam07"' # train a while; faster
mod = '"siam08"' # narrow traintest bounds
mod = '"siam09"' # clean up; nearcube again
mod = '"siam10"' # show true traj
mod = '"siam11"' # return early on test, if necessary
mod = '"siam12"' # narrow
mod = '"siam13"' # nearcube; higher res
mod = '"siam14"' # eliminate a summ
mod = '"siam15"' # eliminate randomized centroid
mod = '"siam16"' # proper ZZ
mod = '"siam17"' # slight clenaup
mod = '"siam18"' # narrow bounds
mod = '"siam19"' # nearcube bounds; nicer vis (occ_memXs[:,0])
mod = '"siam20"' # yet nicer vis (fixing the last missed one)
mod = '"siam21"' # tester; pret 10k 02_s2_m256x128x256_1e-4_F3_d16_M_c1_ktads8i1t_ktads8i1v_siam19
# mean_ious [0.61 0.61 0.59 0.6  0.59 0.58 0.58 0.58]
mod = '"siam22"' # tester; pret 20k 02_s2_m256x128x256_1e-4_F3_d16_M_c1_ktads8i1t_ktads8i1v_siam19
# mean_ious [0.6  0.56 0.54 0.54 0.53 0.53 0.52 0.52]
# ok worse somehow. let's just move on.
mod = '"siam23"' # tester; random features
mod = '"siam24"' # use_window=True (cosine window)
# mean_ious [0.7  0.4  0.21 0.11 0.05 0.03 0.02 0.01]
mod = '"siam25"' # show corr summs
mod = '"siam26"' # tester; pret 10k siam19; cosine window
# mean_ious [0.66 0.6  0.58 0.56 0.57 0.55 0.55 0.55]
mod = '"siam27"' # tester; pret 10k siam19; cosine window; narrow traintest bounds
# mean_ious [0.68 0.53 0.45 0.41 0.38 0.36 0.35 0.34] # ok, so narrow dow not help siamese
mod = '"siam28"' # tester; pret 10k siam19; NO cosine window; nearcube traintest bounds
# mean_ious [0.61 0.61 0.59 0.6  0.59 0.58 0.58 0.58] # indeed repl of siam21
mod = '"siam29"' # tester; no pret; no cosine;
# mean_ious [0.7  0.   0.01 0.   0.   0.   0.   0.  ]


############## define experiment ##############

exps['builder'] = [
    'kitti_siamese', # mode
    # 'kitti_odo_data', # dataset
    'kitti_track_track_data', # dataset
    # 'kitti_regular_train_bounds',
    # 'kitti_narrow_test_bounds',
    'nearcube_traintest_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    # 'pretrained_feat3D',
    # 'train_feat3D',
    # 'train_emb3D',
    # 'train_moc3D',
    # 'fastest_logging',
    'fastest_logging',
]
exps['trainer'] = [
    'kitti_siamese', # mode
    'kitti_track_track_data', # dataset
    'nearcube_traintest_bounds',
    # 'narrow_traintest_bounds',
    '100k_iters',
    'lr4',
    'B2',
    # 'pretrained_feat3D',
    'train_feat3D',
    'train_match',
    'faster_logging',
]
exps['tester'] = [
    'kitti_siamese', # mode
    'kitti_testtrack_data', # dataset
    'nearcube_traintest_bounds',
    # 'kitti_narrower_bounds',
    # 'narrow_traintest_bounds',
    '1k_iters',
    # '100_iters',
    # 'no_vis',
    'no_shuf',
    'do_test', 
    'B1',
    # 'pretrained_feat3D',
    # 'pretrained_match',
    'train_feat3D',
    'train_match',
    'no_backprop',
    'faster_logging',
]
exps['error_finder'] = [
    'kitti_siamese', # mode
    'kitti_track_data', # dataset
    'kitti_narrow_test_bounds',
    '100_iters',
    'no_shuf',
    'do_test', 
    'B1',
    'pretrained_feat3D',
    'train_feat3D',
    'fastest_logging',
]

############## net configs ##############

groups['no_vis'] = ['do_include_vis = False']
groups['do_test'] = ['do_test = True']

groups['kitti_siamese'] = ['do_kitti_siamese = True']


groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 16',
    # 'feat3D_dim = 32',
    # 'feat3D_smooth_coeff = 0.01',
    # 'feat_do_sparse_invar = True', 
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
]

############## datasets ##############

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 8
H = 128
W = 416
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"
dataset_location = "/data6/kitti_processed"

SIZE = 64
SIZE_test = 64

ZZ = int(SIZE/2)
ZY = int(SIZE/2)
ZX = int(SIZE/2)

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

groups['kitti_regular_train_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*1)),
    'X = %d' % (int(SIZE*8)),
]
groups['kitti_narrow_test_bounds'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -2.0', # down (neg is up)
    'YMAX_test = 2.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*1)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['kitti_track_track_data'] = [
    'dataset_name = "ktrack"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ktads8i1t"',
    'trainset_format = "ktrack"', 
    'trainset_seqlen = %d' % S, 
    'trainset_consec = True', 
    'testset = "ktads8i1v"',
    'testset_format = "ktrack"', 
    'testset_seqlen = %d' % S_test, 
    'testset_consec = True', 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_testtrack_data'] = [
    'dataset_name = "ktrack"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "ktads8i1v"',
    'testset_format = "ktrack"', 
    'testset_seqlen = %d' % S_test, 
    'testset_consec = True', 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_track_data'] = [
    'dataset_name = "ktrack"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "ktads8i1a"',
    'testset_format = "ktrack"', 
    'testset_seqlen = %d' % S_test, 
    'testset_consec = True', 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "koads4i4a"',
    'trainset_format = "kodo"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_track_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "koads4i4a"',
    'trainset = "koacs10i2a"',
    'trainset_format = "kodo"', 
    'trainset_seqlen = %d' % S,
    'trainset_consec = False', 
    'testset = "ktads8i1a"',
    'testset_format = "ktrack"', 
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

