from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'trainer'
# current = 'match_trainer'
# current = 'tester'

mod = '"well00"' # 

############## exps ##############

exps['builder'] = [
    'kitti_entity', # mode
    'kitti_odo_trainset_data', # dataset
    'kitti_16-4-16_bounds_train',
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3d',
    'train_feat3d',
    # 'train_entity',
    'train_match',
    'log1',
]
exps['trainer'] = [
    'kitti_entity', # mode
    'kitti_odo_trainset_data', # dataset
    'kitti_odo_testset_data', # dataset
    'kitti_16-4-16_bounds_train', 
    'kitti_16-4-16_bounds_test', 
    '100k_iters',
    'lr4',
    'B1',
    'snap1k',
    'train_feat3d',
    'train_entity',
    'log50',
]
exps['match_trainer'] = [
    'kitti_entity', # mode
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
    'kitti_entity', # mode
    # 'kitti_odo_testset_data', # dataset
    'kitti_odo_testset_two_data', # dataset
    'kitti_16-4-16_bounds_test', 
    # '100_iters',
    # '10_iters',
    '3_iters',
    'lr4',
    'B1',
    'pretrained_feat3d',
    # 'pretrained_entity',
    'train_feat3d',
    'train_match',
    'no_shuf',
    'do_test',
    'log1',
    # 'log5',
]
# exps['tester'] = [
#     'kitti_entity', # mode
#     'kitti_odo_testset_data', # dataset
#     'kitti_16-4-16_bounds_test', 
#     '100_iters',
#     'lr4',
#     'B1',
#     'train_feat3d',
#     'train_entity',
#     'do_test',
#     'log50',
# ]

############## groups ##############


groups['kitti_entity'] = ['do_kitti_entity = True']
groups['do_test'] = ['do_test = True']

groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 8',
    # 'feat3d_smooth_coeff = 0.01',
]
groups['train_entity'] = [
    'do_entity = True',
    'entity_t_l2_coeff = 1.0',
    'entity_deg_l2_coeff = 0.1',
    'entity_num_scales = 2',
    'entity_num_rots = 11',
    'entity_max_deg = 4.0',
    'entity_max_disp_z = 2',
    'entity_max_disp_y = 1',
    'entity_max_disp_x = 2',
    'entity_synth_prob = 0.0',
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
