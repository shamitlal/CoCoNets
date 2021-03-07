from exp_base import *


############## choose an experiment ##############

current = 'builder'
# current = 'trainer'

mod = '"well86"' # feat3d
mod = '"well87"' # size = 16; 32-16-32
mod = '"well88"' # size = 20; 32-16-32
mod = '"well89"' # size = 24; 32-16-32
mod = '"well90"' # no feat
mod = '"well91"' # show resolution in autoname
mod = '"well92"' # be more conservative with free
mod = '"well93"' # erode once more
mod = '"well94"' # shuffle, so i can see more
mod = '"well95"' # zi smooth 0.1 instead of 0.01
mod = '"well96"' # diff coeffs

mod = '"well97"' # diff coeffs
mod = '"well98"' # do run feat3d, just to get something


mod = '"fre00"' # builder; pret frozen feat3d
mod = '"fre01"' # pret 10k 01_s2_m192x96x192_p128x384_1e-3_F3_d4_O_c1_R_d64_r10_mags7i3t_planA17
mod = '"fre02"' # show occ
mod = '"fre03"' # show rgb
mod = '"fre04"' # higher res
mod = '"fre05"' # do not double-sig < ok this fixes things

# sometimes the renders look a little odd, due to limited resolution
mod = '"fre06"' # 16-8-16
mod = '"fre07"' # show boxes
mod = '"fre08"' # show bev mask
mod = '"fre09"' # max_along_y
mod = '"fre10"' # blot out objects
mod = '"fre11"' # 32-16-32
mod = '"fre12"' # show feat_memi_input
mod = '"fre13"' # put the mask in i coords
mod = '"fre14"' # also show no mask, for comparison
mod = '"fre15"' # additive coeff 1.0
mod = '"fre16"' # 

############## exps ##############

exps['builder'] = [
    'carla_entity', # mode
    'carla_multiview_train_data', # dataset
    'carla_32-16-32_bounds_train',
    # 'carla_16-8-16_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    'no_shuf',
    'pretrained_feat3d',
    'frozen_feat3d',
    'train_feat3d',
    'train_occ',
    'train_render',
    # 'train_entity',
    # 'train_match',
    'log1',
]
exps['trainer'] = [
    'carla_entity', # mode
    'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_16-8-16_bounds_train',
    'carla_32-16-32_bounds_train',
    '5k_iters',
    'lr3',
    'B1',
    # 'no_shuf',
    # 'train_feat3d',
    'train_occ',
    'train_render',
    'log50',
]
exps['feat_trainer'] = [
    'carla_entity', # mode
    'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    'carla_16-8-16_bounds_train',
    # 'carla_32-16-32_bounds_train',
    '2k_iters',
    # '1k_iters',
    # '5k_iters',
    # '100_iters',
    'lr3',
    'B1',
    'train_feat3d',
    'train_occ',
    'train_render',
    'log50',
]

############## groups ##############


groups['carla_entity'] = ['do_carla_entity = True']
groups['do_test'] = ['do_test = True']
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 4',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_rgb_coeff = 1.0',
    # 'render_depth_coeff = 0.1',
    # 'render_smooth_coeff = 0.01',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    # 'occ_smooth_coeff = 0.1',
    'occ_smooth_coeff = 0.1',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    # 'rgb_smooth_coeff = 0.1',
]


############## datasets ##############

# # dims for mem
# SIZE = 32
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
# S = 2
S = 1
S_test = 100
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

SIZE = 24
SIZE_val = 24
SIZE_test = 24
SIZE_zoom = 24

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

# SIZE = 8
# SIZE_val = 8
# SIZE_test = 8
# SIZE_zoom = 8

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
