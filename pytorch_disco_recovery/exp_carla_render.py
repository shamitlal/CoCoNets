from exp_base import *


############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"planA00"' # go for it all day
mod = '"planA01"' # run featnet
mod = '"planA02"' # occ
mod = '"planA03"' # print some shapes, so i can see if cropping/padding is necessary
mod = '"planA04"' # again
mod = '"planA05"' # encoder-decoder
mod = '"planA06"' # train 10k
mod = '"planA07"' # random centroid
mod = '"planA08"' # show warped feat
mod = '"planA09"' # show actual warped feat
mod = '"planA10"' # render view
mod = '"planA11"' # wider 
mod = '"planA12"' # really do that i guess
mod = '"planA13"' # encode in X0
mod = '"planA14"' # 16-8-16 again
mod = '"planA15"' # 32-16-32 again; 100k iters; be more conservative with "free"
mod = '"planA16"' # narrower centroid range; log500
mod = '"planA17"' # fixed saver bug; pret 10k
mod = '"planA18"' # pret 10k A17; even higher resolution 

############## exps ##############

exps['builder'] = [
    'carla_render', # mode
    'carla_multiview_train_data', # dataset
    'carla_16-8-16_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    'no_shuf',
    # 'pretrained_feat3d',
    'train_feat3d',
    'train_occ',
    # 'train_render',
    # 'train_match',
    'log1',
]
exps['trainer'] = [
    'carla_render', # mode
    'carla_multiview_train_data', # dataset
    # 'carla_16-8-16_bounds_train',
    'carla_32-16-32_bounds_train',
    '100k_iters',
    'lr3',
    'B1',
    'pretrained_feat3d',
    'train_feat3d',
    'train_occ',
    'train_render',
    # 'train_rgb',
    'log500',
]

############## groups ##############

groups['carla_render'] = ['do_carla_render = True']
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
    # 'occ_smooth_coeff = 0.1',
    # 'occ_smooth_coeff = 0.01',
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
S = 2
# S = 1
S_test = 100
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

SIZE = 32
SIZE_val = 32
SIZE_test = 32
SIZE_zoom = 32

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
