from exp_base import *


############## choose an experiment ##############

current = 'builder'
current = 'trainer'

# i believe what i am realy proposing here is basically an octree,
# but potentially with an irregular structure
# octree code is outdated in caffe, but the successor is the MinkowskiEngine
# to build what i want, how about a generator designed after reconstruction.py from minko
# and you can even concat the partial info you have, at each resolution's block

mod = '"oct00"' # start on minko
mod = '"oct01"' # again
mod = '"oct02"' # again
mod = '"oct03"' # full data
mod = '"oct04"' # eliminated most of the train/test code 
mod = '"oct05"' # show occ

# ok
# i stepped through reconstruction.py and i think i get it
# what i want here is:
# fit a Generator to some carla examples
# batchsize 1
# hacky post-hoc sparsification 
# fixed centroid
# densify and summ_occ

mod = '"oct06"' #
mod = '"oct07"' # minko loss
mod = '"oct08"' # train
mod = '"oct09"' # summ
mod = '"oct10"' # completor < no. generator again, since completor is complaining
mod = '"oct11"' # custom, with just cleaner names
mod = '"oct12"' # 
mod = '"oct13"' # use xyz_memX0_all
mod = '"oct14"' # show occ_memX0_all
mod = '"oct15"' # completor
mod = '"oct16"' # xyz_mem = ins[xyz_mem] < fail
mod = '"oct17"' # batched_sparse_collate
mod = '"oct18"' # print min_coord
mod = '"oct19"' # add min_coord.reshape
mod = '"oct20"' # SIZE=16
mod = '"oct21"' # same
mod = '"oct22"' # only penalize using last targets
# ok, optimization is slower, but i think it'll get there
mod = '"oct23"' # print some shapes so i know more about out_cls
mod = '"oct24"' # get and use pos_coord, via meshgrid3d
mod = '"oct25"' # balanced loss
mod = '"oct26"' # shuffle neg and trim to the size of pos
mod = '"oct27"' # one more layer in the middle! < slower
mod = '"oct28"' # shallower convs < ok faster; still, 3-5s/iter
mod = '"oct29"' # shallower convs
mod = '"oct30"' # only count it as occ if sig>0.5 < faulty reasoning
mod = '"oct31"' # shallower: 8,8,16,16,32,32,
mod = '"oct32"' # full data; cap to 10k pos/neg pts
mod = '"oct33"' # do not grab min coord
mod = '"oct34"' # grab min from sparse(); use max 5k pos, 10k neg

############## exps ##############

exps['builder'] = [
    'carla_minko', # mode
    'carla_multiview_train10_data', # dataset
    # 'carla_16-8-16_bounds_train',
    'carla_32-32-32_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    # 'no_shuf',
    'no_backprop',
    # 'pretrained_feat3d',
    # 'train_feat3d',
    # 'train_occ',
    # 'train_minko',
    'train_sigen3d',
    # 'train_match',
    'log1',
]
exps['trainer'] = [
    'carla_minko', # mode
    # 'carla_multiview_train10_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_32-16-32_bounds_train',
    '100k_iters',
    'lr4',
    'B1',
    'log50',
]

############## groups ##############

groups['carla_minko'] = ['do_carla_minko = True']
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
groups['train_minko'] = [
    'do_minko = True',
    'minko_stages = 1',
    'minko_coeff = 2.0',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    # 'rgb_smooth_coeff = 0.1',
]
groups['train_sigen3d'] = [
    'do_sigen3d = True',
    'sigen3d_coeff = 1.0',
    'sigen3d_reg_coeff = 0.1',
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

SIZE = 16
SIZE_val = 16
SIZE_test = 16
SIZE_zoom = 16

# SIZE = 12
# SIZE_val = 12
# SIZE_test = 12
# SIZE_zoom = 12

# SIZE = 10
# SIZE_val = 10
# SIZE_test = 10
# SIZE_zoom = 10

# SIZE = 8
# SIZE_val = 8
# SIZE_test = 8
# SIZE_zoom = 8

# SIZE = 4
# SIZE_val = 4
# SIZE_test = 4
# SIZE_zoom = 4

dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
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
