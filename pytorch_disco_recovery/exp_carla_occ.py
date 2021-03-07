from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
current = 'tester'

# the idea here is to improve occ and make it not depend on featnet

mod = '"occ00"' # ok
mod = '"occ01"' # preocc
mod = '"occ02"' # fancier preocc, with real net
mod = '"occ03"' # higher S
mod = '"occ04"' # use preocc coeffs
mod = '"occ05"' # be more conservative with "free" labels: erode by 1 < much easier to fit, apparently
mod = '"occ06"' # random amount of dropout of input, so that it completes more
mod = '"occ07"' # use density_coeff, which is a bit clenaer
mod = '"occ08"' # show input; turn off smooth coeff
mod = '"occ09"' # random centroid
mod = '"occ10"' # use 16-16-16 bounds


mod = '"exp00"' # tester; pret 5k 02_s4_1e-4_P_c1_s.01_mags7i3t_occ09
mod = '"exp01"' # tester; log10
mod = '"exp02"' # fps=16
# mean_proposal_maps_3d [0.58 0.52 0.39 0.21 0.07 0.01 0.  ]
# mean_proposal_maps_2d [0.59 0.54 0.46 0.33 0.19 0.08 0.03]
# mean_proposal_maps_pers [0.64 0.6  0.57 0.51 0.41 0.24 0.09]
mod = '"exp03"' # pret 10k 02_s2_1e-4_P_c1_s.01_mags7i3t_occ09
# mean_proposal_maps_3d [0.59 0.54 0.42 0.24 0.07 0.01 0.  ]
# mean_proposal_maps_2d [0.6  0.56 0.49 0.37 0.23 0.09 0.03]
# mean_proposal_maps_pers [0.64 0.61 0.58 0.53 0.42 0.25 0.09]
mod = '"exp04"' # pret 10k 02_s4_1e-4_P_c1_s.01_mags7i3t_occ09
# mean_proposal_maps_3d [0.57 0.53 0.43 0.27 0.11 0.02 0.  ]
# mean_proposal_maps_2d [0.58 0.55 0.48 0.38 0.25 0.12 0.04]
# mean_proposal_maps_pers [0.63 0.59 0.56 0.52 0.43 0.27 0.12]
# ok, slightly better at the higher iou
mod = '"exp05"' # do not use diff = occ*vis*diff; just diff straight
mod = '"exp06"' # diff = diff * vis_memXAI which has been dilated twice
# mean_proposal_maps_3d [0.45 0.32 0.19 0.09 0.02 0.   0.  ]
# mean_proposal_maps_2d [0.49 0.37 0.25 0.14 0.07 0.02 0.01]
# mean_proposal_maps_pers [0.55 0.48 0.41 0.33 0.24 0.12 0.04]
# worse than exp04, so, it's better to mult by occ when collecting boxes
mod = '"exp07"' # same
mod = '"exp08"' #
# mean_proposal_maps_3d [0. 0. 0. 0. 0. 0. 0.]
# mean_proposal_maps_2d [0. 0. 0. 0. 0. 0. 0.]
# mean_proposal_maps_pers [0. 0. 0. 0. 0. 0. 0.]


############## define experiment ##############

exps['builder'] = [
    'carla_occ', # mode
    # 'carla_multiview_all_data', # dataset
    'carla_multiview_ep09_data', # dataset
    # 'carla_multiview_ep09one_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train_data', # dataset
    # 'carla_nearcube_bounds',
    # 'carla_nearcube_bounds',
    'carla_313_bounds',
    # '100_iters',
    '10_iters',
    # '3_iters',
    'lr4',
    'B1',
    'no_shuf',
    # 'pretrained_feat3D', 
    # 'pretrained_up3D', 
    # 'pretrained_center', 
    # 'pretrained_center', 
    # 'pretrained_seg', 
    'train_feat3D',
    'train_emb3D',
    # 'train_up3D',
    # 'train_center',
    # 'train_view',
    'train_occ',
    # 'train_render',
    # 'no_backprop', 
    'log1',
]
exps['trainer'] = [
    'carla_occ', # mode
    # 'carla_multiview_train_test_data', # dataset
    'carla_multiview_train_data', # dataset
    # 'carla_16-8-16_bounds_train',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_test',
    # 'carla_16-16-16_bounds_train',
    # 'carla_8-4-8_bounds_test',
    # 'carla_12-6-12_bounds_test',
    '300k_iters',
    'lr4',
    'B2',
    # 'pretrained_feat3D',
    # 'pretrained_occ',
    # 'train_feat3D',
    # 'train_emb3D',
    # 'train_occ',
    'train_preocc',
    'log500',
    'snap5k',
]
exps['tester'] = [
    'carla_occ', # mode
    'carla_tatv_testset_data', # dataset
    'carla_16-16-16_bounds_train',
    'carla_16-16-16_bounds_test',
    # 'carla_16-16-16_bounds_zoom',
    # 'carla_12-12-12_bounds_zoom',
    'carla_8-8-8_bounds_zoom',
    # 'carla_8-4-8_bounds_zoom',
    '100_iters',
    # '20_iters',
    # '15_iters',
    'lr4',
    'B1',
    # 'use_cache',
    'no_shuf',
    'pretrained_preocc', 
    'train_preocc',
    'no_backprop',
    'do_test', 
    'log10',
]

############## net configs ##############

groups['do_test'] = ['do_test = True']
groups['carla_occ'] = ['do_carla_occ = True']

groups['train_preocc'] = [
    'do_preocc = True',
    'preocc_coeff = 1.0',
    # 'preocc_smooth_coeff = 0.01',
]

############## datasets ##############

SIZE = 20
SIZE_val = 20
SIZE_test = 20
SIZE_zoom = 20

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
PH = int(H)
PW = int(W)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data4/carla/processed/npzs"

groups['carla_tatv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
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
groups['carla_multiview_all_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mads7i3a"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_all_data_as_test'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mads7i3a"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train100_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3hun"',
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
groups['carla_multiview_train10_val10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3ten"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "mabs7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
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
groups['carla_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-8-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -8.0', # down (neg is up)
    'YMAX_zoom = 8.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_16-16-16_bounds_zoom'] = [
    'XMIN_zoom = -16.0', # right (neg is left)
    'XMAX_zoom = 16.0', # right
    'YMIN_zoom = -16.0', # down (neg is up)
    'YMAX_zoom = 16.0', # down
    'ZMIN_zoom = -16.0', # forward
    'ZMAX_zoom = 16.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_12-12-12_bounds_zoom'] = [
    'XMIN_zoom = -12.0', # right (neg is left)
    'XMAX_zoom = 12.0', # right
    'YMIN_zoom = -12.0', # down (neg is up)
    'YMAX_zoom = 12.0', # down
    'ZMIN_zoom = -12.0', # forward
    'ZMAX_zoom = 12.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_8-4-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -4.0', # down (neg is up)
    'YMAX_zoom = 4.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*4)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
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
groups['carla_8-4-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    # 'XMIN_test = -12.0', # right (neg is left)
    # 'XMAX_test = 12.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -12.0', # forward
    # 'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_12-6-12_bounds_test'] = [
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -6.0', # down (neg is up)
    'YMAX_test = 6.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-6-8_bounds_test'] = [
    # 'XMIN_test = -8.0', # right (neg is left)
    # 'XMAX_test = 8.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -8.0', # forward
    # 'ZMAX_test = 8.0', # forward
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -9.0', # down (neg is up)
    'YMAX_test = 9.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*6)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-8-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
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

