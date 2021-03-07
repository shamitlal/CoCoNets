from exp_base import *

############## choose an experiment ##############

# current = 'builder'
current = 'trainer'

mod = '"r00"' # 1/4 res enc-dec
mod = '"r01"' # 1/2 res enc-dec
mod = '"r02"' # half the y dim
mod = '"r03"' # L1 instead of L2
mod = '"r04"' # do tests too
mod = '"r05"' # use L2 instead of L1
mod = '"r06"' # even narrower bounds; drop some mid layers of enc-dec; 0 to 2
mod = '"r07"' # bounds 0.5 to 2.5
mod = '"r08"' # L1 instead of L2; bounds 1 to 3
mod = '"r09"' # train occ loss too
mod = '"r10"' # same but on matrix 0-28
mod = '"r11"' # 0-36  < illegal instructino
mod = '"r12"' # 0-28 but use scratch
mod = '"r13"' # again
mod = '"r14"' # replace v2v with basic 
mod = '"r15"' # again
mod = '"r17"' # replace viewnet; 0-36 < indeed, this old viewnet is slightly better
mod = '"r18"' # do not deconv in viewnet < no diff
mod = '"r19"' # use logspace sampling < slightly better
mod = '"r20"' # aggregate before calling featnet < slightly faster
mod = '"r21"' # nothing
mod = '"r22"' # train 100k 
mod = '"r26"' # train with new vox_util, coeff 0.25
mod = '"r27"' # adjusted some occ scopes
# < i suspect that up until this point, my voxels were not actually cubes 
mod = '"r28"' # tall bounds
mod = '"r29"' # tall bounds; linear projector
mod = '"r30"' # tall bounds; linear projector; EMA
mod = '"r31"' # big bounds; linear projector; EMA
mod = '"r32"' # big bounds; linear projector; no EMA
mod = '"r33"' # input dropout coeff 0.1
mod = '"r34"' # vox_util coeff 0.1 instead of 0.25
mod = '"r35"' # input dropout coeff 0.9
mod = '"r36"' # tiny tiny smooth coeff
mod = '"r37"' # no input dropout; S=2
mod = '"r38"' # yes EMA


mod = '"r39"' # smaller dims
mod = '"r40"' # stretched vert
mod = '"r41"' # coeff = 0.0
mod = '"r42"' # train on 100
mod = '"r43"' # valid = ones
mod = '"r44"' # valid = ones; full dat
mod = '"r45"' # tiny feat smooth coeff, for interp < died
mod = '"r46"' # tiny feat smooth coeff, for interp


mod = '"r47"' # pret 100k 02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46

############## define experiment ##############

exps['builder'] = [
    'carla_vq3drgb', # mode
    # 'carla_multiview_train10_data', # dataset
    'carla_multiview_train10_val10_data', # dataset
    'carla_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat',
    'train_vq3drgb',
    'train_view',
    'fastest_logging',
]
exps['mini_trainer'] = [
    'carla_vq3drgb', # mode
    'carla_multiview_train_test_data', # dataset
    'carla_bounds', 
    '10k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_vq3drgb',
    'train_view',
    # 'pretrained_feat', 
    # 'pretrained_vq3drgb', 
    # 'pretrained_view', 
    'faster_logging',
]
exps['trainer'] = [
    'carla_vq3drgb', # mode
    # 'carla_multiview_train_data', # dataset
    'carla_multiview_train_val_data', # dataset
    # 'carla_multiview_train10_val10_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train_test_data', # dataset
    # 'carla_multiview_train100_data', # dataset
    # 'carla_bounds', 
    # 'carla_tall_bounds', 
    'carla_big_bounds', 
    '100k_iters',
    'lr5',
    'B2',
    'pretrained_feat',
    'pretrained_vq3drgb',
    'pretrained_view',
    'pretrained_occ',
    'train_feat',
    'train_vq3drgb',
    'train_view',
    'train_occ',
    # 'pretrained_feat', 
    # 'pretrained_vq3drgb', 
    # 'pretrained_view', 
    'fast_logging',
]

############## net configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 64',
    'feat_smooth_coeff = 0.01',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l2_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    'occ_smooth_coeff = 0.001',
]

############## datasets ##############

# 'XMIN = -8.0', # right (neg is left)
# 'XMAX = 8.0', # right
# 'YMIN = -1.0', # down (neg is up)
# 'YMAX = 3.0', # down
# 'ZMIN = 4.0', # forward
# 'ZMAX = 20.0', # forward

# dims for mem
SIZE = 8
Z = int(SIZE*16)
Y = int(SIZE*1)
X = int(SIZE*16)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"

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
    'trainset = "mabs7i3t"',
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
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "mabs7i3v"',
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
