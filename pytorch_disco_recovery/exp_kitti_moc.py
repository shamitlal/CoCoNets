from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'trainer'

mod = '"k00"' # reset
mod = '"k01"' # use S=10 raw data < seems like a lot of motion
mod = '"k02"' # more dat
mod = '"k03"' # dat with static check
mod = '"k04"' # more of that, just_gif=True
mod = '"k05"' # kodo data
mod = '"k06"' # kodo data with seqlen10 and stride2
mod = '"k07"' # ks10i2a

############## define experiment ##############

exps['builder'] = [
    'kitti_moc', # mode
    # 'kitti_multiview_train10_test10_data', # dataset
    # 'kitti_train_data', # dataset
    # 'kitti_all_data', # dataset
    'kitti_odo_data', # dataset
    # 'kitti_multiview_train_data', # dataset
    'kitti_regular_bounds',
    '1k_iters',
    # '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    # 'train_feat3D',
    # 'train_feat3DS',
    # 'train_emb3D',
    # 'train_moc3D',
    # 'fastest_logging',
    'fastest_logging',
]
exps['trainer'] = [
    'kitti_moc', # mode
    # 'kitti_multiview_train_data', # dataset
    'kitti_multiview_train_test_data', # dataset
    # 'kitti_regular_bounds',
    # 'kitti_big_bounds',
    'kitti_narrower_bounds',
    '300k_iters',
    'lr4',
    'B8',
    'train_feat3D',
    'train_occ',
    # 'train_feat3DS',
    'train_emb3D',
    'slow_logging',
]

############## net configs ##############

groups['kitti_moc'] = ['do_kitti_moc = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3D'] = [
    'do_moc3D = True',
    'moc3D_num_samples = 1000',
    'moc3D_coeff = 1.0',
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
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
    'feat2D_smooth_coeff = 0.01',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
]
groups['train_feat3DS'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    'feat3D_sparse = True',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    # 'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.001',
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
Y = int(SIZE*2)
X = int(SIZE*16)

ZZ = int(SIZE*3)
ZY = int(SIZE*3)
ZX = int(SIZE*3)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 3
S_test = 8
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"
# dataset_location = "/data/kitti/processed/npzs"

# dataset_location = "/data5/kitti/processed"
dataset_location = "/data5/kitti_odometry/processed"

groups['kitti_narrow_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['kitti_narrower_bounds'] = [
    'XMIN = -3.0', # right (neg is left)
    'XMAX = 3.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -3.0', # forward
    'ZMAX = 3.0', # forward
    'Z = 96', # forward
    'Y = 32', # forward
    'X = 96', # forward
]
groups['kitti_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['kitti_big_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]


groups['kitti_train10_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train10_test10_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3ten"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train100_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1hun"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1t"',
    'trainset_format = "kitti"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_all_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1a"',
    'trainset_format = "kitti"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ks10i2a"',
    'trainset_format = "kodo"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train10_val10_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "kabs4i1ten"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train_val_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "kabs4i1v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train_val_test_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "kabs4i1v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S,
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_train_test_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "kabs4i1t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_test_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
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

