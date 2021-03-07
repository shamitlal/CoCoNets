from exp_base import *

# the idea here is to train and evaluate dense object nets (DON) in 2d and 2.5d settings

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
current = 'tester'

mod = '"don00"' # start
mod = '"don01"' # eliminate Rs and occXs summ
mod = '"don02"' # show rely masks
mod = '"don03"' # show rely masks again
mod = '"don04"' # show rely masks again
mod = '"don05"' # compute "ok" mask directly
mod = '"don06"' # project ok mask into image
mod = '"don07"' # show X1 stuff too
mod = '"don08"' # run featnet2d
mod = '"don09"' # upsamp
mod = '"don10"' # add empty tri net
mod = '"don11"' # sample and show some shapes
mod = '"don12"' # fix shape bug; use single
mod = '"don13"' # add loss computation
mod = '"don14"' # train a while
mod = '"don15"' # subsample early; experiment with num_samples < higher num_samples improves stability
mod = '"don16"' # use slow net; use low resolution on purpose
mod = '"don17"' # eliminated some computation of unused tensors
mod = '"don18"' # Net2D instead of v2v2d
mod = '"don19"' # parallelize
mod = '"don20"' # go_slow; use native grid sample < err because the sampling is not grid shaped
mod = '"don21"' # parallelize again; return early=True if loss=0
mod = '"don22"' # only return early if valid==0
mod = '"don23"' # (same)
mod = '"don24"' # just don't backprop if loss=0
mod = '"don25"' # fast logging


mod = '"test00"' # test a bit
mod = '"test01"' # show 2d boxes
mod = '"test02"' # gif of 2d boxes
mod = '"test03"' # get and show feat_memX0s
mod = '"test04"' # use valid mask in the vis
mod = '"test05"' # get code from zoom mode
mod = '"test06"' # show obj masks < does not seem centered
mod = '"test07"' # use narrow test bounds
mod = '"test08"' # use the right hyps
mod = '"test09"' # bugfix
mod = '"test10"' # fewer prints
mod = '"test11"' # flesh out
mod = '"test12"' # 1k; faster logging; collect stats; going well but super slow
mod = '"test13"' # only run featnet2d once < still super slow
mod = '"test14"' # half that res, to make it apples with 3d model, and also faster
mod = '"test15"' # pret 02_s2_1e-4_F2_d64_s.001_T2_e100_d1_mabs7i3t_don23
mod = '"test16"' # faster logging
mod = '"test17"' # pret don22


mod = '"test18"' # pret 02_s2_1e-4_F2_d64_T2_e100_d1_mabs7i3t_don25
mod = '"test19"' # same but higher res
mod = '"test20"' # log every iter
mod = '"test21"' # load the right ckpt


mod = '"co00"' # col2d < ok let's wait and see
mod = '"co01"' # test sometimes too
mod = '"co02"' # resolve bounds/size issue
mod = '"co03"' # train a while; test every 250
mod = '"co04"' # no test

mod = '"co05"' # tester; pret 10k 02_s2_1e-4_F2_d64_C2_h100_mabs7i3t_co00
mod = '"co06"' # faster logging
mod = '"co07"' # really load
# OOM at 117 iters, but:
# 0.93 0.41 0.32 0.29 0.27 0.25 0.2  0.19


############## define experiment ##############

exps['builder'] = [
    'carla_don', # mode
    # 'carla_multiview_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_regular_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat2D',
    'train_tri2D',
    # 'train_emb3D',
    # 'train_moc3D',
    # 'fastest_logging',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_don', # mode
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_train_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_regular_bounds',
    'narrow_test_bounds',
    '300k_iters',
    'lr5',
    'B2',
    'train_feat2D',
    # 'train_tri2D',
    'train_col2D',
    'fast_logging',
]
exps['tester'] = [
    'carla_don', # mode
    'carla_multiview_test_data', # dataset
    'narrow_test_bounds',
    '1k_iters',
    'no_shuf',
    'do_test', 
    'B1',
    'pretrained_feat2D',
    'no_backprop',
    'train_feat2D',
    'faster_logging',
]

############## net configs ##############

groups['carla_don'] = ['do_carla_don = True']
groups['no_vis'] = ['do_include_vis = False']
groups['do_test'] = ['do_test = True']


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
groups['train_tri2D'] = [
    'do_tri2D = True',
    # 'tri_2D_ml_coeff = 1.0',
    # 'tri_2D_l2_coeff = 0.1',
    'tri_2D_num_samples = 100',
    'tri_2D_ce_coeff = 1.0',
]
groups['train_col2D'] = [
    'do_col2D = True',
    # 'col2D_l1_coeff = 1.0',
    'col2D_huber_coeff = 100.0',
]
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 64',
    # 'feat2D_smooth_coeff = 0.01',
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
SIZE_test = 32
Z = int(SIZE*16)
Y = int(SIZE*2)
X = int(SIZE*16)

ZZ = int(SIZE*3)
ZY = int(SIZE*3)
ZX = int(SIZE*3)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 8
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"

dataset_location = "/data4/carla/processed/npzs"

groups['carla_narrow_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_narrower_bounds'] = [
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
groups['narrow_test_bounds'] = [
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

groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_big_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]


groups['carla_multiview_some_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabsome"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
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
groups['carla_multiview_train10_test10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3ten"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
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

