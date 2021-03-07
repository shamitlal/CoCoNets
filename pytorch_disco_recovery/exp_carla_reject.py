from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"rej00"' # reset
mod = '"rej01"' # pret
mod = '"rej02"' # faster logging; frozen feats
mod = '"rej03"' # narrow bounds
mod = '"rej04"' # smarter prep (centroid puts points inbounds)
mod = '"rej05"' # rejectnet!
mod = '"rej06"' # yes backprop
mod = '"rej07"' # show inputs
mod = '"rej08"' # show inputs better
mod = '"rej09"' # fire it in test mode
mod = '"rej10"' # 10k iters; S=2 instead of 4
mod = '"rej11"' # collect the outputs and summ in a gif < wrong vis in test mode
mod = '"rej12"' # don't log so much < wrong vis in test mode
mod = '"rej13"' # get the right vis
mod = '"rej14"' # just a linear mapping,
mod = '"rej15"' # nonlin; hidden=128 instead of 64 < helps
mod = '"rej16"' # show input vis too
mod = '"rej17"' # train for 30k instead of 10k
mod = '"rej18"' # show that * occ
mod = '"rej19"' # show dyn and sta, rounded

############## define experiment ##############

exps['builder'] = [
    'carla_reject', # mode
    # 'carla_multiview_train10_test10_data', # dataset
    'carla_multiview_test_data', # dataset
    # 'carla_multiview_train_data', # dataset
    'carla_regular_bounds',
    '1k_iters',
    # '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_reject', # mode
    # 'carla_multiview_train_data', # dataset
    'carla_multiview_train_test_data', # dataset
    # 'carla_regular_bounds',
    # 'carla_big_bounds',
    # 'carla_narrower_bounds',
    'carla_narrow_bounds',
    '30k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D', 
    'train_feat3D', # just to load the right hyps
    'frozen_feat3D',
    'train_reject3D',
    'fast_logging',
]

############## net configs ##############

groups['carla_reject'] = ['do_carla_reject = True']

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
groups['train_reject3D'] = [
    'do_reject3D = True',
    'reject3D_ce_coeff = 1.0',
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

