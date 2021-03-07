from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
# current = 'vq_trainer'

mod = '"au00"' # nothing
mod = '"au01"' # encode in R and pred X0
mod = '"au02"' # add vq and linclass; less logging
mod = '"au03"' # better hyp name; 100k iters

############## define experiment ##############

exps['builder'] = [
    'carla_auto', # mode
    'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train10_val10_data', # dataset
    'carla_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat3D',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_auto', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds',
    '100k_iters',
    'lr4',
    'B2',
    # 'train_feat2D',
    'train_feat3D',
    # 'train_emb2D',
    # 'train_emb3D',
    'train_vq3drgb',
    'train_linclass',
    'train_view',
    'fast_logging',
]
exps['vq_trainer'] = [
    'carla_auto', # mode
    'carla_multiview_train_data', # dataset
    # 'carla_multiview_train_test_data', # dataset
    'carla_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'train_feat2D',
    'train_feat3D',
    'train_emb2D',
    'train_emb3D',
    'train_view',
    'train_vq3drgb',
    'train_linclass',
    'train_occ',
    'pretrained_feat2D', 
    'pretrained_feat3D', 
    'pretrained_view', 
    'pretrained_vq3drgb', 
    'fast_logging',
]

############## net configs ##############

groups['carla_auto'] = ['do_carla_auto = True']

groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 64',
    # 'feat2D_smooth_coeff = 0.01',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
    'vq3drgb_num_embeddings = 512', 
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.001',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 2.0',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]


############## datasets ##############

# dims for mem
SIZE = 8
Z = int(SIZE*16)
Y = int(SIZE*1)
X = int(SIZE*16)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 1
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
dataset_location = "/data/carla/processed/npzs"

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

