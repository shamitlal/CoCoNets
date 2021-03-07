from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"c00"' # start on training the classifier a bit; basic trainer for now
mod = '"c01"' # run empty test mode i hope
mod = '"c02"' # show the 3d boxes
mod = '"c03"' # project the 3d boxes to 2d and show them
mod = '"c04"' # turn the boxes into masks
mod = '"c05"' # clamp
mod = '"c06"' # add hyp for num embeddings
mod = '"c07"' # run and plot eval 
mod = '"c08"' # don't plot if nan
mod = '"c09"' # print more info
mod = '"c10"' # print more info
mod = '"c11"' # use cpu numpy
mod = '"c12"' # plot the nans
mod = '"c13"' # plot mean pool size
mod = '"c14"' # encapsulate
mod = '"c15"' # use two masks, so that "obj" does not include so much bkg < ok, almost no difference in eval
mod = '"c16"' # redo
mod = '"c17"' # train linclass
mod = '"c18"' # show boxes and masks
mod = '"c19"' # linclass; use more realistic vqvae
mod = '"c20"' # use EMA < seems slightly worse
mod = '"c21"' # no EMA; use quant[0:1]
mod = '"c22"' # use valid mask
mod = '"c23"' # repeat but on aws
mod = '"c24"' # pret 100k 04_s1_m64x16x64_1e-3_Vr_n512_d64_r1_l1_L_c1_mabs7i3t_c21
        
############## define experiment ##############

exps['builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_vqrgb',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_vqrgb', # mode
    # 'carla_multiview_train_val_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '100k_iters',
    'lr3',
    'B4',
    'train_vqrgb',
    'train_linclass',
    'pretrained_vqrgb',
    'faster_logging',
]
exps['stage2_builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'train_vqrgb',
    'train_gen2dvq',
    'pretrained_vqrgb',
    'fastest_logging',
]
exps['stage2_trainer'] = [
    'carla_vqrgb', # mode
    # 'carla_multiview_train_val_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B4',
    'pretrained_vqrgb',
    'train_gen2dvq',
    'slow_logging',
]
exps['stage2_radial_builder'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'pretrained_vqrgb',
    'train_sigen2d',
    'fastest_logging',
]
exps['stage2_radial_trainer'] = [
    'carla_vqrgb', # mode
    'carla_multiview_train_val_test_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B8',
    'pretrained_vqrgb',
    'frozen_vqrgb', 
    'train_sigen2d',
    'slower_logging',
]

############## net configs ##############

groups['train_vqrgb'] = [
    'do_vqrgb = True',
    'vqrgb_num_embeddings = 512',
    'vqrgb_emb_dim = 64',
    'vqrgb_recon_coeff = 1.0',
    'vqrgb_latent_coeff = 1.0',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]
groups['train_gen2dvq'] = [
    'do_gen2dvq = True',
    'gen2dvq_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]
groups['train_sigen2d'] = [
    'do_sigen2d = True',
    'sigen2d_coeff = 1.0',
]

############## datasets ##############

# dims for mem
SIZE = 16
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 1
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/scratch"

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
