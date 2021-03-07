from exp_base import *

############## choose an experiment ##############

# current = 'forecast_builder'
# current = 'occ_trainer'
# current = 'forecast_trainer_small'
current = 'forecast_trainer'

mod = '"obj42"' # ah data (partial)
mod = '"obj43"' # compute and vis noyaw traj
mod = '"obj44"' # compute and vis noyaw obj occ
mod = '"obj45"' # feed noyaw inputs to forecaster
mod = '"obj46"' # big trains; yes yaw aligned
mod = '"obj47"' # NOT yaw-aligned
mod = '"obj48"' # yes yaw-aligned; ai data, to see
mod = '"obj49"' # again but bugfix on treut raj and more dat
mod = '"obj50"' # translate and untranslate xyz_camXs
mod = '"obj51"' # untranslate with noyaw
mod = '"obj52"' # use -xyz0 for the _T_
mod = '"obj53"' # forget that 
mod = '"obj54"' # get yaw version inside forecast net and plot that separately
mod = '"obj55"' # fix he pls
mod = '"obj56"' # t = 0
mod = '"obj57"' # center it
mod = '"obj58"' # center it other way
mod = '"obj59"' # ONLY center it: -xyz0, then +xyz_mid
mod = '"obj60"' # matmul3
mod = '"obj61"' # noyaw_occ
mod = '"obj62"' # noyaw_occ in vis
mod = '"obj63"' # big trainer; yes yaw thing
mod = '"obj64"' # not yaw thing
mod = '"obj65"' # yes yaw thing; use _bal
mod = '"obj66"' # yes yaw thing; use balt, balv (more data shifted to trainset)
mod = '"obj67"' # 200k; fast logging, in case the curve is deceiving me; partial aj data
mod = '"obj68"' # 200k; fast logging, in case the curve is deceiving me; more aj data
mod = '"spirit00"' # new al data
mod = '"spirit01"' # do not use exp weights
mod = '"spirit02"' # use exp weights; 16m cube instead of 10
mod = '"spirit03"' # use input>0 as a comp mask for a sparse_invar_bottle3D
mod = '"spirit04"' # use input>0 as a comp mask for a sparse_invar_bottle3D; new cleaner data 
mod = '"spirit05"' # regular bottle3D
mod = '"spirit06"' # sparse invar; regress with no weightmask; still plot both; pret 170k 04_s20_m96x64x96_z64x64x64_1e-4_Fo_e1_ami2s20t_ami2s20v_spirit04


############## define experiments ##############

exps['forecast_builder'] = [
    'intphys_forecast',
    # 'intphys_train_data_aws',
    'intphys_train10_data_aws',
    'intphys_bounds',
    '10_iters',
    'train_forecast',
    'B1',
    'no_shuf',
    'fastest_logging',
]
exps['occ_trainer'] = [
    'intphys_forecast',
    # 'intphys_train_data_aws',
    # 'intphys_trainval_data_aws',
    'intphys_debug_data_aws',
    'intphys_bounds',
    'lr3', 
    '10k_iters',
    # '1k_iters',
    # '1k_iters',
    # '100_iters',
    # '20_iters',
    # '30_iters',
    'train_feat',
    'train_occ',
    # 'no_shuf',
    'B1',
    'faster_logging',
]
exps['forecast_trainer_small'] = [
    'intphys_forecast',
    # 'intphys_train10_data_aws',
    'intphys_train_data_aws',
    'intphys_bounds',
    'lr4', 
    '1k_iters',
    'train_forecast',
    # 'no_shuf',
    'B1',
    'faster_logging',
]
exps['forecast_trainer'] = [
    'intphys_forecast',
    'intphys_trainval_data_aws',
    'intphys_bounds',
    'lr5', 
    '200k_iters',
    # 'train_feat',
    'train_forecast',
    'pretrained_forecast',
    'B4',
    'fast_logging',
]

############## group configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
    # 'feat_smooth_coeff = 0.01',
    'feat_do_sparse_conv = True', 
    # 'feat_do_sparse_invar = True', 
]
groups['train_pri'] = [
    'do_pri = True',
    'pri_ce_coeff = 1.0',
    # 'pri_reg_coeff = 1.0',
    'pri_smooth_coeff = 1.0',
]
groups['train_rpo'] = [
    'do_rpo = True',
    'rpo_forward_coeff = 1.0',
    'rpo_reverse_coeff = 1.0',
]
groups['train_forecast'] = [
    'do_forecast = True',
    # 'forecast_maxmargin_coeff = 1.0',
    # # 'forecast_smooth_coeff = 0.1', 
    # 'forecast_num_negs = 500',
    'forecast_l2_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_det'] = [
    'do_det = True',
    'det_prob_coeff = 1.0',
    'det_reg_coeff = 1.0',
]


############## datasets ##############

# mem resolution
SIZE = 16
X = int(SIZE*6)
Y = int(SIZE*4)
Z = int(SIZE*6)

ZX = int(SIZE*4)
ZY = int(SIZE*4)
ZZ = int(SIZE*4)

# these params need cleaning; also, 3 only works if you do not count occluders
N = 3 # max objects
K = 3 # objects to consider 

S = 20
H = 288
W = 288
V = 50000
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['intphys_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -7.0', # down (neg is up)
    'YMAX = 5.0', # down
    'ZMIN = 0.1', # forward
    'ZMAX = 12.1', # forward
]    
groups['intphys_trainval_data_matrix'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "train"',
    'valset = "val"',
    'dataset_format = "raw"',
    'dataset_location = "/projects/katefgroup/datasets/intphys/train"',
]
groups['intphys_trainval_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ami2s20t"',
    'valset = "ami2s20v"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
]
groups['intphys_train_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ami2s20t"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
]
groups['intphys_train10_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ami2s20ten"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
]
groups['intphys_train1_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "ami2s20one"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
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
