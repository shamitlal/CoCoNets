from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"00"' # 
mod = '"01"' # S = 1; show grayscale
mod = '"02"' # halfsize
mod = '"03"' # basic net
mod = '"04"' # train
mod = '"05"' # train; no val set
mod = '"06"' # show recon
mod = '"07"' # again
mod = '"08"' # get samples
mod = '"09"' # get halfheight samples
mod = '"10"' # full data
mod = '"11"' # full data; 100k; B4
mod = '"12"' # full data; 100k; B4; log less freq; gen full image (deleted by accident)
mod = '"13"' # redo
mod = '"14"' # slow logging

############## define experiment ##############

exps['builder'] = [
    'carla_gengray', # mode
    'carla_multiview_train10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_gengray',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_gengray', # mode
    'carla_multiview_train_val_data', # dataset
    # 'carla_multiview_train10_val10_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    'carla_bounds', 
    '100k_iters',
    'lr3',
    'B4',
    'train_gengray',
    'slow_logging',
]

############## net configs ##############

groups['train_gengray'] = [
    'do_gengray = True',
    'gengray_coeff = 1.0',
    # 'gengray_smooth_coeff = 2.0',
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

groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
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
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
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
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
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
