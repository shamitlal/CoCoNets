from exp_base import *

############## choose an experiment ##############

current = 'det_builder'
current = 'det_trainer'

mod = '"reset01"' # go all 1k
mod = '"reset02"' # fix rgb_camR
mod = '"reset03"' # uncomment the rest
mod = '"reset04"' # multiview train
mod = '"reset05"' # new multiview data
mod = '"reset06"' # allow bikes; more dat; train a bit
mod = '"reset07"' # 
mod = '"reset08"' # diff ious
mod = '"reset09"' # S=1
mod = '"reset10"' # only skip if half the batch is bad
mod = '"reset11"' # more dat
mod = '"reset12"' # more dat
mod = '"reset13"' # split data from model
mod = '"reset14"' # 0.0,None,True
mod = '"reset15"' # ten data
mod = '"reset16"' # trainval
mod = '"reset17"' # better dat name


mod = '"stand00"' # builder
mod = '"stand01"' # B2
mod = '"stand02"' # put summs into scope det/
mod = '"stand03"' # cleaned one more summ
mod = '"stand04"' # show map; 5k
mod = '"stand05"' # parse boxes properly
mod = '"stand06"' # log50

mod = '"stand07"' # detnet standalone, with occ_memX0 as input
mod = '"stand08"' # log50
mod = '"stand09"' # log500; B4 < slow at first but then it sped up to normal.
mod = '"stand10"' # summ feat
mod = '"stand11"' # num_workers=2
mod = '"stand12"' # allow rgbd input

############## define experiments ##############

exps['det_builder'] = [
    'carla_det', # mode
    'carla_multiview_10_data', # dataset
    'carla_16-8-16_bounds_train',
    # '3_iters',
    '5k_iters',
    'lr3', 
    'train_feat',
    'train_det',
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    'log50',
]
exps['det_trainer'] = [
    'carla_det', # mode
    'carla_multiview_train_val_data', # dataset
    'carla_16-8-16_bounds_train', 
    'carla_16-8-16_bounds_val', 
    '200k_iters',
    'lr4',
    'B2',
    # 'train_feat',
    'train_det',
    'log500', 
]

############## group configs ##############

groups['train_feat'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    'feat3D_skip = True',
]
groups['train_det'] = [
    'do_det = True',
    'det_prob_coeff = 1.0',
    'det_reg_coeff = 1.0',
    'snap_freq = 5000',
]


############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)

K = 8 # how many proposals to consider
N = 8 # how many objects in each npz

H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

S = 1
groups['carla_multiview_10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mags7i3v"',
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
