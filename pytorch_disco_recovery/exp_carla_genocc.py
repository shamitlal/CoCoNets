from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"roo00"' # roost
mod = '"roo01"' # delete unused nets 
mod = '"roo02"' # only compute occ sup
mod = '"roo03"' # clean up
mod = '"roo04"' # 
mod = '"roo05"' # add a basic genocc net
mod = '"roo06"' # compile some of the voxelcnn code
mod = '"roo07"' # 
mod = '"roo08"' # debug inputs to the genoccnet
mod = '"roo09"' # 
mod = '"roo10"' # run the thing
mod = '"roo11"' # 
mod = '"roo12"' # gen 
mod = '"roo13"' # gen and vis it
mod = '"roo14"' # gen and vis it
mod = '"roo15"' # apply loss
mod = '"roo16"' # coeff 1.0
mod = '"roo17"' # train a bit
mod = '"roo18"' # don't log som uch
mod = '"roo19"' # show input and output feats
mod = '"roo20"' # show loss vis
mod = '"roo21"' # gen occ samples (at fullres)
mod = '"roo22"' # halfres samples
mod = '"roo23"' # only generate samples on val iters
mod = '"roo24"' # do it for real
mod = '"roo25"' # do it for real
mod = '"roo26"' # val every 50
mod = '"roo27"' # show sample_feat and sample_occ
mod = '"roo28"' # show the sample directly, since i think it's 1d
mod = '"roo29"' # smaller gen (Z4)
nmod = '"roo30"' # mid gen again (Z2), but add smooth loss
mod = '"roo31"' # call it logit instead of feat
mod = '"roo32"' # fixed a scope 
mod = '"roo33"' # give the sampler access to occX_half
mod = '"roo34"' # only sample for real if it's not known to be free or occ
mod = '"roo35"' # only sample for real if it's not known to be free or occ
mod = '"roo36"' # halfres sample
mod = '"roo37"' # every 50
mod = '"roo38"' # p(drop) = 0.75
mod = '"roo39"' # use dropout mask
mod = '"roo40"' # again
mod = '"roo41"' # 10k iters < actually this was 1k
mod = '"roo42"' # 100k
mod = '"roo43"' # no dropout
mod = '"roo44"' # swap embedding with 1x1x1 conv < worse
mod = '"roo45"' # 


############## define experiment ##############

exps['builder'] = [
    'carla_genocc', # mode
    'carla_multiview_train10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_genocc',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_genocc', # mode
    'carla_multiview_train10_val10_data', # dataset
    'carla_bounds', 
    '100k_iters',
    'lr3',
    'B1',
    'train_genocc',
    'faster_logging',
]

############## net configs ##############

groups['train_genocc'] = [
    'do_genocc = True',
    'genocc_coeff = 1.0',
    'genocc_smooth_coeff = 2.0',
]

############## datasets ##############

# dims for mem
SIZE = 16
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 7
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
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mabs7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
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
