from exp_base import *

############## choose an experiment ##############

current = 'pwc_builder'
current = 'pwc_eval'

mod = '"test0"'
mod = '"t00"' # compute and show depth
mod = '"t01"' # compute and show gt egoflow
mod = '"t02"' # 
mod = '"t03"' # backwarp
mod = '"t04"' # compute and vis the valid mask
mod = '"t05"' # compute and vis stabflow_e
mod = '"t06"' # also show flow_e-stabflow_g
mod = '"t07"' # compute 3d stabilized flow*occ
mod = '"t08"' # show both 3d flows
mod = '"t09"' # backwarp the stabdepth to get a 3d displacement field and vis 
mod = '"t10"' # transpose 
mod = '"t11"' # *valid
mod = '"t12"' # also alt version
mod = '"t13"' # mask with inbound0 for the 2d vis
mod = '"t14"' # clean v0 to v3
mod = '"t15"' # discover and vis, with v0
mod = '"t16"' # do things at half res, because the coeffs are tuned for that
mod = '"t17"' # do not vis; do compute cumu maps
mod = '"t18"' # do not vis; do compute cumu maps
mod = '"t19"' # compute all cumu maps
mod = '"t20"' # compute all cumu maps; 1k
mod = '"t21"' # include v5, which should be better than v3


############## define experiments ##############

exps['pwc_builder'] = [
    'carla_pwc', # mode
    'carla_moving_data', # dataset
    # 'kitti_pwc_data', # dataset
    '10_iters',
    'lr4',
    'B1',
    'no_shuf',
    'no_backprop',
    'fastest_logging',
    'include_summs',
]
exps['pwc_eval'] = [
    'carla_pwc', # mode
    'carla_moving_data', # dataset
    '1k_iters',
    'B1',
    'no_shuf',
    'no_backprop',
    'fastest_logging',
    'eval_map',
]

############## net configs ##############


############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_moving_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "cabs16i3c0o1t"',
    'valset = "cabs16i3c0o1v"',
    'dataset_list_dir = "/data/carla/npzs"',
    'dataset_location = "/data/carla/npzs"',
    'dataset_format = "npz"',
]
groups['kitti_pwc_data'] = ['dataset_name = "kitti"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caas2i1c0o1t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/kitti/npzs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/kitti/npzs"',
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
