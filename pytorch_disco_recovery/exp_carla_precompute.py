from exp_base import *

############## choose an experiment ##############

# current = 'precompute_builder'
current = 'precompute_full'

mod = '"precomp00"' # 
mod = '"precomp01"' # compute flowRs
mod = '"precomp02"' # compute flowRs
mod = '"precomp03"' # egocancel
mod = '"precomp04"' # typo fix
mod = '"precomp05"' # run full train or val; save outs to disk; S=7
mod = '"precomp06"' # don't save
mod = '"precomp07"' # don't save
mod = '"precomp08"' # one sess
mod = '"precomp09"' # do save outputs
mod = '"precomp10"' # do save outputs; no logging

############## define experiments ##############

exps['precompute_builder'] = [
    'carla_precompute', # mode
    'carla_precompute1_data', # dataset
    '3_iters',
    'B1',
    'no_shuf',
    'no_backprop',
    # 'save_outputs',
    'fastest_logging',
]
exps['precompute_full'] = [
    'carla_precompute', # mode
    'carla_precompute_train_data', # dataset
    # 'carla_precompute_val_data', # dataset
    'B1',
    'no_shuf',
    'no_backprop',
    'save_outputs',
    'no_logging',
]

############## group configs ##############

############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 7
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_precompute1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "picked"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
groups['carla_precompute_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cabs16i3c0o1t"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
    'max_iters = 4313',
]
groups['carla_precompute_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cabs16i3c0o1v"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
    'max_iters = 2124',
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
