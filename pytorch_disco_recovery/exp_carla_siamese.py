from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
# current = 'tester'

mod = '"ta00"' # here i took over model_carla_siamese, using code from ssiamese mostly
mod = '"ta01"' # 
mod = '"ta02"' # trainer
mod = '"ta03"' # better autoname
mod = '"ta04"' # new log dir; log500
mod = '"ta05"' # better scopes
mod = '"ta06"' # 000[0,1] data
mod = '"ta07"' # more data; [0,3]
mod = '"ta08"' # more data; [0,3]
mod = '"ta09"' # full [0,3]
mod = '"ta10"' # tester
mod = '"ta11"' # tester with cleaner setup
mod = '"ta12"' # trainer again
mod = '"ta13"' # test every 50
mod = '"ta14"' # S_test = 50
mod = '"ta15"' # log50
mod = '"ta16"' # log500; train 100k

############## define experiments ##############

exps['builder'] = [
    'carla_siamese', # mode
    # 'carla_train_data', # dataset
    # 'carla_traintest_data', # dataset
    # 'carla_traintest1_data', # dataset
    'carla_traj_train_data', # dataset
    'carla_16-16-16_bounds_train', 
    '10_iters',
    # 'pretrained_feat3d', 
    'train_feat3d', 
    'train_match', 
    # 'train_rigid', 
    # 'train_robust', 
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    'log1',
]
exps['trainer'] = [
    'carla_siamese', # mode
    'carla_traj_data_train', # dataset
    'carla_traj_data_test', # dataset
    'carla_16-16-16_bounds_train', # bounds
    'carla_16-16-16_bounds_test', # bounds
    '100k_iters',
    'train_feat3d', 
    'train_match', 
    'B4',
    'lr4', 
    'log500',
]
exps['tester'] = [
    'carla_siamese', # mode
    'carla_traj_data_test', # dataset
    'carla_16-16-16_bounds_test', 
    '1k_iters',
    # '100_iters',
    # 'no_vis',
    'no_shuf',
    'do_test', 
    'B1',
    # 'pretrained_feat3d',
    # 'pretrained_match',
    'train_feat3d',
    'train_match',
    'no_backprop',
    'log50',
]

############## net configs ##############

groups['do_test'] = ['do_test = True']

groups['include_summs'] = [
    'do_include_summs = True',
]
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 64',
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
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
