from exp_base import *

############## choose an experiment ##############

current = 'stage2_builder'
current = 'stage2_trainer'

# at 16x16x16 resolution, it takes 5.2 seconds, so 787v/s
# at 32x32x32 resolution, i expect it to take 42 seconds < actually it's 292, so 5 mins; 112v/s

mod = '"bag00"' # nothning
mod = '"bag01"' # gen3d mode
mod = '"bag02"' # builder
mod = '"bag03"' # run things in R
mod = '"bag04"' # run test mode too; add viewnet to test
mod = '"bag05"' # return quant and quant_up
mod = '"bag06"' # S=3
mod = '"bag07"' # S=4; train
mod = '"bag08"' # S=2; train
mod = '"bag09"' # S into autoname; builder again
mod = '"bag10"' # builder; fixed batch issue
mod = '"bag11"' # trainer
mod = '"bag12"' # add freezing to autonames
mod = '"bag13"' # try higher res
mod = '"bag14"' # try higher res; fixed a 4/8 bug; builder
mod = '"bag15"' # trainer; fast logging  < this is too frequent, since 250 iters takes only 90s
mod = '"bag16"' # trainer; slower logging 
mod = '"bag17"' # focal loss
mod = '"bag18"' # slowest logging
mod = '"bag19"' # regular CE

############## define experiment ##############

exps['stage2_builder'] = [
    'clevr_gen3dvq', # mode
    'clevr_multiview_train_val_test_data', # dataset
    'clevr_bounds', 
    '10_iters',
    'lr0',
    'B2',
    'pretrained_feat',
    'pretrained_vq3drgb',
    'pretrained_view',
    'frozen_feat', 
    'frozen_vq3drgb', 
    'frozen_view', 
    'train_gen3dvq',
    'fastest_logging',
]
exps['stage2_trainer'] = [
    'clevr_gen3dvq', # mode
    'clevr_multiview_train_val_test_data', # dataset
    'clevr_bounds', 
    '300k_iters',
    'lr3',
    'B2',
    'pretrained_feat',
    'pretrained_vq3drgb',
    'pretrained_view',
    'frozen_feat', 
    'frozen_vq3drgb', 
    'train_gen3dvq',
    'frozen_view', # to render
    'slowest_logging',
]

############## net configs ##############

groups['clevr_vq3drgb'] = ['do_clevr_vq3drgb = True']
groups['clevr_gen3dvq'] = ['do_clevr_gen3dvq = True']

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_num_embeddings = 512',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l2_coeff = 1.0',
]
groups['train_gen3dvq'] = [
    'do_gen3dvq = True',
    'gen3dvq_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]

############## datasets ##############

# dims for mem
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*4)
X = int(SIZE*4)
S = 4
H = 256
W = 256
V = 65536
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# there seem to be 24 views per pickle
groups['clevr_multiview_train_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train_val_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "bgv"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train_val_test_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "bgv"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "bgv"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
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
