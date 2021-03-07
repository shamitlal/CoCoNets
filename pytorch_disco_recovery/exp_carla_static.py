from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer_basic'
# current = 'tester_basic'

mod = '"ret00"' # goal: set up fancy retrieval metric that uses semantic labels across different scenes
mod = '"ret01"' # load pret guy; run a few test iters
mod = '"ret02"' # fix test bugs
mod = '"ret03"' # feed none into weight loader
mod = '"ret04"' # show boxes in X and R
mod = '"ret05"' # show boxes in X and R
mod = '"ret06"' # show masks
mod = '"ret07"' # show masks
mod = '"ret08"' # clean up the model; show plural masks
mod = '"ret09"' # 
mod = '"ret10"' # do multiview and multiscene precision
mod = '"ret11"' # cleaned up
mod = '"ret12"' # 1k iters, to see more eval
mod = '"ret13"' # make precision 0.0 if nothing valid
mod = '"ret14"' # print isnan
mod = '"ret15"' # only append if not isnan
mod = '"ret16"' # use shape[0] instead of len
mod = '"ret17"' # 
mod = '"ret18"' # return early if not at least 2 obj vox in each; (the last 17 did this too) < ok looks stable
mod = '"ret19"' # clean up
mod = '"ret20"' # clean up

############## define experiment ##############

exps['builder'] = [
    'carla_static', # mode
    'carla_multiview_10_data', # dataset
    'carla_bounds', 
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'fastest_logging',
]
exps['tester_basic'] = [
    'carla_static', # mode
    'carla_multiview_test_data', # dataset
    'carla_bounds', 
    '1k_iters',
    # 'lr3',
    'B2',
    'train_feat',
    'train_occ',
    'no_backprop',
    'pretrained_feat', 
    'pretrained_occ',
    'frozen_feat', 
    'frozen_occ', 
    'fastest_logging',
]
exps['trainer_basic'] = [
    'carla_static', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds', 
    '300k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'fast_logging',
]
exps['trainer_accu'] = [
    'carla_sta', # mode
    'carla_stat_stav_data', # dataset
    '200k_iters',
    # 'carla_stat_stav_data', # dataset
    # '200k_iters',
    'lr3',
    'B4',
    'train_feat',
    'train_occ',
    'train_view_accu_render_unps_gt',
    'train_emb2D',
    'train_emb3D',
    'faster_logging',
]
exps['trainer_render'] = [
    'carla_sta', # mode
    'carla_sta1_data', # dataset
    '10k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_occ_no_coeffs',
    'train_render',
    'faster_logging',
]
exps['res_trainer'] = [
    'carla_sta', # mode
    'carla_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B4',
    'train_feat_res',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'faster_logging',
    'resume'
]
exps['res_trainer_accu_render'] = [
    'carla_sta', # mode
    'carla_sta_data', # dataset
    '200k_iters',
    'lr3',
    'B4',
    'train_feat_res',
    'train_occ',
    'train_view_accu_render_unps_gt',
    'faster_logging',
    'resume'
]
exps['emb_trainer'] = [
    'carla_sta', # mode
    'carla_static_data', # dataset
    '300k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_occ',
    'train_emb_view',
    'faster_logging',
]


############## net configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
    # 'feat_cluster = True',
    # 'feat_cluster_num_objs_views = 3000', # Set to num_samples*views in dataset.
    # 'feat_quantize = True',
    # 'feat_quantize_dictsize = 512',
    # 'feat_quantize_init = "vqvae/kmeans_cluster_centers.npy"',
    # 'feat_do_rt = True',
    # 'feat_do_flip = True',
]
groups['train_feat_res'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_rt = True',
    'feat_do_flip = True',
    'feat_do_resnet = True',
]
groups['train_feat_sb'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_sb = True',
    'feat_do_resnet = True',
    'feat_do_flip = True',
    'feat_do_rt = True',
]
groups['train_occ_no_coeffs'] = [
    'do_occ = True',
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
groups['train_render'] = [
    'do_render = True',
    'render_depth = 32',
    'render_l1_coeff = 1.0',
]
groups['train_view_accu_render'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
]
groups['train_view_accu_render_unps_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_unps = True',
    'view_accu_render_gt = True',
]
groups['train_view_accu_render_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_gt = True',
]
groups['train_occ_notcheap'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
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
############## datasets ##############

# dims for mem
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*1)
X = int(SIZE*4)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_multiview_10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
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
