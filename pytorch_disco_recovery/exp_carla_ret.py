from exp_base import *

############## choose an experiment ##############

current = 'test_hardvis_random'
# current = 'trainer'
# current = 'vq_trainer'

mod = '"ret00"' # nothing
mod = '"ret01"' # pret and freeze
mod = '"ret02"' # pret 100k instead of 90k
# pret 20k 02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_O_c.1_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq14
mod = '"ret03"' # 
mod = '"ret04"' # show inds 0 to 9 < ok, some of them are encouraging
mod = '"ret05"' # gather up some pointclouds
mod = '"ret06"' # project onto pixels
mod = '"ret07"' # draw masked rgbs
mod = '"ret08"' # bigger masks; show on R and X
mod = '"ret09"' # show 0.5 opac < ok, but one dict el i was excited about is actually floating above cars
mod = '"ret10"' # show all xyz_mem
mod = '"ret11"' # show inboundsa
mod = '"ret12"' # choose boxes and show them
mod = '"ret13"' # no enum
mod = '"ret14"' # alt util
mod = '"ret15"' # round
mod = '"ret16"' # use full xyz
mod = '"ret17"' # use half xyz R
mod = '"ret18"' # flatten before nonzero
mod = '"ret19"' # better scope
mod = '"ret20"' # print sizes
mod = '"ret21"' # again
mod = '"ret22"' # delta = 4.5 < the metric deltas are showing up as 4.5 also, which seems suspicious
mod = '"ret23"' # do it in fullres mem coords
mod = '"ret24"' # plot the sized boxes
mod = '"ret25"' # use halfdelta for y
mod = '"ret26"' # avoid the borders
mod = '"ret27"' # 
mod = '"ret28"' # show new boxes
mod = '"ret29"' # fewer
mod = '"ret30"' # again
mod = '"ret31"' # print zyx
mod = '"ret32"' # fewr
mod = '"ret33"' # fewer
mod = '"ret34"' # start z in the middle a bit
mod = '"ret35"' # start z in the middle a bit
mod = '"ret36"' # show occs too
mod = '"ret37"' # 5x5
mod = '"ret38"' # req 3 observed pts in each crop
mod = '"ret39"' # cleaned up
mod = '"ret40"' # hardvis bigname pretrain
mod = '"ret41"' # hardvis smallname pretrain
mod = '"ret42"' # hardvis bigname pretrain random crops
mod = '"ret43"' # hardvis smallname pretrain random crops

############## define experiment ##############

exps['builder'] = [
    'carla_ret', # mode
    'carla_multiview_train10_data', # dataset
    'carla_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_vq3drgb',
    'frozen_feat3D', 
    'frozen_vq3drgb', 
    'train_feat3D',
    'fastest_logging',
]

exps['test_hardvis'] = [
    'carla_ret', # mode
    'carla_multiview_test_data', # dataset
    'carla_bounds',
    'hard_vis',
    '100k_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_vq3drgb',
    'frozen_feat3D', 
    'frozen_vq3drgb', 
    'train_feat3D',
    'faster_logging',
    'break_constraint',
    # 'debug',
]

exps['test_hardvis_random'] = [
    'carla_ret', # mode
    'carla_multiview_test_data', # dataset
    'carla_bounds',
    'hard_vis_random',
    '100k_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_vq3drgb',
    'frozen_feat3D', 
    'frozen_vq3drgb', 
    'train_feat3D',
    'faster_logging',
    'break_constraint',
    # 'debug',
]

exps['trainer'] = [
    'carla_ret', # mode
    'carla_multiview_train_data', # dataset
    'carla_bounds',
    '100k_iters',
    'lr4',
    'B2',
    # 'train_feat2D',
    'train_feat3D',
    # 'train_emb2D',
    # 'train_emb3D',
    'train_vq3drgb',
    'train_linclass',
    'train_view',
    'fast_logging',
]
exps['vq_trainer'] = [
    'carla_ret', # mode
    'carla_multiview_train_data', # dataset
    # 'carla_multiview_train_test_data', # dataset
    'carla_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'train_feat2D',
    'train_feat3D',
    'train_emb2D',
    'train_emb3D',
    'train_view',
    'train_vq3drgb',
    'train_linclass',
    'train_occ',
    'pretrained_feat2D', 
    'pretrained_feat3D', 
    'pretrained_view', 
    'pretrained_vq3drgb', 
    'fast_logging',
]

############## net configs ##############

groups['carla_ret'] = ['do_carla_ret = True']

groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 64',
    # 'feat2D_smooth_coeff = 0.01',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
    'vq3drgb_num_embeddings = 512', 
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.001',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 2.0',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]

groups['hard_vis'] = [
    'do_hard_vis = True',
    'box_size_xz = 16',
    'box_size_y = 16',
    'break_constraint = True',
]

groups['hard_vis_random'] = [
    'do_hard_vis = True',
    'box_size_xz = 16',
    'box_size_y = 16',
    'break_constraint = True',
    'use_random_boxes = True'

]

groups['debug'] = [
    'do_debug = True'
]
groups['break_constraint'] = [
    'break_constraint = True'
]

############## datasets ##############

# dims for mem
SIZE = 8
Z = int(SIZE*16)
Y = int(SIZE*1)
X = int(SIZE*16)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
import socket
if "compute" in socket.gethostname():
    dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
else:
    dataset_location = "/media/mihir/dataset/carla/processed/npzs"

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


groups['carla_multiview_train100_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3hun"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]


groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mabs7i3t"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
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
groups['carla_multiview_train_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
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

