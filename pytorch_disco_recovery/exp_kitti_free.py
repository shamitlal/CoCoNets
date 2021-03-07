from exp_base import *

############## choose an experiment ##############

# the idea here is to do some bkg subtraction with NO PARAMETERS -- i.e., for free

current = 'builder'
# current = 'trainer'
current = 'tester'

mod = '"kit00"' # builder; kitti_free start, copied from carla_free
# 100 iters
# mean_proposal_maps_3d [0.16 0.13 0.09 0.05 0.03 0.   0.  ]
# mean_proposal_maps_2d [0.17 0.14 0.1  0.07 0.04 0.02 0.  ]
# mean_proposal_maps_pers [0.18 0.15 0.15 0.13 0.1  0.08 0.04]
# mean_track_maps_3d [0.01 0.01 0.01 0.   0.   0.   0.  ]
# mean_track_maps_2d [0.03 0.01 0.01 0.01 0.01 0.   0.  ]
# mean_track_maps_pers [0.03 0.02 0.01 0.01 0.01 0.   0.  ]

mod = '"kit01"' # 200 iters; return before tracking; stationary camera
# mean_proposal_maps_3d [0.11 0.09 0.06 0.03 0.01 0.   0.  ]
# mean_proposal_maps_2d [0.11 0.1  0.08 0.05 0.02 0.01 0.01]
# mean_proposal_maps_pers [0.13 0.12 0.11 0.08 0.06 0.03 0.02]

mod = '"kit02"' # do not return early for inb
# mean_proposal_maps_3d [0.12 0.1  0.07 0.05 0.03 0.01 0.  ]
# mean_proposal_maps_2d [0.13 0.11 0.09 0.06 0.04 0.03 0.02]
# mean_proposal_maps_pers [0.15 0.13 0.12 0.1  0.08 0.05 0.02]
# actually better

mod = '"kit03"' # any cam motion; any inb
# mean_proposal_maps_3d [0.19 0.14 0.1  0.06 0.04 0.02 0.  ]
# mean_proposal_maps_2d [0.21 0.16 0.12 0.08 0.06 0.05 0.03]
# mean_proposal_maps_pers [0.27 0.21 0.17 0.14 0.11 0.08 0.04]
mod = '"kit04"' # 100 iters; show summs
# mean_proposal_maps_3d [0.19 0.14 0.1  0.07 0.04 0.02 0.  ]
# mean_proposal_maps_2d [0.21 0.16 0.12 0.09 0.06 0.05 0.03]
# mean_proposal_maps_pers [0.26 0.21 0.17 0.14 0.11 0.08 0.04]

mod = '"kit05"' # show summs; use 0.99 to measure movemnet

############## define experiment ##############

exps['builder'] = [
    'kitti_free', # mode
    # 'kitti_multiview_train_data', # dataset
    # 'kitti_multiview_test_data', # dataset
    # 'kitti_tatt_trainset_data', # dataset
    'kitti_tafa_testset_data', # dataset
    # 'kitti_16-16-16_bounds_train',
    'kitti_16-16-16_bounds_test',
    # 'kitti_32-32-32_bounds_test',
    # '10k_iters',
    '1k_iters',
    # '100_iters',
    # '20_iters',
    # '10_iters',
    # '5_iters',
    'use_cache', 
    'lr4',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'train_sub',
    'no_backprop',
    # 'do_test', 
    'log1',
    # 'log10',
    # 'log50',
    # 'log5',
]
exps['trainer'] = [
    'kitti_free', # mode
    # 'kitti_tat1_trainset_data', # dataset
    # 'kitti_tat10_trainset_data', # dataset
    'kitti_tat100_trainset_data', # dataset
    'kitti_16-16-16_bounds_train',
    # 'kitti_16-16-16_bounds_val',
    '100k_iters',
    'lr3',
    'B1',
    'use_cache',
    'pretrained_feat3D', 
    'pretrained_occ',
    'pretrained_occrel',
    'frozen_feat3D', 
    'frozen_occ', 

    'frozen_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'train_sub',
    'log10',
]
exps['tester'] = [
    'kitti_free', # mode
    'kitti_tafa_testset_data', # dataset
    'kitti_16-16-16_bounds_train',
    'kitti_16-16-16_bounds_test',
    # 'kitti_16-16-16_bounds_zoom',
    # 'kitti_12-12-12_bounds_zoom',
    'kitti_8-8-8_bounds_zoom',
    # 'kitti_8-4-8_bounds_zoom',
    '100_iters',
    # '20_iters',
    # '15_iters',
    'lr4',
    'B1',
    'use_cache',
    'no_shuf',
    # 'pretrained_feat3D', 
    # 'pretrained_occ', 
    # 'pretrained_occrel', 
    # 'train_feat3D',
    # 'train_occ',
    # 'train_occrel',
    # 'train_flow', 
    'no_backprop',
    'do_test', 
    'log1',
]
# exps['tester'] = [
#     'kitti_free', # mode
#     # 'kitti_tast_trainset_data', # dataset
#     'kitti_tatv_testset_data', # dataset
#     'kitti_16-8-16_bounds_train',
#     'kitti_16-8-16_bounds_test',
#     '25_iters',
#     # '10k_iters',
#     # '100k_iters',
#     # 'do_test', 
#     'B1',
#     'pretrained_feat3D', 
#     'pretrained_occ',
#     'pretrained_mot',
#     'frozen_feat3D', 
#     'frozen_occ', 
#     'frozen_mot', 
#     'train_feat3D',
#     'train_occ',
#     'train_mot',
#     'log1',
# ]
exps['render_trainer'] = [
    'kitti_free', # mode
    'kitti_multiview_train_data', # dataset
    'kitti_multiview_ep09_data', # dataset
    # 'kitti_multiview_ep09one_data', # dataset
    # 'kitti_multiview_one_data', # dataset
    # 'kitti_wide_nearcube_bounds',
    # 'kitti_nearcube_bounds',
    'kitti_narrow_nearcube_bounds',
    # 'kitti_narrow_flat_bounds',
    # '5k_iters',
    '500k_iters',
    'lr3',
    'B1',
    'pretrained_latents',
    # 'train_vq3d',
    # 'train_up3D',
    'train_occ',
    'train_render',
    # 'no_shuf',
    'snap50',
    'log50',
    # 'log50',
]
exps['center_trainer'] = [
    'kitti_free', # mode
    'kitti_multiview_train_data', # dataset
    'kitti_wide_cube_bounds',
    '100k_iters',
    'lr3',
    'B2',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_center', 
    'train_feat3D',
    'train_occ',
    'train_center',
    'log50',
]
exps['seg_trainer'] = [
    'kitti_free', # mode
    'kitti_multiview_all_data', # dataset
    'kitti_wide_cube_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D', 
    'pretrained_up3D', 
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'train_feat3D',
    'train_up3D',
    'train_occ',
    'train_center',
    'train_seg',
    'snap5k',
    'log500',
]
exps['vq_trainer'] = [
    'kitti_free', # mode
    'kitti_multiview_all_data', # dataset
    'kitti_wide_cube_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D', 
    'pretrained_vq3d', 
    'pretrained_up3D',
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'train_feat3D',
    'train_up3D',
    'train_occ',
    'train_center',
    'train_seg',
    'train_vq3d',
    # # 'frozen_feat3D',
    # 'frozen_up3D',
    # # 'frozen_vq3d',
    # 'frozen_occ',
    # 'frozen_center',
    # 'frozen_seg',
    'snap5k',
    'log500',
]
exps['vq_exporter'] = [
    'kitti_free', # mode
    'kitti_multiview_all_data_as_test', # dataset
    'kitti_wide_cube_bounds',
    '5k_iters', # iter more than necessary, since we have some augs
    # '100_iters', 
    'no_shuf',
    'do_test', 
    'do_export_inds', 
    'lr4',
    'B1',
    'pretrained_feat3D', 
    'pretrained_up3D', 
    'pretrained_vq3d', 
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'frozen_feat3D',
    'frozen_up3D',
    'frozen_vq3d',
    'frozen_occ',
    'frozen_center',
    'frozen_seg',
    'log50',
]

############## net configs ##############

groups['do_test'] = ['do_test = True']
groups['do_export_inds'] = ['do_export_inds = True']
groups['use_cache'] = ['do_use_cache = True']
groups['kitti_free'] = ['do_kitti_free = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat3D_skip = True',
]
groups['train_flow'] = [
    'do_flow = True',
    # 'flow_l1_coeff = 1.0',
    'flow_l2_coeff = 1.0',
    'flow_heatmap_size = 7',
]
groups['train_up3D'] = [
    'do_up3D = True',
    # 'up3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ce_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    # 'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_vq3d'] = [
    'do_vq3d = True',
    'vq3d_latent_coeff = 1.0',
    'vq3d_num_embeddings = 512', 
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l1_coeff = 1.0',
    # 'view_smooth_coeff = 1.0',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_l2_coeff = 10.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    # 'occ_smooth_coeff = 0.1',
]
groups['train_occrel'] = [
    'do_occrel = True',
    'occrel_coeff = 1.0',
]
groups['train_sub'] = [
    'do_sub = True',
    'sub_coeff = 1.0',
    'sub_smooth_coeff = 2.0',
]
groups['train_center'] = [
    'do_center = True',
    'center_prob_coeff = 1.0',
    'center_size_coeff = 0.1', # this loss tends to be large
    'center_rot_coeff = 1.0',
]
groups['train_seg'] = [
    'do_seg = True',
    'seg_prob_coeff = 1.0',
    'seg_smooth_coeff = 0.001',
]
groups['train_mot'] = [
    'do_mot = True',
    'mot_prob_coeff = 1.0',
    'mot_smooth_coeff = 0.01',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]


############## datasets ##############

# dims for mem
# SIZE = 20
# Z = int(SIZE*16)
# Y = int(SIZE*16)
# X = int(SIZE*16)
# SIZE = 20
# Z = 180
# Y = 60
# X = 180
# Z_test = 180
# Y_test = 60
# X_test = 180

# SIZE = 16
# SIZE_test = 16
SIZE = 20
SIZE_val = 20
SIZE_test = 20
SIZE_zoom = 20

# Z = 160
# Y = 80
# X = 160
# Z_test = 160
# Y_test = 80
# X_test = 160
# # Z = 128
# Y = 64
# X = 128
# Z_test = 128
# Y_test = 64
# X_test = 128

K = 2 # how many objects to consider
N = 16 # how many objects per npz
S = 100
S_val = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"
# dataset_location = "/data/kitti/processed/npzs"
# dataset_location = "/data4/kitti/processed/npzs"

groups['kitti_tafa_testset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tafs50i1a"',
    'testset_format = "ktrack"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_train1_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_all_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mads7i3a"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_ep09_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_ep09one_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_one_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_all_data_as_test'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mads7i3a"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_train10_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_train100_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3hun"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_train_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_val_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "mags7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_test_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_multiview_train10_val10_data'] = [
    'dataset_name = "kitti"',
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
groups['kitti_multiview_train_val_data'] = [
    'dataset_name = "kitti"',
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
groups['kitti_multiview_train_val_test_data'] = [
    'dataset_name = "kitti"',
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
groups['kitti_multiview_train_test_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_taqv_testset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tasa_testset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tass100i2a"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tat1_trainset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2one"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tatt_trainset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2t"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tat10_trainset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2ten"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tat100_trainset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2hun"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tatv_testset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_tatv_testset_data'] = [
    'dataset_name = "kitti"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['kitti_16-8-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['kitti_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['kitti_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_8-8-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -8.0', # down (neg is up)
    'YMAX_zoom = 8.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['kitti_16-16-16_bounds_zoom'] = [
    'XMIN_zoom = -16.0', # right (neg is left)
    'XMAX_zoom = 16.0', # right
    'YMIN_zoom = -16.0', # down (neg is up)
    'YMAX_zoom = 16.0', # down
    'ZMIN_zoom = -16.0', # forward
    'ZMAX_zoom = 16.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['kitti_12-12-12_bounds_zoom'] = [
    'XMIN_zoom = -12.0', # right (neg is left)
    'XMAX_zoom = 12.0', # right
    'YMIN_zoom = -12.0', # down (neg is up)
    'YMAX_zoom = 12.0', # down
    'ZMIN_zoom = -12.0', # forward
    'ZMAX_zoom = 12.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['kitti_8-4-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -4.0', # down (neg is up)
    'YMAX_zoom = 4.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*4)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['kitti_32-32-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_16-8-16_bounds_val'] = [
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['kitti_16-8-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_8-4-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    # 'XMIN_test = -12.0', # right (neg is left)
    # 'XMAX_test = 12.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -12.0', # forward
    # 'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_12-6-12_bounds_test'] = [
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -6.0', # down (neg is up)
    'YMAX_test = 6.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_8-6-8_bounds_test'] = [
    # 'XMIN_test = -8.0', # right (neg is left)
    # 'XMAX_test = 8.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -8.0', # forward
    # 'ZMAX_test = 8.0', # forward
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -9.0', # down (neg is up)
    'YMAX_test = 9.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*6)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['kitti_8-8-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]

groups['kitti_313_bounds'] = [
    'XMIN = -18.0', # right (neg is left)
    'XMAX = 18.0', # right
    'YMIN = -6.0', # down (neg is up)
    'YMAX = 6.0', # down
    'ZMIN = -18.0', # forward
    'ZMAX = 18.0', # forward
]
groups['kitti_flat_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['kitti_narrow_nearcube_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['kitti_narrow_flat_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['kitti_cube_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['kitti_wide_nearcube_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]
groups['kitti_wide_cube_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -32.0', # down (neg is up)
    'YMAX = 32.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
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

