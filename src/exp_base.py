import pretrained_nets_carla as pret_carla
import socket
hostname = socket.gethostname()

exps = {}
groups = {}

############## dataset settings ##############

# dataset_location = "/data/carla/processed/npzs"
import getpass
username = getpass.getuser()
if "matrix" in hostname:
    dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
    if username == "mprabhud":
        home_dataset_location = "/home/mprabhud/dataset/carla/processed/npzs"
    else:
        home_dataset_location = "/home/shamitl/datasets/carla/processed/npzs"
else:
    # dataset_location = "/media/mihir/dataset/carla/npzs"
    dataset_location = "/home/mihir/datasets/coconets/processed/npzs"    
    # dataset_location = "/home/mihirpd_google_com/datasets/coconets/processed/npzs"
    home_dataset_location = dataset_location


S_train = 2
S_val = 2
S_test = 10

H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

SIZE_train = 8
SIZE_val = 8
SIZE_test = 8

ZZ = int(SIZE_train*8)
ZY = int(SIZE_train*8)
ZX = int(SIZE_train*8)

data_mod = 'ag'
data_mod = 'ah'
data_mod = 'ab'



groups['carla_complete_data_test_for_traj'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'pseudo_traj = True',
    'testset = "complete_smallv"',
    'testset_format = "complete"', 
    'S_test = 10',
    'testset_seqlen = %d' % 10, 
    'dataset_location = "%s"' % home_dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_complete_data_test_for_traj_closecam'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'pseudo_traj = True',
    'testset = "new_complete_ad_s10_i1v"',
    'testset_format = "complete"', 
    'S_test = 10',
    'testset_seqlen = %d' % 10, 
    'dataset_location = "%s"' % home_dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_complete_data_train_for_multiview'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'pseudo_multiview = True',
    'trainset = "complete_smallt"',
    'trainset_format = "complete"', 
    'trainset_seqlen = %d' % 1, 
    'dataset_location = "%s"' % home_dataset_location,
    'dataset_filetype = "npz"'
]



groups['carla_multiview_data_train'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'trainset = "sm%ss5i8t"' % 'ab',
    'trainset_format = "surveil_multiview"', 
    'trainset_seqlen = %d' % S_train, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_drone_traj_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 768,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'testset = "dtaas50i1a"',
    'testset_format = "surveil_traj"', 
    'S_test = 10',
    'testset_seqlen = %d' % 10, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_drone_traj_test_data_latest_closecam'] = [
    'dataset_name = "carla"',
    'H = %d' % 256,
    'W = %d' % 768,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'testset = "new_drone_traj_aa_s50_i1t"',
    'testset_format = "surveil_traj"', 
    'S_test = 10',
    'testset_seqlen = %d' % 10, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_drone_traj_test_data_temp'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "m%ss7i3t"' % data_mod,
    # 'trainset = "surveil_traj_%s_s50_i1_new"' % data_mod,
    # 'trainset = "st%ss50i1t"' % data_mod,
    'testset = "stahs50i1t"',
    'testset_format = "surveil_traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_traj_data_train'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "taus100i2t"',
    'trainset_format = "oldtraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S_train, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_data_test'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taus100i2v"',
    'testset_format = "oldtraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_sta_data_single_car'] = [
    'dataset_name = "carla_mix"',
    'H = %d' % 256,
    'W = %d' % 256,
	'PH = int(H)',
	'PW = int(W)',
    'trainset = "mc_cart"',
    'low_res = True',
    # 'root_keyword = "katefgroup"',
    'trainset_format = "3dq"', 
    'trainset_seqlen = %d' % 2, 
    'tsdf_dataset_dir = "/home/mprabhud/dataset/carla_mc_car_tsdf"',
    # 'dataset_list_dir = "/projects/katefgroup/datasets/shamit_carla_correct/npys"',
    'dataset_location = "/projects/katefgroup/datasets/shamit_carla_correct/npys"',
    'dataset_filetype = "npz"',
]

groups['carla_16-8-16_bounds_train'] = [
    'XMIN_train = -16.0', # right (neg is left)
    'XMAX_train = 16.0', # right
    'YMIN_train = -8.0', # down (neg is up)
    'YMAX_train = 8.0', # down
    'ZMIN_train = -16.0', # forward
    'ZMAX_train = 16.0', # forward
    'Z_train = %d' % (int(SIZE_train*16)),
    'Y_train = %d' % (int(SIZE_train*8)),
    'X_train = %d' % (int(SIZE_train*16)),
]
groups['carla_16-8-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*16)),
]
groups['carla_16-16-16_bounds_train'] = [
    'XMIN_train = -16.0', # right (neg is left)
    'XMAX_train = 16.0', # right
    'YMIN_train = -16.0', # down (neg is up)
    'YMAX_train = 16.0', # down
    'ZMIN_train = -16.0', # forward
    'ZMAX_train = 16.0', # forward
    'Z_train = %d' % (int(SIZE_train*16)),
    'Y_train = %d' % (int(SIZE_train*16)),
    'X_train = %d' % (int(SIZE_train*16)),
]
groups['carla_16-4-16_bounds_train'] = [
    'XMIN_train = -16.0', # right (neg is left)
    'XMAX_train = 16.0', # right
    'YMIN_train = -4.0', # down (neg is up)
    'YMAX_train = 4.0', # down
    'ZMIN_train = -16.0', # forward
    'ZMAX_train = 16.0', # forward
    'Z_train = %d' % (int(SIZE_train*16)),
    'Y_train = %d' % (int(SIZE_train*4)),
    'X_train = %d' % (int(SIZE_train*16)),
]
groups['carla_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*16)),
    'X_test = %d' % (int(SIZE_test*16)),
]
groups['carla_16-4-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*16)),
]


############## preprocessing/shuffling ##############

############## modes ##############

groups['zoom'] = ['do_zoom = True']
groups['carla_mot'] = ['do_carla_mot = True']
groups['carla_static'] = ['do_carla_static = True']
groups['carla_flo'] = ['do_carla_flo = True']
groups['carla_reloc'] = ['do_carla_reloc = True']
groups['carla_obj'] = ['do_carla_obj = True']
groups['carla_focus'] = ['do_carla_focus = True']
groups['carla_track'] = ['do_carla_track = True']
groups['carla_siamese'] = ['do_carla_siamese = True']
groups['carla_genocc'] = ['do_carla_genocc = True']
groups['carla_gengray'] = ['do_carla_gengray = True']
groups['carla_vqrgb'] = ['do_carla_vqrgb = True']
groups['carla_vq3drgb'] = ['do_carla_vq3drgb = True']
groups['carla_precompute'] = ['do_carla_precompute = True']
groups['carla_propose'] = ['do_carla_propose = True']
groups['carla_det'] = ['do_carla_det = True']
groups['intphys_det'] = ['do_intphys_det = True']
groups['intphys_forecast'] = ['do_intphys_forecast = True']
groups['carla_forecast'] = ['do_carla_forecast = True']
groups['carla_pipe'] = ['do_carla_pipe = True']
groups['intphys_test'] = ['do_intphys_test = True']
groups['mujoco_offline'] = ['do_mujoco_offline = True']
groups['carla_pwc'] = ['do_carla_pwc = True']
groups['pointfeat_ransac'] = ['pointfeat_ransac = True']
groups['point_contrast'] = ['point_contrast = True']
groups['point_contrast_og'] = ['point_contrast_og = True']




############## extras ##############
groups['low_res'] = ['low_res = True']
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_rgb_coeff = 10.0',
    # 'render_depth_coeff = 0.1',
    # 'render_smooth_coeff = 0.01',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    # 'occ_smooth_coeff = 0.1',
    # 'occ_smooth_coeff = 0.01',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    # 'rgb_smooth_coeff = 0.1',
]


groups['implicit_camX'] = [
    'implicit_camX = True',
]

groups['do_tsdf_implicit_occ'] = [
    'do_tsdf_implicit_occ = True',
]

groups['idx0_dataloader'] = [
    'idx0_dataloader = True',
]

groups['eval_boxes'] = [
    'eval_boxes = True',
]
groups['eval_boxes'] = [
    'eval_boxes = True',
]
groups['make_dense'] = [
    'make_dense = True',
]

groups['do_implicit_occ'] = [
    'do_implicit_occ = True',
]


groups['include_summs'] = [
    'do_include_summs = True',
]
groups['decay_lr'] = ['do_decay_lr = True']
groups['clip_grad'] = ['do_clip_grad = True']
# groups['quick_snap'] = ['snap_freq = 500']
# groups['quicker_snap'] = ['snap_freq = 50']
# groups['quickest_snap'] = ['snap_freq = 5']
groups['snap50'] = ['snap_freq = 50']
groups['snap100'] = ['snap_freq = 100']
groups['snap500'] = ['snap_freq = 500']
groups['snap1k'] = ['snap_freq = 1000']
groups['snap5k'] = ['snap_freq = 5000']

groups['no_shuf'] = ['shuffle_train = False',
                     'shuffle_val = False',
                     'shuffle_test = False',
]
groups['time_flip'] = ['do_time_flip = True']
groups['no_backprop'] = ['backprop_on_train = False',
                         'backprop_on_val = False',
                         'backprop_on_test = False',
]
groups['train_on_trainval'] = ['backprop_on_train = True',
                               'backprop_on_val = True',
                               'backprop_on_test = False',
]
groups['gt_ego'] = ['ego_use_gt = True']
groups['precomputed_ego'] = ['ego_use_precomputed = True']
groups['aug3D'] = ['do_aug3D = True']
groups['aug2D'] = ['do_aug2D = True']
groups['use_lrt_not_box'] = ['use_lrt_not_box = True']


groups['sparsify_pointcloud_10k'] = ['do_sparsify_pointcloud = 10000']
groups['sparsify_pointcloud_1k'] = ['do_sparsify_pointcloud = 1000']

groups['horz_flip'] = ['do_horz_flip = True']
groups['synth_rt'] = ['do_synth_rt = True']
groups['piecewise_rt'] = ['do_piecewise_rt = True']
groups['synth_nomotion'] = ['do_synth_nomotion = True']
groups['aug_color'] = ['do_aug_color = True']
# groups['eval'] = ['do_eval = True']
groups['eval_recall'] = ['do_eval_recall = True']
groups['eval_map'] = ['do_eval_map = True']
groups['no_eval_recall'] = ['do_eval_recall = False']
groups['save_embs'] = ['do_save_embs = True']
groups['save_ego'] = ['do_save_ego = True']
groups['save_vis'] = ['do_save_vis = True']
groups['save_outputs'] = ['do_save_outputs = True']

groups['nearest_neighbour'] = ['nearest_neighbour = True']
groups['do_concat'] = ['do_concat = True']


groups['profile'] = ['do_profile = True',
                     'log_freq_train = 100000000',
                     'log_freq_val = 100000000',
                     'log_freq_test = 100000000',
                     'max_iters = 20']



groups['summ_pca_points_3d'] = ['summ_pca_points_3d = True']
groups['summ_pca_points_2d'] = ['summ_pca_points_2d = True']



groups['B1'] = ['trainset_batch_size = 1']
groups['B2'] = ['trainset_batch_size = 2']
groups['B4'] = ['trainset_batch_size = 4']
groups['B6'] = ['trainset_batch_size = 6']
groups['B8'] = ['trainset_batch_size = 8']
groups['B10'] = ['trainset_batch_size = 10']
groups['B12'] = ['trainset_batch_size = 12']
groups['B16'] = ['trainset_batch_size = 16']
groups['B24'] = ['trainset_batch_size = 24']
groups['B32'] = ['trainset_batch_size = 32']
groups['B64'] = ['trainset_batch_size = 64']
groups['B128'] = ['trainset_batch_size = 128']
groups['vB1'] = ['valset_batch_size = 1']
groups['vB2'] = ['valset_batch_size = 2']
groups['vB4'] = ['valset_batch_size = 4']
groups['vB8'] = ['valset_batch_size = 8']
groups['lr0'] = ['lr = 0.0']
groups['lr1'] = ['lr = 1e-1']
groups['lr2'] = ['lr = 1e-2']
groups['lr3'] = ['lr = 1e-3']
groups['2lr4'] = ['lr = 2e-4']
groups['5lr4'] = ['lr = 5e-4']
groups['lr4'] = ['lr = 1e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['lr9'] = ['lr = 1e-9']
groups['lr12'] = ['lr = 1e-12']
groups['1_iters'] = ['max_iters = 1']
groups['2_iters'] = ['max_iters = 2']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['6_iters'] = ['max_iters = 6']
groups['9_iters'] = ['max_iters = 9']
groups['21_iters'] = ['max_iters = 21']
groups['7_iters'] = ['max_iters = 7']
groups['10_iters'] = ['max_iters = 10']
groups['15_iters'] = ['max_iters = 15']
groups['20_iters'] = ['max_iters = 20']
groups['25_iters'] = ['max_iters = 25']
groups['30_iters'] = ['max_iters = 30']
groups['50_iters'] = ['max_iters = 50']
groups['100_iters'] = ['max_iters = 100']
groups['150_iters'] = ['max_iters = 150']
groups['200_iters'] = ['max_iters = 200']
groups['250_iters'] = ['max_iters = 250']
groups['300_iters'] = ['max_iters = 300']
groups['397_iters'] = ['max_iters = 397']
groups['400_iters'] = ['max_iters = 400']
groups['447_iters'] = ['max_iters = 447']
groups['500_iters'] = ['max_iters = 500']
groups['850_iters'] = ['max_iters = 850']
groups['1000_iters'] = ['max_iters = 1000']
groups['2000_iters'] = ['max_iters = 2000']
groups['2445_iters'] = ['max_iters = 2445']
groups['3000_iters'] = ['max_iters = 3000']
groups['4000_iters'] = ['max_iters = 4000']
groups['4433_iters'] = ['max_iters = 4433']
groups['5000_iters'] = ['max_iters = 5000']
groups['10000_iters'] = ['max_iters = 10000']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['30k_iters'] = ['max_iters = 30000']
groups['40k_iters'] = ['max_iters = 40000']
groups['50k_iters'] = ['max_iters = 50000']
groups['60k_iters'] = ['max_iters = 60000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['100k10_iters'] = ['max_iters = 100010']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['400k_iters'] = ['max_iters = 400000']
groups['500k_iters'] = ['max_iters = 500000']

groups['resume'] = ['do_resume = True']
groups['reset_iter'] = ['reset_iter = True']
groups['hypervoxel'] = ['hypervoxel = True']
groups['use_delta_mem_coords'] = ['use_delta_mem_coords = True']
groups['debug'] = ['debug = True']


groups['log1'] = [
    'log_freq_train = 1',
    'log_freq_val = 1',
    'log_freq_test = 1',
]
groups['log5'] = [
    'log_freq_train = 5',
    'log_freq_val = 5',
    'log_freq_test = 5',
]
groups['log10'] = [
    'log_freq_train = 10',
    'log_freq_val = 10',
    'log_freq_test = 10',
]
groups['log50'] = [
    'log_freq_train = 50',
    'log_freq_val = 50',
    'log_freq_test = 50',
]
groups['log11'] = [
    'log_freq_train = 11',
    'log_freq_val = 11',
    'log_freq_test = 11',
]
groups['log53'] = [
    'log_freq_train = 53',
    'log_freq_val = 53',
    'log_freq_test = 53',
]
groups['log100'] = [
    'log_freq_train = 100',
    'log_freq_val = 100',
    'log_freq_test = 100',
]
groups['log500'] = [
    'log_freq_train = 500',
    'log_freq_val = 500',
    'log_freq_test = 500',
]
groups['log5000'] = [
    'log_freq_train = 5000',
    'log_freq_val = 5000',
    'log_freq_test = 5000',
]



groups['no_logging'] = [
    'log_freq_train = 100000000000',
    'log_freq_val = 100000000000',
    'log_freq_test = 100000000000',
]

groups['train_localdecoder_render'] = ['do_localdecoder_render = True']

groups['train_localdecoder'] = ['do_localdecoder = True']

# ############## pretrained nets ##############
groups['pretrained_sigen3d'] = [
    'do_sigen3d = True',
    'sigen3d_init = "' + pret_carla.sigen3d_init + '"',
]

groups['pretrained_localdecoder'] = ['do_localdecoder = True',
    'localdecoder_init = "' + pret_carla.localdecoder_init + '"',
]


groups['pretrained_localdecoder_render'] = ['do_localdecoder_render = True',
    'localdecoder_render_init = "' + pret_carla.localdecoder_render_init + '"',
]

groups['pretrained_conf'] = [
    'do_conf = True',
    'conf_init = "' + pret_carla.conf_init + '"',
]
groups['pretrained_up3D'] = [
    'do_up3D = True',
    'up3D_init = "' + pret_carla.up3D_init + '"',
]
groups['pretrained_center'] = [
    'do_center = True',
    'center_init = "' + pret_carla.center_init + '"',
]
groups['pretrained_seg'] = [
    'do_seg = True',
    'seg_init = "' + pret_carla.seg_init + '"',
]
groups['pretrained_motionreg'] = [
    'do_motionreg = True',
    'motionreg_init = "' + pret_carla.motionreg_init + '"',
]
groups['pretrained_gen3d'] = [
    'do_gen3d = True',
    'gen3d_init = "' + pret_carla.gen3d_init + '"',
]
groups['pretrained_vq2d'] = [
    'do_vq2d = True',
    'vq2d_init = "' + pret_carla.vq2d_init + '"',
    'vq2d_num_embeddings = %d' % pret_carla.vq2d_num_embeddings,
]
groups['pretrained_vq3d'] = [
    'do_vq3d = True',
    'vq3d_init = "' + pret_carla.vq3d_init + '"',
    'vq3d_num_embeddings = %d' % pret_carla.vq3d_num_embeddings,
]
groups['pretrained_feat2D'] = [
    'do_feat2D = True',
    'feat2D_init = "' + pret_carla.feat2D_init + '"',
    'feat2D_dim = %d' % pret_carla.feat2D_dim,
]
groups['pretrained_feat3D'] = [
    'do_feat3D = True',
    'feat3D_init = "' + pret_carla.feat3D_init + '"',
    'feat3D_dim = %d' % pret_carla.feat3D_dim,
]
groups['pretrained_feat3d'] = [
    'do_feat3d = True',
    'feat3d_init = "' + pret_carla.feat3d_init + '"',
    # 'feat3d_dim = %d' % pret_carla.feat3d_dim,
]

groups['pretrained_feat3docc'] = [
    'do_feat3docc = True',
    'feat3docc_init = "' + pret_carla.feat3docc_init + '"',
    'localdecoder_render_init = "' + pret_carla.localdecoder_render_init + '"',
    # 'feat3d_dim = %d' % pret_carla.feat3d_dim,
]

groups['pretrained_ego'] = [
    'do_ego = True',
    'ego_init = "' + pret_carla.ego_init + '"',
]
groups['pretrained_match'] = [
    'do_match = True',
    'match_init = "' + pret_carla.match_init + '"',
]
groups['pretrained_mot'] = [
    'do_mot = True',
    'mot_init = "' + pret_carla.mot_init + '"',
]
groups['pretrained_rigid'] = [
    'do_rigid = True',
    'rigid_init = "' + pret_carla.rigid_init + '"',
]
# groups['pretrained_pri2D'] = [
#     'do_pri2D = True',
#     'pri2D_init = "' + pret_carla.pri2D_init + '"',
# ]
groups['pretrained_det'] = [
    'do_det = True',
    'det_init = "' + pret_carla.det_init + '"',
]
groups['pretrained_forecast'] = [
    'do_forecast = True',
    'forecast_init = "' + pret_carla.forecast_init + '"',
]
groups['pretrained_view'] = [
    'do_view = True',
    'view_init = "' + pret_carla.view_init + '"',
    'view_depth = %d' %  pret_carla.view_depth,
    'feat2D_dim = %d' %  pret_carla.feat2D_dim,
    # 'view_use_halftanh = ' + str(pret_carla.view_use_halftanh),
    # 'view_pred_embs = ' + str(pret_carla.view_pred_embs),
    # 'view_pred_rgb = ' + str(pret_carla.view_pred_rgb),
]
groups['pretrained_flow'] = ['do_flow = True',
                             'flow_init = "' + pret_carla.flow_init + '"',
]
# groups['pretrained_tow'] = ['do_tow = True',
#                             'tow_init = "' + pret_carla.tow_init + '"',
# ]
groups['pretrained_emb2D'] = ['do_emb2D = True',
                              'emb2D_init = "' + pret_carla.emb2D_init + '"',
                              # 'emb_dim = %d' % pret_carla.emb_dim,
]
groups['pretrained_occ'] = [
    'do_occ = True',
    'occ_init = "' + pret_carla.occ_init + '"',
]
groups['pretrained_resolve'] = [
    'do_resolve = True',
    'resolve_init = "' + pret_carla.resolve_init + '"',
]
groups['pretrained_occrel'] = [
    'do_occrel = True',
    'occrel_init = "' + pret_carla.occrel_init + '"',
]
groups['pretrained_preocc'] = [
    'do_preocc = True',
    'preocc_init = "' + pret_carla.preocc_init + '"',
]
groups['pretrained_vis'] = ['do_vis = True',
                            'vis_init = "' + pret_carla.vis_init + '"',
                            # 'occ_cheap = ' + str(pret_carla.occ_cheap),
]
groups['total_init'] = ['total_init = "' + pret_carla.total_init + '"']
groups['pretrained_optim'] = ['optim_init = "' + pret_carla.optim_init + '"']
groups['pretrained_latents'] = ['latents_init = "' + pret_carla.latents_init + '"']

groups['frozen_localdecoder'] = ['do_freeze_localdecoder = True', 'do_localdecoder = True']
groups['frozen_conf'] = ['do_freeze_conf = True', 'do_conf = True']
groups['frozen_mot'] = ['do_freeze_mot = True', 'do_mot = True']
groups['frozen_motionreg'] = ['do_freeze_motionreg = True', 'do_motionreg = True']
groups['frozen_feat2D'] = ['do_freeze_feat2D = True', 'do_feat2D = True']
groups['frozen_feat3d'] = ['do_freeze_feat3d = True', 'do_feat3d = True']
groups['frozen_feat3docc'] = ['do_freeze_feat3docc = True', 'do_feat3docc = True']

groups['frozen_up3D'] = ['do_freeze_up3D = True', 'do_up3D = True']
groups['frozen_vq3d'] = ['do_freeze_vq3d = True', 'do_vq3d = True']
groups['frozen_view'] = ['do_freeze_view = True', 'do_view = True']
groups['frozen_center'] = ['do_freeze_center = True', 'do_center = True']
groups['frozen_seg'] = ['do_freeze_seg = True', 'do_seg = True']
groups['frozen_vis'] = ['do_freeze_vis = True', 'do_vis = True']
groups['frozen_flow'] = ['do_freeze_flow = True', 'do_flow = True']
groups['frozen_match'] = ['do_freeze_match = True', 'do_match = True']
groups['frozen_emb2D'] = ['do_freeze_emb2D = True', 'do_emb2D = True']
groups['frozen_pri2D'] = ['do_freeze_pri2D = True', 'do_pri2D = True']
groups['frozen_occ'] = ['do_freeze_occ = True', 'do_occ = True']
groups['frozen_resolve'] = ['do_freeze_resolve = True', 'do_resolve = True']
groups['frozen_occrel'] = ['do_freeze_occrel = True', 'do_occrel = True']
groups['frozen_vq2d'] = ['do_freeze_vq2d = True', 'do_vq2d = True']
groups['frozen_vq3d'] = ['do_freeze_vq3d = True', 'do_vq3d = True']
groups['frozen_sigen3d'] = ['do_freeze_sigen3d = True', 'do_sigen3d = True']
groups['frozen_gen3d'] = ['do_freeze_gen3d = True', 'do_gen3d = True']
# groups['frozen_ego'] = ['do_freeze_ego = True', 'do_ego = True']
# groups['frozen_inp'] = ['do_freeze_inp = True', 'do_inp = True']
