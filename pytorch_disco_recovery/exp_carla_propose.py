from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'trainer'
# current = 'tester'

mod = '"tak00"' # take over _propose by copying _compose
mod = '"tak01"' # pret 02_s2_m160x80x160_1e-4_F3_d32_O_c1_E3_n2_c1_mags7i3t_taqs100i2v_com56 
mod = '"tak02"' # cleaned up a bit
mod = '"tak03"' # cleaned up more
mod = '"tak04"' # tr 
mod = '"tak05"' # do_test; collect stats
mod = '"tak06"' # print stats
mod = '"tak07"' # don't return early
mod = '"tak08"' # go through without bugs
# mean_best_ious 0.6574731
# mean_selected_ious 0.10202998
# mean_worst_ious 0.10202998
# > hm, it seems to be selecting the worse option consistently

mod = '"tak09"' # print more, so i can see the diffs and ious per iter
# indeed, the worse iou has smaller scene diff
mod = '"tak10"' # S = 30 instead of 10, to exaggerate the diffs
# mean_best_ious 0.52440894
# mean_selected_ious 0.11359135
# mean_worst_ious 0.11359135
# ok, maybe it's bc some tries are in the sky
mod = '"tak11"' # y part of noise = 0
# mean_best_ious 0.5334123
# mean_selected_ious 0.25652424
# mean_worst_ious 0.24752088
# ok still not really working
mod = '"tak12"' # render all scenes with the same centroid; use smaller noise i think
# mean_best_ious 0.5314626
# mean_selected_ious 0.46079364
# mean_worst_ious 0.45373994
mod = '"tak13"' # log10
# mean_best_ious 0.5571946
# mean_selected_ious 0.50199884
# mean_worst_ious 0.49560624
# ok, same result; slightly faster though. (30s instead of 45)
mod = '"tak14"' # wider noise again, except the render point
# mean_best_ious 0.5508354
# mean_selected_ious 0.5508354
# mean_worst_ious 0.072836
# ok nice
mod = '"tak15"' # log1; 4 tries
# mean_best_ious 0.52415407
# mean_selected_ious 0.49252543
# mean_worst_ious 0.038664613
# ok good
mod = '"tak16"' # set the y part to be zero, to be more realistic
mod = '"tak17"' # collect the avg iou too; log5
# mean_best_ious 0.5508545
# mean_selected_ious 0.49577147
# mean_avg_ious 0.29584986
# mean_worst_ious 0.08068404
# great



mod = '"pro00"' # generate feat and occ
mod = '"pro01"' # use middle of scene, instead of an object centroid
# occupancy looks worse than i was hoping
# maybe CS in featurespace will be better actually
mod = '"pro02"' # make a box on every cell, sized 1
# ok, that's a ton of boxes. and a lot seem to be floating
mod = '"pro03"' # trim the centroids using occ>0.5
# ok, better, but these boxes are tiny (1vox i guess), and too high up
mod = '"pro04"' # centroid propose half a meter lower
# ok this looks fine
# next step is center-surround, i think
mod = '"pro05"' # 0.8 thresh < none left
mod = '"pro06"' # log1
mod = '"pro07"' # 0.6 thresh < ok
mod = '"pro08"' # 0.7 thresh < none left
mod = '"pro09"' # 0.6; try generating the full masklist < OOM
mod = '"pro10"' # generate masks one at a time; show the first ten
mod = '"pro11"' # fewer prints
mod = '"pro12"' # get the top10 boxes
mod = '"pro13"' # shuffle and halve the boxes at the outset
mod = '"pro14"' # show scores please
mod = '"pro15"' # again; 3 iters
mod = '"pro16"' # proper scorelist
mod = '"pro17"' # reject if <0.51
mod = '"pro18"' # 10 iters
mod = '"pro19"' # make the boxes wider
mod = '"pro20"' # use boxes at two angles
mod = '"pro21"' # don't hard drop
mod = '"pro22"' # propre cam centroids
mod = '"pro23"' # allow 45deg proposals; bring back hard drop
mod = '"pro24"' # allow 22.5 and 67.5 deg proposals
mod = '"pro25"' # again
mod = '"pro26"' # allow 0.5
mod = '"pro27"' # proper rotation finally
mod = '"pro28"' # thresh 0.6
mod = '"pro29"' # hard 0.52
mod = '"pro30"' # consider lens and lens*0.75; also ::8 in the initial filtering
# time 162
mod = '"pro31"' # just use mask_1 and mask_3 please
# time 113 
mod = '"pro32"' # K=20
# ok this is plenty of boxes, and they look quite reasonable. highly overlapping though.

mod = '"pro32"' # K=20
mod = '"pro33"' # overwrite occ with available gt
mod = '"pro34"' # mult by 1.0-free
mod = '"pro35"' # do not agg please; it's single-view data
mod = '"pro36"' # don't use that sup
mod = '"pro37"' # just use the occ support
mod = '"pro38"' # drop invalid lrts, in a step toward iou
mod = '"pro39"' # drop invalid boxes, to see how that func is supposed to work
mod = '"pro40"' # tara data 
mod = '"pro41"' # decode full boxlist < ok looks good
mod = '"pro42"' # trim this thing
mod = '"pro43"' # measure ap
mod = '"pro44"' # don't show scores, to clean up summ; show gt in bev too
mod = '"pro45"' # include ious at 0.1, 0.2
mod = '"pro46"' # mult by 1.0-free < not better
mod = '"pro47"' # K = 30 < still not better
mod = '"pro48"' # don't touch occ_memX0 < seems worse
mod = '"pro49"' # bring back the occ and free sup
mod = '"pro50"' # K = 10


mod = '"pro51"' # 100 iters, so i can see more happening
mod = '"pro52"' # rescore with inboundd
mod = '"pro53"' # return early if no objects inbound
mod = '"pro54"' # print #objects; log10
mod = '"pro55"' # use gt occ
mod = '"pro56"' # use new occ from skip net; log1; 10 iters
mod = '"pro57"' # hard trim to 100 for sensibility
mod = '"pro58"' # training resolution
mod = '"pro59"' # don't crop_feat

mod = '"pro60"' # conv3d layer
mod = '"pro61"' # just slice out again, to see if i can catch that bug
mod = '"pro62"' # pret 02_s2_m128x64x128_1e-3_F3s_d32_O_c1_s.1_mags7i3t_pro39 < somehow this one works!
mod = '"pro63"' # same
mod = '"pro64"' # :500 instead of :100
mod = '"pro64"' # :200
mod = '"pro65"' # :200; ckpt 02_s2_m128x64x128_1e-4_F3s_d32_O_c1_s.1_mags7i3t_pro41
mod = '"pro66"' # :100; show boxes on occ_memX0
mod = '"pro67"' # one box version instead of 5
mod = '"pro68"' # straight boxes; don't take ::32
mod = '"pro69"' # 


mod = '"pro70"' # get boxes from flow mag; do nothing with them yet
mod = '"pro71"' # show boxes
mod = '"pro72"' # print inside a bit
mod = '"pro73"' # allow huge
mod = '"pro74"' # don't check ymin/ymax
                                # # be less than huge
                                # (hei < 10.0) and
                                # (wid < 10.0) and
                                # (dep < 10.0) and


############## define experiment ##############

exps['builder'] = [
    'carla_propose', # mode
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_test_data', # dataset
    'carla_tart_testset_data', # dataset
    'carla_16-8-16_bounds_train',
    'carla_16-8-16_bounds_test',
    '10_iters',
    'do_test', 
    'lr4',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'train_feat3D',
    # 'train_emb3D',
    'train_occ',
    'no_backprop', 
    'log1',
    # 'log10',
    # 'log5',
]
exps['trainer'] = [
    'carla_propose', # mode
    'carla_multiview_train_test_data', # dataset
    # 'carla_16-8-16_bounds_train',
    'carla_16-16-16_bounds_train',
    # 'carla_8-4-8_bounds_test',
    'carla_12-6-12_bounds_test',
    '300k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D',
    'pretrained_occ',
    'train_feat3D',
    'train_emb3D',
    'train_occ',
    'log500',
]
exps['tester'] = [
    'carla_propose', # mode
    'carla_multiview_test_data', # dataset
    'carla_16-8-16_bounds_train',
    # 'carla_8-4-8_bounds_test',
    'carla_12-6-12_bounds_test',
    # 'carla_8-6-8_bounds_test',
    # 'carla_8-8-8_bounds_test',
    # 'carla_16-8-16_bounds_test',
    '50_iters',
    'do_test', 
    'lr0',
    'B1',
    'pretrained_feat3D',
    'pretrained_occ',
    'train_feat3D',
    'train_emb3D',
    'train_occ',
    'train_render',
    'no_shuf',
    'log10',
]
exps['render_trainer'] = [
    'carla_propose', # mode
    'carla_multiview_train_data', # dataset
    'carla_multiview_ep09_data', # dataset
    # 'carla_multiview_ep09one_data', # dataset
    # 'carla_multiview_one_data', # dataset
    # 'carla_wide_nearcube_bounds',
    # 'carla_nearcube_bounds',
    'carla_narrow_nearcube_bounds',
    # 'carla_narrow_flat_bounds',
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
    'carla_propose', # mode
    'carla_multiview_train_data', # dataset
    'carla_wide_cube_bounds',
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
    'carla_propose', # mode
    'carla_multiview_all_data', # dataset
    'carla_wide_cube_bounds',
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
    'carla_propose', # mode
    'carla_multiview_all_data', # dataset
    'carla_wide_cube_bounds',
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
    'carla_propose', # mode
    'carla_multiview_all_data_as_test', # dataset
    'carla_wide_cube_bounds',
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
groups['carla_propose'] = ['do_carla_propose = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    'feat3D_skip = True',
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
SIZE_test = 20

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
N = 8 # how many objects per npz
S = 2
S_test = 30
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data4/carla/processed/npzs"

groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_all_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mads7i3a"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_all_data_as_test'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mads7i3a"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
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
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
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
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taqv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tara_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tars100i2a"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tart_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tars100i2t"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tarv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tars100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_16-8-16_bounds_train'] = [
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
groups['carla_16-16-16_bounds_train'] = [
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
groups['carla_16-16-16_bounds_test'] = [
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
groups['carla_16-8-16_bounds_test'] = [
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
groups['carla_8-4-8_bounds_test'] = [
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
groups['carla_12-6-12_bounds_test'] = [
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
groups['carla_8-6-8_bounds_test'] = [
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
groups['carla_8-8-8_bounds_test'] = [
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

groups['carla_313_bounds'] = [
    'XMIN = -18.0', # right (neg is left)
    'XMAX = 18.0', # right
    'YMIN = -6.0', # down (neg is up)
    'YMAX = 6.0', # down
    'ZMIN = -18.0', # forward
    'ZMAX = 18.0', # forward
]
groups['carla_flat_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_narrow_nearcube_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_narrow_flat_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_cube_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_wide_nearcube_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]
groups['carla_wide_cube_bounds'] = [
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

