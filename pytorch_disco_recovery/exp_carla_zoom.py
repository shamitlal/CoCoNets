from exp_base import *

############## choose an experiment ##############
# dtaas50i1a.txt
current = 'builder'
# current = 'trainer'
current = 'tester'

mod = '"z00"' # reset
mod = '"z01"' # train a while
mod = '"z02"' # pret 02_s2_m128x16x128_1e-4_F3_d64_E3_n4_d16_c1_mabs7i3t_fags16i3v_moc15
mod = '"z03"' # regular bounds
mod = '"z04"' # measure test perf too
mod = '"z05"' # only output test vis if we proceed into estimation
mod = '"z06"' # tester
mod = '"z07"' # include_vis=False
mod = '"z08"' # narrow bounds
mod = '"z09"' # include feat and traj vis always
mod = '"z10"' # get and feed prev lrt
mod = '"z11"' # unpacked the util
mod = '"z12"' # remake vox util on every step, using the answer from t-1
mod = '"z13"' # actually do this on all nonzero steps; show the revox results < worse, and the registration to 0 looks wrong
mod = '"z14"' # add +delta to xyzI
mod = '"z15"' # add -delta to xyzI
mod = '"z16"' # add +delta to xyzI in cam coords, and then go back < very very good
mod = '"z17"' # narrower bounds
mod = '"z18"' # show reloc occ_memX0 instead of occ_memX
mod = '"z19"' # also show occ0; zoom even farther


mod = '"z20"' # train a bit

# note the loss is wonky at the start because of no dict init; this doesn't matter for FS nets, but it does here

mod = '"z21"' # not totally random zoom < bug
mod = '"z22"' # req 10 pts inbound across the batch < yes. clear improvement over z20
mod = '"z23"' # req 100 pts inbound across the batch < no diff really from z22
mod = '"z24"' # req 100 pts inbound across the batch AND within each seq el < this does seem slightly better
mod = '"z25"' # plot num points at test time
mod = '"z26"' # show num inb on test too
mod = '"z27"' # train a long time
mod = '"z28"' # narrow bounds instead of narrower; this also puts a 4x1x4 ratio, which is different from the 8x1x8 old narrow


mod = '"z29"' # random init; narrow bounds; test
mod = '"z30"' # no feats; test
mod = '"z31"' # random init; narrower bounds; test
mod = '"z32"' # no feats; test; show vis
mod = '"z33"' # collect stats; pret moc15
mod = '"z34"' # again but also summ 
mod = '"z35"' # include_vis=True
mod = '"z36"' # moc15
mod = '"z37"' # moc15 but halfres


mod = '"z38"' # test with random features
# 0.99 0.13 0.07 0.07 0.06 0.05 0.03 0.03 0.03 0.02 0.02 0.01 0.01 0.01 0.01 0.01


mod = '"had00"' # tester  
mod = '"had01"' # actually pret 02_s2_m128x16x128_1e-4_F3_d64_E3_n4_d16_c1_mabs7i3t_fags16i3v_moc15
mod = '"had02"' # again, but narrower bounds instead of narrow, and 313 bounds < good, this seems to reproduce 
mod = '"had03"' # log10 < 2h 5m
mod = '"had04"' # log100 < 1h 15m; much faster
# 0.99 0.78 0.69 0.64 0.6  0.57 0.55 0.53 0.52 0.49 0.48 0.46 0.46 0.43 0.42 0.41
# ok, this is pretty good, but it's not quite 0.61
mod = '"had05"' # narrow_test_bounds
mod = '"had06"' # narrow bounds, but S_test = 8
# mean_ious [0.99 0.71 0.56 0.46 0.39 0.36 0.34 0.31]
mod = '"had07"' # narrower bounds, but S_test = 8
mod = '"had08"' # retry that on faster node

# in an old restored ckpt:
# 01_s2_m128x32x128_F3_d64_E3_n2_d16_c1_faks16i3v_z24
# 100 iters:
# mean_ious [0.99 0.8  0.77 0.73 0.68 0.62 0.63 0.62]
# 100 iters again:
# mean_ious [0.99 0.82 0.75 0.72 0.69 0.64 0.61 0.58]
# 1k iters
# mean_ious [0.98 0.79 0.72 0.68 0.65 0.62 0.57 0.55]
# so, when i computed the results for "ours" i probably computed it at 100 iters
# i wonder if the 3d siamese used 100 or 1k iters.
# anyway, this odes not change the ordering of methods

mod = '"had09"' # again but 100 iters
# mean_ious [0.99 0.8  0.71 0.64 0.61 0.57 0.56 0.52]
mod = '"had10"' # carla_narrow_test_bounds
# mean_ious [0.99 0.81 0.76 0.72 0.68 0.65 0.63 0.58]
# ok maybe this is close enough?
mod = '"had11"' # again
# mean_ious [0.99 0.81 0.76 0.72 0.68 0.65 0.63 0.57]
mod = '"had12"' # again
# mean_ious [0.99 0.81 0.76 0.72 0.68 0.65 0.63 0.57]
# ok, so, somehow, we're still missing a bit. i think this is due to shuffling
mod = '"had13"' # shuffle
# mean_ious [0.99 0.81 0.75 0.68 0.59 0.59 0.53 0.56]
mod = '"had14"' # don't shuffle; return early if dist < 1.0; 200 iters
# mean_ious across 160 tests [0.99 0.8  0.75 0.68 0.65 0.61 0.58 0.56]
mod = '"had15"' # don't shuffle; return early if dist > 1.0; 200 iters
# mean_ious across 35 tests [0.99 0.89 0.87 0.84 0.8  0.73 0.66 0.6 ]
mod = '"had16"' # return early if dist>1 again; 400 iters
# mean_ious across 66 tests [0.99 0.88 0.85 0.86 0.82 0.77 0.69 0.64]
# yes.



mod = '"mot00"' # prep to export mot-style data < these boxes are tiny. did i normalize?
mod = '"mot01"' # don't normlaize
mod = '"mot02"' # write multi-timestep thing
mod = '"mot03"' # do this 400 steps
# mean_ious across 388 tests [0.99 0.81 0.75 0.71 0.66 0.64 0.6  0.56]

# following your suggestion, we...


mod = '"nerv00"' # builder; pret 02_s2_m128x128x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu12
mod = '"nerv01"' # tester
mod = '"nerv02"' # parse boxes
mod = '"nerv03"' # run_new_test
mod = '"nerv04"' # clean up the track code a bit and fire it
mod = '"nerv05"' # fire it
mod = '"nerv06"' # run old test please
mod = '"nerv07"' # crop orig_xyz
mod = '"nerv08"' # print more, to see the nans
mod = '"nerv09"' # fixed a coordinate issue i think
mod = '"nerv10"' # print ious and pause
mod = '"nerv11"' # run the output 
mod = '"nerv12"' # use 300k ckpt instead of 10k
mod = '"nerv13"' # 400 steps; fewer prints
mod = '"nerv14"' # no crop; pret 40k 02_s2_m128x128x128_1e-4_F3_d64_O_c1_E3_n2_d16_c1_mags7i3t_bu21
mod = '"nerv15"' # pret bu20 < seems similar to when we pret bu12 (in nerv13)
mod = '"nerv16"' # aws



mod = '"rep00"' # aws; run whatever this is
mod = '"rep01"' # pret moc15 < somehow this is succeeding to load... but the tracking results stink.
mod = '"rep02"' # 5 iters; print some loading info
mod = '"rep03"' # fixed the loading maybe
mod = '"rep04"' # 16-4-16
mod = '"rep05"' # 10 iters


current = '{}'.format(os.environ["exp_name"])
mod = '"{}"'.format(os.environ["run_name"]) 

############## define experiment ##############

exps['builder'] = [
    'carla_zoom', # mode
    # 'carla_multiview_train10_test10_data', # dataset
    'carla_multiview_test_data', # dataset
    # 'carla_multiview_test_data', # dataset
    # 'carla_multiview_train_data', # dataset
    'carla_16-16-16_bounds_test',
    # '1k_iters',
    '1k_iters',
    # '10_iters',
    'no_backprop',
    # 'lr0',
    'do_test', 
    'B1',
    'no_shuf',
    'train_feat3d',
    # 'train_feat3dS',
    # 'train_emb3d',
    # 'train_moc3d',
    # 'fastest_logging',
    'log1',
]
exps['trainer'] = [
    'carla_zoom', # mode
    'carla_multiview_train_test_data', # dataset
    'carla_narrow_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'pretrained_feat3d',
    'train_feat3d',
    'train_occ',
    'train_emb3d',
    'log500',
]
exps['tester_debug_complete'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    'summ_pca_points_3d',
    'summ_pca_points_2d',
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    'debug',
    # '5_iters',
    'B1',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'train_feat3d',
    # 'pretrained_feat3d',
    # 'pretrained_localdecoder',
    # 'pointfeat_ransac',    
    # 'make_dense',
    # 'log100',
    'log1',
]


exps['tester_drone'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_complete_data_test_for_traj', # dataset
    'carla_drone_traj_test_data',
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    # '5_iters',
    'B1',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    'train_feat3d',
    # 'log100',
    'log1',
]

exps['tester_drone_closecam'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_complete_data_test_for_traj', # dataset
    'carla_drone_traj_test_data_latest_closecam',
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    # '5_iters',
    'B1',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    'train_feat3d',
    # 'log100',
    'log1',
]
#########################
exps['tester'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_complete_data_test_for_traj', # dataset
    # 'carla_complete_data_test_for_traj',
    'carla_complete_data_test_for_traj',
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '50_iters',
    # '5_iters',
    'B1',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    'train_feat3d',
    # 'log100',
    'log100',
]


exps['tester_pointfeat'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_complete_data_test_for_traj_closecam', # dataset
    # '400_iters',
    '50_iters',
    # '5_iters',
    'B1',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    'train_feat3d',
    # 'log100',
    'log1',
    'use_lrt_not_box',
]

exps['tester_closecam'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_complete_data_test_for_traj', # dataset
    # 'carla_complete_data_test_for_traj',
    'carla_complete_data_test_for_traj',

    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # 'make_dense',
    # '400_iters',
    '50_iters',
    # 'make_dense',
    # '5_iters',
    'B1',
    'make_dense',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log1',
]


exps['tester_pointfeat_drn'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_drone_traj_test_data', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # 'make_dense',
    # '400_iters',
    '50_iters',
    # 'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log1',
]

exps['tester_pointfeat_tsdf'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '50_iters',
    # 'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'train_feat3docc',
    'pretrained_feat3docc',
    'pointfeat_ransac',
    'train_feat3d',
    'make_dense',
    # 'log100',
    'log1',
]

exps['tester_pointfeat_tsdf_closecam'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_complete_data_test_for_traj_closecam', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '50_iters',
    # 'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'train_feat3docc',
    'pretrained_feat3docc',
    'pointfeat_ransac',
    'train_feat3d',
    'make_dense',
    # 'log100',
    'log1',
    'use_lrt_not_box'
]

exps['tester_pointfeat_dense'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '50_iters',
    'make_dense',
    # 'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log100',
]

#########################


exps['tester_pointfeat_nearest_concat'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    # 'hypervoxel',
    'nearest_neighbour',
    'use_delta_mem_coords',
    'do_concat',    
    'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',    
    # 'make_dense',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log1',
]



exps['tester_pointfeat_nearest_delta'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    # 'hypervoxel',
    'nearest_neighbour',
    'use_delta_mem_coords',
    'hypervoxel',
    # 'do_concat',    
    'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',    
    # 'make_dense',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log1',
]

exps['tester_pointfeat_nearest_delta_hyper'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '10_iters',
    'hypervoxel',
    'nearest_neighbour',
    'use_delta_mem_coords',

    # 'do_concat',    
    'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',    
    # 'make_dense',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log1',
]


# exps['tester_pointfeat_nearest_delta'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     'carla_complete_data_test_for_traj', # dataset
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '10_iters',
#     # 'hypervoxel',
#     'nearest_neighbour',
#     'use_delta_mem_coords',
#     # 'do_concat',    
#     # '5_iters',
#     'B1',
#     # 'make_dense',
#     # 'use_delta_mem_coords',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]




# exps['tester_pointfeat_nearest_delta_hyper'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     'carla_complete_data_test_for_traj', # dataset
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '10_iters',
#     # 'hypervoxel',
#     'nearest_neighbour',
#     'use_delta_mem_coords',
#     'hypervoxel',
#     # 'do_concat',    
#     # '5_iters',
#     'B1',
#     # 'make_dense',
#     # 'use_delta_mem_coords',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]



# exps['tester_pointfeat_nearest_delta'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     'carla_complete_data_test_for_traj', # dataset
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '10_iters',
#     # 'hypervoxel',
#     'nearest_neighbour',
#     'use_delta_mem_coords',
#     'hypervoxel',
#     # 'do_concat',    
#     # '5_iters',
#     'B1',
#     # 'make_dense',
#     'use_delta_mem_coords',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]
 




# exps['tester_pointfeat_hyper_delta'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     'carla_complete_data_test_for_traj', # dataset
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '10_iters',
#     'hypervoxel',
#     # '5_iters',
#     'B1',
#     # 'make_dense',
#     'use_delta_mem_coords',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]


# exps['tester_pointfeat_dense'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     # 'carla_complete_data_test_for_traj', # dataset
#     'carla_drone_traj_test_data',
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '3_iters',
#     # '5_iters',
#     'B1',
#     # 'make_dense',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]


# exps['tester_pointfeat_dense_hyper_delta'] = [
#     'carla_zoom', # mode
#     # 'carla_traj_test_data', # dataset
#     # 'carla_complete_data_test_for_traj', # dataset
#     'carla_drone_traj_test_data',
#     # 'carla_regular_bounds',
#     # 'carla_narrow_bounds',
#     # 'narrow_test_bounds',
#     # 'carla_narrower_test_bounds',
#     # 'carla_narrow_test_bounds',
#     # 'carla_16-16-16_bounds_test',
#     'carla_16-16-16_bounds_test',
#     # '400_iters',
#     '10_iters',
#     'hypervoxel',
#     # '5_iters',
#     'B1',
#     'make_dense',
#     'use_delta_mem_coords',
#     'no_shuf',
#     'do_test',
#     'pretrained_feat3d',
#     'pretrained_localdecoder',
#     'pointfeat_ransac',
#     'train_feat3d',
#     # 'log100',
#     'log100',
# ]

############## net configs ##############

groups['carla_zoom'] = ['do_carla_zoom = True']
groups['do_test'] = ['do_test = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3d'] = [
    'do_moc3d = True',
    'moc3d_num_samples = 1000',
    'moc3d_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    # 'emb_3d_ml_coeff = 1.0',
    # 'emb_3d_l2_coeff = 0.1',
    # 'emb_3d_mindist = 16.0',
    # 'emb_3d_num_samples = 2',
    'emb_3d_mindist = 16.0',
    'emb_3d_num_samples = 2',
    'emb_3d_ce_coeff = 1.0',
]
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
    'feat2D_smooth_coeff = 0.01',
]
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 64',
]

groups['train_feat3docc'] = [
    'do_feat3docc = True',
    'feat3d_dim = 64',
    'do_tsdf_implicit_occ = True',
]

groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    # 'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    'occ_smooth_coeff = 0.001',
]

############## datasets ##############

# 'XMIN = -8.0', # right (neg is left)
# 'XMAX = 8.0', # right
# 'YMIN = -1.0', # down (neg is up)
# 'YMAX = 3.0', # down
# 'ZMIN = 4.0', # forward
# 'ZMAX = 20.0', # forward

# dims for mem
SIZE = 16
Z = int(SIZE*3)
Y = int(SIZE*1)
X = int(SIZE*3)
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

ZZ = int(SIZE*3)
ZY = int(SIZE*3)
ZX = int(SIZE*3)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
dataset_location = "/data/carla/processed/npzs"

# dataset_location = "/data4/carla/processed/npzs"

SIZE_test = 8

groups['carla_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*16)),
    'Y = %d' % (int(SIZE*16)),
    'X = %d' % (int(SIZE*16)),
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

groups['regular_train_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*1)),
    'X = %d' % (int(SIZE*8)),
]
groups['narrow_test_bounds'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -2.0', # down (neg is up)
    'YMAX_test = 2.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*1)),
    'X_test = %d' % (int(SIZE_test*4)),
]


groups['carla_narrow_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_narrower_bounds'] = [
    'XMIN = -3.0', # right (neg is left)
    'XMAX = 3.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -3.0', # forward
    'ZMAX = 3.0', # forward
    'Z = 96', # forward
    'Y = 32', # forward
    'X = 96', # forward
]
groups['carla_narrow_test_bounds'] = [
    'XMIN_test = -4.0', # right (neg is left)
    'XMAX_test = 4.0', # right
    'YMIN_test = -1.0', # down (neg is up)
    'YMAX_test = 1.0', # down
    'ZMIN_test = -4.0', # forward
    'ZMAX_test = 4.0', # forward
    'Z_test = 128', # forward
    'Y_test = 32', # forward
    'X_test = 128', # forward
]
groups['carla_narrower_test_bounds'] = [
    'XMIN_test = -3.0', # right (neg is left)
    'XMAX_test = 3.0', # right
    'YMIN_test = -1.0', # down (neg is up)
    'YMAX_test = 1.0', # down
    'ZMIN_test = -3.0', # forward
    'ZMAX_test = 3.0', # forward
    'Z_test = 96', # forward
    'Y_test = 32', # forward
    'X_test = 96', # forward
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_big_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]


groups['carla_multiview_some_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabsome"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
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
groups['carla_multiview_train10_test10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "taus100i2ten"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "taus100i2v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_test_data'] = [
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
groups['carla_traj_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taus100i2t"',
    'testset_format = "oldtraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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

