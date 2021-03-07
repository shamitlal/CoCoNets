from exp_base import *
import ipdb 
st = ipdb.set_trace
# the idea here is to train with momentum contrast (MOC), in a simple clean way for eccv

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
current = 'trainer_implicit'

mod = '"moc00"' # reset
mod = '"moc01"' # drop the 2d code
mod = '"moc02"' # train a bit
mod = '"moc03"' # add test code
mod = '"moc04"' # drop moc for a bit
mod = '"moc05"' # drop all moc
mod = '"moc06"' # train a while
mod = '"moc07"' # add embnet3d
mod = '"moc08"' # add embnet3d; train a while
mod = '"moc09"' # ce loss; 10k pool
mod = '"moc10"' # 100k pool < slows iterations; 
mod = '"moc11"' # use slow net < slightly better all around

# diff machine
mod = '"moc12"' # classic arch; use slow net; 20k pool
mod = '"moc13"' # resnet arch
mod = '"moc14"' # 300k; fast logging
mod = '"moc15"' # classic arch again; choose obj mask resolution dynamically < best, surprisingly. is it the slow net helping? < later exps say yes
mod = '"moc16"' # DCAA
mod = '"moc17"' # like moc15 but no slow net < indeed, a bit worse than with the dict
mod = '"moc18"' # clean repeat of moc15 < indeed, following moc15 pretty nicely, up to ~80k



mod = '"sp00"' # start on sparse
mod = '"sp01"' # no testset; use slow
mod = '"sp02"' # use sparse resnet
mod = '"sp03"' # train a while, just to see speeds
mod = '"sp04"' # actually use sparse please < but sparse is not used in loss

mod = '"sp05"' # builder again; show valid_memR < not so sparse!
mod = '"sp06"' # feat = feat * mask < still not sparse
mod = '"sp07"' # separately return feat and mask, and show mask
mod = '"sp08"' # mult by mask AFTER l2 norm, just to be safe 
mod = '"sp09"' # compute valid mask better
mod = '"sp10"' # compute valid mask for altfeat too
mod = '"sp11"' # train a while
mod = '"sp12"' # encode in X0 < somehow a lot is falling out of bounds; often "not enough valid samples! returning a few fakes"
mod = '"sp13"' # carla_big_bounds
mod = '"sp14"' # carla_big_bounds; use the true mask instead of >0 < nice; this is maybe new best, better than moc15
mod = '"sp15"' # carla_big_bounds; go back to encoding in Xs, but use the true mask < this is on par with moc15 < after longer it seems to dip under < after even longer it catches up

mod = '"sp16"' # builder; middle arch
mod = '"sp17"' # move coords to gpu
mod = '"sp18"' # coords on cpu again; drop the batch norm
mod = '"sp19"' # custom3d arch
mod = '"sp20"' # custom3d arch; encode in X0
mod = '"sp21"' # dilate 3 on every layer; print shapes!!
mod = '"sp22"' # dilate 3 on every layer; train a while < seems worse than sp22, meaning dilate1 is better for now

mod = '"sp23"' # builder; not so many sparse-to-dense conversions
mod = '"sp24"' # not so many sparse-to-dense conversions; train a while < seems worse than sp22. why is this?

mod = '"sp25"' # resnet-style arch again, but without BN, just to see (since i need to remove/change BN before doing my density fix) < doing great < still improving after 300k
mod = '"sp26"' # mult by mc after each res layer < neck and neck with sp25 < no diff from sp25 i'd say
mod = '"sp27"' # mult by mc in downsamp layer too < terrible <  i think i killed it too early; these evals are noisy

# other machine
mod = '"sp28"' # careful sparse resnet; this should be as good as sp15, and maybe slightly better since batch norms are 3d now
mod = '"sp29"' # regular sparse resnet (repeat of sp15 i hope) < no, seems different; not sure what's wrong
mod = '"sp30"' # regular sparse resnet; higher res < this is a winner; very strong: iou@8:0.3. (oddly it's better than sp31)
mod = '"sp31"' # careful sparse resnet; higher res
mod = '"sp32"' # regular sparse resnet, but add some clones; do not encode in X0 < after 300k, this is very slightly better than sp33
mod = '"sp33"' # regular sparse resnet, but add some clones; do encode in X0


# other machine
mod = '"sp34"' # nonsparse Net3D; do encode in X0 < bad in comparison
# mod = '"sp35"' # custom3d; no encode in X0



mod = '"test00"' # builder; 10 iters; just use frame0 box and check ious
mod = '"test01"' # builder; 1k iters; faster logging
mod = '"test02"' # no image summs; fastest



mod = '"nar00"' # narrower bounds
mod = '"nar01"' # on a busy gpu2, train for occ also

# machine3
mod = '"nar02"' # same as nar01
mod = '"nar03"' # higher batchdim; slow logging




mod = '"pho00"' # nothing; run builder as-is if possible
mod = '"pho01"' # added data_mod
mod = '"pho02"' # use scorelist in the summ
mod = '"pho03"' # train a bit; regular bounds
mod = '"pho04"' # crop_feat before ml
mod = '"pho05"' # log500
mod = '"pho06"' # crop_feat with crop=2; this leaves 60x4x60 in emb
mod = '"pho07"' # crop=2; fewer prints
mod = '"pho08"' # no crop; log500
mod = '"pho09"' # no crop; log500; do not encode in X0


mod = '"bu00"' # builder; start on final
mod = '"bu01"' # emb3dnet
mod = '"bu02"' # emb3dnet
mod = '"bu03"' # trainer
mod = '"bu04"' # do not encode in x0
mod = '"bu05"' # 
mod = '"bu06"' # occ

# ok very quickly next, update the test setup to produce the proper vis

# these models are broken since they do not use crop/pad properly; maybe that was an issue with my old resnet exps here too


mod = '"bu07"' # just occ; log50
mod = '"bu08"' # pad and upsample feat before occ
mod = '"bu09"' # crop the invalid  
mod = '"bu10"' # crop one more voxel on each side
mod = '"bu11"' # ml
mod = '"bu12"' # log500; snap5k
mod = '"bu13"' # wider bounds; updated utils in emb3dnet
mod = '"bu14"' # fixed some scopes, so that valid shows up properly; updated utils.*
mod = '"bu15"' # occ sup at half resolution < illegal instruction
mod = '"bu16"' # occ sup at half resolution; 0-38 < illegal instruction
mod = '"bu17"' # occ sup at half resolution; 1-14
mod = '"bu18"' # 16-16-16; 0-20 < slow data read 
mod = '"bu19"' # 16-16-16; 0-18
mod = '"bu20"' # pret 300k bu12
mod = '"bu21"' # 500k iters; skipnet3d; no crop
mod = '"bu22"' # like bu21 but use proper out_dim in feat3dnet
mod = '"bu23"' # pret if you can < nope, fails. 
mod = '"bu24"' # altnet3d, which is simpler and maybe deeper
mod = '"bu25"' # replication padding in the encoder half


mod = '"bu26"' # altnet on aws, partial data (623 ex)
mod = '"bu27"' # full data; pret 5k from other; full data
mod = '"bu28"' # pret 5k bu27
mod = '"bu29"' # new 4x_v100 machine
mod = '"bu30"' # pret 5k 02_s2_m128x128x128_1e-4_F3_d64_O_c1_s.1_E3_n2_d16_c1_mags7i3t_bu28
mod = '"bu31"' # 16-8-16
mod = '"bu32"' # pret 5k bu31
mod = '"bu33"' # shifted things to _base
mod = '"bu34"' # cleaned up more
mod = '"bu35"' # cleaned more, and cube


current = '{}'.format(os.environ["exp_name"])
mod = '"{}"'.format(os.environ["run_name"]) 

############## define experiment ##############


###### Carla occupancies #########
exps['trainer_implicit_carla_lr4'] = [
    'carla_moc', # mode
    # 'carla_complete_data_train_for_multiview',
    'carla_sta_data_single_car',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    # 'train_emb3d',
    'train_localdecoder',
    'do_tsdf_implicit_occ',
    'log100',
    'snap5k', 
    # 'debug'
]

exps['trainer_implicit_carla_lr3'] = [
    'carla_moc', # mode
    # 'carla_complete_data_train_for_multiview',
    'carla_sta_data_single_car',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr3',
    'B2',
    'train_feat3d',
    # 'train_occ',
    # 'train_emb3d',
    'train_localdecoder',
    'do_tsdf_implicit_occ',
    'log1',
    'snap5k', 
    # 'debug'
]

exps['trainer_implicit_carla_5lr4'] = [
    'carla_moc', # mode
    # 'carla_complete_data_train_for_multiview',
    'carla_sta_data_single_car',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    '5lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    # 'train_emb3d',
    'train_localdecoder',
    'do_tsdf_implicit_occ',
    'log100',
    'snap5k', 
    # 'debug'
]

exps['trainer_implicit_carla_idx0'] = [
    'carla_moc', # mode
    # 'carla_complete_data_train_for_multiview',
    'carla_sta_data_single_car',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'idx0_dataloader',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    # 'train_emb3d',
    'train_localdecoder',
    'do_tsdf_implicit_occ',
    'log10',
    'snap5k', 
    # 'debug'
]
##################################
exps['builder'] = [
    'carla_moc', # mode
    # 'carla_multiview_train10_test10_data', # dataset
    'carla_multiview_train10_data', # dataset
    # 'carla_multiview_test_data', # dataset
    # 'carla_multiview_all_data', # dataset
    'carla_regular_bounds',
    # '1k_iters',
    # '100_iters',
    '5_iters',
    'lr0',
    'B1',
    'no_shuf',
    # 'train_feat3D',
    # 'train_feat3DS',
    # 'train_emb3D',
    # 'train_moc3D',
    # 'fastest_logging',
    'log1',
]
'''
['rgb_camXs', 'xyz_camXs', 'origin_T_camXs', 'origin_T_camRs', 'pix_T_cams', 'lrt_traj_camR', 'score_traj', 'full_lrtlist_camRs', 'full_scorelist_s', 'full_tidlist_s']
'''


exps['trainer_implicit_debug_concat'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    # 'debug',
    'summ_pca_points_3d',
    'summ_pca_points_2d',
    '500k_iters',
    'lr4',
    'B1',
    # 'train_occ',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'nearest_neighbour',
    'use_delta_mem_coords',
    'do_concat',
    'train_feat3d',
    'make_dense',
    'train_emb3d',
    'train_localdecoder',
    'debug',
    'log1',
    'snap5k', 
]



exps['trainer_implicit_occ'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    # 'debug',
    # 'summ_pca_points_3d',
    # 'summ_pca_points_2d',
    '500k_iters',
    'lr4',
    'B1',
    # 'train_occ',
    # 'pretrained_feat3d',
    # 'pretrained_localdecoder',
    # 'nearest_neighbour',
    'use_delta_mem_coords',
    # 'do_concat',
    'train_feat3d',
    'make_dense',
    # 'train_emb3d',
    'train_localdecoder',
    'do_implicit_occ',
    # 'debug',
    'log100',
    'snap5k', 
]

exps['trainer_implicit_occ_visualize'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    # 'debug',
    # 'summ_pca_points_3d',
    # 'summ_pca_points_2d',
    '500k_iters',
    'lr4',
    'B1',
    # 'train_occ',
    # 'pretrained_feat3d',
    # 'pretrained_localdecoder',
    # 'nearest_neighbour',
    'use_delta_mem_coords',
    # 'do_concat',
    'train_feat3d',
    'make_dense',
    # 'train_emb3d',
    'train_localdecoder',
    'do_implicit_occ',
    # 'debug',
    'log1',
    'snap5k', 
]


#newml3d exps:


exps['trainer_implicit_hyper_delta_nearest'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'nearest_neighbour',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'hypervoxel',
    'use_delta_mem_coords',
    'log100',
    'snap5k', 
]

exps['trainer_implicit_delta_nearest'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'nearest_neighbour',
    'use_delta_mem_coords',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'hypervoxel',
    'log100',
    'snap5k', 
]


exps['trainer_implicit_delta_nearest_concat'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'nearest_neighbour',
    'use_delta_mem_coords',
    'do_concat',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    # 'hypervoxel',
    'log100',
    'snap5k', 
]


#ml3d exps:


exps['trainer_ml'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'log100',
    'snap5k', 
]



exps['trainer_implicit_3dq'] = [
    'carla_moc', # mode
    'carla_sta_data_single_car',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'log100',
    'snap5k', 
]

exps['trainer_implicit'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'log100',
    'snap5k', 
]

exps['trainer_pointcontrast'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'no_shuf',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'point_contrast',
    'log100',
    'snap5k', 
]


exps['trainer_pointcontrast_og'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    # 'pretrained_localdecoder',
    # 'no_shuf',
    '500k_iters',
    'lr4',
    'B2',
    # 'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    # 'train_localdecoder',
    'point_contrast_og',
    'log100',
    'snap5k', 
]



exps['trainer_pointcontrast_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B1',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'point_contrast',
    'log100',
    'snap5k', 
]














exps['trainer_implicit_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'log100',
    'snap5k', 
    'debug'
]

exps['trainer_implicit_hyper_delta'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'hypervoxel',
    'use_delta_mem_coords',
    'log100',
    'snap5k', 
]



exps['trainer_implicit_hyper'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'hypervoxel',
    # 'use_delta_mem_coords',
    'log100',
    'snap5k', 
]





#rgb render exps:

exps['trainer_implicit_render_no_pretrain'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    # 'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'train_localdecoder_render',
    'log100',
    'snap5k', 
]

exps['trainer_implicit_render'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'train_localdecoder_render',
    'log100',
    'snap5k', 
]

exps['trainer_implicit_render_hyper'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'train_localdecoder_render',
    'log100',
    'snap5k', 
]


exps['trainer_implicit_render_hyper_delta'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    #sadf
    'train_feat3d',
    'train_localdecoder_render',
    'use_delta_mem_coords',
    'log100',
    'snap5k', 
]

exps['trainer_render'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'make_dense',
    'train_feat3d',
    'train_render',
    'train_rgb',
    'log100',
    'snap5k', 
]
















exps['trainer_render_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B1',
    'make_dense',
    'train_feat3d',
    'train_render',
    'train_rgb',
    'debug',
    'log1',
    'snap5k', 
]







exps['trainer_implicit_occ_complete_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'debug',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B1',
    'train_feat3d',
    'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'log1',
    'snap5k', 
]


exps['trainer_implicit_render_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    'train_localdecoder_render',
    'log1',
    'debug',
    'snap5k', 
]



exps['trainer_render_debug'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    '500k_iters',
    'lr4',
    'B1',
    'make_dense',
    'train_feat3d',
    'train_render',
    'train_rgb',
    'debug',
    'log1',
    'snap5k', 
]

############## net configs ##############


groups['carla_moc'] = ['do_carla_moc = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3D'] = [
    'do_moc3D = True',
    'moc3D_num_samples = 1000',
    'moc3D_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    # 'emb3d_ml_coeff = 1.0',
    # 'emb3d_l2_coeff = 0.1',
    # 'emb3d_mindist = 16.0',
    # 'emb3d_num_samples = 2',
    'emb3d_mindist = 16.0',
    'emb3d_num_samples = 2',
    'emb3d_ce_coeff = 1.0',
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
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
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

