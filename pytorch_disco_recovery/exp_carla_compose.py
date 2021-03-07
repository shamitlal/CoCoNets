from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
# current = 'tester'

mod = '"com00"' # builder
mod = '"com01"' # run featnet; no latents
mod = '"com02"' # show occ_memX0
mod = '"com03"' # show rgb_memX0
mod = '"com04"' # S=2, smooth loss
mod = '"com05"' # do backprop
mod = '"com06"' # encoder-decoder < not sure this happened
mod = '"com07"' # print some shapes; 
mod = '"com08"' # workign encodr
mod = '"com09"' # dcaa
mod = '"com10"' # dcaa again
mod = '"com11"' # show mask
mod = '"com12"' # show actual mask
mod = '"com13"' # sparse resnet
mod = '"com14"' # encoderdecoder3d
mod = '"com15"' # add the final layer
mod = '"com16"' # look at shapes
mod = '"com17"' # 
mod = '"com18"' # 
mod = '"com19"' # emb3d
mod = '"com20"' # 
mod = '"com21"' # 
mod = '"com22"' # occ too; look at crop guess
mod = '"com23"' # upsample 
mod = '"com24"' # one more res block
mod = '"com25"' # train a while
mod = '"com26"' # log50; fewer prints
mod = '"com27"' # drop the depth stuff
mod = '"com28"' # do normalize
mod = '"com29"' # 313 
mod = '"com30"' # nearcube again, so that occ looks better; randomize centroid
mod = '"com31"' # only apply emb loss in vis_memX0 < hm, relu=false in the up3d seems like a mistake
mod = '"com32"' # up by deconv instead
mod = '"com33"' # up by nn instead
mod = '"com34"' # like com33, but show the feat and altfeat inputs
mod = '"com35"' # one more res layer

mod = '"com36"' # test frequently, to debug 
mod = '"com37"' # B1; log/test every 500
mod = '"com38"' # log50; deal in cropped coords
mod = '"com39"' # log500
mod = '"com40"' # do save checkpoints!!!! gosh. 
mod = '"com41"' # log50; properly parse gt boxes < hell yeah
mod = '"com42"' # log500; properly parse gt boxes < hell yeah

# hm, ious are zero. but the boxes look ok. maybe this is the same bug jing talked about
mod = '"com43"' # log10; S_test = 5; print some of the ious < ok all zero
mod = '"com44"' # log10; S_test = 5; print some of the ious; print xyz too < ok, it looks pretty damn close at S=1, yet it's gettting iou 0.
# did i accidentally change the corner ordering?
mod = '"com45"' # try both ways, take the winner
mod = '"com46"' # log50; S_test = 20
mod = '"com47"' # log500
mod = '"com48"' # S_test = 50 < maybe i never ran this
mod = '"com49"' # transform the lrts
mod = '"com50"' # com48 again < OOM
mod = '"com51"' # print the s
mod = '"com52"' # detach before matmul
mod = '"com53"' # detach the feats
mod = '"com54"' # torch.cuda.empty_cache()
mod = '"com55"' # do not voxelize the entire seq simultaneously on every step
mod = '"com56"' # log500

mod = '"com57"' # old arch; crop 18,18,18 (matching the other net) right after featnet3d
mod = '"com58"' # reshape instead of view; log50, to get through a test iter
mod = '"com59"' # log500
mod = '"com60"' # cleaner impl; log50
mod = '"com61"' # eccv-supp-style summs
mod = '"com62"' # fewer prints

mod = '"tha00"' # tester; pret 02_s2_m160x80x160_1e-4_F3_d32_O_c1_E3_n2_c1_mags7i3t_taqs100i2v_com56
mod = '"tha01"' # show bkg memx0
mod = '"tha02"' # S_test = 10
mod = '"tha03"' # show original feat_memX0
mod = '"tha04"' # show the inputs without pca, bc i'm not convinced about the mask < ok, it's mostly good, but sometimes too conservative
mod = '"tha05"' # coeff = 1.5
mod = '"tha06"' # coeff = 1.2
mod = '"tha07"' # crop obj
mod = '"tha08"' # only summ if not return early
mod = '"tha09"' # fewer prints
mod = '"tha10"' # do the trivial compose
mod = '"tha11"' # use widemask 
mod = '"tha12"' # add 1m on each len < ok, getting there. to fix this completely, i think i will need a gaussian blur
mod = '"tha13"' # apply_4x4 with memI_T_cam0
mod = '"tha14"' # do that in a loop along s; show the gif
mod = '"tha15"' # show the recomposed scene in a gif
mod = '"tha16"' # show bottom vis
mod = '"tha17"' # show bot and top occ < this looks quite good, but also, it shows i need that gaussian
mod = '"tha18"' # blurry mask
mod = '"tha19"' # blurrier mask < ok, but on some iters, the occ looks crazy. (all occ within the box)
mod = '"tha20"' # l2 norm after composing < something weird happening. but it looks related...
mod = '"tha21"' # hard mask 
mod = '"tha22"' # hard mask
mod = '"tha23"' # add mask vis < maybe the mask is touching the ceiling?
mod = '"tha24"' # pad 0.5 < still issue
mod = '"tha25"' # show the other angle of the mask vis, to see if it is touching the ceiling maybe
mod = '"tha26"' # .clone() in crop
mod = '"tha27"' # print some shapes
mod = '"tha28"' # wider bounds; use _test resolutions properly
mod = '"tha29"' # SIZE_test = 20
mod = '"tha30"' # 16-16-16
mod = '"tha31"' # 16-8-16 < ok looks fine, but this gives worse perf than the zoomed-in version. this is consistent with my eccv exps
mod = '"tha32"' # 8-4-8 < ok, back to the artifact
mod = '"tha33"' # *valid
mod = '"tha34"' # 8-6-8
mod = '"tha35"' # same resolution, but 12-9-12 meters
mod = '"tha36"' # 16-8-16
mod = '"tha37"' # same 
mod = '"tha38"' # pret 100k instead of 90k
mod = '"tha39"' # 12-6-12
mod = '"tha40"' # 8-4-8; show occ_obj
mod = '"tha41"' # mult by valid mask
mod = '"tha42"' # mult obj by valid mask too
mod = '"tha43"' # make zeros before the apply_4x4
mod = '"tha44"' # binary_feat=True
mod = '"tha45"' # show valid vis
mod = '"tha46"' # alt vis
mod = '"tha47"' # obj_mask = obj_mask*valid
mod = '"tha48"' # show valid and main mask as oned with norm=Faslse
mod = '"tha49"' # trim the edges AFTER the warp and crop, and also mult after
mod = '"tha50"' # put proper padding in valid
mod = '"tha51"' # eliminate the extra trim; don't even mult occ*valid < looks pretty good, though a scary amount gets lost sometimes
mod = '"tha52"' # 12-6-12 bounds test


mod = '"pre00"' # a few iters; log10; pret 100k 02_s2_m160x80x160_1e-4_F3_d32_O_c1_E3_n2_c1_mags7i3t_taqs100i2v_com56
mod = '"pre01"' # 300k; log500; pret 100k 02_s2_m160x80x160_1e-4_F3_d32_O_c1_E3_n2_c1_mags7i3t_taqs100i2v_com56
mod = '"pre02"' # train as a cube

mod = '"tha53"' # blurry mask
mod = '"tha54"' # S_test = 10 again

# ok, at this stage, i have the 3d scenes
# i can do an experiment now, of rendering these, and checking if matches in projection space correlate better than those of 3d space
# the step after that, is to allow alt hypotheses, and select them based on the assumption of score being helpful

mod = '"pro00"' # project/render; show top_feat_vis < seems like we're projected into the top left corner of a full-res image
mod = '"pro01"' # pca=True
mod = '"pro02"' # show bot feats too
mod = '"pro03"' # print some shapes
mod = '"pro04"' # make raw H,W 128x384, since that's what it looks like the rgb is shaped
mod = '"pro05"' # render full H,W
mod = '"pro06"' # compute diffs 2d and 3d
mod = '"pro07"' # compute corr coeffs
mod = '"pro08"' # get the real 3d values
mod = '"pro09"' # 50 iters
mod = '"pro10"' # compute summary stats of corr_2d and corr_3d
mod = '"pro11"' # similar but straight mean instaed of masked < slightly better

# i think the bigger issue is that the zoom region renders to a small 2d area sometimes;

mod = '"pro12"' # re-run the last thing
mod = '"pro13"' # simplify the code a bit
mod = '"pro14"' # move the obj/bkg code to the front
mod = '"pro15"' # move the assembly code into the func
mod = '"pro16"' # encap scene_memY0
mod = '"pro17"' # reuse feat_memI
mod = '"pro18"' # compute diff3d and get its corr
mod = '"pro19"' # simpler summs
mod = '"pro20"' # compute sceneY0_g and diff_3ds_g
mod = '"pro21"' # use .inverse() for the obj_T_cam
mod = '"pro22"' # caught bug: you NEED to compute feat_memX0i, bc feat_memI has a moving centroid
mod = '"pro23"' # repalce _e with _g when the diff is smaller
mod = '"pro24"' # print iou on every step
mod = '"pro25"' # use a zero-velocity estimate as backup
# mean_ious [0.98 0.88 0.87 0.84 0.87 0.85 0.87 0.86 0.84 0.86] # too high!
mod = '"pro26"' # use a const-velocity estimate as backup; also, vis the new tops
# mean_ious [0.98 0.88 0.86 0.81 0.84 0.8  0.82 0.83 0.83 0.83] # still a bug
# ok bug fixed i think


# single hypothesis, for reference:
# mean_ious [0.98 0.81 0.8 0.76 0.68 0.65 0.62 0.6 0.58 0.56]

mod = '"pro27"' # constvel hypothesis
# mean_ious [0.98 0.83 0.81 0.73 0.68 0.65 0.62 0.6  0.58 0.54]

mod = '"pro28"' # gt hypothesis
# mean_ious [0.98 0.87 0.84 0.81 0.78 0.75 0.76 0.78 0.76 0.74]

mod = '"pro29"' # zeromot hypothesis
# mean_ious [0.98 0.83 0.8  0.73 0.68 0.64 0.62 0.6  0.57 0.51]



mod = '"pro30"' # gt hypothesis; foreground only (using obj_mask_memY0)
# mean_ious [0.98 0.86 0.84 0.79 0.77 0.74 0.76 0.82 0.75 0.76]

mod = '"pro31"' # gt hypothesis; full scene; sharp edges


# train again; i want better occ
mod = '"pro32"' # pret; JUST train for occ
mod = '"pro33"' # skip arch
mod = '"pro34"' # log50, to get an error sooner
mod = '"pro35"' # magv on test iters; on every iter just run_train block
mod = '"pro36"' # small bug; log50
mod = '"pro37"' # JUSt train
mod = '"pro38"' # log500
mod = '"pro39"' # same
mod = '"pro40"' # emb
mod = '"pro41"' # no emb; conv3d on the occ!!! holy smoke
mod = '"pro42"' # no skip
mod = '"pro43"' # faster gpu


############## define experiment ##############

exps['builder'] = [
    'carla_compose', # mode
    # 'carla_multiview_all_data', # dataset
    'carla_multiview_ep09_data', # dataset
    # 'carla_multiview_ep09one_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train_data', # dataset
    # 'carla_nearcube_bounds',
    # 'carla_nearcube_bounds',
    'carla_313_bounds',
    # '100_iters',
    '10_iters',
    # '3_iters',
    'lr4',
    'B1',
    'no_shuf',
    # 'pretrained_feat3D', 
    # 'pretrained_up3D', 
    # 'pretrained_center', 
    # 'pretrained_center', 
    # 'pretrained_seg', 
    'train_feat3D',
    'train_emb3D',
    # 'train_up3D',
    # 'train_center',
    # 'train_view',
    'train_occ',
    # 'train_render',
    # 'no_backprop', 
    'log1',
]
exps['trainer'] = [
    'carla_compose', # mode
    # 'carla_multiview_train_test_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_16-8-16_bounds_train',
    'carla_16-8-16_bounds_test',
    # 'carla_16-16-16_bounds_train',
    # 'carla_8-4-8_bounds_test',
    # 'carla_12-6-12_bounds_test',
    '300k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D',
    'pretrained_occ',
    'train_feat3D',
    'train_emb3D',
    'train_occ',
    'log500',
    'snap5k',
]
exps['tester'] = [
    'carla_compose', # mode
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
    'carla_compose', # mode
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
    'carla_compose', # mode
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
    'carla_compose', # mode
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
    'carla_compose', # mode
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
    'carla_compose', # mode
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
groups['carla_compose'] = ['do_carla_compose = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat3D_skip = True',
]
groups['train_up3D'] = [
    'do_up3D = True',
    # 'up3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ce_coeff = 0.1',
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
    'occ_smooth_coeff = 0.1',
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

SIZE = 16
SIZE_test = 16
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
S_test = 10
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
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
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
groups['carla_16-8-16_bounds_test'] = [
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

