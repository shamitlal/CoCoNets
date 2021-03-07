from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'occ_trainer'
current = 'render_trainer'
# current = 'center_trainer'
# current = 'seg_trainer'
# current = 'vq_trainer'
# current = 'vq_exporter'

mod = '"pr00"' # pretty model
mod = '"pr01"' # 
mod = '"pr02"' # again
mod = '"pr03"' # don't return early
mod = '"pr04"' # 
mod = '"pr05"' # viewnet
mod = '"pr06"' # give viewnet feat_memX0
mod = '"pr07"' # log50
mod = '"pr08"' # turn off loss in the bkg, via valid_camX0
mod = '"pr09"' # train1
mod = '"pr10"' # again, but on 1-14
mod = '"pr11"' # ep09; with mask; 1-14
mod = '"pr12"' # train data
mod = '"pr13"' # log50
mod = '"pr14"' # show _valid vis
mod = '"pr15"' # smaller (single weather) ep09
mod = '"pr16"' # mse instead of l1
mod = '"pr17"' # l1 again; lower down/up scale (2) and therefore not pret up < got killed, don't know why
mod = '"pr18"' # do not l2 normalize
mod = '"pr19"' # drop the conv3d layer in the decoder
mod = '"pr20"' # reset the arch; builder;
mod = '"pr21"' # show proj
mod = '"pr22"' # logspace slices
mod = '"pr23"' # fixed the intrinsics
mod = '"pr24"' # linear
mod = '"pr25"' # use utils_vox, to bring back the bug
mod = '"pr26"' # fix the bug and train
mod = '"pr27"' # exclude view0 from the average
mod = '"pr28"' # s=3 < exceeded mem
mod = '"pr29"' # warp 
mod = '"pr30"' # tell me about the inputs
mod = '"pr31"' # feed zi into viewnet; no featnet < exceeded ram
mod = '"pr32"' # same but builder
mod = '"pr33"' # train a while in builder
mod = '"pr34"' # trainer < again exceeded ram, this time at iter 340
mod = '"pr35"' # yes but noshuf, so that you get the same image eveyr time
mod = '"pr36"' # again but watch the mem more carefully
mod = '"pr37"' # again
mod = '"pr38"' # num_workers=1
mod = '"pr39"' # visualize zi
mod = '"pr40"' # normalize properly
mod = '"pr41"' # normalize .data
mod = '"pr42"' # show hist and feat of .data
mod = '"pr43"' # randn
mod = '"pr44"' # fixed centroid
mod = '"pr45"' # train regular viewnet+featnet combo with fixed centroid
mod = '"pr46"' # provide padded feat inputs to viewnet, since otw the mem calc will fail
mod = '"pr47"' # use logspace slices < nan
mod = '"pr48"' # linear slices; use padding=1 arch, and no upnet; no pret
mod = '"pr49"' # nearcube
mod = '"pr50"' # net3d (with skips) instead of encoder3d
mod = '"pr51"' # regular nearcube instead of wide_nearcube
mod = '"pr52"' # set of vars
mod = '"pr53"' # ep09one, with self.zi
mod = '"pr54"' # 488
mod = '"pr55"' # 431
mod = '"pr56"' # sample_pair = 0
mod = '"pr57"' # sample_pair = 1
mod = '"pr58"' # sample_pair = 2
mod = '"pr59"' # sample_id = [1,2]
mod = '"pr60"' # sample_id = [2,3]
mod = '"pr61"' # sample_id = [3,4]
mod = '"pr62"' # sample_id = [4,5]
mod = '"pr63"' # [1,2] again; higher resolution
mod = '"pr64"' # no biases in viewnet < ok, results are still good
mod = '"pr65"' # print ind along s
mod = '"pr66"' # get data ind
mod = '"pr67"' # shuffle and such 
mod = '"pr68"' # pass around the right zi and right origin_T_cami
mod = '"pr69"' # print ind along s; print diffs
mod = '"pr70"' # 4 decimal places for stats
mod = '"pr71"' # apapretnlty init zi with zeros, to see if the assignment is happening propelry
mod = '"pr72"' # no l2 norm < shows all ones
mod = '"pr73"' # re-declare self.zi every time
mod = '"pr74"' # do not update zi with teh feed inside 
mod = '"pr75"' # again but no shuf < why does it seem zi is changing non-productively?
mod = '"pr76"' # use camX0_T_camXs < ok good. so maybe this was not init properly
mod = '"pr77"' # use camX0_T_camXs
mod = '"pr78"' # use camXs_T_camis
mod = '"pr79"' # update zi_np
mod = '"pr80"' # do shuf
mod = '"pr89"' # sgd opt
mod = '"pr90"' # show total sum diff
mod = '"pr91"' # use set_len
mod = '"pr92"' # adam
mod = '"pr93"' # 
mod = '"pr94"' # 100k
mod = '"pr95"' # similar but just 3 ex in "one", to see if optim is indeed much faster as i'm guessing < yes
mod = '"pr96"' # investigate model params, try to exclude zi; just use model.viewnet.parameters()
mod = '"pr97"' # use separate optim for each zi
mod = '"pr98"' # compute valid in cami; use 1.0-0.5*valid
mod = '"pr99"' # use 0.5*0.5+valid
mod = '"pr100"' # double resolution
mod = '"pr101"' # properly double resolution (in both places)
mod = '"pr102"' # un-double; bias=True in the cnn
# ok, i have some bug where i am rendering the wrong view
mod = '"pr103"' # one; update zi.data (holy smokes, when did i turn this off?)
mod = '"pr104"' # double resolution
mod = '"pr105"' # orig resolution (Z/2); 2x higher lr for the volumes
mod = '"pr106"' # 0.001 smooth coeff
mod = '"pr107"' # 0.01 smooth coeff
mod = '"pr108"' # smaller res (128x64x128, instead of 160x80x160)
mod = '"pr109"' # train occ too
mod = '"pr110"' # occnet just slices out the zeroth chan
mod = '"pr111"' # upnet3d, with 1/4 resolution zi as input
mod = '"pr112"' # log500
mod = '"pr113"' # 1/8 resolution zi with chans*2; 4-scale upnet
mod = '"pr114"' # 09one, to see
mod = '"pr115"' # valid mask with no touch-ups
mod = '"pr116"' # wide_nearcube_bounds
mod = '"pr117"' # nearcube_bounds
mod = '"pr118"' # pret 100k 01_s1_p128x384_1e-4_U_s.001_V_d64_e1_mafs7i3ep09_pr117
mod = '"pr119"' # shared optim, so that you can load more easily
mod = '"pr120"' # frozen view, up
mod = '"pr121"' # actually pret
mod = '"pr122"' # train 300k; frozen up, view;
mod = '"pr123"' # train 300k; frozen view; up scale 2
mod = '"pr124"' # add a res block before up < not sure this had the res block running
mod = '"pr125"' # not frozen view; add view smooth coeff
mod = '"pr126"' # pret 01_s1_p128x384_1e-4_U_V_d64_e1_s.1_mafs7i3ep09_pr125
mod = '"pr127"' # actually do use that res block please
mod = '"pr128"' # feat3d_dim*4 as in dim
mod = '"pr129"' # separate optim for each zi < killed by accident
mod = '"pr130"' # 3 repeats every step
mod = '"pr131"' # 1 repeats every step; up layer is another res3dblock; eliminated the padding before viewnet
mod = '"pr132"' # higher res z; padding=0; pad by 2 after
mod = '"pr133"' # totally avoid returning early


mod = '"ren00"' # rendernet
mod = '"ren01"' # rendernet again
mod = '"ren02"' # train a while 
mod = '"ren03"' # do not l2 normalize
mod = '"ren04"' # do summ the feat; size Z/2  instead of /4
mod = '"ren05"' # train occ slice instead of render
mod = '"ren06"' # train occ slice instead of render
mod = '"ren07"' # single optimizer; no shuf
mod = '"ren08"' # params_to_optimize
mod = '"ren09"' # directly penalize mean(abs(self.zi))
mod = '"ren10"' # don't assign .data
mod = '"ren11"' # print those paramse
mod = '"ren12"' # param group for self.model.zi < ok, works.
mod = '"ren13"' # train for occ
mod = '"ren14"' # smaller params; occ = feat*10.0 < yes, descends much more quickly
mod = '"ren15"' # just use those zi_optim < works too
mod = '"ren16"' # exclude zi from the main optim < ok 
mod = '"ren17"' # set equal to feed < no, this does not work at all
mod = '"ren18"' # do not set; print diff < diff is zero
mod = '"ren19"' # grab zi_np; do use feed < ok, diff is nonzero, and optim is workign
mod = '"ren20"' # mistake 
mod = '"ren21"' # mistake
mod = '"ren22"' # train rendernet
mod = '"ren23"' # train occnet and rendernet < okokokok, it seems i am not passing or handling the camera mats properly. the zi is trying to handle many views
mod = '"ren24"' # assign to [b] for the pose
mod = '"ren25"' # just two txt files, to see if two occ_g alignments appear < indeed yes. ok, so this means i am just not swapping around the zi properly, or, the momentum is leaking
mod = '"ren26"' # SGD instead of adam, to see if there is a momentum leak. < holy smokes, good guess. 
mod = '"ren27"' # train render 
mod = '"ren28"' # alt list of Adam optims < no, still leak 
mod = '"ren29"' # another alt way
mod = '"ren30"' # another alt way adam < SUCCESS
mod = '"ren31"' # train render
mod = '"ren32"' # upnet after zi
mod = '"ren33"' # optimizer.step
mod = '"ren34"' # viewnet uses 1x1x1 fov in the 3d conv
mod = '"ren35"' # no up < still, near perfect rgb. i think viewnet learned some memorization trick
mod = '"ren36"' # ep09 full
mod = '"ren37"' # summ feat better
mod = '"ren38"' # train occ too
mod = '"ren39"' # just view; 300k
mod = '"ren40"' # just occ; 300k
mod = '"ren41"' # just render; 300k
mod = '"ren42"' # just occ
mod = '"ren43"' # better gpu
mod = '"ren44"' # *10 lr for the optim
mod = '"ren45"' # literally one
mod = '"ren46"' # upnet
mod = '"ren47"' # lock the feats to the l2 sphere < this adds some interp and does not hurt performance
mod = '"ren48"' # move features at 100x lr 
mod = '"ren49"' # 0.01 smooth loss on zi < looks slightly more interpretable i think
mod = '"ren50"' # 0.00 smooth loss on zi
mod = '"ren51"' # 0.01 smooth loss on zi
mod = '"ren52"' # rendernet
mod = '"ren53"' # use chans 1:4 of zi_up_pad
mod = '"ren54"' # only objective is rgb
mod = '"ren55"' # l1 instead
mod = '"ren56"' # add nerf rendering code
mod = '"ren57"' # sigmoid instead of tanh, and do it in the raw2output
mod = '"ren58"' # fix a bug in the reshape
mod = '"ren59"' # provide some occ sup
mod = '"ren60"' # raw noise 0.0
mod = '"ren61"' # show disp and acc maps
mod = '"ren62"' # raw noise 1.0 again
mod = '"ren63"' # occ raw
mod = '"ren64"' # no noise
mod = '"ren65"' # up just 4 chans
mod = '"ren66"' # zi is /2 res and just 4 chans and we use it directly < AMAZING results
mod = '"ren67"' # also optim occ
mod = '"ren68"' # fix a bug in the occ sup
mod = '"ren69"' # show depth map instead
mod = '"ren70"' # 
mod = '"ren71"' # clamp to 0.01  instead of 1e-10
mod = '"ren72"' # hist of disp_e, just to ee
mod = '"ren73"' # white bkg
mod = '"ren74"' # print stats, since the hist failed me < aha, it's due to a nan
mod = '"ren75"' # exclusive weights < nice, that was important
mod = '"ren76"' # 0.1 noise
mod = '"ren77"' # 1.0 noise
mod = '"ren78"' # don't mult by 10 in occnet
mod = '"ren79"' # clamp to help the gif
mod = '"ren80"' # smooth, and noise 0.1
mod = '"ren81"' # use 0.1 noise on grid_z_vec
mod = '"ren82"' # use 1.0 noise on grid_z_vec; print it < unstable; maybe it makes oversteps
mod = '"ren83"' # noise_amount = 1.0, uniform, at 0.5*bin size, so that it never oversteps
mod = '"ren84"' # wide nearcube bounds
mod = '"ren85"' # print dists
mod = '"ren86"' # use zmin=1e-4
mod = '"ren87"' # clamp grid to z_near
mod = '"ren88"' # try to render higher res
mod = '"ren89"' # full res voxelgrid
mod = '"ren90"' # render halfres again
mod = '"ren91"' # smooth 0.1 on zi
# there is something weird happening where the ENTIRE rgb image colorization shifts across iters. how is this possible?
# it seems it should be impossible, bc each var is independent; so the rays should optimize separately
mod = '"ren92"' # quarter resolution zi < not very sharp
mod = '"ren93"' # fewer prints
mod = '"ren94"' # smooth 0.0 on zi < still no good
mod = '"ren95"' # not-so-wide nearcube bounds < improves rgb recon slightly
mod = '"ren96"' # white_bkg=False < very bad for some reason
mod = '"ren97"' # white_bkg=True
mod = '"ren98"' # proper exclusive!!!
mod = '"ren99"' # white_bkg=False < ok good, with that bug fixed, black and white give the same output

mod = '"up00"' # 1x1x1 conv to scale up by 2
mod = '"up01"' # drop the l2 norm; fewer prints
mod = '"up02"' # 1x1x1 res layer to help
mod = '"up03"' # another res layer to help
mod = '"up04"' # relu=False after up < smart. looking good now.
mod = '"up05"' # zi is /8; scale 4 < ok, takes longer to fit for sure, and maybe perf is worse overall
mod = '"up06"' # chans=128 instead of 64 inside the upnet < indeed better
mod = '"up07"' # zi dim 128 instead of 64 < ok nice, pretty close to up04 now. 
mod = '"up08"' # zi is /16, scale=8
mod = '"up09"' # upnet produces 32-dim feats; rendernet does a 1x1 conv after projection < ok good job, this is slightly better fit
mod = '"up10"' # resnet then rgb layer < good power, but maybe this is just overfitting. in principle i like it though, bc this can bring out the thin structures
mod = '"up11"' # show pca of projected feat
mod = '"up12"' # train for a while on bigger data
mod = '"up13"' # zi is /4 and 4chan, and we render it directly
mod = '"up14"' # zi is /2 and 4chan, and we render it directly < this looks beautiful

mod = '"up15"' # zi is /4 and 128, and we up by 2; up_dim 4< OOM on compute-0-26 
mod = '"up16"' # compute-0-22 instead < OOM again 
mod = '"up17"' # res1 up1 res3, instead of res1 res1 up1 < OOM
mod = '"up18"' # zi is /8, up scale 4
mod = '"up19"' # v1: res1 up1 res3 conv1_norelu
mod = '"up21"' # v3: output some 32-dim feats, and put a single 1x1 conv rgb layer to render it
mod = '"up22"' # v2: res1, res1, up1, render directly


mod = '"di00"' # render directly; add the dict/vq; train on one ex
mod = '"di01"' # not EMA, to see if it converges faster < yes, but ok let's discard
mod = '"di02"' # vq; EMA; full dat; 


mod = '"up23"' # zi is /2 and 4chan, and we render it directly, and narrow_nearcube_bounds < ok, error is apparently lower, but it's hard to see visually, since so much is OOB 
mod = '"up24"' # zi is /2 and 4chan, and we render it directly, and narrow_flat_bounds



mod = '"sa00"' # save those goddamn npzs; snap50
mod = '"sa01"' # load npzs
mod = '"sa02"' # save better npzs
mod = '"sa03"' # load
# ok works perfectly


############## define experiment ##############

exps['builder'] = [
    'carla_pretty', # mode
    # 'carla_multiview_all_data', # dataset
    # 'carla_multiview_ep09_data', # dataset
    'carla_multiview_ep09one_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train_data', # dataset
    'carla_nearcube_bounds',
    '100_iters',
    # '10_iters',
    # '3_iters',
    'lr4',
    'B1',
    'no_shuf',
    # 'pretrained_feat3D', 
    # 'pretrained_up3D', 
    # 'pretrained_center', 
    # 'pretrained_center', 
    # 'pretrained_seg', 
    # 'train_feat3D',
    # 'train_up3D',
    # 'train_center',
    # 'train_view',
    'train_occ',
    # 'train_render',
    'log1',
]
exps['occ_trainer'] = [
    'carla_pretty', # mode
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_ep09_data', # dataset
    # 'carla_multiview_ep09one_data', # dataset
    'carla_multiview_one_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train_data', # dataset
    # 'carla_wide_cube_bounds',
    # 'carla_wide_nearcube_bounds',
    'carla_nearcube_bounds',
    '5k_iters',
    # '100k_iters',
    # '1k_iters',
    'lr3',
    'B1',
    # 'pretrained_feat3D', 
    # 'pretrained_up3D', 
    # 'pretrained_view', 
    # 'train_feat3D',
    # 'frozen_view',
    # 'frozen_up3D',
    'train_up3D',
    # 'no_shuf', 
    # 'train_view',
    'train_occ',
    'log500',
]
exps['render_trainer'] = [
    'carla_pretty', # mode
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
    'carla_pretty', # mode
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
    'carla_pretty', # mode
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
    'carla_pretty', # mode
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
    'carla_pretty', # mode
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
groups['carla_pretty'] = ['do_carla_pretty = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    'feat3D_smooth_coeff = 0.01',
]
groups['train_up3D'] = [
    'do_up3D = True',
    'up3D_smooth_coeff = 0.01',
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
# Z = 160
# Y = 80
# X = 160
# Z_test = 160
# Y_test = 80
# X_test = 160
Z = 128
Y = 64
X = 128
Z_test = 128
Y_test = 64
X_test = 128

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 1
H = 128*2
W = 384*2
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)
# PH = int(H)
# PW = int(W)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data4/carla/processed/npzs"

groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mafs7i3one"',
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
    'trainset = "mafs7i3ep09"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mafs7i3ep09one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mafs7i3one"',
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
    'trainset = "mafs7i3t"',
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
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_nearcube_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
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

