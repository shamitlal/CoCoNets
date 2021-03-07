from exp_base import *

############## choose an experiment ##############

# the idea here is to do some clean bkg subtraction

current = 'builder'
current = 'trainer'
current = 'tester'

mod = '"sub00"' # builder; sub start, copied from _reloc
mod = '"sub01"' # again, with less dead code
mod = '"sub02"' # get feat_memXAI
mod = '"sub03"' # S = 20 instead of 10
mod = '"sub04"' # show masks in A 
mod = '"sub05"' # req 3k pts ; make summs 
mod = '"sub06"' # show feat_memXAI_input
mod = '"sub07"' # show rgbs
mod = '"sub08"' # show gt occs as inputs
mod = '"sub09"' # do the inb check in A coords
mod = '"sub10"' # check inb by voxelizing and cropping; req 300 < sometimes no object 
mod = '"sub11"' # req 1k pts instead of 300; 
mod = '"sub12"' # get median feat and summ it
mod = '"sub13"' # summ occ too < the median seems still dominated by the obj
mod = '"sub14"' # S = 40
mod = '"sub15"' # S_test = 60
mod = '"sub16"' # S_test = 90
mod = '"sub17"' # compute and show diffs
mod = '"sub18"' # normalize all diffs afterward
mod = '"sub19"' # fps=16 instead of 8
mod = '"sub20"' # prep step to making synced vis
mod = '"sub21"' # show super vis
# mod = '"sub22"' # do the sub with occ
mod = '"sub23"' # vertical vis
mod = '"sub24"' # do the sub with occ
mod = '"sub25"' # compute median on occ directly < this looks better i think
mod = '"sub26"' # vis-aware median; where nan, use safe median
mod = '"sub27"' # permute trick
mod = '"sub28"' # add 0.25 on all sides of the obj, so that tiny objs show up better
mod = '"sub29"' # add 0.5 on all sides of the obj
mod = '"sub30"' # generate it at higher res, and downsample; also, show the orig highres stuff
mod = '"sub31"' # again, so i can compare centroids < indeed, same. rand seed is working
mod = '"sub32"' # S_test = 5, just to investigate the bike mask faster
mod = '"sub33"' # do_test=False; run teh train set
mod = '"sub34"' # tas data
mod = '"sub35"' # S = 80
mod = '"sub36"' # more tas data; 25 iters
mod = '"sub37"' # vis occ in the feat slot
mod = '"sub38"' # straight median; please inspect the median itself < ok! vis-aware one is better
mod = '"sub39"' # 10 iters again; S = 50
mod = '"sub40"' # vis-aware median; norm*vis
mod = '"sub41"' # expand vis by 1vox
mod = '"sub42"' # expand vis by another vox
mod = '"sub43"' # wide vis
mod = '"sub44"' # S = 10; 3 iters; get conns and boxes
mod = '"sub45"' # show diff on anchor frm
mod = '"sub46"' # 5 iters
mod = '"sub47"' # propose boxes in all frames
mod = '"sub48"' # realy do that
mod = '"sub49"' # overlay boxes on diff
mod = '"sub50"' # allow K = 8; print box overflow
mod = '"sub51"' # only include positive diffs (where we have occ)
mod = '"sub52"' # show pers vis too
mod = '"sub53"' # don't show scores
mod = '"sub54"' # show gt boxes in bev too
mod = '"sub55"' # req >2vox
mod = '"sub56"' # don't req occ
mod = '"sub57"' # don't draw boxes with score 0
mod = '"sub58"' # print fram eid
mod = '"sub59"' # do use occ < ok, why no boxes in frame4?
mod = '"sub60"' # print about the boxes 
mod = '"sub61"' # print more 
mod = '"sub62"' # blue vis
mod = '"sub63"' # vis on occ
mod = '"sub64"' # 3 iters; focus on frme2
mod = '"sub65"' # allow any size
mod = '"sub66"' # print each reject
mod = '"sub67"' # allow any size 
mod = '"sub68"' # drop invalid boxes
mod = '"sub69"' # req 1 vox
mod = '"sub70"' # fewer prints
mod = '"sub71"' # K = 16; S = 20
mod = '"sub72"' # show all boxes on pers too
mod = '"sub73"' # transform boxes to camXI
mod = '"sub74"' # properly do that
mod = '"sub75"' # tat data, with 16 object slots
mod = '"sub76"' # more data
mod = '"sub77"' # S = 30 < ok, but why am i still mising some big cars?
mod = '"sub78"' # only drop if >50
mod = '"sub79"' # K = 32 instead of 16 < still missing a big car
mod = '"sub80"' # 20 threshs instead of 12
mod = '"sub81"' # take it as long as not_already_have < ok! i see lots of boxes now; some for the little road bumps
mod = '"sub82"' # req 1vox < gone
mod = '"sub83"' # any again; K = 64; 
mod = '"sub84"' # allow vox >=1 < ah, some of my favorite objects are getting sliced/cropped out
mod = '"sub85"' # 16-16-16 < ok much better 
mod = '"sub86"' # S = 50; 5 iters; <40 for hwd; 
mod = '"sub87"' # S = 10; req mask[
mod = '"sub88"' # S = 50
mod = '"sub89"' # norm=True for diff vis
# mod = '"sub90"' # evaluate iou 
# mod = '"sub91"' # round those occs, so you totally elim the bad ones
# mod = '"sub92"' # use gt occ/free to hard-clean things
mod = '"sub93"' # S = 10; evaluate iou
mod = '"sub94"' # report mean iou across S
mod = '"sub95"' # S = 50
mod = '"sub96"' # S = 10; do the dropping wiht B=1
mod = '"sub97"' # S = 50
mod = '"sub98"' # .round() the occ, to elim the bad ones
mod = '"sub99"' # boost the occ with occ/free
mod = '"sub100"' # boost the occ with just occ
mod = '"sub101"' # log50; 10k iters
mod = '"sub102"' # drop the occ boost; more varied data < ok, actually, sub101 is a bit better
# i think this eval is a bit unfair, since it includes gt objects that are OOB

mod = '"sub103"' # figure out the padding
mod = '"sub104"' # pad using the pad
mod = '"sub105"' # use that new scorelist in the perspective view too
mod = '"sub106"' # drop the _full vis
mod = '"sub107"' # get the pad in halfmem; only summ if score>0
mod = '"sub108"' # again
mod = '"sub109"' # show mean and var of occ
mod = '"sub110"' # diff = relu(norm(thing-mean)-std); also, use straight mean when vis-aware is none
mod = '"sub111"' # use std*2
mod = '"sub112"' # S = 10; 
mod = '"sub113"' # S = 20
mod = '"sub114"' # S = 50
mod = '"sub115"' # S = 10; use temporal diffs, and use diff = relu(norm(thing-median)-mean_temporal_diff) < ok, this is better than sub112
mod = '"sub116"' # S = 50
mod = '"sub117"' # S = 50; back to just regular median
mod = '"sub118"' # use_cache=True
mod = '"sub119"' # S = 10; use_cache=True
mod = '"sub120"' # read from cache hopefully
mod = '"sub121"' # include data_name in cache fn
mod = '"sub122"' # read from cache hopefully
mod = '"sub123"' # save/load cache with better name
mod = '"sub124"' # update the saving/loading a bit 
mod = '"sub125"' # load please < success
mod = '"sub126"' # clean re-go save < 157s
mod = '"sub127"' # clean re-go load < 118s; somehow, lower perf. why?
mod = '"sub128"' # load again
mod = '"sub129"' # save again < 159
mod = '"sub130"' # load again, but use the loaded centroid to help with rescoring
mod = '"sub131"' # save info for 100 iters
mod = '"sub132"' # get some occupancy CS
mod = '"sub133"' # do that 100 iters
mod = '"sub134"' # print time for each step
mod = '"sub135"' # save medians
mod = '"sub136"' # load medians
mod = '"sub137"' # save boxes
mod = '"sub138"' # load boxes < the connlist is apparently huge, so the npzs are 1.3g and this is slow
mod = '"sub139"' # re-save boxes, without connlist
mod = '"sub140"' # don't even compute connlist
mod = '"sub141"' # load boxes < 32s
mod = '"sub142"' # do_test; log10 < 18s
mod = '"sub143"' # do_test; log1 so that we save the early vis; 100 iters
mod = '"sub144"' # re-score boxes, using debugged CS on the occs
mod = '"sub145"' # run through again, without rescoring
mod = '"sub146"' # 100 iters
mod = '"sub147"' # rescore with occupancy CS
mod = '"sub148"' # score natively with this; S = 50
mod = '"sub149"' # don't save the _vis tensors; only compute them when logging is necessary
mod = '"sub150"' # 10 iters; load
mod = '"sub151"' # rescore using CS on diffs
mod = '"sub152"' # do that for 100 iters; log10
mod = '"sub153"' # do that for 100 iters; log10; do_test
mod = '"sub154"' # do that for 100 iters; log10; do_test


mod = '"rel00"' # start on reliability for occnet
mod = '"rel01"' # on train iters, fire feat  
mod = '"rel02"' # again
mod = '"rel03"' # fire occ too
mod = '"rel04"' # occrel
mod = '"rel05"' # train < slow data read time
mod = '"rel06"' # B2 < slow data read time
mod = '"rel07"' # num_workers = 4 < ok, but a lot of early returns
mod = '"rel08"' # log10, so i can see better
mod = '"rel09"' # sum the total loss; only req 300 inb points
mod = '"rel10"' # only try 20 times to find the centroid
mod = '"rel11"' # log50
mod = '"rel12"' # compute-0-18
mod = '"rel13"' # 10; B1
mod = '"rel14"' # num_workers=1 for this guy
mod = '"rel15"' # more powerful occrelnet
mod = '"rel16"' # restart after matrix failure
mod = '"rel17"' # log50; 
mod = '"rel18"' # log50; num_workers=4
mod = '"rel19"' # summ_oned too
mod = '"rel20"' # num_workers=6
mod = '"rel21"' # compute-0-18
mod = '"rel22"' # compute-1-14; mag data
# mod = '"rel23"' # compute-0-18; mag data
mod = '"rel24"' # 1-14
mod = '"rel25"' # 0-18
mod = '"rel26"' # 1-14
mod = '"rel27"' # summ occ*occrel
mod = '"rel28"' # 4-way bal: pos-occ, neg-occ, pos-free, neg-free
mod = '"rel29"' # debug that occ*occrel summ
mod = '"rel30"' # add a val set
mod = '"rel31"' # add a val set; use bounds val and S_val = 2; log10 to check
mod = '"rel32"' # only req that the mean of the batch is ok
mod = '"rel33"' # SIZE_val = 20 
mod = '"rel34"' # fewer prints
mod = '"rel35"' # only req mean inb; log500



mod = '"can00"' # tester; pret occrel 10k 04_s2_m160x80x160_1e-3_F3f_d32_Of_c1_R_c1_mags7i3t_mags7i3v_rel35
mod = '"can01"' # do not "rescore" boxes
mod = '"can02"' # 10 iters; S_test = 10
mod = '"can03"' # occ = occ * occrel
mod = '"can04"' # instead, diff = diff * occrel
mod = '"can05"' # show vis of occ and occrel
mod = '"can06"' # properly can05; also show occrel in the super vis
mod = '"can07"' # pre-normalize diffs, so that they are constant scale
mod = '"can08"' # use 1.0-exp(-diff)
mod = '"can09"' # print stats
mod = '"can10"' # summ max over birdview < looks good
mod = '"can11"' # proper option for that
mod = '"can12"' # don't normalize; S = 20
mod = '"can13"' # don't use occrel, as comparison
mod = '"can14"' # S = 50
mod = '"can15"' # cache and use occrel
mod = '"can16"' # re-compute boxes; ONLY proceed to find boxes where thresh > 0.5
mod = '"can17"' # same
mod = '"can18"' # do not pad before getting boxes; just add crop_guess to the outputs
mod = '"can19"' # if CS<=0.51, drop the box < ok, much cleaner



mod = '"can20"' # redo that last thing
mod = '"can21"' # re-enable box cache
mod = '"can22"' # compute moving scorelist
mod = '"can23"' # use moving instead of full; return after summ
mod = '"can24"' # req that the object be valid at the beginning and the end
mod = '"can25"' # check motion on every frmae
mod = '"can26"' # alt way: check for a change in inb
mod = '"can27"' # req 50 change; print the ones above thresh
mod = '"can28"' # req 10 change; print the ones above thresh
mod = '"can29"' # score = absdiff, so i can see
mod = '"can30"' # allow scores >1 
mod = '"can31"' # don't speed up the gif
mod = '"can32"' # use iou
mod = '"can33"' # set score =iou
mod = '"can34"' # always display score
mod = '"can35"' # use new_scorelist 
mod = '"can36"' # don't rescore with inbound, so that i can see more and verify
mod = '"can37"' # do rescore with inbound; use 0.95 thresh to trigger motion
mod = '"can38"' # do not return so early
mod = '"can39"' # do not use box cache, so that we can regen that bev vis
mod = '"can40"' # use 
mod = '"can41"' # 5 iters ;if global_step<4, return early
mod = '"can42"' # 5 iters ;if global_step<5, return early; show lrtlist_e,g
mod = '"can43"' # drop lrts < 0.01 len; print maps
mod = '"can44"' # print for s==S/2
mod = '"can45"' # do not exclude movign objects, so i can see all the lrts and a lower mAP < ok, indeed, the map is lower now. NO BUG.
mod = '"can46"' # re-enable motion-based rescoring; run for 100 iters; generate new box cache; K=32
mod = '"can47"' # use a solid centroid; delete the whole cache; return early before feats
mod = '"can48"' # pret pro43
mod = '"can49"' # 10 iters, load from cache < ok; 3mins instead of 11, for these steps
mod = '"can50"' # disable box cache
mod = '"can51"' # do return connlist < same speed
mod = '"can52"' # do CS on conn list, with a single voxel dilation
mod = '"can53"' # generate flow < some cuda bug
mod = '"can54"' # 0-14 < ok good
mod = '"can55"' # show flow vis
mod = '"can56"' # show feats too; dilation=1
mod = '"can57"' # clip = 2
mod = '"can58"' # do not relu
mod = '"can59"' # summ occ flow
mod = '"can60"' # summ occ too
mod = '"can61"' # clip 1.0
mod = '"can62"' # do not use that edge mask
mod = '"can63"' # mag vis
mod = '"can64"' # do use edge mask
mod = '"can65"' # show featmag and flowmag vis
mod = '"can66"' # kernel 3 for that 
mod = '"can67"' # don't return early; backwarp the diff; vis this; < floating point exception!!
mod = '"can68"' # print
mod = '"can69"' # stride 1, kernel 1, pad 0 
mod = '"can70"' # do not return early
mod = '"can71"' # only go to S-1
mod = '"can72"' # use cumdiff for the scoring
mod = '"can73"' # use zero flow instead, to make sure that's worse
mod = '"can74"' # real flow; use sum instead of prod
mod = '"can75"' # again
mod = '"can76"' # return after flow; flow*occ*occ_median summ
mod = '"can77"' # again; clip=2.0
mod = '"can78"' # clip = 1.0; occ0*(1.0-occ_median)
mod = '"can79"' # clip = 2.0; occ0*(1.0-occ_median)
mod = '"can80"' # debug;
mod = '"can81"' # debug; clip=1
mod = '"can82"' # again
mod = '"can83"' # again
mod = '"can84"' # again
mod = '"can85"' # again
mod = '"can86"' # proceed and get proposals
mod = '"can87"' # show best lrt
mod = '"can88"' # delete cache, since something is wrong; use multiplicative cumdiff
mod = '"can89"' # small bugfix
mod = '"can90"' # show best iou in the vis 
mod = '"can91"' # one less dilation of vis; reset box cache
mod = '"can92"' # compute flow01 and flow10 then return
mod = '"can93"' # compute consistency
mod = '"can94"' # use norm of (flow10+flow01)
mod = '"can95"' # mask1*mask2
mod = '"can96"' # get flow04 by chaining a bunch
mod = '"can97"' # get flow40  and consistency 
mod = '"can98"' # do it 05; show mask itself
mod = '"can99"' # use visibility mask
mod = '"can100"' # use the right coords please
mod = '"can101"' # proceed further; use flow05_mag to propose boxes
mod = '"can102"' # don't compute vis0
mod = '"can103"' # use cumdiff summed across 6 frames
mod = '"can104"' # use cumdiff to propose objects
mod = '"can105"' # use consistent_flow05_mag to prpose objects, and note, this time we mult by the mask
mod = '"can106"' # use consistent_flow05_mag * diff
mod = '"can107"' # use summed cumdiff again, but /5 so it's brighter
mod = '"can108"' # again
mod = '"can109"' # accumulate one step at a time < this seems slightly better
mod = '"can110"' # mult mask by occrel < this looks worse
mod = '"can111"' # mult flow by occrel directly
mod = '"can112"' # for vis, do not mult by rel0
mod = '"can113"' # break and re-integrate flow05


############## define experiment ##############

exps['builder'] = [
    'carla_sub', # mode
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_test_data', # dataset
    'carla_tatt_trainset_data', # dataset
    # 'carla_tasv_testset_data', # dataset
    'carla_16-16-16_bounds_train',
    'carla_16-16-16_bounds_test',
    # '10k_iters',
    '100_iters',
    # '10_iters',
    # '5_iters',
    # 'use_cache', 
    'lr4',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    # 'no_backprop',
    # 'do_test', 
    # 'log1',
    'log10',
    # 'log50',
    # 'log5',
]
exps['trainer'] = [
    'carla_sub', # mode
    # 'carla_tat1_trainset_data', # dataset
    # 'carla_tatt_trainset_data', # dataset
    'carla_multiview_train_data', # dataset
    'carla_multiview_val_data', # dataset
    'carla_16-8-16_bounds_train',
    'carla_16-8-16_bounds_val',
    '100k_iters',
    'lr3',
    'B4',
    # 'B1',
    'pretrained_feat3D', 
    'pretrained_occ',
    'frozen_feat3D', 
    'frozen_occ', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'log500',
]
exps['tester'] = [
    'carla_sub', # mode
    'carla_tatv_testset_data', # dataset
    'carla_16-16-16_bounds_train',
    'carla_16-16-16_bounds_test',
    '10_iters',
    'lr4',
    'B1',
    'use_cache',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'train_flow', 
    'no_backprop',
    'do_test', 
    'log1',
]
# exps['tester'] = [
#     'carla_sub', # mode
#     # 'carla_tast_trainset_data', # dataset
#     'carla_tatv_testset_data', # dataset
#     'carla_16-8-16_bounds_train',
#     'carla_16-8-16_bounds_test',
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
    'carla_sub', # mode
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
    'carla_sub', # mode
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
    'carla_sub', # mode
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
    'carla_sub', # mode
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
    'carla_sub', # mode
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
groups['use_cache'] = ['do_use_cache = True']
groups['carla_sub'] = ['do_carla_sub = True']

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
S = 2
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
groups['carla_multiview_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "mags7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
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
groups['carla_tasa_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tass100i2a"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tat1_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2one"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatt_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2t"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
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
groups['carla_16-8-16_bounds_val'] = [
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

