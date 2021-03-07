from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'match_trainer'
# current = 'trainer'
# current = 'tester'

# the intent here is to build the forecaster sharing a backbone featnet with the matchnet

mod = '"share00"' # nothing
mod = '"share01"' # on train iters, load vab; on val iters, load taf
mod = '"share02"' # on val, put forecaster 
mod = '"share03"' # on train, show rgb_camXs, on val show rgb_camX0
mod = '"share04"' # just train iters;  
mod = '"share05"' # train a bit
mod = '"share06"' # cleaned up; more data
mod = '"share07"' # builder trainval
mod = '"share08"' # train trainval
mod = '"share09"' # train trainval
mod = '"share10"' # pret mul21; track over seq on test
mod = '"share11"' # show occ_1s
mod = '"share12"' # obj_dr = 0
mod = '"share13"' # place it at -obj_dt
mod = '"share14"' # place it at obj_dt
mod = '"share15"' # show the obj_loc_e on every step
mod = '"share16"' # show the obj_loc_e on every step as a gif
mod = '"share17"' # show gt properly
mod = '"share18"' # show gt properly
mod = '"share19"' # show obj_dt
mod = '"share20"' # unpack and update obj_lrt
mod = '"share21"' # unpack and UPDATE obj_lrt < beautiful
mod = '"share22"' # update R part too: obj_r = obj_r * eul2rotm(rad_e) < perfect
mod = '"share23"' # compute ious
mod = '"share24"' # show vis
mod = '"share25"' # test 100 iters, show the results
mod = '"share26"' # compute stats
# mean_ious [0.97 0.94 0.93 0.92 0.92 0.91 0.91 0.9  0.9  0.9  0.89 0.89 0.88 0.88
#             0.87 0.87 0.87 0.86 0.86 0.85 0.84 0.84 0.84 0.83 0.82 0.82 0.82 0.81
#             0.8  0.8 ]
# holy smokes
# holy smokes.
mod = '"share27"' # try faster, harder data
mod = '"share28"' # use -12,0,12 for this faster data
mod = '"share29"' # back to slow data; trainvaltest
mod = '"share30"' # use -6,0,6
mod = '"share31"' # use complete vis_ab data
mod = '"share32"' # S_test = 30
mod = '"share33"' # valbatch 8 instead of 4
mod = '"share34"' # only val (no train)
mod = '"share35"' # show o_clean
mod = '"share36"' # require all channels clean
mod = '"share37"' # no bkg for >1; use o_new anywhere
mod = '"share38"' # byte < looks good
mod = '"share39"' # try the real val set < looks fine
mod = '"share40"' # tester, with cleanup and new ckpt: 20k 04_s2_m256x128x256_z64x64x64_1e-4_F3_d64_M_c1_r.1_Mrd_p4_f56_k5_e.1_w.001_vabs8i1t_tafs60i1tce_tags90i1vce_share33
# mean_ious [0.96 0.94 0.93 0.93 0.92 0.92 0.92 0.92 0.91 0.91 0.91 0.9  0.9  0.89
#             0.89 0.89 0.89 0.88 0.88 0.88 0.87 0.87 0.86 0.86 0.85 0.84 0.84 0.84
#             0.83 0.83]
# holy smokes, this is even higher. so training with forecasting improved the matcher
mod = '"share41"' # clean more; mul21 ckpt < ok, almost no diff from share40. so this means the net doesn't mind overloading with the motionreg task
# mean_ious [0.97 0.94 0.93 0.92 0.92 0.91 0.91 0.9  0.9  0.9  0.89 0.89 0.88 0.88
#             0.87 0.87 0.87 0.86 0.86 0.85 0.84 0.84 0.84 0.83 0.82 0.82 0.81 0.81
#             0.8  0.8 ]



mod = '"share42"' # deb
mod = '"share43"' # byte



mod = '"rotpred"' # trainer on val; 
mod = '"share44"' # pret 20k share33; on val, use t instead of tce
mod = '"share45"' # don't pret motionreg; k6
mod = '"share46"' # use tag data; use the latest data, which is xyz_camR0s[:,t_past] as the input 
mod = '"share47"' # feed clist_past and futu separately
mod = '"share48"' # compute end stats; log5
mod = '"share49"' # compute end stats; log50
mod = '"share50"' # 5 slots, pret motionreg
mod = '"share51"' # snap500
mod = '"share52"' # print the drops
mod = '"share53"' # dropout the other direction
mod = '"share54"' # show trajs in high res; log5
mod = '"share55"' # flipback after
mod = '"share56"' # log50
mod = '"share57"' # manually bal by txt concat and train on that
mod = '"share58"' # print flips
mod = '"share59"' # proper unflipping
mod = '"share60"' # b4 for both; higher lr
mod = '"share61"' # nothing; log5; trying to fix the vis
mod = '"share62"' # again; flip on every step
mod = '"share63"' # clone the thing you are slicing out
mod = '"share64"' # go through the 20
mod = '"share65"' # eliminate prints
mod = '"share66"' # fix bug in dist comptuation
mod = '"share67"' # train a while
mod = '"share68"' # log50
mod = '"share69"' # additive noise in range -.5,.5 to clist_past
mod = '"share70"' # do not pret motionreg; use hidden_dim=256 instead of 512
mod = '"share71"' # share70 but k=8
mod = '"share72"' # k5; pret and FROZEN feat3d; only val
mod = '"share73"' # none of that noise
mod = '"share74"' # bring back the noise
mod = '"share75"' # compute mindist on every logging iter, not just image ones
mod = '"share76"' # hidden_dim=128
mod = '"share77"' # hidden_dim=256
mod = '"share78"' # retrain that, with more time; snap1k
mod = '"share79"' # traj noise just within [-0.1,0.1]
mod = '"share80"' # self.smoothl1(vels_e[:,:,1:], vels_e[:,:,:-1])
mod = '"share81"' # use xyz_agg in training < bug here, where i allowed the whole future as input
mod = '"share82"' # wider placement error in the input: -0.5,0.5 instead of -0.1,0.1
mod = '"share83"' # nicer vis 
mod = '"share84"' # wider vis 
mod = '"share85"' # (fix of bug from share81:) only allow past to contribute to agg


mod = '"share86"' # redo share85, pret share85; 
mod = '"share87"' # tester of share85
mod = '"share88"' # fix uint8 issue
mod = '"share89"' # S_test = 90



mod = '"share90"' # S_test = 90


mod = '"same00"' # upgraded for speed; also, tap data and S_val = 100
mod = '"same01"' # wider v vis
mod = '"same02"' # output accels; log50
mod = '"same03"' # convert to pos in numpy
mod = '"same04"' # apply loss directly on accel
mod = '"same05"' # print accel_futu
mod = '"same06"' # l2 
mod = '"same07"' # coeff 100; print more
mod = '"same08"' # print less
mod = '"same09"' # weightlist = ones
mod = '"same10"' # wider context input: 64, 64, 64
mod = '"same11"' # pred and train velocity instead of accel
mod = '"same12"' # log500

mod = '"mat00"' # run this on matrix
mod = '"mat01"' # 0-38; scratch data (partial) < indeed, much faster
# mod = '"mat02"' # 0-18, /projects data
mod = '"mat03"' # 0-38; more data
mod = '"mat04"' # 0-38; full data
mod = '"mat05"' # 
mod = '"mat06"' # 8 workers
mod = '"mat07"' # 4 workers
mod = '"mat08"' # 12 workers
mod = '"mat09"' # compute-0-22
mod = '"mat10"' # compute-0-14
mod = '"mat11"' # 0-38; 8 workers again
mod = '"mat12"' # aws again
mod = '"mat13"' # classify
mod = '"mat14"' # bal data
mod = '"mat15"' # 50
mod = '"mat16"' # 30
mod = '"mat17"' # back to predicting position
mod = '"mat18"' # sz=32 in input
mod = '"mat19"' # sz=32 in vis too
mod = '"mat20"' # linspace weightlist
mod = '"mat21"' # vel loss
mod = '"mat22"' # diff coeffs
mod = '"mat23"' # new bal data; test set
mod = '"mat24"' # slightly larger vce data < actually this is false
mod = '"mat25"' # bottle3d
mod = '"mat26"' # slightly larger vce data
mod = '"mat27"' # bottle2d
mod = '"mat28"' # bottle3d; hidden_dim = 128 instead of 256
mod = '"mat29"' # 60k
mod = '"mat30"' # noise in -0.5,0.5 instead of -0.1,0.1
mod = '"mat31"' # B4
mod = '"mat32"' # like mat30 but longer seqlen, since i think this gives more curves



mod = '"ole00"' # builder; vact data
mod = '"ole01"' # show corrlist
mod = '"ole02"' # matc_trainer
mod = '"ole03"' # fewer prints; lr4
mod = '"ole05"' # log10, to see some errors sooner
mod = '"ole06"' # log500
mod = '"ole07"' # 100k
mod = '"ole08"' # use new box parser; log10 to check it < ok good
mod = '"ole09"' # log500; 100k 


############## define experiments ##############

exps['builder'] = [
    'carla_ssiamese', # mode
    # 'carla_vis_data', # dataset
    # 'carla_vabt_taft_data', # dataset
    'trainset_vact_data', # dataset
    # 'carla_traintraintest_data', # dataset
    # 'carla_traintraintest_data', # dataset
    'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '10_iters',
    'train_feat3D', 
    'train_match', 
    'train_motionreg', 
    'B1',
    'no_shuf',
    # 'no_backprop',
    'log1',
]
exps['match_trainer'] = [
    'carla_ssiamese', # mode
    'trainset_vact_data', # dataset
    'testset_taqv_data', # dataset
    # 'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '100k_iters',
    'train_feat3D', 
    'train_match', 
    'snap5k',
    'B4',
    'lr4', 
    'log500',
]
exps['trainer'] = [
    'carla_ssiamese', # mode
    # 'carla_tagtb_data', # dataset
    # 'carla_tap_v_data', # dataset
    'valset_taqtb_data', # dataset
    'testset_tapvce_data', # dataset
    'train_on_trainval',
    # 'nearcube_traintest_bounds_narrow_val', 
    'nearcube_trainvaltest_bounds', 
    '60k_iters',
    # '30k_iters',
    # '30_iters',
    'pretrained_feat3D',
    'pretrained_match',
    'frozen_feat3D', 
    'frozen_match', 
    # 'pretrained_motionreg',
    'train_feat3D', 
    'train_match', 
    'train_motionreg',
    'snap1k',
    # 'no_shuf',
    'B4',
    'vB4', # valset batchsize 
    'lr3', 
    'log500',
]
exps['tester'] = [
    'carla_ssiamese', # mode
    # 'carla_test_on_curved_data', # dataset
    'carla_tagvce_data', # dataset
    # 'carla_fakv_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '100_iters',
    # '20_iters',
    # '5_iters',
    'no_shuf',
    'do_test', 
    'do_export_vis',
    'do_export_stats',
    'B1',
    'pretrained_feat3D',
    'pretrained_match',
    'train_feat3D', 
    'train_match', 
    'no_backprop',
    # 'log5',
    'log1',
]

############## net configs ##############

groups['carla_ssiamese'] = ['do_carla_ssiamese = True']
groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']
groups['do_export_stats'] = ['do_export_stats = True']

groups['include_summs'] = [
    'do_include_summs = True',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    # 'feat3D_dim = 32',
    'feat3D_dim = 64',
    'feat3D_skip = True',
    # 'feat3D_smooth_coeff = 0.01',
    # 'feat_do_sparse_invar = True', 
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
    'match_r_coeff = 0.1', 
]
groups['train_motionreg'] = [
    'do_motionreg = True',
    'motionreg_dropout = True',
    'motionreg_num_slots = 5',
    'motionreg_t_past = 4', 
    'motionreg_t_futu = 36',
    'motionreg_l2_coeff = 1.0',
    'motionreg_weak_coeff = 0.001', # 0.001 or 0.0001 seem OK
    # 'motionreg_smooth_coeff = 0.01', # in old exps this seemed to hurt perf, but right now my trajs look very jagged
    # 'motionreg_vel_coeff = 0.1',
]

############## datasets ##############

# DHW for mem stuff
SIZE = 64
SIZE_val = 64
SIZE_test = 64
# SIZE = 48
# SIZE_test = 48
# SIZE = 32
# SIZE_test = 32

# Z = SIZE*4
# Y = SIZE*1
# X = SIZE*4

# zoom stuff

# ZZ = int(SIZE/2)
# ZY = int(SIZE/2)
# ZX = int(SIZE/2)
# ZZ = 24
# ZY = 24
# ZX = 24
# ZZ = 32
# ZY = 32
# ZX = 32
# ZZ = 40
# ZY = 40
# ZX = 40
# ZZ = 48
# ZY = 48
# ZX = 48
ZZ = 64
ZY = 64
ZX = 64

K = 8 # how many objects to consider
N = 8

# S = 10
S = 2
S_val = 40
S_test = 40

H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/data4/carla/processed/npzs"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/scratch"


groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['nearcube_traintest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['nearcube_traintest_bounds_narrow_val'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_val = -8.0', # right (neg is left)
    'XMAX_val = 8.0', # right
    'YMIN_val = -4.0', # down (neg is up)
    'YMAX_val = 4.0', # down
    'ZMIN_val = -8.0', # forward
    'ZMAX_val = 8.0', # forward
    'Z_val = %d' % (int(SIZE_val*4)),
    'Y_val = %d' % (int(SIZE_val*2)),
    'X_val = %d' % (int(SIZE_val*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['nearcube_trainvaltest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*4)),
    'Y_val = %d' % (int(SIZE_val*2)),
    'X_val = %d' % (int(SIZE_val*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['cube_traintest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['narrow_traintest_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
]

groups['carla_flowtrack_mini_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "fags16i3ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3ten"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    # 'testset_seqlen = 2',
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train1_test1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1one"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train10_test10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1ten"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "fags16i3t"',
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_trainvaltest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintraintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintraintestce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_t_tce_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "faks30i1tce"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vis_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vabs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "vabs8i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vabt_taft_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vabs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "tafs60i1tce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vabt_taft_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vabs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "tafs60i1t"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vabt_tagt_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vabs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "tags90i1t"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vabt_tagtb_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vabs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "tags90i1tb"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_tagtb_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'valset = "tags90i1tb"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['valset_taqtb_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'valset = "taqs100i2tb"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['testset_taqv_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_taft_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'valset = "tafs60i1tce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_taft_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'valset = "tafs60i1tce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_tafv_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'valset = "tafs60i1vce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S_val,
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_fakv_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "faks16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['trainset_vact_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintestce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintrain_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1t"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_curved_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_curved_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1tce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_straight_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1t_straight"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_test_on_straight_test_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "faks30i1t"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'testset = "faks30i1v_straight"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['old_carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['mix_carla_traintest_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1v"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_traintest100_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1hun"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1hun"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1ten"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1ten"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traintest1_simple_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_train1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_trainvaltest1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1one"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3t"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'backprop_on_val = True',
    'testset = "faks30i1one"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
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
