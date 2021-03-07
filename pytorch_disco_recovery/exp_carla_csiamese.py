from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'
current = 'tester'

# the intent here is to build the confidence net

mod = '"conf00"' # builder
mod = '"conf01"' # again, but proper logdir < why are results bad?
mod = '"conf02"' # motionreg too
mod = '"conf03"' # pret actually
mod = '"conf04"' # empty conf net; show estimated and gt box in training
mod = '"conf05"' # show pers box too
mod = '"conf06"' # measure iou in x1 instead < ok same
mod = '"conf07"' # bev_v
mod = '"conf08"' # bev_v proper
mod = '"conf09"' # one more bugfix 
mod = '"conf10"' # print obj_len, to debug two weird frames < looks good
# there's a serious question here, of what should the arch be for this model
# how about, take the corrs as input, compress, and estim
# note you can maybe exaggerate the errors here, by placing the object badly
mod = '"conf11"' # run confnet
mod = '"conf12"' # mse
mod = '"conf13"' # mse and norm
mod = '"conf14"' # train a while
mod = '"conf15"' # show iou_e; clamp before end; log50
mod = '"conf16"' # show iou_e; clamp before end; log50
mod = '"conf17"' # wider motions; use val set too
mod = '"conf18"' # slightly cleaner vis
mod = '"conf19"' # measure corr
mod = '"conf20"' # log500
mod = '"conf21"' # measure the actual corr
mod = '"conf22"' # on val iters run the same thing
mod = '"conf23"' # compress with one conv < great. immediately better perf
mod = '"conf24"' # use 0.001 and 0.999
mod = '"conf25"' # dt_1*2
mod = '"conf26"' # eliminate the np.isclose zero thing; more data; snap5k
mod = '"conf27"' # restore the flatten/fc < ok, noisier and worse than the conv version
mod = '"conf28"' # bottle again; dt_1*1.0
mod = '"conf29"' # bottle again; dr_1*2.0
mod = '"conf30"' # pay higher penalty for higher true ious < abandoned because no, this is not the true problem. the true problem is uncertainty.

# mod = '"test00"' # tester; pret 
# mod = '"test01"' # vis estimated iou instead of true 
# mod = '"test02"' # pret 5k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf26
# mod = '"test03"' # S_test = 90; 50 iters instead, because i don't have all day 
# mod = '"test04"' # pret 10k

mod = '"conf31"' # 4 handmade replicas. how will i know that this is good? at test time i suppose i should see fewer grave errors on the positive side, if i play it conservative (with a min)
mod = '"conf32"' # bottle_chans=16 instead of 32
mod = '"conf33"' # hidden_dim=128 instead of 256
mod = '"conf34"' # 8 replicas instead of 4
# mod = '"conf35"' # measure the corr coeff of the min estimate


mod = '"test05"' # use minconf; pret 5k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf34
mod = '"test06"' # record and save both mean and min conf
mod = '"test07"' # pret 10k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf34
mod = '"test08"' # 100 iters and S_test=30, to see that odd case again, if it reappears
mod = '"test09"' # actually do save all_min_confs in the npy
mod = '"test10"' # faster, by going on empty gpu
mod = '"test11"' # pret 15k


# i should input some visibility signal into the thing 
mod = '"conf35"' # take as input mean of occ_1, concat onto the chan

# i should also include some invis data in training... this means don't use vact; use some other set.
mod = '"conf36"' # talt, talv data

mod = '"test12"' # pret 5k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_vacs8i1t_vacs8i1v_conf35
# still quite bad on that hard iter

mod = '"test13"' # pret 5k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf36
# very good. i should train that guy for longer


mod = '"conf37"' # redo conf36, but on fresh day and go for 3k 


mod = '"test14"' # pret 30k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf37


mod = '"test15"' # same, but tan data
mod = '"test16"' # tap data
mod = '"test17"' # generate vis with true ious
mod = '"test18"' # RUIN the gt at the very outset, just to ensure i do not cheat. < ok, phew, the estimates still look great.
mod = '"test19"' # rx = -rx
mod = '"test20"' # print angles < ah, ok, there is no pitch or roll in this data
mod = '"test21"' # tap s100
mod = '"test22"' # avoid vis
mod = '"test23"' # comment out summs
mod = '"test24"' # trim = 6 < no, then conf does not fire.
mod = '"test25"' # run some timings < ok, interesting, matchnet is dominating the time. it takes 11.3 out of the 13 seconds
mod = '"test26"' # eliminate confnet, so that i can mess with trim
mod = '"test27"' # trim = 6 instead of 5
mod = '"test28"' # trim = 5; properly measure feat time
mod = '"test29"' # print out sizes of things
mod = '"test30"' # in dev.py, it seems like 100 corrs with these shapes should take 0.77 seconds. for me it takes 11. let's add timing to matchnet < corr is 0.1s per iter. *100 is 10, so indeed corr is hard here. but which part?
mod = '"test31"' # use loop over batch in the corr, to avoid the grouped convs < ok, yes.
mod = '"test32"' # fewer prints < 495s in all
# data0_time 0.026440694332122802
# data1_time 0.711906726360321
# feat0_time 0.015351045131683349
# feat1_time 0.13976666450500488
# match_time 3.2898762345314028
# 01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test32; [ 100/ 100]; ttime: 495 (0.00, 4.85); loss: 0.000 (test)
mod = '"test33"' # allocate and assign, rather than append and stack; drop the permute < tiny effect, but i agree it's cleaner
mod = '"test34"' # re-enable conf
mod = '"test35"' # re-enable summs, but note log5
mod = '"test36"' # log50

# # the timing right now seems to be: 
# data0_time 0.026062289873758953
# data1_time 0.69102942943573
# feat0_time 0.015562600559658475
# feat1_time 0.13784689373440212
# match_time 3.286398278342353
# conf_time 0.21101278728908962

# simplified:
# data0_time 0.03
# data1_time 0.70
# feat0_time 0.02
# feat1_time 0.14
# match_time 3.29
# conf_time 0.21

mod = '"test37"' # drop the pret; feat3D_dim=32 < super slow. there seems to be something messy in pytorch. at C>32, it switches algo.
mod = '"test38"' # drop the pret; feat3D_dim=48
mod = '"test39"' # drop the pret; feat3D_dim=33
mod = '"test40"' # print the steps in matchnet
mod = '"test41"' # do not print inside matchnet; eliminate fake inputs from matchnet
mod = '"test42"' # print cuda synced times instead
mod = '"test43"' # return f_time
mod = '"test44"' # measure and return the early part's time
mod = '"test45"' # eliminate the re-stacking 
mod = '"test46"' # eliminate stacking
mod = '"test47"' # bottle_chans=8 instead of 16 < no effect
mod = '"test48"' # 16again; 4 replicas instead of 8 < yes, halftime
mod = '"test49"' # self.trim = 6 < takes slightly longer than 5
mod = '"test50"' # self.trim = 4 < takes forever
mod = '"test51"' # self.trim = 5
# data0_time 0.02695915222167969
# data1_time 0.6618345427513123
# feat0_time 0.015304701328277588
# feat1_time 0.13340885400772096
# match_time 2.384454554252624 < this has improved since test36, but maybe only because i changed the measurement to cuda.record
# altmatch_time 0.0589072281897068
# conf_time 0.12698745489120483
# 01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test51; [ 100/ 100]; ttime: 461 (0.00, 4.56); loss: 0.000 (test)
mod = '"test52"' # compute grid only on first step < very tiny effect
mod = '"test53"' # compute old style time again
mod = '"test54"' # eliminate syncsjN
mod = '"test55"' # pret; back to 8 replicas
# data0_time 0.024748275279998778
# data1_time 0.6621940302848816
# feat0_time 0.015299797058105469
# feat1_time 0.12986051321029662
# match_time 2.9053706312179566 < ok good job, this is lower indeed. so it's either the stacking or the grid. both are good i think.
# altmatch_time 0.0
# conf_time 0.1976485800743103
# 01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test55; [ 100/ 100]; ttime: 464 (0.00, 4.62); loss: 0.000 (test)

mod = '"test56"' # use get_occupancy_single in a loop, since it uses the indexing trick i was thinking of < worse
mod = '"test57"' # voxelize_single on the cpu < worse yet
mod = '"test58"' # back to normal; also, try on matrix
mod = '"test59"' # provide ckpt 
mod = '"test60"' # nothing; aws
mod = '"test61"' # export vis
mod = '"test62"' # no vis; pret conf37
mod = '"test63"' # yes vis
# mean_ious [0.9  0.83 0.76 0.72 0.71 0.7  0.69 0.71 0.68 0.68 0.64 0.65 0.63 0.64
#             0.61 0.61 0.59 0.59 0.56 0.58 0.56 0.56 0.54 0.56 0.52 0.54 0.53 0.53
#             0.52 0.52 0.51 0.5  0.49 0.5  0.48 0.5  0.47 0.48 0.46 0.47 0.46 0.47
#             0.46 0.46 0.46 0.46 0.45 0.46 0.45 0.46 0.43 0.45 0.42 0.44 0.43 0.45
#             0.44 0.44 0.44 0.43 0.43 0.44 0.43 0.44 0.42 0.43 0.41 0.42 0.41 0.42
#             0.41 0.41 0.41 0.4  0.4  0.41 0.4  0.4  0.39 0.4  0.38 0.39 0.38 0.39
#             0.37 0.37 0.37 0.36 0.36 0.37 0.36 0.36 0.35 0.36 0.34 0.35 0.34 0.35
#             0.34 0.34]
mod = '"test64"' # no vis; grouped corr
# mean_ious [0.91 0.83 0.76 0.84 0.84 0.82 0.82 0.8  0.8  0.78 0.8  0.78 0.79 0.76
#             0.77 0.76 0.76 0.75 0.75 0.75 0.75 0.74 0.75 0.73 0.75 0.73 0.75 0.73
#             0.73 0.72 0.72 0.72 0.72 0.71 0.7  0.7  0.7  0.68 0.69 0.67 0.68 0.66
#             0.67 0.65 0.66 0.64 0.64 0.63 0.63 0.62 0.63 0.62 0.62 0.61 0.62 0.6
#             0.61 0.6  0.6  0.6  0.6  0.6  0.59 0.59 0.6  0.59 0.6  0.58 0.59 0.58
#             0.58 0.57 0.57 0.56 0.57 0.56 0.56 0.55 0.56 0.54 0.55 0.53 0.54 0.53
#             0.52 0.53 0.53 0.53 0.52 0.52 0.52 0.51 0.52 0.51 0.52 0.51 0.51 0.51
#             0.52 0.52]
mod = '"test65"' # yes vis
mod = '"test66"' # no vis; 
mod = '"test67"' # model and matchnet from 1b59fccfed8284d78144281a01cb79522a73a7f0
# mean_ious [0.96 0.91 0.9  0.89 0.88 0.87 0.86 0.85 0.84 0.84 0.83 0.82 0.83 0.82
#             0.82 0.81 0.81 0.81 0.81 0.81 0.81 0.81 0.8  0.8  0.79 0.79 0.79 0.78
#             0.78 0.78 0.78 0.77 0.77 0.76 0.77 0.76 0.76 0.75 0.74 0.74 0.73 0.73
#             0.73 0.72 0.72 0.72 0.71 0.71 0.71 0.7  0.7  0.7  0.69 0.69 0.69 0.69
#             0.69 0.69 0.68 0.68 0.68 0.68 0.67 0.67 0.67 0.67 0.67 0.66 0.66 0.66
#             0.65 0.65 0.66 0.65 0.64 0.64 0.63 0.63 0.63 0.62 0.62 0.62 0.61 0.61
#             0.61 0.61 0.6  0.6  0.6  0.6  0.6  0.6  0.6  0.6  0.6  0.6  0.6  0.6
#             0.6  0.59]
# ok, good job recovering, i guess
# this is still slightly lower than test34, but it's probably different examples?
mod = '"test68"' # eliminate summs
mod = '"test69"' # same but non-grouped corr
# mean_ious [0.86 0.89 0.8  0.85 0.79 0.82 0.76 0.81 0.75 0.79 0.74 0.78 0.73 0.77
#             0.72 0.76 0.71 0.75 0.69 0.72 0.68 0.72 0.67 0.71 0.67 0.7  0.66 0.7
#             0.65 0.69 0.65 0.68 0.64 0.67 0.63 0.66 0.62 0.65 0.61 0.63 0.6  0.63
#             0.59 0.62 0.58 0.61 0.58 0.61 0.57 0.6  0.56 0.59 0.55 0.57 0.54 0.57
#             0.53 0.56 0.53 0.55 0.53 0.55 0.52 0.55 0.52 0.55 0.52 0.54 0.52 0.54
#             0.51 0.53 0.51 0.53 0.5  0.52 0.49 0.52 0.49 0.51 0.48 0.5  0.48 0.49
#             0.46 0.49 0.46 0.48 0.45 0.47 0.45 0.47 0.44 0.46 0.44 0.46 0.44 0.46
#             0.43 0.45]
# how could i have been so wrong? i thought i checked this. what even is happening with the grouped conv?
mod = '"test70"' # grouped corr; allow vis

# anyway, test64 had grouped corr also, but worse results. so there must still be an unidentified bug 

mod = '"test71"' # restored new matchnet, minus grid part
mod = '"test72"' # no groups < clearly worse
mod = '"test73"' # print shapes in both 
mod = '"test74"' # loop outside
mod = '"test75"' # groups or not, but deglist = [0] < ok, with or without groups, when there is one deg, things work out.
mod = '"test76"' # no group; solid search region < false
mod = '"test77"' # grouped version, but solid, and iterate over N < false
mod = '"test78"' # pack again
mod = '"test79"' # step through the pack
mod = '"test80"' # redo
mod = '"test81"' # alt version < uh looks fine.
mod = '"test82"' # reg version
mod = '"test83"' # alt version again, because i don't believe it
mod = '"test84"' # alt version one more time
mod = '"test85"' # loop over batch within the tracker
mod = '"test86"' # eliminate the loop within matchnet  < still fine
mod = '"test87"' # git checkout those files



############## define experiments ##############

exps['builder'] = [
    'carla_csiamese', # mode
    # 'carla_vis_data', # dataset
    # 'carla_vact_taft_data', # dataset
    'carla_vact_data',
    # 'carla_vact_tagtb_tagvce_data',
    # 'carla_vact_data', # dataset
    # 'carla_traintraintest_data', # dataset
    # 'carla_traintraintest_data', # dataset
    # 'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '10_iters',
    'pretrained_feat3D',
    'pretrained_match',
    'train_feat3D', 
    'train_match', 
    'train_conf', 
    # 'train_motionreg', 
    'B1',
    'no_shuf',
    # 'no_backprop',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_csiamese', # mode
    # 'carla_vact_data', # dataset
    # 'carla_vact_vacv_data', # dataset
    'carla_talt_talv_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '30k_iters',
    'pretrained_feat3D',
    'pretrained_match',
    'frozen_feat3D', 
    'frozen_match', 
    'train_feat3D', 
    'train_match', 
    'train_conf',
    'snap5k',
    # 'no_shuf',
    'B4',
    # 'vB4', # valset batchsize 
    'lr4', 
    'log500',
]
exps['tester'] = [
    'carla_csiamese', # mode
    # 'carla_test_on_curved_data', # dataset
    # 'carla_tagvce_data', # dataset
    # 'carla_tan_t_data', # dataset
    'carla_tap_t_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '100_iters',
    # '50_iters',
    # '20_iters',
    # '5_iters',
    'no_shuf',
    'do_test', 
    # 'do_export_vis',
    'do_export_stats',
    'B1',
    'pretrained_feat3D',
    'pretrained_match',
    'pretrained_conf',
    'train_feat3D', 
    'train_match', 
    'train_conf', 
    'no_backprop',
    # 'log5',
    'log500',
    # 'log1',
]

############## net configs ##############

groups['carla_csiamese'] = ['do_carla_csiamese = True']
groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']
groups['do_export_stats'] = ['do_export_stats = True']

groups['include_summs'] = [
    'do_include_summs = True',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    # 'feat3D_dim = 32',
    # 'feat3D_dim = 34',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.01',
    # 'feat_do_sparse_invar = True', 
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
    'match_r_coeff = 0.1', 
]
groups['train_conf'] = [
    'do_conf = True',
    'conf_coeff = 1.0', 
    # 'conf_num_replicas = 4', 
]
groups['train_motionreg'] = [
    'do_motionreg = True',
    'motionreg_dropout = True',
    'motionreg_num_slots = 5',
    'motionreg_t_past = 4', 
    'motionreg_t_futu = 56',
    # 'motionreg_l1_coeff = 1.0',
    'motionreg_l2_coeff = 1.0',
    'motionreg_weak_coeff = 0.01', # 0.001 or 0.0001 seem OK
    # 'motionreg_smooth_coeff = 0.01', # in old exps this seemed to hurt perf, but right now my trajs look very jagged
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
S_val = 60
# S_test = 60
S_test = 100

H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

dataset_location = "/data4/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
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
    'trainset = "vacs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "vacs8i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vact_taft_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
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
groups['carla_vact_taft_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
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
groups['carla_vact_tagt_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
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
groups['carla_vact_tagtb_tagvce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
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
groups['carla_tan_t_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tans60i2t"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_t_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taps100i2t"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_v_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taps100i2v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tal_v_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tals90i1v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_talt_talv_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tals90i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S, 
    'valset = "tals90i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
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
groups['carla_vact_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
]
groups['carla_vact_vacv_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "vacs8i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = False', 
    'trainset_seqlen = %d' % S,
    'valset = "vacs8i1v"',
    'valset_format = "traj"', 
    'valset_consec = False', 
    'valset_seqlen = %d' % S,
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
