from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'mini_trainer'
# current = 'trainer'
# current = 'tester'

mod = '"f00"' # start; builder; do nothing but prep
mod = '"f01"' # trainset; make noyaw data
mod = '"f02"' # print sizes
mod = '"f03"' # use hyp.XMIN < ok nice and wide, but i don't see traj
mod = '"f04"' # print some shapes
mod = '"f05"' # place object 
mod = '"f06"' # transform traj
mod = '"f07"' # same
mod = '"f08"' # bigger ZZ
mod = '"f09"' # sigma=1
mod = '"f10"' # print shapes
mod = '"f11"' # use consec
mod = '"f12"' # new data
mod = '"f13"' # new data
mod = '"f14"' # s90 data
mod = '"f15"' #
# seems like the object is not being placed straight up
mod = '"f16"' # tad t and tce
mod = '"f17"' # tad t and tce and vce
mod = '"f18"' # tae t and tce and vce
mod = '"f19"' # tae t and tce and vce
mod = '"f20"' # wider zoom
mod = '"f21"' # older data
mod = '"f22"' # yet older
mod = '"f23"' # use neg rot < good
mod = '"f24"' # new data
mod = '"f25"' # show trajs in R0
mod = '"f26"' # show trajs in R0 and X0
mod = '"f27"' # show trajs in R0 and X0
mod = '"f28"' # bugfix
mod = '"f29"' # generate features
mod = '"f30"' # motionreg
mod = '"f31"' # deeper bottle
mod = '"f32"' # linear layers
mod = '"f33"' # linear layers; many shapes
mod = '"f34"' # loss, vis
mod = '"f35"' # train a while on 10 samples
mod = '"f36"' # ZZ*2 instead of *3
mod = '"f37"' # lr4
mod = '"f38"' # 32, 32 inside the bottle
mod = '"f39"' # one ex
mod = '"f40"' # debugged vis
mod = '"f41"' # ten again
mod = '"f42"' # full trainer; t,tce,vce
mod = '"f43"' # show gt alongside; use 2d bottle
mod = '"f44"' # val batchsize 4
mod = '"f45"' # smooth l1
mod = '"f46"' # weak l1 on the rest, to ensure they are not totally useless < yes, this brings them close
mod = '"f47"' # l1 on accel < looks good
mod = '"f48"' # flip xdim half the time < bug
mod = '"f49"' # put accel on the FULL thing; flip the X part 
mod = '"f50"' # don't waste time voxelizing
mod = '"f51"' # don't waste time voxelizing
mod = '"f52"' # lower res
mod = '"f53"' # kernel_size=4 in the bottle2d; so one less layer, and hidden=512


mod = '"t00"' # test f53
mod = '"t01"' # test f53 taf data
mod = '"t02"' # fewer prints


# mod = '"f54"' # pret f53 and compute centroid dist in meters

mod = '"t03"' # compute centroid dist in meters

mod = '"f54"' # train a while
mod = '"f55"' # better scopes; weightlist = ones
mod = '"f56"' # better scopes; weightlist proper (accidentally deleted)
mod = '"f57"' # same; 
mod = '"f58"' # drop past ind
mod = '"f59"' # with weak, penalize the worst, so that it learns to be more adaptive
mod = '"f60"' # weightlist=ones

# killing all these to save money
mod = '"f61"' # proper weightlist
mod = '"f62"' # penalize mean instead of max
mod = '"f63"' # drop

mod = '"f64"' # collect more detailed stats, to see the error on every step
mod = '"f65"' # print the clists when you summ; summ eg; just summ details unscaled; log50
mod = '"f66"' # sub bias from input, add bias to output; scale pred *Z 
mod = '"f67"' # plot norm instead of sqnorm
mod = '"f68"' # log500
mod = '"f69"' # drop; concat mask
mod = '"f70"' # fancy conditional to fix gt vis; hidden_dim=256 instead of 512 < seems the vis was ineffecive here, but kicked in in f71
mod = '"f71"' # only drop on non-test iters



############## define experiments ##############

exps['builder'] = [
    'carla_forecast', # mode
    # 'carla_tae_t_tce_vce_data', # dataset
    # 'carla_tag_data', # dataset
    'carla_tap_data', # dataset
    'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '10_iters',
    'train_feat3D',
    'train_motionreg',
    'no_backprop',
    'B1',
    # 'lr4', 
    'log1',
]
exps['mini_trainer'] = [
    'carla_forecast', # mode
    'carla_tae10_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '1k_iters',
    'train_feat3D',
    'train_motionreg',
    'B1',
    'lr4', 
    'log50',
]
exps['trainer'] = [
    'carla_forecast', # mode
    'carla_taf_t_tce_vce_data', # dataset
    'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '100k_iters',
    'train_feat3D',
    'train_motionreg',
    'B2',
    'vB2', # valset batchsize  
    'lr5', 
    'log500',
]
exps['tester'] = [
    'carla_forecast', # mode
    # 'carla_taf_t_tce_vce_data', # dataset
    'carla_taf_data', # dataset
    'nearcube_trainvaltest_bounds',
    'no_shuf',
    '100_iters',
    'pretrained_feat3D',
    'pretrained_motionreg',
    'train_feat3D',
    'train_motionreg',
    'B1',
    'log1',
]
exps['tester2'] = [
    'carla_forecast', # mode
    'carla_taf_t_tce_vce_data', # dataset
    'nearcube_trainvaltest_bounds', 
    '100_iters',
    'no_shuf',
    'do_test', 
    'do_export_vis',
    'do_export_stats',
    'B1',
    'pretrained_feat3D',
    'pretrained_motionreg',
    'train_feat3D', 
    'train_motionreg', 
    'no_backprop',
    'log1',
]


############## group configs ##############

groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']
groups['do_export_stats'] = ['do_export_stats = True']

groups['train_motionreg'] = [
    'do_motionreg = True',
    'motionreg_dropout = True',
    # 'Motionreg_num_slots = 5',
    # 'motionreg_num_slots = 2',
    # 'motionreg_num_slots = 8',
    'motionreg_num_slots = 3',
    'motionreg_t_past = 4',
    'motionreg_t_futu = 56',
    # 'motionreg_l1_coeff = 1.0',
    'motionreg_l2_coeff = 0.1',
    'motionreg_weak_coeff = 0.001', # 0.001 or 0.0001 seem OK
    # 'motionreg_smooth_coeff = 0.01', # this hurts perf
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat_smooth_coeff = 0.01',
    # 'feat_do_sparse_conv = True', 
    # 'feat_do_sparse_invar = True', 
]
groups['train_pri'] = [
    'do_pri2D = True',
    'pri2D_ce_coeff = 1.0',
    # 'pri2D_smooth_coeff = 0.01',
    'pri2D_reg_coeff = 0.002',
]
groups['train_rpo'] = [
    'do_rpo2D = True',
    'rpo2D_forward_coeff = 2.0',
    'rpo2D_reverse_coeff = 1.0',
]
groups['train_forecast'] = [
    'do_forecast = True',
    # 'forecast_maxmargin_coeff = 1.0',
    # # 'forecast_smooth_coeff = 0.1', 
    # 'forecast_num_negs = 500',
    'forecast_l2_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_det'] = [
    'do_det = True',
    'det_prob_coeff = 1.0',
    'det_reg_coeff = 1.0',
]


############## datasets ##############

# mem resolution
SIZE = 64
SIZE_val = 64
SIZE_test = 64
# X = int(SIZE*32)
# Y = int(SIZE*4)
# Z = int(SIZE*32)
# Z = SIZE*4
# Y = SIZE*1
# X = SIZE*4

ZZ = 64
ZY = 64
ZX = 64

# ZX = int(SIZE*32)
# ZY = int(SIZE*4)
# ZZ = int(SIZE*32)

# these params need cleaning; also, 3 only works if you do not count occluders
N = 3 # max objects
K = 3 # objects to consider 

S = 100
S_test = 100
H = 128
W = 384
V = 50000
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

dataset_location = "/data4/carla/processed/npzs"

groups['carla_fak_t_tce_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S,
    'valset = "faks30i1tce"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S,
    'testset = "faks30i1vce"',
    'testset_format = "traj"',
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"',
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

groups['carla_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taf_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tafs60i1t"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tafs60i1v"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tag_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tags90i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tags90i1v"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "taps100i2t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "taps100i2v"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taf10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tafs60i1ten"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tafs60i1ten"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taf1_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tafs60i1one"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tafs60i1one"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tac_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tacs60i1t"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tacs60i1v"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tadt_tadtce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tads90i1t"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tads90i1tce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taf_t_tce_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tafs60i1t"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tafs60i1tce"',
    'valset_format = "simpletraj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'testset = "tafs60i1vce"',
    'testset_format = "simpletraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taf_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tafs60i1vce"',
    'testset_format = "simpletraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tactce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tacs60i1tce"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tact_tactce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tacs60i1t"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'trainset = "tacs60i1tce"',
    'trainset_format = "simpletraj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_forecast_train_data'] = [
    'dataset_name = "carla_forecast"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faks30i1t"',
    'dataset_location = "%s"' % dataset_location,
    'dataset_format = "npz"'
]
groups['carla_forecast10_data'] = [
    'dataset_name = "carla_forecast"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faes16i3ten"',
    'dataset_list_dir = "./npzs"',
    'dataset_location = "./npzs"',
    # 'dataset_list_dir = "/projects/katefgroup/datasets/carla/processed/npzs"',
    # 'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_format = "npz"'
]
groups['carla_forecast1_data'] = [
    'dataset_name = "carla_forecast"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "faes16i3one"',
    'dataset_list_dir = "./npzs"',
    'dataset_location = "./npzs"',
    # 'dataset_list_dir = "/projects/katefgroup/datasets/carla/processed/npzs"',
    # 'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
    'dataset_format = "npz"'
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
