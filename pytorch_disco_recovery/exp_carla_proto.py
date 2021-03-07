from exp_base import *


############## choose an experiment ##############

# current = 'builder'
# current = 'trainer'
# current = 'cs_trainer'
current = 'softarg_trainer'

# i want a model that reconstructs the scene by using a dictionary of high-resolution prototypes

# i need to find an e


mod = '"steal00"' # one < looks great
mod = '"steal01"' # show occ_mem00s, occ_mem10s
mod = '"steal02"' # show occ_memRs0 and occ_memR0s
mod = '"steal03"' # cac data 
mod = '"steal04"' # show depth camR
mod = '"steal05"' # diff ex
mod = '"steal06"' # 16-8-16 bounds
mod = '"steal07"' # 8-4-8 bounds
mod = '"steal08"' # run_train
mod = '"steal09"' # run_train
mod = '"steal10"' # render
mod = '"steal11"' # render
mod = '"steal12"' # render a video
mod = '"steal13"' # richer inputs
mod = '"steal14"' # zoompix_T_cam
mod = '"steal15"' # valid box
mod = '"steal16"' # zoom 0.25-0.75

mod = '"steal17"' # get rgb_g cropped too
mod = '"steal18"' # actual render a movie in Rs, not r0s
mod = '"steal19"' # compute loss
mod = '"steal20"' # avoid confusing rendernet summs due to loop
mod = '"steal21"' # run centernet
mod = '"steal22"' # rgb_g*valid
mod = '"steal23"' # centernet self.K=1; show boxes detected on each frame
mod = '"steal24"' # show bev vis
mod = '"steal25"' # thresh 0.0
mod = '"steal26"' # show some centernet vis
mod = '"steal27"' # S = 1, to see if center vis matches outside vis
mod = '"steal28"' # use halfmem cropped occ
mod = '"steal29"' # again full occ < ok it matches up
mod = '"steal30"' # declare and show obj feat
mod = '"steal31"' # show obj_featRs
mod = '"steal32"' # use Z2 for zoom < good
mod = '"steal33"' # get and show masks
mod = '"steal34"' # dilate them masks
mod = '"steal35"' # again
mod = '"steal36"' # dilate actually
mod = '"steal37"' # 1 rot bin < good
mod = '"steal38"' # get and summ bkg feat
mod = '"steal39"' # render the bkg feat on each frame
mod = '"steal40"' # S = 4
mod = '"steal41"' # S = 8
mod = '"steal42"' # place object at loc; pad and upsample bkg
mod = '"steal43"' # train a while
mod = '"steal44"' # S = 4
mod = '"steal45"' # S = 9
mod = '"steal46"' # higher res again, since the low-res clips out too much stuff
mod = '"steal47"' # freeze feat; init occ channel of obj with ones
mod = '"steal48"' # reduce prints; higher lr < hm, it seems that a strategy is to make the object tensor huge
mod = '"steal49"' # S = 6
mod = '"steal50"' # allow featnet to train actually
mod = '"steal51"' # detach feat in the main rendering loss; apply direct loss on full rgb
mod = '"steal52"' # snap500; just train feat3d a bit
mod = '"steal53"' # train occ
mod = '"steal54"' # add second viewpred loss < nah, the other views are not so good
mod = '"steal55"' # pret 1k 01_s1_m192x96x192_p128x384_1e-3_F3_d4_O_c1_s.1_R_d64_r10_cacs10i2one_steal53
mod = '"steal56"' # freeze feat3d now, re-enable entenret
mod = '"steal57"' # construct occ loss after re-assembling
mod = '"steal58"' # again
mod = '"steal59"' # clamp size to 0.01
mod = '"steal60"' # show halfmem occ
mod = '"steal61"' # 0.01 regularization loss on size>1.0
mod = '"steal62"' # fix bug caused by new occ sup
mod = '"steal63"' # eliminate summ of non-supervised occ
mod = '"steal64"' # S = 6
mod = '"steal65"' # add size prior on both sides: 0.5, 1.0
mod = '"steal66"' # use mean l across the seq
mod = '"steal67"' # use high-res occ loss, but show the small one
mod = '"steal68"' # harder penalty, but 0.5 and 4.0
mod = '"steal69"' # clamp to 0.1m , and use obj_mask_memRs[:,s] within occnet loss, so that the obj does not shrink to avoid loss
mod = '"steal70"' # softplus + 0.25
mod = '"steal71"' # softplus + 0.5; only grab sizes>4.0
mod = '"steal72"' # coeff 1.0 on the size
mod = '"steal73"' # on every iter, set obj.data[:,0] = 10.0
mod = '"steal74"' # print stats, to see why size loss disappeared < nan!
mod = '"steal75"' # fix nan
mod = '"steal76"' # print sizes; softplus+1.0
mod = '"steal77"' # allow any occ
mod = '"steal78"' # higher lr
mod = '"steal79"' # size loss 2.0


mod = '"cs00"' # start on cs trainer collect occ_memRs
mod = '"cs01"' # get a box and show masks
mod = '"cs02"' # get cs loss and apply it
mod = '"cs03"' # K = 4
mod = '"cs04"' # use 0.9 for center
mod = '"cs05"' # use K=8
mod = '"cs06"' # mediate cs_loss by scorelist_s
mod = '"cs07"' # do not add occ loss to total
mod = '"cs08"' # normalize scorelist_s, so that it cannot cheat by just reducing all scores


mod = '"sa00"' # train softargnet
mod = '"sa01"' # get low/mid/high feats
mod = '"sa02"' # run softargnet and show corr 
mod = '"sa03"' # summ
mod = '"sa04"' # make boxes please
mod = '"sa05"' # show perspective 
mod = '"sa06"' # size 3.0; compute cs
mod = '"sa07"' # backprop
mod = '"sa08"' # freeze feat3d again
mod = '"sa09"' # smooth coeff 0.0
mod = '"sa10"' # unfreeze
mod = '"sa11"' # show boxes on cropped feat
mod = '"sa12"' # show boxes on cropped feat
mod = '"sa13"' # please actually collect and apply loss
mod = '"sa14"' # construct obj_featR tesnors
mod = '"sa15"' # render
mod = '"sa16"' # log11
mod = '"sa17"' # collect occ loss across full seq
mod = '"sa18"' # optimize the var too please
mod = '"sa19"' # again
mod = '"sa20"' # lz = 5.0
mod = '"sa21"' # lx=2.5,ly=1.5,lz=5.0
mod = '"sa22"' # show median and diffs
mod = '"sa23"' # occ median
mod = '"sa24"' # fix bug in occ: do it in R0
mod = '"sa25"' # fix that bug in the obj-occ too
mod = '"sa26"' # S = 8
mod = '"sa27"' # fixed scope bug
mod = '"sa28"' # show diffs gif
mod = '"sa29"' # S = 10
mod = '"sa30"' # frozen feat3d, to save memory

############## exps ##############

exps['builder'] = [
    'carla_proto', # mode
    # 'carla_multiview_train10_data', # dataset
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    # 'carla_32-16-32_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    # 'no_shuf',
    'no_backprop',
    # 'train_bkg',
    # 'train_occ',
    'pretrained_feat3d',
    'train_feat3d',
    'train_render',
    'train_center',
    'log1',
]
exps['center_trainer'] = [
    'carla_proto', # mode
    'carla_multiview_train_data', # dataset
    'carla_multiview_val_data', # dataset
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    # 'carla_multiview_val10_data', # dataset
    # 'carla_32-32-32_bounds_train',
    # 'carla_32-32-32_bounds_val',
    'carla_32-16-32_bounds_train',
    'carla_32-16-32_bounds_val',
    '100k_iters',
    # 'train_bkg',
    # 'train_occ',
    # 'no_shuf',
    'pretrained_feat3d',
    # 'frozen_feat3d', 
    'train_feat3d',
    # 'train_render',
    'train_center',
    'lr3',
    'B4',
    'log53',
    # 'log500',
]
exps['trainer'] = [
    'carla_proto', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    '100k_iters',
    'pretrained_feat3d',
    'frozen_feat3d',
    'train_feat3d',
    'train_occ', 
    'train_render',
    'train_center',
    'lr2',
    'B1',
    'log53',
    'snap500',
    # 'log500',
]
exps['cs_trainer'] = [
    'carla_proto', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    '100k_iters',
    'pretrained_feat3d',
    'frozen_feat3d',
    'train_feat3d',
    'train_occ',
    'train_center',
    # 'no_backprop', 
    'lr3',
    'B1',
    'log53',
    # 'log500',
]
exps['softarg_trainer'] = [
    'carla_proto', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    '100k_iters',
    'pretrained_feat3d',
    'frozen_feat3d',
    'train_feat3d',
    'train_occ',
    'train_softarg',
    'train_render',
    # 'no_backprop', 
    'lr3',
    'B1',
    'log11',
    # 'log53',
    # 'log500',
]

############## groups ##############

groups['carla_proto'] = ['do_carla_proto = True']
groups['do_test'] = ['do_test = True']
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 4',
]
groups['train_bkg'] = [
    'do_bkg = True',
    'bkg_coeff = 1.0',
    'bkg_epsilon = 0.75',
]
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
    'occ_smooth_coeff = 0.1',
]
groups['train_center'] = [
    'do_center = True',
    # 'center_prob_coeff = 1.0',
    # # 'center_size_coeff = 10.0',
    # 'center_size_coeff = 1.0',
    # 'center_rot_coeff = 1.0',
    # # 'center_offset_coeff = 10.0',
    # 'center_offset_coeff = 1.0',
    # 'center_peak_coeff = 0.1',
    'center_smooth_coeff = 0.001', 
]
groups['train_softarg'] = [
    'do_softarg = True',
    'softarg_coeff = 1.0', 
]
groups['train_proto'] = [
    'do_proto = True',
    'proto_stages = 1',
    'proto_coeff = 2.0',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    # 'rgb_smooth_coeff = 0.1',
]
groups['train_sigen3d'] = [
    'do_sigen3d = True',
    'sigen3d_coeff = 1.0',
    'sigen3d_reg_coeff = 0.1',
]


############## datasets ##############

# # dims for mem
# SIZE = 32
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 10
# S_val = 5
S_val = 2
S_test = 5
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

# SIZE = 32
# SIZE_val = 32
# SIZE_test = 32
# SIZE_zoom = 32

SIZE = 24
SIZE_val = 24
SIZE_test = 24
SIZE_zoom = 24

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

# SIZE = 16
# SIZE_val = 16
# SIZE_test = 16
# SIZE_zoom = 16

# SIZE = 12
# SIZE_val = 12
# SIZE_test = 12
# SIZE_zoom = 12

# SIZE = 10
# SIZE_val = 10
# SIZE_test = 10
# SIZE_zoom = 10

# SIZE = 8
# SIZE_val = 8
# SIZE_test = 8
# SIZE_zoom = 8

# SIZE = 4
# SIZE_val = 4
# SIZE_test = 4
# SIZE_zoom = 4

# dataset_location = "/data/carla/processed/npzs"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla_odometry/processed"

groups['carla_8-4-8_bounds_train'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
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
groups['carla_32-16-32_bounds_train'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_32-16-32_bounds_val'] = [
    'XMIN_val = -32.0', # right (neg is left)
    'XMAX_val = 32.0', # right
    'YMIN_val = -16.0', # down (neg is up)
    'YMAX_val = 16.0', # down
    'ZMIN_val = -32.0', # forward
    'ZMAX_val = 32.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_32-16-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_32-32-32_bounds_train'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -32.0', # down (neg is up)
    'YMAX = 32.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_32-32-32_bounds_val'] = [
    'XMIN_val = -32.0', # right (neg is left)
    'XMAX_val = 32.0', # right
    'YMIN_val = -32.0', # down (neg is up)
    'YMAX_val = 32.0', # down
    'ZMIN_val = -32.0', # forward
    'ZMAX_val = 32.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*8)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_32-32-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
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
groups['carla_multiview_train1_test1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "mags7i3one"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3ten"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_complete_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cacs10i2ten"',
    'trainset_format = "complete"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_complete_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cacs10i2one"',
    'trainset_format = "complete"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_val10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "mags7i3ten"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
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
