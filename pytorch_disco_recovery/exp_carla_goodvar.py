from exp_base import *


############## choose an experiment ##############

current = 'builder'
current = 'feat3d_trainer'
current = 'traj_trainer'
current = 'scene_trainer'
current = 'traj_via_scene_trainer'

mod = '"bkg00"' # latent init 200 01_s20_m128x64x128_1e-3_F3f_d4_O_c1_t1_s.1_cacs20i2one_traj12
mod = '"bkg01"' # init self.bkg[:,0:1] with median occ, then optimize free 
mod = '"bkg02"' # 100 iters
mod = '"bkg03"' # optimize bkg var
mod = '"bkg04"' # use pos loss too
mod = '"bkg05"' # use 0.1 smooth loss
mod = '"bkg06"' # again
mod = '"bkg07"' # show latent occ
mod = '"bkg08"' # fixed bug in scopes; higher smooth
mod = '"bkg09"' #

# next:
# project and inflate the 3d traj box to get a 2d one
# turn that into a mask
# train bkg color, excluding the obj region
# train obj full, only within the obj region

mod = '"bkg10"' # repeat whatever this is
mod = '"bkg11"' # complete_ac_s30_i2


mod = '"bust00"' # feat3d 
mod = '"bust01"' # backprop
mod = '"bust02"' # do not pret
mod = '"bust03"' # lower lr
mod = '"bust04"' # 1k with S=2
mod = '"bust05"' # si arch
mod = '"bust06"' # downsamp in arch
mod = '"bust07"' # 6 blocks instead of 4
mod = '"bust08"' # similar but each block is just a conv
mod = '"bust09"' # one compressor then three proper res blocks
mod = '"bust10"' # go to quarter res with compressor
mod = '"bust11"' # pret 400 01_s2_m128x64x128_1e-5_F3_d4_O_c1_t1_s.1_cacs30i2one_bust09


mod = '"bust12"' # pret feat06
mod = '"bust13"' # show medians
mod = '"bust14"' # show medians on every step, to help it show up
mod = '"bust15"' # vis-aware median
mod = '"bust16"' # use gt occ
mod = '"bust17"' # use vis*diff
mod = '"bust18"' # init with true argmax (not soft)
mod = '"bust19"' # show traj optim
mod = '"bust20"' # higher smooth coeff
mod = '"bust21"' # elastic coeff
mod = '"bust22"' # lock in the middle
mod = '"bust23"' # clone
mod = '"bust24"' # set middle to mean of traj_init
mod = '"bust25"' # set middle to median
mod = '"bust26"' # smooth coeff 0.0

# ok, the traj is not great. but maybe this is a hard example, and anyway, maybe we will still learn an ok object
# no, that sounds super unlikely actually
# what i ought to do is: take one proposal and track it with the eccv method, just like i did for neurips

mod = '"bust27"' # diff example
mod = '"bust28"' # blur those diffs by 4
mod = '"bust29"' # fixed bug in elastic loss
# good now
# let's try again the harder data
mod = '"bust30"' # harder (s30)
# ok kind of reasonable but let's stay easy
mod = '"bust31"' # s20
mod = '"bust32"' # soft argmax
mod = '"bust33"' # hard=False
mod = '"bust34"' # hard=False < super similar


# ok, now i have a traj
# i would like to try to optimize for the scene now
# maybe i can/should optimize for obj and bkg simultaneously

# btw maybe this is another place where some convolutional priors might help... anyway let's see

mod = '"bust35"' # init latents with (traj from) 01_s20_m128x64x128_1e-2_F3f_d4_O_c1_t1_s.5_cacs20i2one_bust34; just show some latents and return
mod = '"bust36"' # show traj
mod = '"bust37"' # show traj on occ
mod = '"bust38"' # get median and apply loss on bkg var
mod = '"bust39"' # do backprop
mod = '"bust40"' # compose obj and bkg
mod = '"bust41"' # render
mod = '"bust42"' # train obj for occ too
mod = '"bust43"' # only load data on step==1, since it's fixed
mod = '"bust44"' # re-use occ/free data after first iter; also, nothing special in init of obj < but this is overwritten by the init i think
mod = '"bust45"' # again
mod = '"bust46"' # better scope for scene occ; smooth loss on full scene
mod = '"bust47"' # 1k; log53
mod = '"bust48"' # symmetry loss
mod = '"bust49"' # show scene occ gif
mod = '"bust50"' # be more conservative with "free" < indeed worth it
mod = '"bust51"' # fix bug in vis
mod = '"bust52"' # snap100


mod = '"bust53"' # try again for traj optim, using the debugged free/vis


mod = '"bust54"' # smooth loss directly on obj too


mod = '"bust55"' # traj again but cleaner (no feat3d)
mod = '"bust56"' # train traj via scene
mod = '"bust57"' # log11; 200 iters; pret latents 500 01_s20_m128x64x128_p128x384_1e-2_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust54

mod = '"bust58"' # train scene again, but with proper occ sup, where bkg is trained with median and obj is trained with fullscene


mod = '"bust59"' # traj via scene, but pret bust54 this time for real
mod = '"bust60"' # coeff 0 on elastic; 

mod = '"bust61"' # train scene again; pret with bust58; 2.0m car; smaller lr, so that it does not diverge

mod = '"bust62"' # train_via_scene; noise_amount=0.0, so i can get a hint of whether this is optimizable at all

# why isn't traj moving?
# why is the render loss going up when i train for scene right now?
mod = '"bust63"' # again but 10k iters, just to see if there's any motion overnight
# nope.
# i think there is some grad problem with differentiability
# also, bust61 had loss going steadily up. let's fix that first.


mod = '"bust64"' # train scene; replicate teh issue from scene61
mod = '"bust65"' # flip randomly, instead of using a symm loss
mod = '"bust66"' # make objvar half, with a hard mirror
mod = '"bust67"' # compute-0-36
mod = '"bust68"' # 1-14 again
mod = '"bust69"' # do not load obj, so that we get proper mirror init
# overall, these results go the opposite direction from bust61. so maybe that was just a gpu bug
mod = '"bust70"' # traj_via_scene; pret 01_s20_m128x64x128_p128x384_1e-3_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust69 < too early

mod = '"bust71"' # train scene; use coeffs
mod = '"bust72"' # put latents earlier in autoname; pret 400 01_s20_m128x64x128_p128x384_1e-3_O_c1_t1_s.5_R_d64_r10_cacs20i2one_bust69
mod = '"bust73"' # separate coeffs/stats for bkg and obj occ
mod = '"bust74"' # render coeff 1
mod = '"bust75"' # scale the render loss by /S
mod = '"bust76"' # added hyp for weighing the total render
# oddly enough, it seems a lot like self.obj has a seq dim, allowing it to behave differently on different steps
mod = '"bust77"' # pret 800 bust69; show unsqueezed obj feats; log11; 
mod = '"bust78"' # show occ_objs
mod = '"bust79"' # show occ_objs better
mod = '"bust80"' # apply occ loss with help of full scene

# i think what i need to do is:
# first resize the image to its target size, with F.interpolate, then move the pixels one by one

mod = '"bust81"' # bring back noise in rendering < ok, render loss is a bit jumpier, but all ok.
mod = '"bust82"' # traj_via_scene; pret 1k 01_s20_m128x64x128_p128x384_1e-3_L_oo2_bo1_os2_ss1_r1_O_c1_t1_s.5_R_d64_r1_cacs20i2one_bust80
# maybe i need to not detach the occ, to backprop into the traj?

mod = '"bust83"' # print a few things, to see where the grads are
mod = '"bust84"' # disable elastic and diff, to see if grads still appear in traj
# ok, we're still getting grads.
mod = '"bust85"' # one more step to disable traj elastic
# ok, with high enough lr, it starts to move.
mod = '"bust86"' # bring back elastic
mod = '"bust87"' # also train occ
mod = '"bust88"' # put render total into latents scope
mod = '"bust89"' # allow obj to optimize too, with lr mult 1
mod = '"bust90"' # more coeffs

############## exps ##############

exps['builder'] = [
    'carla_goodvar', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    '3_iters',
    'lr5',
    'B1',
    'no_backprop',
    'train_occ',
    # 'train_render',
    # 'train_center',
    'log1',
]
exps['feat3d_trainer'] = [
    'carla_goodvar', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    # '10k_iters',
    '1k_iters',
    # '10_iters',
    'lr5',
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    # 'frozen_feat3d',
    # 'pretrained_feat3d',
    'train_feat3d',
    'train_occ',
    # 'train_render',
    # 'pretrained_latents',
    # 'log53',
    'snap100',
    'log11',
    # 'log1',
]
exps['traj_trainer'] = [
    'carla_goodvar', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    # '10k_iters',
    '200_iters',
    # '3_iters',
    'lr2',
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    'frozen_feat3d', 
    'pretrained_feat3d',
    'train_feat3d',
    'train_occ',
    'train_trajvar',
    # 'train_render',
    # 'pretrained_latents',
    # 'log53',
    'snap100',
    'log11',
    # 'log1',
]
exps['scene_trainer'] = [
    'carla_goodvar', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    # '10k_iters',
    '1k_iters',
    # '50_iters',
    'lr3',
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    # 'frozen_feat3d', 
    # 'pretrained_feat3d',
    'pretrained_latents',
    # 'train_feat3d',
    'train_scenevar',
    'train_occ',
    'train_render',
    # 'pretrained_latents',
    'snap100',
    'log53',
    # 'log11',
]
exps['traj_via_scene_trainer'] = [
    'carla_goodvar', # mode
    'carla_complete_train1_data', # dataset
    'carla_8-4-8_bounds_train',
    '10k_iters',
    # '1k_iters',
    # '200_iters',
    'lr3',
    'B1',
    # 'no_shuf',
    # 'no_backprop',
    # 'frozen_feat3d', 
    # 'pretrained_feat3d',
    'pretrained_latents',
    'train_traj_via_scene',
    'train_occ',
    'train_render',
    'log53',
    # 'log11',
]

############## groups ##############

groups['carla_goodvar'] = ['do_carla_goodvar = True']

groups['train_trajvar'] = [
    'train_trajvar = True',
    'latent_traj_elastic_coeff = 1.0', 
    'latent_traj_diff_coeff = 1.0', 
]
groups['train_scenevar'] = [
    'train_scenevar = True',
    'latent_bkg_occ_coeff = 1.0', 
    'latent_obj_occ_coeff = 2.0', 
    'latent_scene_smooth_coeff = 1.0', 
    'latent_obj_smooth_coeff = 2.0',
    'latent_render_coeff = 1.0',
]
groups['train_traj_via_scene'] = [
    'train_traj_via_scene = True',
    'latent_render_coeff = 1.0',
    'latent_traj_elastic_coeff = 0.1', 
    'latent_scene_smooth_coeff = 1.0', 
    'latent_obj_smooth_coeff = 2.0',
    'latent_bkg_occ_coeff = 1.0', 
    'latent_obj_occ_coeff = 2.0', 
]

groups['do_test'] = ['do_test = True']
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 4',
]
# groups['train_bkg'] = [
#     'do_bkg = True',
#     'bkg_coeff = 1.0',
#     'bkg_epsilon = 0.75',
# ]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_rgb_coeff = 1.0',
    # 'render_depth_coeff = 0.1',
    # 'render_smooth_coeff = 0.01',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_temporal_coeff = 1.0',
    'occ_smooth_coeff = 0.5',
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
# S = 2
S = 20
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

# SIZE = 24
# SIZE_val = 24
# SIZE_test = 24
# SIZE_zoom = 24

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

SIZE = 16
SIZE_val = 16
SIZE_test = 16
SIZE_zoom = 16

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
    'trainset = "cacs20i2ten"',
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
    'trainset = "cacs20i2one"',
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
