from exp_base import *

############## choose an experiment ##############

current = 'stage2_builder'
current = 'stage2_trainer_uncond'
# current = 'stage2_trainer_cond'
current = 'stage2_tester_uncond'

# at 16x16x16 resolution, it takes 5.2 seconds, so 787v/s
# at 32x32x32 resolution, i expect it to take 42 seconds < actually it's 292, so 5 mins; 112v/s

mod = '"earth00"' # massachussets
mod = '"earth01"' # val too
mod = '"earth02"' # to get a baseline, let's try training gen3dvq
mod = '"earth03"' # pre-agg
mod = '"earth04"' # faster logging; just trainval
mod = '"earth05"' # vis the training quant outs
mod = '"earth06"' # dy=0
mod = '"earth07"' # test 
mod = '"earth08"' # only generate the top half
mod = '"earth09"' # do this every 1k iters
mod = '"earth10"' # stop early
mod = '"earth11"' # use a different pret model
mod = '"earth12"' # use a different pret model; big bounds
mod = '"earth13"' # stop even earlier
mod = '"earth14"' # do not pre-agg
mod = '"earth15"' # sigen builder
mod = '"earth16"' # activate sigen
mod = '"earth17"' # train a while
mod = '"earth18"' # input mask 0.05 to 0.95
mod = '"earth19"' # faster logging, just to see some test errors
mod = '"earth20"' # only compute half
mod = '"earth21"' # proper suffixes
mod = '"earth22"' # goal = 5000 
mod = '"earth23"' # goal = 1000 ; show feats before and after; just_gif < good vis, but 1000 is not enough.
mod = '"earth24"' # goal = 5000
mod = '"earth25"' # generate two completions; set bottom half to zero always
mod = '"earth26"' # 300k trainer; just train train train. we'll test after 10k
mod = '"earth27"' # see what 10k looks like
mod = '"earth28"' # 300k trainer; radius 5 instead of 3
mod = '"earth29"' # 300k trainer; layers 8-end are not masked (instead of 5-end)
mod = '"earth30"' # summ output embs; radius 3 again
mod = '"earth31"' # diff lr; otw should be equiv
mod = '"earth32"' # layers 6-end are regular convs, instead of 8-end

mod = '"pret00"' # layers 8-end regular; pret 02_s5_m128x16x128_p64x192_1e-3_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth30
mod = '"pret01"' # no shuf
mod = '"pret02"' # smaller dims
mod = '"pret03"' # double time

mod = '"earth33"' # use all views as input
mod = '"earth34"' # use all views as input; only sup in visible area
mod = '"earth35"' # show vis_memR; don't compute extras; also better gpu
mod = '"earth36"' # ckpt=02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.01_V_d64_e1_mabs7i3t_mabs7i3v_r41
mod = '"earth37"' # redo 
mod = '"earth38"' # vis=ones
mod = '"earth39"' # vis=visR; S=6 < practically no difference
mod = '"earth40"' # add layer11, layer12 < bug here, where layer10 was reused for the last two
mod = '"earth41"' # fix bug with layer11, layer12 < not doing better, for some reason
mod = '"earth42"' # fix second bug with layer11, layer12, and with masking < not doing better, for some reason
mod = '"earth43"' # S = 6; train with dropouts in [0.05,0.5]; end at layer10
mod = '"earth44"' # end at layer15


# mod = '"pret04"' # pret 02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38
# mod = '"pret05"' # uncomment init lines
# mod = '"pret06"' # don't pret sigen3d
# mod = '"pret07"' # clamp valid
# mod = '"pret08"' # do pret; and speedup 4
# mod = '"pret09"' # again, to see diversity
# mod = '"pret10"' # bugfix, to make it more stable at end
# mod = '"pret11"' # drop the last two layers, since i think they are not in earth38
# mod = '"pret12"' # speedup=8
# mod = '"pret13"' # proper freeze
# mod = '"pret14"' # pret 02_s6_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth39
# mod = '"pret15"' # fire occnet after
# mod = '"pret16"' # vq3drgb with the same ckpt


mod = '"earth45"' # pret 40k 02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r44




# mod = '"pret16"' # pret 30k 02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38
# mod = '"pret17"' # again
# mod = '"pret18"' # eliminate layers 11-15 (speedup=8) < ok, this looks better tahn pret16; 
mod = '"pret19"' # same as pret18 speedup=1, to see if things look a lot better


# pret 60k 02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r44;
# idea here: radial test
mod = '"earth46"' # builder; nothing 
mod = '"radial00"' # show frontier
mod = '"radial01"' # show frontier*total_vis
mod = '"radial02"' # round, and also clamp outer
mod = '"radial03"' # apply loss at the frontier
mod = '"radial04"' # train a while
mod = '"radial05"' # add back layers 11-15
mod = '"radial06"' # delta (0.1, 0.0, 0.1)
mod = '"radial07"' # like radial06 but no grid input to sigen3d
mod = '"radial08"' # like radial07 but no delta

# pret 20k 02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46
mod = '"radial09"' # radial08 but diff ckpt < died

# turned off grid, turned off layers 11-15
mod = '"pret20"' # speed=8, pret 10k 02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_radial04;

# 20k 02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth45
# restored layers 11-15; turned on grid;
mod = '"pret21"' # 


# pret 50k 02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38
# eliminated layers 11-15
mod = '"pret22"' # 
mod = '"pret23"' # again
mod = '"pret24"' # pret occ with 02_s2_m128x8x128_p64x192_1e-4_F64_V3r_n512_l1_O_c.1_s.01_V_d64_e1_mabs7i3t_mabs7i3v_r41



mod = '"radial10"' # radial08 but ckpt 30k 02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46
mod = '"radial11"' # delta (0.1, 0.0, 0.1), and no grid


# turned on grid
# pret 50k 02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38
mod = '"pret25"'
# pret 100k 02_s5_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Vf_d64_mabs7i3t_mabs7i3v_earth38
mod = '"pret26"'

# pret 50k 02_s4_m128x8x128_p64x192_1e-4_F64f_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_radial11
mod = '"pret27"'
mod = '"pret28"' # again



mod = '"raidal12"' # like radial11 but ckpt 100k 02_s2_m128x8x128_p64x192_1e-4_F64_s.01_V3r_n512_l1_O_c.1_s.001_V_d64_e1_mabs7i3t_mabs7i3v_r46
mod = '"raidal13"' # similar but delta 0.0

mod = '"pret29"' # pret radial12

# pret 02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_O_c.1_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq14
mod = '"go00"' # use new featnet3D
mod = '"go01"' # pixelcnn
mod = '"go02"' # pixelcnn, but 10 layers with conv5 filters, so it is more like my cond model
mod = '"go03"' # sigen, but B+1,S=1,faster logging for simpler vis < hm, frontier is showing up empty
mod = '"go04"' # show outer vis too
mod = '"go05"' # show total vis too
mod = '"go06"' # use total=ones
mod = '"go07"' # 10 iters; show a couple frontiers


# this model, if it works, represents slightly higher quality (more colorful) conditional generation
mod = '"pretgo00"' # pret 02_s4_m128x8x128_p64x192_1e-4_F3f_d64_V3rf_n512_S3i_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_go00
mod = '"pretgo01"' # again, to see diversity
mod = '"pretgo02"' # condition on the opposite half
mod = '"pretgo03"' # proper ckpt, proper cond
mod = '"pretgo04"' # again, for diversity

# # uncond sampling
# mod = '"unc_go02"' # pret 02_s4_m128x8x128_p64x192_1e-4_F3f_d64_V3rf_n512_G3v_c1_Of_Vf_d64_mabs7i3t_mabs7i3v_go02
# mod = '"unc_go03"' # really load it this time
# mod = '"unc_go04"' # seriously < ok, looks good. 
# mod = '"unc_go05"' # do not stop early



mod = '"car00"' # just try the builder; see what happens
mod = '"car01"' # decode into up
mod = '"car02"' # sample too < nice and fast
mod = '"car03"' # decode center too 
mod = '"car04"' # decode seg too 
mod = '"car05"' # again
mod = '"car06"' # took over go() block
mod = '"car07"' # run the model
mod = '"car08"' # again
mod = '"car09"' # again
mod = '"car10"' # clean up summs; put more into the sample/ scope
mod = '"car11"' # load the dataset onto gpu
mod = '"car12"' # train a while < mem leak
mod = '"car13"' # just the training set


mod = '"car14"' # test 10 iters
mod = '"car15"' # pret 20k 04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car13 for gen3dvq
mod = '"car16"' # see what it's like to generate a 16x16x16 scene instead
mod = '"car17"' # pret 40k 
mod = '"car18"' # actually generate 8x8x8 again, since i think otw the fov changes at test time; anyway, the main lesson is that 16x16x16 is pretty tolerable
mod = '"car19"' # pret 160k
mod = '"car20"' # show occ samples
mod = '"car21"' # show center samples; then train a while
mod = '"car22"' # put 16,16,16
mod = '"car23"' # pret 30k 04_s1_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_G3v_c1_Of_Cf_Sf_mads7i3t_car22
mod = '"car24"' # pret the gen3d the same way
mod = '"car25"' # pret 40k 
mod = '"car26"' # train cond
mod = '"car27"' # pret 50k < this is shockingly better than the 40k ckpt
mod = '"car28"' # train cond
mod = '"car29"' # pret 60k
mod = '"car30"' # train cond; apply loss at the frontier of a random mask
mod = '"car31"' # pret 120k car22
mod = '"car32"' # temporarily restore the l2 norm in upnet
mod = '"car33"' # l2 norm again, ckpt 01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14
mod = '"car34"' # l2 norm, train with vq14 and inds/01_s3_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_Of_Cf_Sf_mads7i3a_ns_ex02_ind_list.npy


mod = '"car35"' # try 50k car34


############## define experiment ##############

exps['stage2_builder'] = [
    'carla_gen3dvq', # mode
    # 'carla_multiview_train_val_data', # dataset
    # 'carla_multiview_train_val_test_data', # dataset
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_all_data', # dataset
    'carla_multiview_train_test_data', # dataset
    'carla_wide_cube_bounds', 
    '5_iters',
    'lr0',
    'B4',
    'pretrained_feat3D',
    'pretrained_up3D',
    'pretrained_vq3d',
    'pretrained_occ',
    'pretrained_center',
    'pretrained_seg',
    'frozen_feat3D', 
    'frozen_vq3d', 
    'frozen_up3D',
    'frozen_occ',
    'frozen_center',
    'frozen_seg',
    # 'train_sigen3d',
    'train_gen3d',
    'no_shuf',
    'log1',
]
exps['stage2_trainer_uncond'] = [
    'carla_gen3dvq', # mode
    'carla_multiview_train_data', # dataset
    'ind_dataset', # actual dataset
    'carla_wide_cube_bounds',
    '300k_iters',
    'lr4',
    'B4',
    'pretrained_feat3D', # we definitely need this
    'pretrained_vq3d', # we definitely need this
    'pretrained_up3D', # it's helpful to load this so that we have the occ weights built in
    'pretrained_occ', # it's helpful to load this so that we have the occ weights built in
    'pretrained_center', # it's helpful to load this so that we have the occ weights built in
    'pretrained_seg', # it's helpful to load this so that we have the occ weights built in
    'frozen_feat3D', 
    'frozen_vq3d', 
    'frozen_occ', 
    'frozen_center',
    'frozen_seg',
    'train_gen3d',
    'log500',
]
exps['stage2_trainer_cond'] = [
    'carla_gen3dvq', # mode
    'carla_multiview_train_data', # dataset
    'ind_dataset', # actual dataset
    'carla_wide_cube_bounds',
    '300k_iters',
    'lr4',
    'B4',
    'pretrained_feat3D', # we definitely need this
    'pretrained_vq3d', # we definitely need this
    'pretrained_up3D', # it's helpful to load this so that we have the occ weights built in
    'pretrained_occ', # it's helpful to load this so that we have the occ weights built in
    'pretrained_center', # it's helpful to load this so that we have the occ weights built in
    'pretrained_seg', # it's helpful to load this so that we have the occ weights built in
    'frozen_feat3D', 
    'frozen_vq3d', 
    'frozen_occ', 
    'frozen_center',
    'frozen_seg',
    'train_sigen3d',
    'log500',
]
exps['stage2_tester_uncond'] = [
    'carla_gen3dvq', # mode
    'carla_multiview_test_data', # dataset
    'carla_wide_cube_bounds',
    'ind_dataset', 
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3D',
    'pretrained_up3D',
    'pretrained_vq3d',
    'pretrained_occ',
    'pretrained_center',
    'pretrained_seg',
    'pretrained_gen3d',
    'frozen_feat3D', 
    'frozen_up3D', 
    'frozen_vq3d', 
    'frozen_occ', 
    'frozen_center',
    'frozen_seg',
    'frozen_gen3d',
    'train_feat3D',
    'log1',
]

############## net configs ##############

groups['ind_dataset'] = [
    # 'ind_dataset = "inds/01_s3_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_Of_Cf_Sf_mads7i3a_ns_ex00_ind_list.npy"',
    # 'ind_dataset = "inds/01_s3_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_Of_Cf_Sf_mads7i3a_ns_ex01_ind_list.npy"',
    'ind_dataset = "inds/01_s3_m160x160x160_1e-4_F3f_d32_U_V3rf_n512_Of_Cf_Sf_mads7i3a_ns_ex02_ind_list.npy"',
]    

groups['carla_gen3dvq'] = ['do_carla_gen3dvq = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_num_embeddings = 512',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'feat2D_dim = 64',
    'view_depth = 32',
    'view_l2_coeff = 1.0',
]
groups['train_gen3d'] = [
    'do_gen3d = True',
    'gen3d_coeff = 1.0',
]
groups['train_sigen3d'] = [
    'do_sigen3d = True',
    'sigen3d_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]

############## datasets ##############

Z = 160
Y = 160
X = 160
Z_test = 160
Y_test = 160
X_test = 160

# # dims for mem
# SIZE = 8
# Z = int(SIZE*16)
# Y = int(SIZE*1)
# X = int(SIZE*16)
K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 1
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"

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
    'trainset = "mads7i3t"',
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
    'trainset = "mads7i3a"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "mads7i3a"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mads7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
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
