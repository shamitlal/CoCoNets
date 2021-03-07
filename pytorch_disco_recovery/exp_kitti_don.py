from exp_base import *

############## choose an experiment ##############

# current = 'builder'
# current = 'trainer'
current = 'tester'

mod = '"b00"' # see what happens

mod = '"test00"' # pret 02_s2_1e-4_F2_d64_s.001_T2_e100_d1_mabs7i3t_don23
mod = '"test01"' # faster logging < OOM
mod = '"test02"' # put the matmul on cpu
mod = '"test03"' # have a dummy trainset, to avoid mem leak < ineffective; lowering the resolution helped though
# after 84 iters:
# mean_ious [0.96 0.57 0.44 0.34 0.31 0.27 0.24 0.22]

# let's train from this init and see if results go up or down
mod = '"test03"' # train a while


mod = '"test04"' # tester; pret 120k 02_s2_1e-4_F2_d64_T2_e100_d1_koacs10i2a_ktads8i1a_test03
# after 84 iters
# mean_ious [0.98 0.7  0.62 0.57 0.55 0.49 0.43 0.34]
# quite good it seems

mod = '"col00"' # start on colorization
mod = '"col01"' # trainer
mod = '"col02"' # show lab dropout
mod = '"col03"' # fastest
mod = '"col04"' # back2color and prep
mod = '"col05"' # on cpu
mod = '"col06"' # show L A B
mod = '"col07"' # show L A B
mod = '"col08"' # proper ranges for these funny tools
mod = '"col09"' # cuda issue
mod = '"col10"' # fix
mod = '"col11"' # run featnet2D
mod = '"col12"' # feed rgb at halfres
mod = '"col13"' # col2D
mod = '"col14"' # show rgb_e
mod = '"col15"' # train a while
mod = '"col16"' # use huber
mod = '"col17"' # track on test iters

mod = '"col18"' # show some lab stats
mod = '"col19"' # train a while; do the loss in lab space
mod = '"col20"' # feed lab input to featnet2d
mod = '"col21"' # div corr 0.1 < looks good
mod = '"col22"' # div corr 0.5 < bad
mod = '"col23"' # div corr 0.07
mod = '"col24"' # SIZE=SIZE_TEST=32 instead of 16
mod = '"col25"' # again, but max_pts=1000, to avoid mem leak
mod = '"col26"' # small smooth coeff < memleak
mod = '"col27"' # feat_camXs.detach().cpu() in test mode < actually other bug
mod = '"col28"' # keep the matmul on cpu
mod = '"col29"' # fix the bug < no, too much on cpu
mod = '"col30"' # heat_b.cuda()
mod = '"col31"' # heat_b.cuda() earlier, to get past those min/max lines < very slow
mod = '"col32"' # use v2v and quarter res < slow
mod = '"col33"' # stay on gpu
mod = '"col34"' # fast logging instead of faster < ok, still a bit slow i don't know why < worse than the other net
mod = '"col35"' # no test, to avoid memleak


mod = '"dis00"' # tester; pret 30k 02_s2_1e-4_F2_d32_s.01_C2_h100_koacs10i2a_ktads8i1a_col31
# OOM after 336 iters, but:
# mean_ious [0.78 0.3  0.24 0.2  0.14 0.11 0.09 0.1 ]

############## define experiment ##############

exps['builder'] = [
    'kitti_don', # mode
    'kitti_odo_data', # dataset
    'kitti_regular_train_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat2D',
    # 'train_tri2D',
    'train_col2D',
    # 'fastest_logging',
    'fastest_logging',
]
exps['trainer'] = [
    'kitti_don', # mode
    # 'kitti_odo_track_data', # dataset
    'kitti_odo_data', # dataset
    'kitti_regular_train_bounds',
    'kitti_narrow_test_bounds',
    '300k_iters',
    'lr4',
    'B2',
    # 'pretrained_feat2D',
    'train_feat2D',
    # 'train_tri2D',
    'train_col2D',
    'fast_logging',
]
exps['tester'] = [
    'kitti_don', # mode
    'kitti_odo_track_data', # dataset
    # 'kitti_track_data', # dataset
    'kitti_regular_train_bounds',
    'kitti_narrow_test_bounds',
    '1k_iters',
    'no_shuf',
    'do_test', 
    'B1',
    'pretrained_feat2D',
    'train_feat2D',
    'no_backprop',
    'faster_logging',
]

############## net configs ##############

groups['kitti_don'] = ['do_kitti_don = True']
groups['no_vis'] = ['do_include_vis = False']
groups['do_test'] = ['do_test = True']


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
groups['train_emb3D'] = [
    'do_emb3D = True',
    # 'emb_3D_ml_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    # 'emb_3D_mindist = 16.0',
    # 'emb_3D_num_samples = 2',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
    'emb_3D_ce_coeff = 1.0',
]
groups['train_tri2D'] = [
    'do_tri2D = True',
    # 'tri_2D_ml_coeff = 1.0',
    # 'tri_2D_l2_coeff = 0.1',
    'tri_2D_num_samples = 100',
    'tri_2D_ce_coeff = 1.0',
]
groups['train_col2D'] = [
    'do_col2D = True',
    # 'col2D_l1_coeff = 1.0',
    'col2D_huber_coeff = 100.0',
]
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
p    # 'feat2D_smooth_coeff = 0.01',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
]
groups['train_feat3DS'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    'feat3D_sparse = True',
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
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.001',
]

############## datasets ##############

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 8
H = 128
W = 416
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"
dataset_location = "/data6/kitti_processed"

SIZE = 32
SIZE_test = 32
groups['kitti_regular_train_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*1)),
    'X = %d' % (int(SIZE*8)),
]
groups['kitti_narrow_test_bounds'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -2.0', # down (neg is up)
    'YMAX_test = 2.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*1)),
    'X_test = %d' % (int(SIZE_test*4)),
]
groups['kitti_track_data'] = [
    'dataset_name = "ktrack"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "ktads8i1a"',
    'testset_format = "ktrack"', 
    'testset_seqlen = %d' % S_test, 
    'testset_consec = True', 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "koacs10i2a"',
    'trainset_format = "kodo"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['kitti_odo_track_data'] = [
    'dataset_name = "kodo"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "koads4i4a"',
    'trainset = "koacs10i2a"',
    'trainset_format = "kodo"', 
    'trainset_seqlen = %d' % S,
    'trainset_consec = False', 
    'testset = "ktads8i1a"',
    'testset_format = "ktrack"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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

