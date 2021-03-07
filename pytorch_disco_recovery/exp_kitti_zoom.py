from exp_base import *

############## choose an experiment ##############

# current = 'builder'
# current = 'trainer'
current = 'tester'
# current = 'error_finder'

mod = '"z00"' # reset
mod = '"z01"' # load new ktacs8i1t
mod = '"z02"' # include vis
mod = '"z03"' # odo on train; track on test
mod = '"z04"' # allow ktrack consec
mod = '"z05"' # delete R stuff
mod = '"z06"' # show indiv boxes on rgbs
mod = '"z07"' # show gif of full boxes
mod = '"z08"' # show traj too
mod = '"z09"' # center on object in test
mod = '"z10"' # test with pret; narrow bounds
mod = '"z11"' # test with pret; narrow bounds again; exclude some vis
mod = '"z12"' # use S_test
mod = '"z13"' # narrower bounds
mod = '"z14"' # zeromot
mod = '"z15"' # narrower bounds; 1k instead of 100
mod = '"z16"' # narrow bounds; 1k instead of 100
mod = '"z17"' # narrow bounds; 1k instead of 100; shuffle valid and sink invalid
mod = '"z18"' # bugfix
mod = '"z19"' # one more bugfix; also vis
mod = '"z20"' # 1k instead of 100; less vis
mod = '"z21"' # narrower bounds
mod = '"z22"' # do_test mode
mod = '"z23"' # ensure template exists on frame0
mod = '"z24"' # narrow bounds 
mod = '"z25"' # fewer prints; avoid double summ of gt
mod = '"z26"' # zeromot
mod = '"z27"' # zeromot with early return
# (partial) new dat, with 130k pts
mod = '"z28"' # narrow bounds
mod = '"z29"' # return early for impossible objects
mod = '"z30"' # zeromot
# full new dat
mod = '"z31"' # full new dat
mod = '"z32"' # full new dat; zeromot < crashed due to convex hull thing
mod = '"z33"' # add 0.01 noise in iou eval < crashed too
mod = '"z34"' # clamp lenlist to min 0.01 < fixed things
mod = '"z35"' # req full traj to be valid
mod = '"z36"' # real model; narrow bounds
mod = '"z37"' # data6



mod = '"t00"' # train a bit
mod = '"t01"' # diff bounds/resolution on train and test < size error
mod = '"t02"' # faster logging; more prints
mod = '"t03"' # see if vox util is doing what i watn
mod = '"t04"' # don't use hyp so much in vox util
mod = '"t05"' # fewer prints; faster logging
mod = '"t06"' # higher res on train (SIZE=32)
mod = '"t07"' # regular res on train (SIZE=16); 200k iters; faster logging < super slow; i'll kill it at 10k
mod = '"t08"' # regular res on train (SIZE=16); 200k iters; fast logging
mod = '"t09"' # use W=416; use lighter odo data < seems worse, at least at the start; maybe it's coming back
mod = '"t10"' # like t09 but carefully compute a valid mask


mod = '"test00"' # pret 10k 02_s3_m128x16x128_1e-5_F3_d64_O_c.1_s.001_E3_n2_d16_c1_koacs10i2a_ktads8i1a_t05< indeed, slightly higher, but not necessarily beyond noise level
mod = '"test01"' # better autoname; collect stable stats
# mean_ious [1.01 0.67 0.58 0.53 0.52 0.49 0.49 0.44]
mod = '"test02"' # pret moc15 instead
# mean_ious [0.99 0.66 0.57 0.53 0.51 0.48 0.45 0.42]
# ok, good, so t05 outperforms moc15 slightly
# but these two had shuf

mod = '"test03"' # pret moc15; no shuf
# mean_ious [1.01 0.65 0.56 0.52 0.51 0.46 0.43 0.42]
mod = '"test04"' # pret 10k t05; no shuf
# mean_ious [1.01 0.67 0.59 0.54 0.51 0.48 0.46 0.43]
# alright, t05 still wins. it's great news that this is in the right direction

mod = '"test05"' # pret 20k 02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_t09
# mean_ious [1.01 0.67 0.58 0.53 0.52 0.49 0.49 0.44]
# hey! slightly better over most timestpes. so it's going up. let's see for how long

mod = '"test06"' # pret 30k 02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_t09
# mean_ious [1.01 0.67 0.59 0.56 0.5  0.48 0.46 0.43]
# ok not clearly better, but this does respect the plot, where iou@3 was on a hump

mod = '"b00"' # builder; show 10 iters
mod = '"b01"' # compute that reliability thing
mod = '"b02"' # also show not_ok
mod = '"b03"' # also show not_ok*occ
mod = '"b04"' # show this at higher res
mod = '"b05"' # req that you confirm the voxel in 2 views
mod = '"b06"' # for freespace, req only 1 view; also use free to reject occ
mod = '"b07"' # use max for occ agg in not_ok
mod = '"b08"' # show that max
mod = '"b09"' # assure we are looking at the same thing across S
mod = '"b10"' # totally new method
mod = '"b11"' # show ok_occ
mod = '"b12"' # pre-handle disagreement
mod = '"b13"' # check for either*both
mod = '"b14"' # get rid of the prep
mod = '"b15"' # clean up, go to half res
mod = '"b16"' # train with this, see if it outperforms t09
mod = '"b17"' # yes masking, but old data

mod = '"test07"' # pret 30k 02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_b17
# mean_ious [1.01 0.67 0.6  0.57 0.52 0.51 0.48 0.46]
# ok, indeed better 

mod = '"b18"' # like b17 but train occ too


mod = '"test08"' # pret 40k 02_s3_m128x16x128_1e-5_F3_d64_E3_n2_d16_c1_koads4i4a_ktads8i1a_b16
# mean_ious [1.01 0.67 0.61 0.58 0.53 0.5  0.48 0.45]
# compared to test07, this is better on early steps and worse on late

mod = '"s00"' # show test feats; i want to make them slow


# one idea is to propagate the object with const velocity, and propagate the entire remaining scene as if it's static

mod = '"s01"' # 100 iters; if iou is above some threshold, return early. i want to see the errors

# sometimes the location itself gets completely occluded, e.g., by a tree. on these steps we could use just the propagated box

# try using ALL points in the box, not just box*occ
mod = '"s02"' # tester; use full mask
mod = '"s03"' # no backprop; when sum>1k, take occ only
mod = '"s04"' # no backprop; when sum>1k, take a random 1k
# mean_ious [0.96 0.61 0.56 0.53 0.49 0.46 0.44 0.41]
# ok, this is a bit worse than test08, meaning this random 1k is worse than the occ set
# ah, but this is not apples to apples, since i am not rejecting the same frames

mod = '"s05"' # zero out the freespace in the heatmap < reverse of what's good
mod = '"s06"' # bugfix attempt
mod = '"s07"' # make freespace min value, make max value 0
mod = '"s08"' # for apples to apples eval, reject frames where mask*occ <=8
# mean_ious [0.88 0.65 0.59 0.57 0.52 0.51 0.48 0.45]
# ok, this is similar to test08; i think this means the extra points are not helping much
mod = '"s09"' # allow up to 2k pts
# mean_ious [0.95 0.66 0.6  0.57 0.54 0.51 0.48 0.46]
# ok, 2k points is better than 1k. and now we cleanly outperform occ only. this is good, because a 2d method would be definitely limited to occ
mod = '"s10"' # ransac_steps = 256 instead of 128
# mean_ious [0.96 0.66 0.6  0.58 0.54 0.52 0.49 0.46] < new winner 
# nice: this is slightly better than s09, meaning steps=256 is better than steps=128.
# also it's interesting that this mode looked worse in the tb
mod = '"s11"' # take the occ points plus up to 2k non-occ non-free pts within the mask
# mean_ious [0.95 0.65 0.59 0.56 0.52 0.51 0.48 0.45]
# not better than s10, oddly. 
mod = '"s12"' # repeat of s10 i hope: use 2k points within the object mask
# mean_ious [0.96 0.66 0.6  0.58 0.54 0.52 0.48 0.45]
# ok, maybe this is within the randomness tol?
mod = '"s13"' # pret 10k 02_s3_m128x16x128_1e-5_F3_d64_O_c.1_s.001_E3_n2_d16_c1_koacs10i2a_ktads8i1a_b18
# mean_ious [0.96 0.66 0.6  0.58 0.54 0.51 0.49 0.46]
# pretty similar

mod = '"zer00"' # zeromot; actually i loaded moc15 by accident
mod = '"zer01"' # zeromot
mod = '"zer02"' # zermot again, but maybe with a bugfix? < no it's the same on the curve. apparently my table is a bit off though.
# mean_ious [0.97 0.59 0.4  0.29 0.22 0.18 0.16 0.14]

mod = '"g00"' # basic trainer again, as a baseline/replicate
mod = '"g01"' # same as g00 but higher lr
mod = '"g02"' # pret 230k moc15 (i think this is a real replicate)
mod = '"g03"' # pret and allow all pts (be_safe=False)


mod = '"t00"' # test 02_s2_m128x16x128_1e-4_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_g02
# mean_ious [0.96 0.65 0.59 0.57 0.53 0.51 0.49 0.46]
mod = '"t01"' # test 02_s2_m128x16x128_1e-4_F3_d64_E3_n2_d16_c1_koacs10i2a_ktads8i1a_g03
# mean_ious [0.96 0.62 0.56 0.53 0.5  0.45 0.41 0.39]
mod = '"t02"' # test moc15
# mean_ious [0.96 0.68 0.63 0.58 0.55 0.53 0.49 0.47]
# ok, so it seems that with kitti self-sup we are very slightly ruining things.
# this is not great.
mod = '"t03"' # test random features!!!!
# mean_ious [0.5  0.11 0.07 0.04 0.03 0.03 0.02 0.02]

mod = '"g04"' # pret moc15; fastest; 5 iters; show feat gifs too
mod = '"g05"' # show occs too
mod = '"g06"' # 20 iters
mod = '"g07"' # summ oned for occs_vis
mod = '"g08"' # shuf=True

############## define experiment ##############

exps['builder'] = [
    'kitti_zoom', # mode
    # 'kitti_odo_data', # dataset
    'kitti_track_data', # dataset
    # 'kitti_regular_train_bounds',
    'kitti_narrow_test_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    # 'pretrained_feat3D',
    # 'train_feat3D',
    # 'train_emb3D',
    # 'train_moc3D',
    # 'fastest_logging',
    'fastest_logging',
]
exps['trainer'] = [
    'kitti_zoom', # mode
    'kitti_odo_track_data', # dataset
    'kitti_regular_train_bounds',
    'kitti_narrow_test_bounds',
    '20k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D',
    'train_feat3D',
    # 'train_occ',
    'train_emb3D',
    'fast_logging',
]
exps['tester'] = [
    'kitti_zoom', # mode
    'kitti_track_data', # dataset
    'kitti_narrow_test_bounds',
    # 'kitti_narrower_bounds',
    '20_iters',
    # '1k_iters',
    # '100_iters',
    # 'no_vis',
    # 'no_shuf',
    'do_test', 
    'B1',
    'pretrained_feat3D',
    'train_feat3D',
    'no_backprop',
    # 'faster_logging',
    'fastest_logging',
]
exps['error_finder'] = [
    'kitti_zoom', # mode
    'kitti_track_data', # dataset
    'kitti_narrow_test_bounds',
    '100_iters',
    'no_shuf',
    'do_test', 
    'B1',
    'pretrained_feat3D',
    'train_feat3D',
    'fastest_logging',
]

############## net configs ##############

groups['no_vis'] = ['do_include_vis = False']
groups['do_test'] = ['do_test = True']

groups['kitti_zoom'] = ['do_kitti_zoom = True']

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
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
    'feat2D_smooth_coeff = 0.01',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.001',
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
    'occ_smooth_coeff = 0.001',
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

SIZE = 16
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
    'trainset = "koads4i4a"',
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

