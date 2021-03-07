from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'trainer'
# current = 'center_trainer'
current = 'seg_trainer'
# current = 'vq_trainer'
# current = 'vq_exporter'

mod = '"bot64"' # partial data on singlegpu aws. using aws so that i can compile spconv
mod = '"bot65"' # more data
mod = '"bot66"' # sparse < indeed, this is faster. good job. 
mod = '"bot67"' # higher res
mod = '"bot68"' # full data; wide bounds
mod = '"bot69"' # kernel 4 instead of 3 in the sparse blocks
mod = '"bot70"' # one more sparse conv; swap the down res with a regular res
mod = '"bot71"' # kernel 3, pad 1
mod = '"bot72"' # one more downsamp during the sparse part; upsamp 16 at end
mod = '"bot73"' # pad1 on all kernels
mod = '"bot74"' # more powerful (more layers, higher chans), but some pad0
mod = '"bot75"' # avoid computing the smooth loss < no effect on speed
mod = '"bot76"' # start at half that resolution; downsample less, so you end up with full res outputs
mod = '"bot77"' # provide dense input! i am throwing away the encoder anyway!
mod = '"bot78"' # 2x2x2 in that up layer. (3x3x3 caused an error)
mod = '"bot79"' # S = 5; pret 90k 01_s3_m160x160x160_1e-4_F3_d32_O_c1_mabs7i3t_bot78
mod = '"bot80"' # similar but on matrix and no pret


mod = '"cen00"' # start on centernet; 10 iters; show centers
mod = '"cen01"' # run centernet; crop the mask
mod = '"cen02"' # radius 2
mod = '"cen03"' # soft=True
mod = '"cen04"' # soft=True
mod = '"cen05"' # show hist of that, so i can see some values; also min and max
mod = '"cen06"' # sigma 1
mod = '"cen07"' # return early if there is no object in the crop
mod = '"cen08"' # print 
mod = '"cen09"' # print 
mod = '"cen10"' # properly return early
mod = '"cen11"' # properly return early
mod = '"cen12"' # sigma 2
mod = '"cen13"' # sigma 1; pret 10k 01_s7_m160x160x160_1e-4_F3_d32_O_c1_mabs7i3t_bot80
mod = '"cen14"' # train centernet with focal
mod = '"cen15"' # add center loss to total < okok nan! 
mod = '"cen16"' # assert crop==crop_guess. 
mod = '"cen17"' # 10 iters
mod = '"cen18"' # clamped sigmoid before loss
mod = '"cen19"' # print some itnermeds
mod = '"cen20"' # show me the neg weights
mod = '"cen21"' # train 10k like this; see if it goes down
mod = '"cen22"' # freeze feat and occ
mod = '"cen23"' # use mean of pos and neg loss
mod = '"cen24"' # really use mean please
mod = '"cen25"' # unfreeze
mod = '"cen26"' # redo after nodes died
mod = '"cen27"' # actual sigmoid instead of clamp
mod = '"cen28"' # sigma=2
mod = '"cen29"' # coeff 100.0

mod = '"seg00"' # summ seg0
mod = '"seg01"' # new data actually
mod = '"seg02"' # get onehot and show it with pca
mod = '"seg03"' # unproject one-hots and round and show
mod = '"seg04"' # again
mod = '"seg05"' # downsamp before unproj
mod = '"seg06"' # squeeze
mod = '"seg07"' # use valid in summs 
mod = '"seg08"' # convert to index seg; use color codes
mod = '"seg09"' # get to memX0 in one shot
mod = '"seg10"' # again
mod = '"seg11"' # do it in two steps
mod = '"seg12"' # only acquire labels 1:
mod = '"seg13"' # shift the summ colors
mod = '"seg14"' # use the mode instead of max < this gives all zeros, obviously
mod = '"seg15"' # use all labels again, and max
mod = '"seg16"' # print min and max of mem
mod = '"seg17"' # print min and max of mem
mod = '"seg18"' # one shot
mod = '"seg19"' # print some shapes < ah, the rgb here is high res!
mod = '"seg20"' # do NOT downsamp the seg < hm. looks bad. 
mod = '"seg21"' # two steps < OK looks good, but a bit dirty.
mod = '"seg22"' # only allow points with >0.6
mod = '"seg23"' # only allow points with >0.9 < yes, much cleaner
mod = '"seg24"' # do this after mean
mod = '"seg25"' # eliminate all values <0.9
mod = '"seg26"' # eliminate all values where nonzeromin and max do not agree
mod = '"seg27"' # reduce masked mean, then elim values <0.98
mod = '"seg28"' # use the carla colors < looks great
mod = '"seg29"' # just elim values <0.8 < even better. now some lane markings come through


mod = '"cen30"' # pret 10k 01_s7_m160x160x160_1e-4_F3_d32_O_c1_C_c100_mabs7i3t_cen29
mod = '"cen31"' # same but apply ce loss instead of that weird focal
mod = '"cen32"' # same
mod = '"cen33"' # better summs (avoid overwriting prob_loss)
mod = '"cen34"' # 



mod = '"seg30"' # re-run
mod = '"seg31"' # parse boxes the new way
mod = '"seg32"' # summ lrtlists
mod = '"seg33"' # summ lrtlists
mod = '"seg34"' # pack up the seg parser
mod = '"seg35"' # mostly empty segnet
mod = '"seg36"' # mostly empty segnet
mod = '"seg37"' # show seg_e and g
mod = '"seg38"' # balance even further
mod = '"seg39"' # train a while 
mod = '"seg40"' # only show occ voxels of seg_e
mod = '"seg41"' # show all of seg_e, but at free voxels, apply a free loss
mod = '"seg42"' # show prob_los image summ; ckpt 10k 01_s5_m160x160x160_1e-4_F3_d32_O_c1_C_p1_mabs7i3t_cen31
mod = '"seg43"' # ckpt 10k 01_s5_m160x160x160_1e-3_F3_d32_O_c1_C_p1_mabs7i3t_cen32


mod = '"up00"' # upnet
mod = '"up01"' # higher chans in the encoder; more pret where possible. 
mod = '"up02"' # one less down; scale up 8 (instead of 16)
mod = '"up03"' # couple more layers padding=0


mod = '"si00"' # builder; prep size pred
mod = '"si01"' # feed lrtlist_mem to centernet; compute and show clist_mask, to see if my clist is good
mod = '"si02"' # again. < ok looks good. 
mod = '"si03"' # see if the gt boxes look good
mod = '"si04"' # again
mod = '"si05"' # again, smaller thickness  < ok i guess
mod = '"si06"' # sample out the locations in the pred map; apply a regression loss
mod = '"si07"' # center_e_clean via nms with kernel 3
mod = '"si08"' # pret center
mod = '"si09"' # kernel 7
mod = '"si10"' # suppress <0.8
mod = '"si11"' # kernel 9
mod = '"si12"' # kernel 15 < ok, i have the object centroids, but they are a bit rough
mod = '"si13"' # radius=1 for objectness training (for later)
mod = '"si14"' # get the topk and turn those into centroids
mod = '"si15"' # and mask
mod = '"si16"' # predict orientation 
mod = '"si17"' # predict everything and parse it into boxes
mod = '"si18"' # summ the boxes in perspective too
mod = '"si19"' # alt bev vis
mod = '"si20"' # print sizes, since it's odd that my preds are too large
mod = '"si21"' # start training the sizes
mod = '"si22"' # train 
mod = '"si23"' # fix summ bug; log50
mod = '"si24"' # builder for rotation 
mod = '"si25"' # construct rot with argmax
mod = '"si26"' # train a while
mod = '"si27"' # better summs; balance the rot loss across bins
mod = '"si28"' # default snap freq 5k, since this is slow days; centernet thresh 0.9 instead of 0.8


mod = '"vq00"' # train latent loss for a bit; see how high it is
mod = '"vq01"' # allow no boxes
mod = '"vq02"' # log50
mod = '"vq03"' # try the exporter
mod = '"vq04"' # pret 40k up3
mod = '"vq05"' # do_test
mod = '"vq06"' # wide cube bounds for test
mod = '"vq07"' # train a while; pret 10k 02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_si27
mod = '"vq08"' # more data: 3000 instead of 881 < nan!
mod = '"vq09"' # export a bit, so i can see < looks fine
mod = '"vq10"' # retry i guess
mod = '"vq11"' # smaller lr, since i think the dict needs to learn more; also even more dat, and pret 15k 02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq10
mod = '"vq12"' # pret vq too
mod = '"vq13"' # elim focal loss, since it sometimes gives nan; pret 30k 02_s5_m160x160x160_1e-3_F3_d32_U_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_si27; train JUST the dict for a while



mod = '"ex00"' # export inds for 02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq12

mod = '"vq14"' # only down/up 4 scales! ; pret 5k 02_s5_m160x160x160_1e-9_F3_d32_U_V3r_n512_l1_Of_c1_Cf_p1_s.1_r1_Sf_p1_mads7i3a_vq13
mod = '"vq15"' # pret 10k 01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14; small smooth coeffs

mod = '"ex01"' # export inds for 15k 01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14

mod = '"vq16"' # smoothness loss directly on up; drop the l2 norm constraint; pret 5k 02_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_s.001_C_p1_s.1_r1_S_p1_s.001_mads7i3a_vq15


mod = '"ex02"' # export inds for 55k 01_s5_m160x160x160_1e-4_F3_d32_U_V3r_n512_l1_O_c1_C_p1_s.1_r1_S_p1_mads7i3a_vq14


############## define experiment ##############

exps['builder'] = [
    'carla_bottle', # mode
    'carla_multiview_all_data', # dataset
    'carla_wide_cube_bounds',
    # '10_iters',
    '3_iters',
    'lr0',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_up3D', 
    'pretrained_center', 
    # 'pretrained_center', 
    # 'pretrained_seg', 
    'train_feat3D',
    'train_up3D',
    'train_center',
    'log1',
]
exps['trainer'] = [
    'carla_bottle', # mode
    # 'carla_multiview_train1_data', # dataset
    # 'carla_multiview_train10_data', # dataset
    'carla_multiview_train_data', # dataset
    # 'carla_regular_bounds',
    # 'carla_nearcube_bounds',
    'carla_wide_cube_bounds',
    # 'carla_cube_bounds',
    # '100k_iters',
    '100k_iters',
    'lr4',
    'B1',
    # 'pretrained_feat3D', 
    # 'pretrained_occ', 
    'train_feat3D',
    # 'train_emb3D',
    'train_occ',
    'train_center',
    # 'log500',
    'log50',
]
exps['center_trainer'] = [
    'carla_bottle', # mode
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
    'carla_bottle', # mode
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
    'carla_bottle', # mode
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
    'carla_bottle', # mode
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
groups['carla_bottle'] = ['do_carla_bottle = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat3D_smooth_coeff = 0.01',
]
groups['train_up3D'] = [
    'do_up3D = True',
    'up3D_smooth_coeff = 0.001',
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
    'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.001',
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
Z = 160
Y = 160
X = 160
Z_test = 160
Y_test = 160
X_test = 160

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 5
H = 128*2
W = 384*2
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data4/carla/processed/npzs"

groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3one"',
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
    'trainset = "mabs7i3t"',
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

