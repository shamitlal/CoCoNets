from exp_base import *

############## choose an experiment ##############

# current = 'builder'
current = 'trainer'
# current = 'vq_trainer'

mod = '"ml00"' # nothing
mod = '"ml01"' # release go
mod = '"ml02"' # do_ml3d
mod = '"ml03"' # get feat and altfeat
mod = '"ml04"' # 3d ml
mod = '"ml05"' # train a bit
mod = '"ml06"' # use rounded valid
mod = '"ml07"' # get featnet2D happening
mod = '"ml08"' # get emb2D happening
mod = '"ml09"' # also rgb
mod = '"ml10"' # improved summs
mod = '"ml11"' # eliminated smooth coeffs in emb nets
mod = '"ml12"' # better hyp name
mod = '"ml13"' # use altfeat_memR for viewpred

mod = '"vq00"' # vq_trainer
mod = '"vq01"' # train and test
mod = '"vq02"' # start adding a linear stopgrad classifier
mod = '"vq03"' # prep for test properly, so the log goes in the right place
mod = '"vq04"' # linclass
mod = '"vq05"' # train linclass and show accs
mod = '"vq06"' # use halfres crops
mod = '"vq07"' # add linclass loss; only run linclass if len
mod = '"vq08"' # feed the feats into linclass
mod = '"vq09"' # train occ too
mod = '"vq10"' # summ_lrtlist (for debug)
mod = '"vq11"' # 100k, to see if acc keeps rising
mod = '"vq12"' # higher featdim, for perfect apples to apples
mod = '"vq13"' # higher quant coeff


mod = '"vq14"' # aws; train occ too
mod = '"vq15"' # pret (minus occ) 90k 02_s3_m128x8x128_p64x192_1e-4_F2_d64_F3_d64_V3r_n512_l2_V_d64_e.1_E2_s1_m.1_e2_n32_E3_m1_e.1_n2_d16_L_c1_mabs7i3t_vq13
mod = '"vq16"' # same but bigger ydim
mod = '"vq17"' # nothing; builder; put most things into base; just prep common tensors
mod = '"vq18"' # show the boxes
mod = '"vq19"' # again
mod = '"vq20"' # trainer
mod = '"vq21"' # trainer; just emb3d
mod = '"vq22"' # train occ too
mod = '"vq23"' # really train occ
mod = '"vq24"' # check for 100 inb points; use regular bounds
mod = '"vq25"' # no occ, just to go faster for a bit
mod = '"vq26"' # occ again; summ default axis
mod = '"vq27"' # resnet3d with no padding; no occ, bc i'm worried about dims < indeed: feat_memX0s torch.Size([2, 2, 64, 57, 1, 57])
mod = '"vq28"' # again
mod = '"vq29"' # higher input resolution
mod = '"vq30"' # more cube-ish < 57, 9, 57
mod = '"vq31"' # mindist 0
mod = '"vq32"' # back to net3d
mod = '"vq33"' # again
mod = '"vq34"' # bring back l2
mod = '"vq35"' # resnet
mod = '"vq36"' # use Encoder3D
mod = '"vq37"' # do not pad
mod = '"vq38"' # print less; chans=32
mod = '"vq39"' # 100k, log500
mod = '"vq40"' # 

############## define experiment ##############

exps['builder'] = [
    'carla_ml', # mode
    'carla_multiview_train10_data', # dataset
    # 'carla_multiview_train10_val10_data', # dataset
    'carla_bounds',
    '10_iters',
    'lr0',
    'B1',
    'no_shuf',
    'train_feat3D',
    'train_emb3D',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_ml', # mode
    'carla_multiview_train_data', # dataset
    # 'carla_regular_bounds',
    'carla_nearcube_bounds',
    '100k_iters',
    'lr4',
    'B2',
    'train_feat3D',
    'train_emb3D',
    # 'train_occ',
    'log500',
]
exps['vq_trainer'] = [
    'carla_ml', # mode
    'carla_multiview_train_data', # dataset
    # 'carla_multiview_train_test_data', # dataset
    'carla_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'train_feat3D',
    'train_emb3D',
    'train_view',
    'train_vq3drgb',
    'train_linclass',
    'train_occ',
    'pretrained_feat3D', 
    'pretrained_view', 
    'pretrained_vq3drgb', 
    'fast_logging',
]

############## net configs ##############

groups['carla_ml'] = ['do_carla_ml = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ce_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    # 'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
    'vq3drgb_num_embeddings = 512', 
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
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 2.0',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]


############## datasets ##############

# dims for mem
SIZE = 10
Z = int(SIZE*16)
Y = int(SIZE*8)
X = int(SIZE*16)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 3
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

