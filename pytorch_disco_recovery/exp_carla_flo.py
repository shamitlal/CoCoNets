from exp_base import *


############## choose an experiment ##############

current = 'flo_builder'
# current = 'flo_trainer_small'
current = 'flo_trainer'

mod = '"hun00"' # on time but she late
mod = '"hun01"' # uncomment hsv to rgb
mod = '"hun02"' # builder; get synth flow and show it
mod = '"hun03"' # on real_flow iters do nothing
mod = '"hun04"' # backwarp the right thing 
mod = '"hun05"' # run feat and flo
mod = '"hun06"' # train a bit
mod = '"hun07"' # revox
mod = '"hun08"' # show the occ_memX0s
mod = '"hun09"' # back to v1, to see the (worse) occ_memX0s
mod = '"hun10"' # v2
mod = '"hun11"' # train with l1 and drop the synth arg
mod = '"hun12"' # 0.1 motion
mod = '"hun13"' # 0.1 motion
mod = '"hun14"' # import vox util, to see if voxels are square
mod = '"hun15"' # make them square
mod = '"hun16"' # back to 0.1,1.0 motion
mod = '"hun17"' # heatmap size 5
mod = '"hun18"' # l2 instead of l1
mod = '"hun20"' # negative_slope=0.1 in featnet3D Net3D also < no difference
mod = '"hun21"' # reverted neg slope


mod = '"flo00"' # builder
mod = '"flo01"' # again
mod = '"flo02"' # debug=True, so i can see some shapes
mod = '"flo03"' # make a grid; mult by heat
# ok, looks quite reasonable, except at the borders < actually, this is just due to tiny flows; the model finds corresps inward due to invalid/none matches oob
mod = '"flo04"' # scales = [1.0]
mod = '"flo05"' # clip = 5.0
mod = '"flo06"' # clip = 2.0
mod = '"flo07"' # heatmap 7
mod = '"flo08"' # clip = 1.0
mod = '"flo09"' # clip according to g; scales 0.5, 0.75, 1.0
mod = '"flo10"' # print cc in corner and middle, so i can see the cc values and determine the border strategy
# ok, OOB cc is 0.0; inbound cc is mostly 1.0, meaning near-perfect match
mod = '"flo11"' # relu before the softmax, to make neg coors similar to OOB
mod = '"flo12"' # dilation_patch=2.0, to search wider
mod = '"flo13"' # clip = 4.0; use dilation in the grid too
mod = '"flo14"' # clip = 3.0; use dilation in the grid too
mod = '"flo15"' # dilation = 3
# ok, that's a good way to get larger motion estimates
# but overall, it seems everyone is staying put.
# mod = '"flo16"' # t_amount = 2.0 instead of 0.1
mod = '"flo17"' # scales = [1.0]
mod = '"flo18"' # t_amount = 2.0 instead of [0.1 or 1.0]
mod = '"flo19"' # show all the flow, not flow*occg
mod = '"flo20"' # t_amount = 1.0
mod = '"flo21"' # generate mask with zeros at the border, with width max_disp
mod = '"flo22"' # print that, and summ it separately
# somehow, the flow borders are still there
mod = '"flo23"' # pca vis, to see what's going on with that flow
mod = '"flo24"' # flip that mask
mod = '"flo25"' # use that mask in loss
mod = '"flo26"' # train a while
mod = '"flo27"' # pret; fewer prints
mod = '"flo28"' # pret; builder
mod = '"flo29"' # train a while
mod = '"flo30"' # t_amount = 2.0 instead of 1.0
mod = '"flo31"' # again, on fresh node

############## define experiments ##############

exps['flo_builder'] = [
    'carla_flo', # mode
    'carla_tatt_trainset_data', # dataset
    'carla_16-16-16_bounds_train',
    '10_iters',
    'train_feat3D',
    'train_flow',
    'lr4',
    'B1',
    'no_shuf',
    'no_backprop',
    # 'time_flip',
    'log1',
]
exps['flo_trainer_small'] = [
    'carla_flo', # mode
    'carla_flow_data', # dataset
    'carla_bounds',
    '100k_iters',
    'train_feat3D',
    'train_flow',
    'lr4',
    'B2',
    'faster_logging',
]
exps['flo_trainer'] = [
    'carla_flo', # mode
    'carla_tatt_trainset_data', # dataset
    'carla_16-16-16_bounds_train',
    '300k_iters',
    'pretrained_feat3D',
    'train_feat3D',
    'train_flow',
    'lr4',
    'B2',
    'log50',
]

############## net configs ##############

groups['load_pb'] = [
    'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z22"', # 4k ft
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat_do_flip = True',
]
groups['train_flow'] = [
    'do_flow = True',
    # 'flow_l1_coeff = 1.0',
    'flow_l2_coeff = 1.0',
    'flow_heatmap_size = 7',
]

############## datasets ##############

# DHW for mem stuff
SIZE = 16
Z = SIZE*8
Y = SIZE*1
X = SIZE*8

K = 8 # how many objects to consider
N = 8

S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

SIZE = 20
SIZE_val = 20
SIZE_test = 20

K = 2 # how many objects to consider
N = 16 # how many objects per npz
S = 2
S_val = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
PH = int(H)
PW = int(W)

dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"

# groups['carla_flo10_data'] = [
#     'dataset_name = "carla_flow"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "lacs2i3ten"',
#     'dataset_list_dir = "./npzs"',
#     'dataset_location = "./npzs"',
#     'dataset_filetype = "npz"',
# ]
# groups['carla_flo_train_data'] = [
#     'dataset_name = "carla_flow"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "lacs2i3t"',
#     'dataset_list_dir = "./npzs"',
#     'dataset_location = "./npzs"',
#     'dataset_filetype = "npz"',
# ]
# groups['carla_flo_trainval_data'] = [
#     'dataset_name = "carla_flow"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "lacs2i3t"',
#     'valset = "lacs2i3v"',
#     'dataset_list_dir = "./npzs"',
#     'dataset_location = "./npzs"',
#     'dataset_filetype = "npz"',
# ]
# groups['carla_forecast10_data'] = [
#     'dataset_name = "carla_forecast"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "fags16i3ten"',
#     'dataset_list_dir = "./npzs"',
#     'dataset_location = "./npzs"',
#     # 'dataset_list_dir = "/projects/katefgroup/datasets/carla/processed/npzs"',
#     # 'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"',
#     'dataset_filetype = "npz"'
# ]
groups['carla_tatt_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2t"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_flowtrack_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "lacs2i3ten"',
    'trainset_format = "seq"', 
    'trainset_seqlen = 2', 
    'testset = "fags16i3ten"',
    'testset_format = "traj"',
    'testset_seqlen = 10',
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_ep14_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "lacep14"',
    'trainset_format = "seq"', 
    'trainset_seqlen = 2', 
    'testset = "fagep14"',
    'testset_format = "traj"',
    'testset_seqlen = 10',
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_ep16_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "lacep16"',
    'trainset_format = "seq"', 
    'trainset_seqlen = 2', 
    # 'testset = "fagep16"',
    # 'testset_format = "traj"',
    # 'testset_seqlen = 10',
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_flow_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "lacs2i3ten"',
    'trainset_format = "seq"', 
    'trainset_seqlen = 2', 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]


groups['carla_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
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
