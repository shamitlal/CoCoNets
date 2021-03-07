from exp_base import *
import pretrained_nets_carla as pret_carla

############## choose an experiment ##############

current = 'time_builder'
# current = 'time_mini_trainer'
current = 'time_trainer'

mod = '"time00"' # clean up, reset exps
mod = '"time01"' # show hists
mod = '"time02"' # normalize to -0.5, 0.5
mod = '"time03"' # fix
mod = '"time04"' # show loc on occ
mod = '"time05"' # use a sampling coords loss < ok similar, but anyway now i can add rot
mod = '"time06"' # normalize by resolution 
mod = '"time07"' # coeff=10; show hist of coords
mod = '"time08"' # add gt rots < bug: mixing R and X0
mod = '"time09"' # X0 
mod = '"time10"' # estimate rots
mod = '"time11"' # show perspective box
mod = '"time12"' # builder;  show bird box
mod = '"time13"' # mult by occ
mod = '"time14"' # masked mean by occ
mod = '"time15"' # swap x and z for vis
mod = '"time16"' # transpose the rgb actually
mod = '"time17"' # totally different vis strat
mod = '"time18"' # fix coord bug
mod = '"time19"' # use gt
mod = '"time20"' # other method again
mod = '"time21"' # summ
mod = '"time22"' # clean up
mod = '"time23"' # show e and g
mod = '"time24"' # don't log so much
mod = '"time25"' # fast logging, but flush_secs=100000, max_queue=10000; flush on every log_this
mod = '"time26"' # faster logging
mod = '"time27"' # 10k
mod = '"time28"' # feat loss
mod = '"time29"' # rename xyz to samp 
mod = '"time30"' # split up losses to separate funcs
mod = '"time31"' # no feat loss please
mod = '"time32"' # added a loop over forward steps
mod = '"time33"' # make gifs of inputs; builder
mod = '"time34"' # replace another
mod = '"time35"' # don't summ write inside locnet
mod = '"time36"' # traj
mod = '"time37"' # proper gt
mod = '"time38"' # fewer summs
mod = '"time39"' # show many timesteps in occ
mod = '"time40"' # sum up the feat loss over time
mod = '"time41"' # also plot the mini feat losses
mod = '"time42"' # list then mean
mod = '"time43"' # show bev gifs
mod = '"time44"' # show perspective gifs
mod = '"time45"' # go forward and backward
mod = '"time46"' # make all gifs use cycles
mod = '"time47"' # train 
mod = '"time49"' # only compute summs on summ iters
mod = '"time50"' # see what happens when you vary S
mod = '"time51"' # fix bug relating to when to crop; 100k
mod = '"time52"' # new aws
mod = '"time53"' # ckpt 02_s1_m128x32x128_1e-4_F3_d32_L_c10_fags16i3t_time52; look at val error too < big gap!
mod = '"time54"' # don't ckpt; use val along the way, to see what is happening


mod = '"search00"' # show search region
mod = '"search01"' # just use search util to upgrade the lrt
mod = '"search02"' # rotation *= 0.1
mod = '"search03"' # factor 3
mod = '"search04"' # prints
mod = '"search05"' # fix the coordinates issue for the rotation
mod = '"search06"' # delte optimonal searhc reigonstuff

mod = '"search07"' # train with this

mod = '"search08"' # back to builder; rotate and translate after getting to obj coords 
mod = '"search09"' # rotation *= 10 (back to normal)
mod = '"search10"' # t_max=4.0

mod = '"search11"' # train again; safe inverse in the locnet param converter, for better grads
mod = '"search12"' # train again; additive_pad=0.0 everywhere

mod = '"search13"' # builder again;
mod = '"search14"' # train again; estimate the box in mem coords
mod = '"search15"' # hun data; show input and final obj
mod = '"search16"' # again, but gpu2
mod = '"search17"' # proper summ of this
mod = '"search18"' # train again (14 repeat, since i killed by accident)

mod = '"search19"' # mini trainer; fixed another bug in the summ
mod = '"search20"' # print obj lens
mod = '"search21"' # fastest logging
mod = '"search22"' # show obj_len_e
mod = '"search23"' # no rot
mod = '"search24"' # use neg xy
mod = '"search25"' # use rot.T


# something is very wrong. i'm getting tiny-looking boxes, whose positions don't even make sense,
# despite the fact that length is staying constant

mod = '"search26"' # simpler method
mod = '"search27"' # get the scale and bias
mod = '"search28"' # again
mod = '"search29"' # proper scaling and bias, by passing in ZYX

# i think the problem i had before is related to this mixing of raw_scene coords with feat_scene coords




############## define experiments ##############

exps['time_builder'] = [
    'carla_time', # mode
    'carla_track_train_data', # dataset
    'carla_tall_bounds',
    '10_iters',
    'train_feat3D',
    'train_loc',
    'lr4',
    'B1',
    'no_shuf',
    'no_backprop',
    'fastest_logging',
]
exps['time_mini_trainer'] = [
    'carla_time', # mode
    'carla_track_train100_data', # dataset
    'carla_tall_bounds',
    '1k_iters',
    'train_feat3D',
    'train_loc',
    'lr4',
    'B2',
    'fastest_logging',
]
exps['time_trainer'] = [
    'carla_time', # mode
    'carla_track_trainval_data', # dataset
    'carla_tall_bounds',
    '300k_iters',
    # 'pretrained_feat3D', 
    # 'pretrained_loc', 
    'train_feat3D',
    'train_loc',
    'lr4',
    'B2',
    'fast_logging',
]

############## net configs ##############

groups['carla_time'] = ['do_carla_time = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
]
groups['train_loc'] = [
    'do_loc = True',
    'loc_samp_coeff = 10.0',
    # 'loc_feat_coeff = 1.0',
]
groups['train_flow'] = [
    'do_flow = True',
    'flow_l2_coeff = 1.0',
    'flow_heatmap_size = 5',
]
groups['pretrained_loc'] = [
    'do_loc = True',
    'loc_init = "' + pret_carla.loc_init + '"',
]


############## datasets ##############

# mem resolution
SIZE = 16
Z = SIZE*8
Y = SIZE*2
X = SIZE*8

# ZZ = int(SIZE/2)
# ZY = int(SIZE/2)
# ZX = int(SIZE/2)
ZZ = int(SIZE)
ZY = int(SIZE)
ZX = int(SIZE)

K = 8 # how many objects to consider
N = 8
S = 1 # timesteps

# input resolution
H = 128
W = 384
# proj resolution
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
dataset_location = "/data2/carla/processed/npzs"

groups['carla_time10_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "lacs2i3ten"',
    'trainset_format = "seq"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_track_train_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_track_train100_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3hun"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_track_trainval_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "fags16i3t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "fags16i3v"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
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
