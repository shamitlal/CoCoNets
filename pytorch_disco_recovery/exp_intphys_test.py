from exp_base import *

############## choose an experiment ##############

current = 'test_basic'

mod = '"00"' # 
mod = '"01"' # pret det and fore 
mod = '"02"' # cleaned up test mode a bi
mod = '"03"' # no nets; use gt for now
mod = '"04"' # load fullseq data
mod = '"05"' # loops
mod = '"06"' # show free and vis with ray_add=0.25
mod = '"07"' # ray_add in the right place, and clamp(with+without)
mod = '"08"' # ab data, with more sky hopefully
mod = '"09"' # ac data, with full sky 
mod = '"10"' # sample 
mod = '"11"' # round the raylists
mod = '"12"' # *_ validlist
mod = '"13"' # vis vislist with mean
mod = '"14"' # print first ten < ok, sometimes it works but sometimes not. maybe the rays are not long enough
mod = '"15"' # show a single occ and vis, to see the ray extension
mod = '"16"' # no sum and clmp
mod = '"17"' # ad data
mod = '"18"' # clamp(free+occ)
mod = '"19"' # no sky
mod = '"20"' # ray extend 0.5
mod = '"21"' # ray extend 1.0; this time really no sky
mod = '"22"' # summ occ
mod = '"23"' # 100 samps
mod = '"24"' # 200 samps < wow looks a lot better
mod = '"25"' # fewer prints < ok looks great now. 
mod = '"26"' # hypo*2 samps (so 230 in this case)
mod = '"27"' # extend by 2.0
mod = '"28"' # plot ray vis match acc
mod = '"29"' # one scalar vis, instead of per object
mod = '"30"' # extend by 1.0 instead < worse than 2
mod = '"31"' # extend by 3.0 instead < worse than 2
# let's extend by 2 and call it a day
mod = '"32"' # extend by 0.5 and 2 and add and clamp < no difference from 30
mod = '"33"' # extend by 1.5 < same as 2.0
mod = '"34"' # ae npzs, with 50k and some sky
mod = '"35"' # dilate
mod = '"36"' # dilate but ray_add=1.0 instead of 1.5
mod = '"36"' # dilate but ray_add=1.0 instead of 1.5
mod = '"37"' # no dilate. ray_add=2.0
mod = '"38"' # more iters
mod = '"39"' # eliminate X
mod = '"40"' # more 
mod = '"41"' # pret feat
mod = '"42"' # pret feat
mod = '"43"' # get rid of the req that lr>0
mod = '"44"' # tiny lr
mod = '"45"' # det
mod = '"46"' # det
mod = '"47"' # use boxlist_mem_e in the finder
mod = '"48"' # evaluate found matches
mod = '"49"' # be within 20 vox
mod = '"50"' # print in finder
mod = '"51"' # do things in cam coords; thresh=3.0
mod = '"52"' # make tidlist the right length 
mod = '"53"' # do not include image summs
mod = '"54"' # only log every 50
mod = '"55"' # include all
mod = '"56"' # run every step, even if not log_this or do_backprop
mod = '"57"' # fix bug of None returns (which happened on iter50)
mod = '"58"' # start getting those tracklets; 10 iters; print if a trilpet looks ok
mod = '"59"' # log, so i can see what's going on
mod = '"60"' # choose max number of objects
mod = '"61"' # seqlen 10
mod = '"62"' # do it in cam coords
mod = '"63"' # report the raw distance
mod = '"64"' # 
mod = '"65"' # assoc and show it

############## define experiments ##############

exps['test_basic'] = [
    'intphys_test',
    'intphys_fullseq_train_data_aws',
    'intphys_bounds',
    'test_basic',
    'pretrained_feat',
    'pretrained_det',
    # 'pretrained_forecast',
    '3_iters',
    'B1',
    'no_shuf',
    'fastest_logging',
    # 'faster_logging',
    'lr9',
    'no_backprop',
]

############## group configs ##############

groups['test_basic'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_sparse_conv = True', 
    'do_det = True',
    # 'do_forecast = True',
]
groups['train_pri'] = [
    'do_pri = True',
    'pri_ce_coeff = 1.0',
    # 'pri_reg_coeff = 1.0',
    'pri_smooth_coeff = 1.0',
]
groups['train_rpo'] = [
    'do_rpo = True',
    'rpo_forward_coeff = 1.0',
    'rpo_reverse_coeff = 1.0',
]
groups['train_forecast'] = [
    'do_forecast = True',
    # 'forecast_maxmargin_coeff = 1.0',
    # # 'forecast_smooth_coeff = 0.1', 
    # 'forecast_num_negs = 500',
    'forecast_l2_coeff = 1.0',
]

############## datasets ##############

# mem resolution
SIZE = 16
X = int(SIZE*6)
Y = int(SIZE*4)
Z = int(SIZE*6)

ZX = int(SIZE*4)
ZY = int(SIZE*4)
ZZ = int(SIZE*4)

# these params need cleaning; also, 3 only works if you do not count occluders
N = 3 # max objects
K = 3 # objects to consider 

S = 10
H = 288
W = 288
V = 100000
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['intphys_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -7.0', # down (neg is up)
    'YMAX = 5.0', # down
    'ZMIN = 0.1', # forward
    'ZMAX = 12.1', # forward
]    
groups['intphys_fullseq_train_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "fsaei2s20t"',
    'trainset = "fsaei2s10t"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
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
