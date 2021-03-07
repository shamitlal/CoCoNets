from exp_base import *

############## choose an experiment ##############

current = 'track_builder'
# current = 'track_vis'

mod = '"snap00"' # 
mod = '"snap01"' # run the frozen flow graph inside
mod = '"snap02"' # only declare each net once
mod = '"snap03"' # use actual inputs
mod = '"snap04"' # vis feats
mod = '"snap05"' # vis flow
mod = '"snap06"' # vis flow
mod = '"snap07"' # do it in X0 < looks good
mod = '"snap08"' # match the metric bounds of the original model
mod = '"snap09"' # S=2, use first ex; use pt wrapper
mod = '"snap10"' # S=5; run every pair
mod = '"snap11"' # S=4 for a bit more speed; better scope
mod = '"snap12"' # fewer summs
mod = '"snap13"' # S=2
mod = '"snap14"' # R
mod = '"snap15"' # show boxesR
mod = '"snap16"' # R0
mod = '"snap17"' # R0
mod = '"snap18"' # show obj/box and obj/mask
mod = '"snap19"' # 
mod = '"snap20"' # only collect flow in occ
mod = '"snap21"' # clean up
mod = '"snap22"' # *2
mod = '"snap23"' # compute iou
mod = '"snap24"' # show box _e,g
mod = '"snap25"' # show mask e,g 
mod = '"snap26"' # same
mod = '"snap27"' # real flow
mod = '"snap28"' # 10 iters, to see if there is something declining, or it's random
mod = '"snap29"' # 32 ransac tries instead of 16
mod = '"snap30"' # packed up a util
mod = '"snap31"' # cleaner
mod = '"snap32"' # cumulative; do full S
mod = '"snap33"' # again
mod = '"snap34"' # S = 5
mod = '"snap35"' # go 1 to S-1
mod = '"snap36"' # show actual cumu mem flow, and actually use the backwarp
mod = '"snap37"' # clean up some vars
mod = '"snap38"' # cam0_T_camI
mod = '"snap39"' # warp the features instead
mod = '"snap40"' # vis that
mod = '"snap41"' # show mask_mems_e and g
mod = '"snap42"' # set 0 to gt for 3d masks vis
mod = '"snap43"' # show box on rgb
mod = '"snap44"' # warp using cumu flow directly
mod = '"snap45"' # go all the way to S
mod = '"snap46"' # show box camRs gif
mod = '"snap47"' # coeff 1.0
mod = '"snap48"' # 128 ransac steps instead of 16, and reset the perm < better indeed
mod = '"snap49"' # take residual_flow1 as velocity, and add it for backwarps
mod = '"snap50"' # no velocity; use coeff 0.9
mod = '"snap51"' # deleted some  unused lines
mod = '"snap52"' # print npz_filanem
mod = '"snap53"' # save outs
mod = '"snap54"' # save outputs in Rs too
mod = '"snap55"' # actually do it
mod = '"snap56"' # save also Rs
mod = '"snap57"' # save also flows

mod = '"jump00"' # just run; no save
mod = '"jump01"' # return early if no object
mod = '"jump02"' # return if scorelist is unhappy
mod = '"jump03"' # use obj_lrtlits_camR0s[0]
mod = '"jump04"' # disable summs
mod = '"jump05"' # 1k 
mod = '"jump06"' # eval
mod = '"jump07"' # aggregate
mod = '"jump08"' # S = 10
mod = '"jump09"' # vis 10
mod = '"jump10"' # vis 20; include summs
mod = '"jump11"' # vis 20; include summs
mod = '"jump12"' # use obj_
mod = '"jump13"' # show occR0s, unpR0s
mod = '"jump14"' # get to R0 to calc inbounds
mod = '"jump15"' # 1k
mod = '"jump16"' # show mean and std


mod = '"jump17"' # vis
mod = '"jump18"' # just_gif; show in X
mod = '"jump19"' # lock in the scores and ids
mod = '"jump20"' # also show gt
mod = '"jump21"' # use obj[0]


mod = '"jump22"' # re-run the last thing, but on matrix
mod = '"jump23"' # don't shuffle and sink; track_builder (for eval)
mod = '"jump24"' # do shuffle and sink
# 25
# 26
# 27
mod = '"jump28"' # use old shuffle-and-sink func
mod = '"jump29"' # use old collect func
mod = '"jump30"' # mostly new; ONLY old collect func
mod = '"jump31"' # again
mod = '"jump32"' # wrap tracker
mod = '"jump33"' # wrap tracker; feed proper obj_ tensors
mod = '"jump34"' # if (1) 
mod = '"jump35"' # re-translate it < looks good this time. < no. that was still if(1)
mod = '"jump36"' # if (0)
mod = '"jump37"' # cleaned up the vars
mod = '"jump38"' # use lrt_camI_g in both places
mod = '"jump39"' # eliminated lrtlist_camRs
mod = '"jump40"' # eliminated scorelist_s
mod = '"jump41"' # cleaned more
mod = '"jump42"' # got rid of alt impl
mod = '"jump43"' # enable vis
mod = '"jump44"' # quant again


############## define experiments ##############

exps['track_builder'] = [
    'carla_track', # mode
    'carla_val_data', # dataset
    '1k_iters',
    'B1',
    'no_shuf',
    'no_backprop',
    'eval_map',
    'load_pb',
    'fastest_logging',
]
exps['track_vis'] = [
    'carla_track', # mode
    'carla_val_data', # dataset
    '20_iters',
    'B1',
    'no_shuf',
    'no_backprop',
    'include_summs',
    'eval_map',
    'load_pb',
    'fastest_logging',
]

############## net configs ##############

groups['load_pb'] = [
    'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z22"', # 4k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z21"', # 4k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z20"', # 3k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z19"', # 3k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z18"', # another 2k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z17"', # another 2k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z16"', # 2k ft
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z15"', # iclr
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z14"', # iclr
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z10"', # no cycles
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z11"', # short cycles
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z12"',
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z12"',
    # 'pb_model = "01_m32x128x128_p64x192_F16_F_caws2i6c0o1t_caws2i6c0o1v_z13"',
]
groups['include_summs'] = [
    'do_include_summs = True',
]

############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 10
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# groups['carla_track1_data'] = [
#     'dataset_name = "carla"',
#     'H = %d' % H,
#     'W = %d' % W,
#     'trainset = "picked"',
#     'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
#     'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
#     'dataset_format = "npz"',
# ]
groups['carla_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cabs16i3c0o1v"',
    # 'dataset_list_dir = "/data/carla/npzs"',
    # 'dataset_location = "/data/carla/npzs"',
    'dataset_list_dir = "/projects/katefgroup/datasets/carla/processed/npzs"', 
    'dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"', 
    'dataset_format = "npz"',
    'max_iters = 2124',
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
