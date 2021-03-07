from exp_base import *

############## choose an experiment ##############

current = 'builder'
# current = 'tester'

mod = '"ful00"' # reset exp list and logs
mod = '"ful01"' # show vis clean of zeroth hypothesis
mod = '"ful02"' # create func for vis
mod = '"ful03"' # collapse past   
mod = '"ful04"' # 
mod = '"ful05"' # refresh future
mod = '"ful06"' # print hypotheeses
mod = '"ful07"' # better collapse
mod = '"ful08"' # put things into camR0
mod = '"ful09"' # only show 30 steps, to eliminate the red dot < good
mod = '"ful10"' # rename s to step
mod = '"ful11"' # dummy loop; always select 1 
mod = '"ful12"' # show gif
mod = '"ful13"' # S_cap = 30
mod = '"ful14"' # simplify
mod = '"ful15"' # get yaws
mod = '"ful16"' # convert to boxes then lrts
mod = '"ful17"' # put len before rot
mod = '"ful18"' # extract right len
mod = '"ful19"' # subtract pi/2
mod = '"ful20"' # show gt too, to see if the alignment is right
mod = '"ful21"' # smooth the xyzs a bit < error
mod = '"ful22"' # smooth=1
mod = '"ful23"' # smooth=2
mod = '"ful24"' # show all k trajs
mod = '"ful25"' # -arctan - np.pi/2.
mod = '"ful26"' # make it begin straight, by yaw = yaw * (1.0-np.exp(-s))
mod = '"ful27"' # make it begin straight, using angle0
mod = '"ful28"' # -arctan+np.pi/2
mod = '"ful29"' # self.S_cap = 5
mod = '"ful30"' # self.S_cap = 20, but only show every other step in the bev box
mod = '"ful31"' # smooth=3 < ok good job
mod = '"ful32"' # moved util to utils_geom
mod = '"ful33"' # hypotheses are lrts; show the zeroth one 
mod = '"ful34"' # do collapse and refresh
mod = '"ful35"' # summ one
mod = '"ful36"' # show boxes
mod = '"ful37"' # eliminate that averaging/smoothing step, since raw angles are not really available here
mod = '"ful38"' # use past to fuel the future
mod = '"ful39"' # use the latest xyz
mod = '"ful40"' # slightly cleaner impl
mod = '"ful41"' # visualize the trajs as boxes, on the last step i guess
mod = '"ful42"' # visualize the trajs as boxes, on the zeroth  step
mod = '"ful43"' # do not collapse/refresh
mod = '"ful44"' # only collapse on step0, with the 
mod = '"ful45"' # if conf>0.1, collapse with this
mod = '"ful46"' # S_cap = 40
mod = '"ful47"' # use gt position to centralize the vis
mod = '"ful48"' # 0.05 threhs
mod = '"ful49"' # visualize each traj's boxes
mod = '"ful50"' # show the the hypo boxes on each step within the main vis 
mod = '"ful51"' # JUST show the boxes
mod = '"ful52"' # JUST show the boxes proper
mod = '"ful53"' # S_cap = 60
mod = '"ful54"' # show boxes on top of trajs
mod = '"ful55"' # avoid singularity
mod = '"ful56"' # thresh 0.03
mod = '"ful57"' # show gt box under the others
mod = '"ful58"' # thresh 0.05 again
mod = '"ful59"' # show the scores in bev
mod = '"ful60"' # compute match/conf of each try
mod = '"ful61"' # eliminate the confidence in the summ, since it's confusing; S_cap = 80
mod = '"ful62"' # only show traj_g up to S_cap
mod = '"ful63"' # show conf of the best guy
mod = '"ful64"' # set end of trajs; use .3f for the scores
mod = '"ful65"' # don't set conf to 1.0 after collapse
mod = '"ful66"' # set hyp0 to be zero mot
mod = '"ful67"' # use color0 for gt
mod = '"ful68"' # use color19
mod = '"ful69"' # use color1 for 
mod = '"ful70"' # use 2: for e
mod = '"ful71"' # 20 iters
mod = '"ful72"' # S_cap=10, just to see pers
mod = '"ful73"' # fix shape issue
mod = '"ful74"' # show box scores; fix return isue
mod = '"ful75"' # S_cap = 80; 5 iters; no shuf
mod = '"ful76"' # conf_thresh = 0.04
mod = '"ful77"' # no shuf
mod = '"ful78"' # replace each hypothesis with the matcher's best answer
mod = '"ful79"' # in topleft corner write the frame number
mod = '"ful80"' # cap 10 for see
mod = '"ful81"' # cap 10 for see
mod = '"ful82"' # 0,0
mod = '"ful83"' # write it in not black, at 20,10
mod = '"ful84"' # 10,20
mod = '"ful85"' # S_cap = 90
mod = '"ful86"' # put the frame number in white
mod = '"ful87"' # cap 10 for a sec
mod = '"ful88"' # cap 90
mod = '"ful89"' # 5,20
mod = '"ful90"' # cap 90
mod = '"ful91"' # re-run, just to see
mod = '"ful92"' # pret share82, so that i have motionreg happening
mod = '"ful93"' # S_test=30, to speed up
mod = '"ful94"' # S_test=30 for real
mod = '"ful95"' # confnet, pret with 30k 04_s2_m256x128x256_z64x64x64_1e-4_F3f_d64_Mf_c1_r.1_C_c1_tals90i1t_tals90i1v_conf37
mod = '"ful96"' # print more; 
mod = '"ful97"' # thresh 0.8
mod = '"ful98"' # thresh 0.7
mod = '"ful99"' # an data; S_cap = 60
mod = '"ful100"' # ap data
mod = '"ful101"' # skip the track/whatever
mod = '"ful102"' # show the gt boxes
mod = '"ful103"' # use correct pix_T_cam
mod = '"ful104"' # show 90 frames
mod = '"ful105"' # 60 frames; tans60i2a
mod = '"ful106"' # matrix; tap data; S_test = 100; S_cap = 100; 

mod = '"pip00"' # pret 20k 02_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f26_k5_e100_w.001_taps100i2tb_taps100i2vce_mat28 < 121s
mod = '"pip01"' # elim rgb_camXs vis < ok 78s; those summs are expensive
mod = '"pip02"' # really run teh model
mod = '"pip03"' # return early; summ every second step < ok 46s
mod = '"pip04"' # return early; summ every second step < ok 46s
mod = '"pip05"' # really run the mdoel
mod = '"pip06"' # clean up
mod = '"pip07"' # val set
mod = '"pip08"' # again
mod = '"pip09"' # show full traj
mod = '"pip10"' # use 1.0 scores for past
mod = '"pip11"' # use the ans to update the hyp for the next timestep, even if not past conf thresh
mod = '"pip12"' # print more, because i am confused
mod = '"pip13"' # show past boxes UNDER the hyps  
mod = '"pip14"' # only update the ans for hyp0
mod = '"pip15"' # shuffle, to see more
mod = '"pip16"' # update hypothesis0 greedily
mod = '"pip17"' # eliminate that confidence hack
mod = '"pip18"' # again, on this new day
mod = '"pip19"' # no shuf
mod = '"pip20"' # pret mat28
mod = '"pip21"' # pret 60k mat30
mod = '"pip22"' # avoid updating 0 with the other guys
mod = '"pip23"' # same but: do NOT use multiple hypotheses; attempt to use basic matchnet < ok looks fine
mod = '"pip24"' # evaluate
mod = '"pip25"' # export ious and vis; S_cap = 30, for quicker bug
mod = '"pip26"' # S_cap = 100
mod = '"pip27"' # use all hypotheses # 599 seconds
mod = '"pip28"' # do not vis so much # 180 seconds. ok good.
mod = '"pip29"' # pret 20k 04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f36_k5_e1_w.001_taps100i2tb_taps100i2vce_mat32
mod = '"pip30"' # do vis
mod = '"pip31"' # conf thresh 0.5
mod = '"pip32"' # pret instead 26k 04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f36_k5_e100_w.001_taps100i2tb_taps100i2vce_mat32 < ok, a bit better
mod = '"pip33"' # 20 iters; vis every 5
mod = '"pip34"' # pret 30k instead of 26k; no vis
mod = '"pip35"' # K=1
mod = '"pip36"' # replicate pip33, by pret 26k and allowing K
mod = '"pip37"' # 50 iters
mod = '"pip38"' # conf = 0.7 instead of 0.5; log10 < yes, this is slightly better than using 0.5
mod = '"pip39"' # conf = 0.5
mod = '"pip40"' # pret 50k 04_s2_m256x128x256_z64x64x64_1e-3_F3f_d64_Mf_c1_r.1_Mrd_p4_f36_k5_e100_w.001_taps100i2tb_taps100i2vce_mat32
mod = '"pip41"' # K = 1; conf = 0.7
mod = '"pip42"' # back to 26k ckpt; full K; ONLY update lrt for k==0 < much worse actually
mod = '"pip43"' # update on each iter (reverting pip42); save time, with thresh 0.5 on the centroid diff
mod = '"pip44"' # thresh 0.1 on the centroid diff
mod = '"pip45"' # thresh 0.2 on the centroid diff
mod = '"pip46"' # thresh 0.5 on centroid diff AND 3 degrees thresh for rot diff
mod = '"pip47"' # thresh 0.5 on centroid diff AND 6 degrees thresh for rot diff < ok looks fine
mod = '"pip48"' # add empty template strat
mod = '"pip49"' # add template tool
mod = '"pip50"' # add the first three good steps as templates
mod = '"pip51"' # refresh templates according to conf
mod = '"pip52"' # just 2 templates
mod = '"pip53"' # 4 templates; print more < turns out slightly worse than 3 templates
mod = '"pip54"' # 3 templates; show the updates in tb < winner. this should be equivalent to pip51, but it's not.'
mod = '"pip55"' # better summs; log5
mod = '"pip56"' # only export vis if log_this
mod = '"pip57"' # show gt at least
mod = '"pip58"' # compute careful lrt
mod = '"pip59"' # more dat
mod = '"pip60"' # more dat
mod = '"pip61"' # packed up the box parser


############## define experiments ##############

exps['builder'] = [
    'carla_pipe', # mode
    # 'carla_tag_t_tce_vce_data', # dataset
    # 'carla_tag_vce_data', # dataset
    # 'carla_tal_v_data', # dataset
    # 'carla_tan_v_data', # dataset
    # 'carla_tap_t_data', # dataset
    # 'carla_tap_v_data', # dataset
    # 'carla_tap_vce_data', # dataset
    'carla_taq_a_data', # dataset
    'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '5_iters',
    'pretrained_match', 
    'pretrained_feat3D', 
    'pretrained_motionreg', 
    'pretrained_conf', 
    'train_feat3D',
    'train_motionreg',
    'train_match',
    'train_conf',
    'no_backprop',
    'no_shuf',
    'B1',
    # 'lr4', 
    'log1',
]
exps['tester'] = [
    'carla_pipe', # mode
    'carla_tap_vce_data', # dataset
    'train_on_trainval',
    'nearcube_trainvaltest_bounds', 
    '50_iters',
    'pretrained_match', 
    'pretrained_feat3D', 
    'pretrained_motionreg', 
    'pretrained_conf', 
    'train_feat3D',
    'train_motionreg',
    'train_match',
    'train_conf',
    'frozen_feat3D', 
    'frozen_match', 
    'frozen_conf', 
    'frozen_motionreg', 
    'do_test', 
    'do_export_stats', 
    'do_export_vis', 
    'no_backprop',
    'no_shuf',
    'B1',
    # 'lr4', 
    # 'log1',
    'log5',
    # 'log500',
]

############## group configs ##############

groups['do_test'] = ['do_test = True']
groups['do_export_vis'] = ['do_export_vis = True']
groups['do_export_stats'] = ['do_export_stats = True']

groups['train_motionreg'] = [
    'do_motionreg = True',
    'motionreg_t_past = 4',
    # 'motionreg_t_futu = 26',
    'motionreg_t_futu = 36',
    'motionreg_num_slots = 5',
    # 'motionreg_l1_coeff = 1.0',
    # 'motionreg_l2_coeff = 0.1',
    # 'motionreg_weak_coeff = 0.0001',
]
groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
]
groups['train_match'] = [
    'do_match = True',
    'match_coeff = 1.0', 
    'match_r_coeff = 0.1', 
]
groups['train_conf'] = [
    'do_conf = True',
    'conf_coeff = 1.0', 
    # 'conf_num_replicas = 4', 
]


############## datasets ##############

# mem resolution
SIZE = 64
SIZE_val = 64
SIZE_test = 64
# X = int(SIZE*32)
# Y = int(SIZE*4)
# Z = int(SIZE*32)
# Z = SIZE*4
# Y = SIZE*1
# X = SIZE*4

ZZ = 64
ZY = 64
ZX = 64

# ZX = int(SIZE*32)
# ZY = int(SIZE*4)
# ZZ = int(SIZE*32)

# these params need cleaning; also, 3 only works if you do not count occluders
N = 3 # max objects
K = 3 # objects to consider 

S = 100
S_val = 100
S_test = 100
H = 128
W = 384
V = 50000
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/data4/carla/processed/npzs"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"

groups['carla_tag_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tags90i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tags90i1v"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tag_t_tce_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tags90i1t"',
    'trainset_format = "traj"', 
    'trainset_consec = True', 
    'trainset_seqlen = %d' % S, 
    'valset = "tags90i1tce"',
    'valset_format = "traj"', 
    'valset_consec = True', 
    'valset_seqlen = %d' % S, 
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tag_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tags90i1vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tal_v_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tals90i1v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tan_v_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tans60i2a"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_v_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taps100i2v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_vce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taps100i2vce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taq_a_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2a"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tap_t_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taps100i2t"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tag_vocc_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tags90i1vocc"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tag_tce_data'] = [
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tags90i1tce"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['nearcube_trainvaltest_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*4)),
    'Y = %d' % (int(SIZE*2)),
    'X = %d' % (int(SIZE*4)),
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*4)),
    'Y_val = %d' % (int(SIZE_val*2)),
    'X_val = %d' % (int(SIZE_val*4)),
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*2)),
    'X_test = %d' % (int(SIZE_test*4)),
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
