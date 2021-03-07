from exp_base import *

############## choose an experiment ##############

current = 'det_builder'
current = 'occ_trainer'
current = 'det_trainer'


mod = '"int00"' # 
mod = '"int01"' # summ rgbs
mod = '"int02"' # fastest logging
mod = '"int03"' # just_gif
mod = '"int04"' # show 3d inputs too
mod = '"int05"' # asdf
mod = '"int06"' # new bounds, defined as hyps
mod = '"int07"' # new bounds, defined as hyps
mod = '"int08"' # do shuf
mod = '"int09"' # proper lens
mod = '"int10"' # show depth
mod = '"int11"' # maxdepth 50
mod = '"int12"' # maxdepth 20
mod = '"int13"' # V = 100k
mod = '"int14"' # maxdepth 15
mod = '"int15"' # print stats
mod = '"int16"' # log vis
mod = '"int17"' # log vis
mod = '"int18"' # 20m limit
mod = '"int19"' # vis boxlist
mod = '"int20"' # do not rotate those boxes
mod = '"int21"' # use more colors again
mod = '"int22"' # 
mod = '"int23"' # bright colors
mod = '"int24"' # do not use +1 for indexing into frames_js
mod = '"int25"' # show boxes on depth
mod = '"int26"' # pad=0
mod = '"int27"' # neg yaw < not better
mod = '"int28"' # neg all < not better
mod = '"int29"' # no shuf; 1 frame per scene < nothing to see
mod = '"int30"' # seed 11
mod = '"int31"' # neg yaw; 7 iters, since the 7th frame is intersting
mod = '"int32"' # also use neg yaw in cam < this make the boxes not match position-wise
mod = '"int33"' # put -rx in the ref_T_obj thing
mod = '"int34"' # show occ_memRs, to ensure things are flat
mod = '"int35"' # add np.pi/2.0 to yaw < 
mod = '"int36"' # sub np.pi/2 < these do nothing, because the thing is a cube anyway
mod = '"int37"' # pos yaw < not better
mod = '"int38"' # eliminate double deg2rad
mod = '"int39"' # eliminate double deg2rad
mod = '"int40"' # shuf
mod = '"int41"' # shuf
mod = '"int42"' # allow occluders too
mod = '"int43"' # allow 10 frames
mod = '"int44"' # -pitch
mod = '"int45"' # stride 5 in the dat
mod = '"int46"' # no occluders
mod = '"int47"' # occlu; 4,.2,2
mod = '"int48"' # occlu; 4,.2,2
mod = '"int49"' # +80
mod = '"int50"' # +80
mod = '"int51"' # ordering
mod = '"int52"' # actually use the shit
mod = '"int53"' # do not shift < the box is centered at the object bottom
mod = '"int54"' # shift up half the height
mod = '"int55"' # shift up half the height
mod = '"int56"' # apply rotm then do it then unapply
mod = '"int57"' # apply rotm then do it then unapply
mod = '"int58"' # apply
mod = '"int59"' # use regular angles < no
mod = '"int60"' # -pitch, -roll
mod = '"int61"' # no shift
mod = '"int62"' # shift 
mod = '"int63"' # -yaw also
mod = '"int64"' # use -ry,-rz in generating the rotm
mod = '"int65"' # just -pitch
mod = '"int66"' # -offset
mod = '"int67"' # 
mod = '"int68"' # apply rotm.T
mod = '"int69"' # other way
mod = '"int70"' # wider bounds
mod = '"int71"' # first way
mod = '"int72"' # more
mod = '"int73"' # more
mod = '"int74"' # run featnet
mod = '"int75"' # 
mod = '"int76"' # show better occ_sup vsi
mod = '"int77"' # beter vis
mod = '"int78"' # print the stats again
mod = '"int79"' # tighter
mod = '"int80"' # show horz view
mod = '"int81"' # 1m down
mod = '"int82"' # 1m up
mod = '"int83"' # wi
mod = '"int84"' # clip the dpeth
mod = '"int85"' # keep all nonsky; keep some clipped sky
mod = '"int86"' # vis frree better
mod = '"int87"' # train a bit
mod = '"int88"' # fancier 
mod = '"int89"' # 1m up, to account for those objects near cam
mod = '"int90"' # start 0.01 in front of cam
mod = '"int91"' # zmax 8.01
mod = '"int92"' # dimbug fixed
mod = '"int93"' # zmax 4.01
mod = '"int94"' # zmax 4.01
mod = '"int95"' # zmax 2.01
mod = '"int96"' # zmax 2.01; builder
mod = '"int97"' # zmax 8.01

mod = '"int98"' # zmax 10.01
mod = '"int99"' # use a mindepth = 1e-4
mod = '"in100"' # ensure nonzero
mod = '"in101"' # no sky
mod = '"in102"' # 0.1 to 10.1 < fixed the floating dot artifact
mod = '"in103"' # bring sky back < ok still ok
mod = '"in104"' # show axboxlist_camXs
mod = '"in105"' # train detnet a bit
mod = '"in106"' # 
mod = '"in107"' # clip boxes to 1e-2, 10.0
mod = '"in108"' # clip boxes to 1e-2, 20.0
mod = '"in109"' # clip boxes to 1e-1, 20.0
mod = '"in110"' # clip boxes to 0.5, 20.0
mod = '"in111"' # clip boxes to 0.5, 20.0
mod = '"in112"' # only use the regular objects
mod = '"in113"' # use all objects, but do not compute map
mod = '"in114"' # 113 again
mod = '"in115"' # 112 again
mod = '"in116"' # 112 but log more frequently, to maybe accel the mem leak
mod = '"in117"' # P_thresh 0.5; only proceed into nms and vis if some inds exist
mod = '"in118"' # only proceed if not None
mod = '"in119"' # weak feat smooth coeff; standard logging
mod = '"in120"' # do nms on cpu then come back immediately
# mod = '"in121"' # like 120 but fastest logging, to see if i errro < seems ok after 200 iters
mod = '"in122"' # actually apply the weak smooth coeff


mod = '"ob00"' # use new single data 
mod = '"ob01"' # use vislist
mod = '"ob02"' # two tfrs
mod = '"ob03"' # ten tfrs
mod = '"ob04"' # do not rescore
mod = '"ob05"' # rescore conservatively
mod = '"ob06"' # shuf
mod = '"ob07"' # big cleanup
mod = '"ob08"' # big cleanup
mod = '"ob09"' #
mod = '"ob10"' # more data; include det 
mod = '"ob11"' # more data; cleaned more
mod = '"ob12"' # changed scope of det stuff; compute map with axbox 
mod = '"ob13"' # sparse convs
mod = '"ob14"' # fixed a pca bug
mod = '"ob15"' # full data


mod = '"time00"' # new ad data (stopped prematurely)
mod = '"time01"' # again
mod = '"time02"' # pret 200k 08_s1_m96x64x96_1e-3_F32s_D12_p1_r1_adst_adsv_time01 


############## define experiments ##############

exps['det_builder'] = [
    'intphys_det',
    'intphys_train_data_aws',
    'intphys_bounds',
    '10_iters',
    # '100_iters',
    # '1k_iters',
    'train_feat',
    'train_occ',
    # 'train_det',
    'B1',
    'no_shuf',
    # 'no_backprop',
    # 'faster_logging',
    'fastest_logging',
]
exps['occ_trainer'] = [
    'intphys_det', # mode
    'intphys_trainval_data_aws', # dataset
    'intphys_bounds',
    '10k_iters',
    'lr3',
    'B1',
    'train_feat',
    'train_occ',
    'faster_logging',
]
exps['det_trainer'] = [
    'intphys_det', # mode
    'intphys_trainval_data_aws', # dataset
    'intphys_bounds',
    '200k_iters',
    'lr4',
    'B8',
    'train_feat',
    # 'train_occ',
    'train_det',
    'pretrained_feat',
    'pretrained_det',
    'fast_logging', 
]

############## group configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_sparse_conv = True', 
    # 'feat_smooth_coeff = 0.01',
    # 'feat_do_rt = True',
    # 'feat_do_flip = True',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_det'] = [
    'do_det = True',
    'det_prob_coeff = 1.0',
    'det_reg_coeff = 1.0',
]


############## datasets ##############

# mem resolution
SIZE = 16
X = int(SIZE*6)
Y = int(SIZE*4)
Z = int(SIZE*6)

K = 3 # how many proposals to consider
N = 3 # how many objects are possible

S = 1
H = 288
W = 288
V = 50000
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['intphys_train_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "adst"',
    'dataset_list_dir = "/data/intphys/npzs"',
    'dataset_location = "/data/intphys/npzs"',
    'dataset_format = "npz"',
]
groups['intphys_trainval_data_aws'] = [
    'dataset_name = "intphys"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "adst"',
    'valset = "adsv"',
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
