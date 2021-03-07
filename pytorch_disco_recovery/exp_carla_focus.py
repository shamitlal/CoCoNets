from exp_base import *

############## choose an experiment ##############

# current = 'focus_builder'
current = 'focus_trainer'
# current = 'obj_trainer'
# current = 'obj_trainer'
# current = 'sta_builder'

mod = '"workitout00"' # 
mod = '"workitout01"' # 
mod = '"workitout02"' # 10k; not just gif 
mod = '"workitout03"' # also run obj-centric viewnet
mod = '"workitout04"' # import utils_basic
mod = '"workitout05"' # compute crop
mod = '"workitout06"' # do not use image aspect ratio
mod = '"workitout07"' # don't pad, to see if the crops match
mod = '"workitout08"' # align_corners=False< no,  not an option
mod = '"workitout10"' # extract crop also for the fake box2d
mod = '"workitout11"' # use a full size fake box
mod = '"workitout12"' # back to half box; drop the round from get_occupancy
mod = '"workitout13"' # clamp  to eliminate an eps
mod = '"workitout14"' # clamp  more things
mod = '"workitout15"' # don't do z+eps when it's numer
mod = '"workitout16"' # clamp
mod = '"workitout17"' # clamp 1e-4
mod = '"workitout18"' # depth=64
mod = '"workitout19"' # dept=128
mod = '"workitout20"' # train a bit for obj prediction
mod = '"workitout21"' # 
mod = '"workitout22"' # add embnet
mod = '"workitout23"' # add obj-centric embnet2d < no, bug. 
mod = '"workitout24"' # fix a bug (somehow i had 3 viewnets)
mod = '"workitout25"' # obj-centric embnet2d;  do obj projection at halfres, to fix shape issue
mod = '"workitout26"' # compute but do not bprop into obj losses, to see if they are being learned really
mod = '"workitout27"' # do bprop; also use embnet3D
mod = '"workitout28"' # 

############## define experiments ##############

exps['focus_builder'] = [
    'carla_focus', # mode
    'carla_focus10_data', # dataset
    '3_iters',
    'B1',
    'no_shuf',
    'no_backprop',
    'fastest_logging',
]
exps['focus_trainer'] = [
    'carla_focus', # mode
    'carla_focus10_data', # dataset
    '10k_iters',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'lr3',
    'B1',
    'faster_logging',
]


############## net configs ##############

groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_rt = True',
                        'feat_do_flip = True',
                        # 'feat_do_resnet = True',
                        # 'feat_do_sparse_invar = True',
]
groups['train_occ'] = ['do_occ = True',
                       'occ_coeff = 1.0', 
                       'occ_smooth_coeff = 2.0', 
]
groups['train_view'] = ['do_view = True',
                        'view_depth = 64',
                        'view_l1_coeff = 1.0',
]
groups['train_occ_notcheap'] = ['do_occ = True',
                                'occ_coeff = 1.0',
                                'occ_smooth_coeff = 0.1',
]
groups['train_emb2D'] = ['do_emb2D = True',
                         'emb_2D_smooth_coeff = 0.01', 
                         'emb_2D_ml_coeff = 1.0', 
                         'emb_2D_l2_coeff = 0.1', 
                         'emb_2D_mindist = 32.0',
                         'emb_2D_num_samples = 2', 
]
groups['train_emb3D'] = ['do_emb3D = True',
                         'emb_3D_smooth_coeff = 0.01', 
                         'emb_3D_ml_coeff = 1.0', 
                         'emb_3D_l2_coeff = 0.1', 
                         'emb_3D_mindist = 16.0',
                         'emb_3D_num_samples = 2', 
]
groups['train_sup_flow'] = ['do_flow = True',
                            'flow_heatmap_size = 3',
                            'flow_l1_coeff = 1.0',
                            # 'do_time_flip = True',
]
groups['train_unsup_flow'] = ['do_flow = True',
                              'flow_heatmap_size = 5',
                              'flow_warp_coeff = 1.0',
                              # 'flow_hinge_coeff = 1.0',
                              # 'flow_warp_g_coeff = 1.0',
                              'flow_cycle_coeff = 0.5', # 
                              'flow_smooth_coeff = 0.1',
                              # 'flow_do_synth_rt = True',
                              # 'flow_synth_l1_coeff = 1.0',
                              # 'flow_synth_l2_coeff = 1.0',
                              # 'do_time_flip = True',
                              # 'flow_cycle_coeff = 1.0',
]

############## datasets ##############

# DHW for mem stuff
SIZE = 40
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_focus1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0one"',
    # 'trainset = "cabs16i3c1o0one"',
    # 'trainset = "cabs16i3c0o1one"',
    # 'trainset = "picked"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
groups['carla_focus10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "cabs16i3c0o1ten"',
    'trainset = "caas7i6c1o0ten"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
groups['carla_focus_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0t"',
    # 'trainset = "cabs16i3c0o1t"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
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
