from exp_base import *

############## choose an experiment ##############

# current = 'obj_builder'
current = 'obj_trainer'
# current = 'obj_trainer'
# current = 'sta_builder'

mod = '"wasted00"' # 
mod = '"wasted01"' # load the thing
mod = '"wasted02"' # load the thing better
mod = '"wasted03"' # load the real thing and use it as init
mod = '"wasted04"' # better vis of the outs
mod = '"wasted05"' # not optim_init < good performance!!!
mod = '"wasted06"' # show masks from track init
mod = '"wasted07"' # better vis; higher lr
mod = '"wasted08"' # load flow_memRs and print shape
mod = '"wasted09"' # show flow_memRs < good performance!!! practically this is 05 but higher lr.
mod = '"wasted10"' # use these newer flows < bad! why i wonder?
mod = '"wasted11"' # use occ in masked mean < similar and maybe worse
mod = '"wasted12"' # 0.1 coeff on flow_l2 < even worse. these new flows are hurting.
mod = '"wasted13"' # 0.001 coeff on flow_l2
mod = '"wasted14"' # go back to old flow for the loss, just to see
mod = '"wasted15"' # extract flow_t the other way
mod = '"wasted16"' # use 0.01 coeff
mod = '"wasted17"' # no ml, for smoother and nonrandom descent < working well, especially at lr3
mod = '"wasted18"' # back to new flows < better at the start
mod = '"wasted19"' # use halfres object for assembly < seems better at first, but then takes a dive. 

# there is some weird hump at the beginning, where other losses are higher. this might be occ smoothness



mod = '"wasted20"' # turn off occ losses (leaving only view); my hypothesis is this will be bad

mod = '"wasted21"' # do nothing but show occXs_crop
mod = '"wasted22"' # 5m pad
mod = '"wasted23"' # pad less in vert dim
mod = '"wasted24"' # assemble double and cropped feat
mod = '"wasted25"' # backprop into this
mod = '"wasted26"' # backprop into this
mod = '"wasted27"' # train zoomed occ
mod = '"wasted28"' # also train l2 ml; add suffix for occnet
mod = '"wasted29"' # like wasted28 but no crop

############## define experiments ##############

exps['obj_builder'] = ['carla_obj', # mode
                       'carla_obj1_data', # dataset
                       '3_iters',
                       'test_feat',
                       'pretrained_feat',
                       'frozen_feat',
                       'lr4',
                       'B1',
                       'no_shuf',
                       'no_backprop',
                       'fastest_logging',
]
exps['obj_trainer'] = ['carla_obj', # mode
                       'carla_obj1_data', # dataset
                       '20k_iters',
                       'test_feat',
                       'pretrained_feat',
                       'frozen_feat',
                       'train_view',
                       'pretrained_view',
                       'frozen_view',
                       'train_emb2D',
                       'pretrained_emb2D',
                       'frozen_emb2D',
                       'train_emb3D', # nothing to pret or freeze here
                       'train_occ',
                       'pretrained_occ',
                       'frozen_occ',
                       # 'total_init',
                       # 'quick_snap',
                       'lr4',
                       # 'pretrained_obj',
                       # 'pretrained_bkg',
                       # 'pretrained_optim',
                       'B1',
                       'no_shuf',
                       'faster_logging',
]
exps['sta_builder'] = ['carla_sta', # mode
                       # 'carla_stat_stav_data', # dataset
                       'carla_sta10_data', # dataset
                       '3_iters',
                       # '20_iters',
                       'lr0',
                       'B1',
                       'no_shuf',
                       'train_feat',
                       'train_occ',
                       'train_view',
                       'train_emb2D',
                       'train_emb3D',
                       'fastest_logging',
                       # 'slow_logging',
]


############## net configs ##############

groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_rt = True',
                        'feat_do_flip = True',
                        # 'feat_do_resnet = True',
                        'feat_do_sparse_invar = True',
]
groups['test_feat'] = ['do_feat = True',
                       'feat_dim = 32',
                       # 'feat_do_resnet = True',
                       # 'feat_do_sparse_invar = True',
]
groups['train_occ'] = ['do_occ = True',
                       'occ_coeff = 1.0', 
                       'occ_smooth_coeff = 2.0', 
]
groups['train_view'] = ['do_view = True',
                       'view_depth = 32',
                       'view_l1_coeff = 1.0',
]
groups['train_occ_notcheap'] = ['do_occ = True',
                                'occ_coeff = 1.0',
                                'occ_smooth_coeff = 0.1',
]
groups['train_emb2D'] = ['do_emb2D = True',
                         'emb_2D_smooth_coeff = 0.01', 
                         # 'emb_2D_ml_coeff = 1.0', 
                         'emb_2D_l2_coeff = 0.1', 
                         'emb_2D_mindist = 32.0',
                         'emb_2D_num_samples = 2', 
]
groups['train_emb3D'] = ['do_emb3D = True',
                         'emb_3D_smooth_coeff = 0.01', 
                         # 'emb_3D_ml_coeff = 1.0', 
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
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 5
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_sta1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caus2i6c1o0one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_format = "tf"',
]
groups['carla_sta10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caus2i6c1o0ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_sta_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caus2i6c1o0t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_format = "tf"',
]
# groups['carla_stat_stav_data'] = ['dataset_name = "carla"',
#                                   'H = %d' % H,
#                                   'W = %d' % W,
#                                   'trainset = "caus2i6c1o0t"',
#                                   'valset = "caus2i6c1o0v"',
#                                   'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
#                                   'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
#                                   'dataset_format = "tf"',
# ]
groups['carla_flo1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caws2i6c0o1one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_format = "tf"',
]
groups['carla_flo10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caws2i6c0o1ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_flot_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caws2i6c0o1t"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_format = "tf"',
]
groups['carla_flot_flov_data'] = ['dataset_name = "carla"',
                                  'H = %d' % H,
                                  'W = %d' % W,
                                  'trainset = "caws2i6c0o1t"',
                                  'valset = "caws2i6c0o1v"',
                                  'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_format = "tf"',
]
groups['carla_flov_flov_data'] = ['dataset_name = "carla"',
                                  'H = %d' % H,
                                  'W = %d' % W,
                                  'trainset = "caws2i6c0o1v"',
                                  'valset = "caws2i6c0o1v"',
                                  'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                                  'dataset_format = "tf"',
]
groups['carla_stat_stav_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0t"',
    'valset = "caas7i6c1o0v"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"'
]
groups['carla_quicktest_data'] = [
    'dataset_name = "carla"',
    'testset = "quicktest9"', # sequential
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
    'dataset_format = "tf"',
]
groups['carla_obj1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             # 'trainset = "cabs16i3c0o1one"',
                             'trainset = "picked"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
                             'dataset_format = "npz"',
]
groups['carla_obj10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caws2i6c0o1ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_format = "tf"',
]
groups['carla_obj_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            'trainset = "caws2i6c0o1t"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_format = "tf"',
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
