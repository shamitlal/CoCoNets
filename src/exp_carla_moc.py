from exp_base import *
import ipdb 
st = ipdb.set_trace
# the idea here is to train with momentum contrast (MOC), in a simple clean way for eccv

############## choose an experiment ##############

current = '{}'.format(os.environ["exp_name"])
mod = '"{}"'.format(os.environ["run_name"]) 

############## define experiment ##############


exps['trainer_implicit'] = [
    'carla_moc', # mode
    'carla_complete_data_train_for_multiview',
    'carla_16-16-16_bounds_train',
    # 'carla_16-8-16_bounds_train',
    # 'carla_16-16-16_bounds_train',
    # 'carla_32-32-32_bounds_train',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    '500k_iters',
    'lr4',
    'B2',
    'train_feat3d',
    # 'train_occ',
    'train_emb3d',
    'train_localdecoder',
    'log100',
    'snap5k', 
]


############## net configs ##############


groups['carla_moc'] = ['do_carla_moc = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3D'] = [
    'do_moc3D = True',
    'moc3D_num_samples = 1000',
    'moc3D_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    # 'emb3d_ml_coeff = 1.0',
    # 'emb3d_l2_coeff = 0.1',
    # 'emb3d_mindist = 16.0',
    # 'emb3d_num_samples = 2',
    'emb3d_mindist = 16.0',
    'emb3d_num_samples = 2',
    'emb3d_ce_coeff = 1.0',
]
groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 32',
    'feat2D_smooth_coeff = 0.01',
]
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 64',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    # 'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
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

