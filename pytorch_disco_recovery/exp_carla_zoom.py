from exp_base import *

############## choose an experiment ##############


current = '{}'.format(os.environ["exp_name"])
mod = '"{}"'.format(os.environ["run_name"]) 

############## define experiment ##############



exps['tester_pointfeat_dense'] = [
    'carla_zoom', # mode
    # 'carla_traj_test_data', # dataset
    # 'carla_drone_traj_test_data', # dataset
    'carla_complete_data_test_for_traj', # dataset
    # 'carla_regular_bounds',
    # 'carla_narrow_bounds',
    # 'narrow_test_bounds',
    # 'carla_narrower_test_bounds',
    # 'carla_narrow_test_bounds',
    # 'carla_16-16-16_bounds_test',
    'carla_16-16-16_bounds_test',
    # '400_iters',
    '50_iters',
    'make_dense',
    # 'make_dense',
    # '5_iters',
    'B1',
    'summ_pca_points_2d',
    'summ_pca_points_3d',
    'no_shuf',
    'do_test',
    'pretrained_feat3d',
    'pretrained_localdecoder',
    'pointfeat_ransac',
    'train_feat3d',
    # 'log100',
    'log100',
]


############## net configs ##############

groups['carla_zoom'] = ['do_carla_zoom = True']
groups['do_test'] = ['do_test = True']

groups['train_moc2D'] = [
    'do_moc2D = True',
    'moc2D_num_samples = 1000',
    'moc2D_coeff = 1.0',
]
groups['train_moc3d'] = [
    'do_moc3d = True',
    'moc3d_num_samples = 1000',
    'moc3d_coeff = 1.0',
]
groups['train_emb3d'] = [
    'do_emb3d = True',
    # 'emb_3d_ml_coeff = 1.0',
    # 'emb_3d_l2_coeff = 0.1',
    # 'emb_3d_mindist = 16.0',
    # 'emb_3d_num_samples = 2',
    'emb_3d_mindist = 16.0',
    'emb_3d_num_samples = 2',
    'emb_3d_ce_coeff = 1.0',
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

groups['train_feat3docc'] = [
    'do_feat3docc = True',
    'feat3d_dim = 64',
    'do_tsdf_implicit_occ = True',
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
    'occ_coeff = 0.1',
    'occ_smooth_coeff = 0.001',
]

############## datasets ##############

# 'XMIN = -8.0', # right (neg is left)
# 'XMAX = 8.0', # right
# 'YMIN = -1.0', # down (neg is up)
# 'YMAX = 3.0', # down
# 'ZMIN = 4.0', # forward
# 'ZMAX = 20.0', # forward

# dims for mem
SIZE = 16
Z = int(SIZE*3)
Y = int(SIZE*1)
X = int(SIZE*3)
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

ZZ = int(SIZE*3)
ZY = int(SIZE*3)
ZX = int(SIZE*3)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
dataset_location = "/data/carla/processed/npzs"

# dataset_location = "/data4/carla/processed/npzs"

SIZE_test = 8

groups['carla_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*16)),
    'Y = %d' % (int(SIZE*16)),
    'X = %d' % (int(SIZE*16)),
]
groups['carla_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*16)),
    'Y_test = %d' % (int(SIZE_test*16)),
    'X_test = %d' % (int(SIZE_test*16)),
]

groups['regular_train_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*1)),
    'X = %d' % (int(SIZE*8)),
]
groups['narrow_test_bounds'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -2.0', # down (neg is up)
    'YMAX_test = 2.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*4)),
    'Y_test = %d' % (int(SIZE_test*1)),
    'X_test = %d' % (int(SIZE_test*4)),
]


groups['carla_narrow_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_narrower_bounds'] = [
    'XMIN = -3.0', # right (neg is left)
    'XMAX = 3.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -3.0', # forward
    'ZMAX = 3.0', # forward
    'Z = 96', # forward
    'Y = 32', # forward
    'X = 96', # forward
]
groups['carla_narrow_test_bounds'] = [
    'XMIN_test = -4.0', # right (neg is left)
    'XMAX_test = 4.0', # right
    'YMIN_test = -1.0', # down (neg is up)
    'YMAX_test = 1.0', # down
    'ZMIN_test = -4.0', # forward
    'ZMAX_test = 4.0', # forward
    'Z_test = 128', # forward
    'Y_test = 32', # forward
    'X_test = 128', # forward
]
groups['carla_narrower_test_bounds'] = [
    'XMIN_test = -3.0', # right (neg is left)
    'XMAX_test = 3.0', # right
    'YMIN_test = -1.0', # down (neg is up)
    'YMAX_test = 1.0', # down
    'ZMIN_test = -3.0', # forward
    'ZMAX_test = 3.0', # forward
    'Z_test = 96', # forward
    'Y_test = 32', # forward
    'X_test = 96', # forward
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_big_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]


groups['carla_multiview_some_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabsome"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
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
groups['carla_multiview_train10_test10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "taus100i2ten"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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
    'testset = "taus100i2v"',
    'testset_format = "traj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taus100i2v"',
    'testset_format = "oldtraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_traj_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taus100i2t"',
    'testset_format = "oldtraj"', 
    'testset_consec = True', 
    'testset_seqlen = %d' % S_test, 
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

