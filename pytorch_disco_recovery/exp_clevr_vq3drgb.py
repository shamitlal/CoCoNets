from exp_base import *

############## choose an experiment ##############


current = 'builder'
current = 'trainer'
# current = 'tester'

# at 16x16x16 resolution, it takes 5.2 seconds, so 787v/s
# at 32x32x32 resolution, i expect it to take 42 seconds

mod = '"00"' # show rgb data
mod = '"01"' # show 3d data 
mod = '"02"' # use proper hyps for the data
mod = '"03"' # bigger vox
mod = '"04"' # maxval 15 for depth vis
mod = '"05"' # show occRs too < some small calibration error visible, but maybe we can ignore this
mod = '"06"' # introduce some nets
mod = '"07"' # train a bit
mod = '"08"' # discard alpha
mod = '"09"' # train a bit
mod = '"10"' # don't log so much
mod = '"11"' # train occ too
mod = '"12"' # train vq part 
mod = '"13"' # num_embeddings as hyp; set it to 64 instead of 512
mod = '"14"' # print net, to see if the params are there and visible < indeed it's there. so i just messed up in the sparse version somehow
mod = '"15"' # simpler net
mod = '"16"' # v2v net; B=1
mod = '"17"' # no front or back layers; just encoderdecoder, minus one upsamp < nice features! less square artifacts in bird view
mod = '"18"' # go to quarter res instead
mod = '"19"' # pixelshuffle in occnet < bad
mod = '"20"' # pixelshuffle right after quant, so that occnet and viewpred both use it < bad
mod = '"21"' # summ the upsampled quants; don't do occ < good pictures, good job of using the dictionary
mod = '"22"' # skip that upsamp < still good rgb, but somehow the dict is less-used
mod = '"24"' # S=2 again; even lower res (1/16); do upsamp; do use a val set < noticeably lower quality, but OK

# i moved the pret stuff to exp_clevr_gen3dvq


mod = '"r00"' # add rendernet (and occ)
mod = '"r01"' # put feat things in X1
mod = '"r02"' # run occ  and render
mod = '"r03"' # just occ
mod = '"r04"' # train a bit < why so slow?
mod = '"r05"' # diff gpu
mod = '"r06"' # S=2,B=1
mod = '"r07"' # no occ
mod = '"r08"' # yes occ; render a bit
mod = '"r09"' # builder on aws
mod = '"r10"' # trainer
mod = '"r11"' # run rendernet
mod = '"r12"' # train on 10, to see the artifacts settle
mod = '"r13"' # train on 1
mod = '"r14"' # no occ loss 
mod = '"r15"' # symlink; occ loss but stronger rgb loss
mod = '"r16"' # ten
mod = '"r17"' # occ+view 
mod = '"r18"' # 8 data workers < hm, something is slow
mod = '"r19"' # 4 workers, pin=False
mod = '"r20"' # 4 workers, pin=True < slow!
mod = '"r22"' # some data on data2 < ok, smaller drive does seem faster
mod = '"r23"' # more some data on data2
mod = '"r25"' # rendernet at halfres 
mod = '"r26"' # bigger hyps
mod = '"r28"' # new viewnet arch, with some resnet inside < indeed, this is faster & better (on train) than r26
mod = '"r29"' # deeper viewnet: 6 residual instead of 3 < similar to r28
mod = '"r30"' # prep layer does 1x1 and relu and batchnorm, instead of just 1x1 < slightly worse than r29
mod = '"r31"' # 1x1x1 then proj then 3x3x3 < not better than r30
mod = '"r32"' # 1x1x1, relu, batchnorm, proj, 2D, like r29, but now decoder has a conv transpose, so we predict full res
mod = '"r33"' # project to 1/4 res instead of 1/2, and use one more deconv
mod = '"r34"' # S=7
mod = '"r35"' # r34 with vq
mod = '"r36"' # r34 with vqema < slower to get going, and later seems equivalent. anyway i guess we can keep it. 
mod = '"r38"' # cleaned up; run a basic feat2net
mod = '"r39"' # use feat2 as additional unproj input < indeed slightly better than r36, but slower
mod = '"r40"' # unp *= occ, feat2 *= occ < seems slightly worse actually
mod = '"r41"' # aggregate views[1:] BEFORE encoding with 3d featnet < surprisingly, not much faster than before, and not better either
mod = '"r42"' # shallower 2d v2v; no upsampling
mod = '"r43"' # back to normal 2d v2v; lower voxel resolution 
mod = '"r44"' # eliminate l2 norm from featnet < descends more smoothly, but somehow uses the dict less
mod = '"r45"' # do not mult inputs by occ < ok seems better! uses the dict more.
mod = '"r46"' # convtranspose3d to start viewnet < good
mod = '"r47"' # lower l2 coeff (1.0)
mod = '"r48"' # bring back the l2 normalization, so that the dict lies in unit space < no that's bad after all
mod = '"r49"' # higher res

############## define experiment ##############

exps['builder'] = [
    'clevr_vq3drgb', # mode
    'clevr_multiview_train_data', # dataset
    'clevr_bounds', 
    '3_iters',
    'lr0',
    'B1',
    # 'no_shuf',
    'train_feat', 
    'train_occ', 
    # 'train_render', 
    # 'train_view', 
    'fastest_logging',
]
exps['trainer'] = [
    'clevr_vq3drgb', # mode
    # 'clevr_multiview_train_data', # dataset
    # 'clevr_multiview_train10_data', # dataset
    'clevr_multiview_train1_data', # dataset
    # 'clevr_multiview_train_val_data', # dataset
    'clevr_bounds', 
    '100k_iters',
    'lr3',
    'B8',
    'train_feat2', 
    'train_feat', 
    # 'train_occ', 
    # 'train_render', 
    'train_view', 
    'train_vq3drgb',
    'fast_logging',
]
exps['tester'] = [
    'clevr_vq3drgb', # mode
    'clevr_multiview_train_data', # dataset
    'clevr_bounds', 
    '10_iters',
    'lr0',
    'B1',
    'pretrained_feat', 
    'pretrained_vq3drgb', 
    'pretrained_view', 
    'fastest_logging',
]

############## net configs ##############

groups['clevr_vq3drgb'] = ['do_clevr_vq3drgb = True']

groups['train_feat2'] = [
    'do_feat2 = True',
    'feat2_dim = 32',
]
groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 64',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_num_embeddings = 512',
    'vq3drgb_latent_coeff = 1.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.1',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l2_coeff = 1.0',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 32',
    'render_rgb = True',
    'render_l2_coeff = 10.0',
]
groups['train_gen3dvq'] = [
    'do_gen3dvq = True',
    'gen3dvq_coeff = 1.0',
    # 'vqrgb_smooth_coeff = 2.0',
]

############## datasets ##############

# dims for mem
SIZE = 16
Z = int(SIZE*4)
Y = int(SIZE*4)
X = int(SIZE*4)
S = 7
H = 256
W = 256
V = 65536
# H and W for proj stuff
PH = int(H/4.0)
PW = int(W/4.0)
# PH = int(H/2.0)
# PW = int(W/2.0)
# PH = int(H)
# PW = int(W)
 

# there seem to be 24 views per pickle
groups['clevr_multiview_train1_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    # 'trainset = "bgone"',
    'trainset = "some"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    # 'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    # 'dataset_location = "/data/clevr_veggies/npys"',
    'dataset_location = "/data2/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train10_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    # 'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    # 'dataset_location = "/data/clevr_veggies/npys"',
    'dataset_location = "/data2/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    # 'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    # 'dataset_location = "/data/clevr_veggies/npys"',
    'dataset_location = "/data2/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train_val_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "bgv"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    # 'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    # 'dataset_location = "/data/clevr_veggies/npys"',
    'dataset_location = "/data2/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
]
groups['clevr_multiview_train_val_test_data'] = [
    'dataset_name = "clevr"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "bgt"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "bgv"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "bgv"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    # 'dataset_location = "/home/aharley/datasets/clevr_veggies/npys"',
    # 'dataset_location = "/data/clevr_veggies/npys"',
    'dataset_location = "/data2/clevr_veggies/npys"',
    'dataset_filetype = "pickle"'
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
