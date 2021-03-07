from exp_base import *

############## choose an experiment ##############

current = 'builder'
current = 'trainer'

mod = '"bench00"' # nothing
mod = '"bench01"' # train some
mod = '"bench02"' # train full
mod = '"bench03"' # fast logging; 200k iters
mod = '"bench04"' # builder; make an empty test mode, with traj data
mod = '"bench05"' # show feat_memXs
mod = '"bench06"' # use seqlen=5 at test time
mod = '"bench07"' # memX0
mod = '"bench08"' # get the affinity matrix per voxel
mod = '"bench09"' # compute object boxes
mod = '"bench10"' # do not make the boxes uniform size
mod = '"bench11"' # show object crop
mod = '"bench12"' # show occ object crop too
mod = '"bench14"' # higher res crop, to see something < ok, i believe these are objects
mod = '"bench15"' # show heatmap
mod = '"bench16"' # convert heatmaps to soft argmaxes, and show thsoe
mod = '"bench17"' # show traj
mod = '"bench18"' # make boxes uniform size
mod = '"bench19"' # compute features after cropping from the thing
mod = '"bench20"' # norot boxes
mod = '"bench21"' # pret
mod = '"bench22"' # no mult by 10
mod = '"bench23"' # sum instead of mean
mod = '"bench24"' # better summs
mod = '"bench25"' # wider zoom
mod = '"bench26"' # take the mean of the argmaxes, rather than the argmax of the mean heat < leads to near zero
mod = '"bench27"' # back to the other way
mod = '"bench28"' # hard argmax
mod = '"bench29"' # crop obj_feat out of the tensor < ok, this looks better
mod = '"bench30"' # hard=False
mod = '"bench31"' # get true norot boxes
mod = '"bench32"' # use proper centers
mod = '"bench33"' # show gt traj too
mod = '"bench34"' # newer ckpt
mod = '"bench35"' # compute and plot centroid dist in cam coords
mod = '"bench36"' # compute mean and median
mod = '"bench37"' # quarter res sampling
mod = '"bench38"' # hard argmax
mod = '"bench39"' # argmax_of_mean=False
mod = '"bench40"' # coeff ZZ < ok, slightly fixed 
mod = '"bench41"' # coeff Z; B2
mod = '"bench42"' # argmax_of_mean=True; mult by occ
mod = '"bench43"' # no mult by occ < pretty similar
mod = '"bench44"' # train again, and measure centroid error in test, to see it go down
mod = '"bench45"' # builder with new 40k ckpt instead of 30k < not better on this debug set. ok no big deal.
mod = '"bench46"' # no pret < yes, strictly worse, but not by much!!

# i think the resampling is hurting things a lot
# i can fix this, right? how about nn sampling?

mod = '"bench47"' # nn sampling
mod = '"bench48"' # pret
mod = '"bench49"' # use nn for the input (vis) too 
mod = '"bench50"' # hard argmax!!
mod = '"bench51"' # hard argmax!!; argmaxof_mean=Fasle < nope, not better

# is there still a chance that the resampling is hurting me?
# i don't think so... the nn sampling gives the same features i would get from hard indexing, though a different set of them
# but ok can i try that hard walk very very briefly please

mod = '"bench52"' # hard step through obj vectors, via indexing
mod = '"bench53"' # fix bug in mask gen/resolution; mult mask by occ
mod = '"bench54"' # hard=True < worse
mod = '"bench55"' # B1, to see
mod = '"bench56"' # upsample before argmax
mod = '"bench57"' # back to Z4; normalize before vis and sum and heat
mod = '"bench58"' # normalize the other dim
mod = '"bench59"' # F.relu(corr_)
mod = '"bench60"' # use a single voxel on the object
mod = '"bench61"' # show all locs that have the argmax value
mod = '"bench62"' # visualize and measure the discretization error
mod = '"bench63"' # *Z
mod = '"bench64"' # use the moving mask for gt
mod = '"bench65"' # train a long time


mod = '"bench66"' # builder again
mod = '"bench67"' # better scopes
mod = '"bench68"' # better scopes
mod = '"bench69"' # cleaned up and fixed bugs
mod = '"bench70"' # use true boxes
mod = '"bench71"' # clean up more; B2
mod = '"bench72"' # show point counts
mod = '"bench73"' # show mean point count
mod = '"bench74"' # get total mean iou 
mod = '"bench75"' # train a long time
mod = '"bench76"' # inspect summ bug
mod = '"bench77"' # return early if less than 8 pts in the mask
mod = '"bench78"' # return early if less than 8 pts in mask0
mod = '"bench79"' # fast logging
mod = '"bench80"' # S=10 for test
mod = '"bench81"' # S_test = 10 (proper coeff)
mod = '"bench82"' # just_gif on test

mod = '"res00"' # smaller bounds
mod = '"res01"' # use obj c for delta
mod = '"res02"' # -delta
mod = '"res03"' # pos delta; centered mem bounds
mod = '"res04"' # show R0 < ok, looks fine.
mod = '"res04"' # get random scene centroids on training iters
mod = '"res05"' # fixed bug in centroid computation 
mod = '"res06"' # nothing
mod = '"res07"' # use R0 for test
mod = '"res08"' # fixed a var name
mod = '"res09"' # switch to R0 for test
mod = '"res10"' # feed vox_util to the summ writer
mod = '"res11"' # use obj_clis_e instead of rough centroids
mod = '"res12"' # track over 8 frames instead of 10, to see if it stays inbounds
mod = '"res13"' # again
mod = '"res14"' # carla_regular_bounds, so that resolution is effectively lower
mod = '"res15"' # retry
mod = '"res16"' # narrower centroid range
mod = '"res17"' # narrower centroid range: 1-2
mod = '"res18"' # ok train
mod = '"res19"' # make vox util centroid generalize over B
mod = '"res20"' # try B=2
mod = '"res21"' # set B=1 for test iters
mod = '"res22"' # fixed a bug where hyp.S got used
mod = '"res23"' # train a bit < worse than before; is it the randomized centroid, or the resolution (which was a bug, causing noncube vox), or eval in R0?
mod = '"res24"' # lower Y res to fix the bug
# i am at a stage where i want to try many things. i should use the 4gpu machine
mod = '"res25"' # solid centroid


mod = '"res26"' # solid centroid; builder; no test
mod = '"res27"' # test10 data
mod = '"res28"' # train; solid centroid on train, obj0 centroid on test;
mod = '"res29"' # use solid scene centroid on ALL iters (y=1.0)
mod = '"res30"' # use varying centroid on train iters (y in [0.5, 1.5])
mod = '"res31"' # on test iters, use obj0; on train iters, use random centroid 
mod = '"res32"' # use solid scene centroid on all iters (like res29); test in X0
mod = '"res33"' # builder; print some bounds < looks ok. bounds match what i have in exp_base

mod = '"res34"' # trainer; faster logging; add numerical stability to utils_track
mod = '"res36"' # varying centroid on train iters; stable centroid on test
mod = '"res40"' # varying centroid on train iters; clist_camX0 on test
mod = '"res41"' # eliminate some extra nets
mod = '"res42"' # use (new) DCAA net
mod = '"res43"' # dcaa again, debugged; 
mod = '"res44"' # 11 layers instead of 8 (1 more at each scale)
mod = '"res45"' # same thing but drop the max idea in the argmax, and drop the extra mult by Z
mod = '"res46"' # same thing but drop the max idea in the argmax, and drop the extra mult by Z
mod = '"res47"' # mult by Z*10
mod = '"res48"' # basic net; drop the util; paste in what we had
mod = '"res49"' # basic centroid
mod = '"res51"' # basic net; use the util; feed vox_util to tracker
mod = '"res52"' # moving centroid on train, obj centroid on test, and print it, to see if it's out of distr 
mod = '"res53"' # y in -1.5, 3.0
mod = '"res54"' # again, but on other aws
mod = '"res55"' # learn in memX0, and no warp! 
mod = '"res56"' # regular again (Xs and warp to Rs for loss); diff coeffs
mod = '"res57"' # use view again

mod = '"res58"' # repeat, but with proper disk usage

mod = '"zo00"' # carla_regular_bounds (baseline)
mod = '"zo01"' # narrow bounds
mod = '"zo02"' # narrow bounds; avoid warps and do things in X0
mod = '"zo03"' # narrow bounds; avoid warps and do things in X0; SIZE=8
mod = '"zo04"' # regular bounds; avoid warps and do things in X0; SIZE=8 < killed accidentally; seems worse than zo05 though
mod = '"zo05"' # regular bounds; encode in X and warp to R; SIZE=8 (baseline)

mod = '"zo06"' # regular bounds; encode in X and warp to R; old encoderdecoder arch
mod = '"ml00"' # regular bounds; encode in X and warp to R; old encoderdecoder arch; train a bit
mod = '"ml01"' # enabled ce measurement; ce loss
mod = '"ml02"' # regular ml, but enable ce measurement < killed by accidnet
mod = '"ml03"' # ce loss; sparse-invariant
mod = '"ml04"' # dict size 50k instead of 100k

############## define experiment ##############

exps['builder'] = [
    'carla_bench', # mode
    # 'carla_multiview_train10_data', # dataset
    'carla_multiview_train10_test10_data', # dataset
    # 'carla_multiview_some_data', # dataset
    # 'carla_narrow_bounds',
    'carla_regular_bounds',
    '10_iters',
    'lr0',
    'B2',
    'no_shuf',
    # 'pretrained_feat3D', 
    'train_feat3D',
    'train_emb3D',
    'fastest_logging',
]
exps['trainer'] = [
    'carla_bench', # mode
    'carla_multiview_train_val_test_data', # dataset
    'carla_regular_bounds',
    # 'carla_narrow_bounds',
    '300k_iters',
    'lr4',
    'B2',
    # 'train_feat2D',
    'train_feat3D',
    # 'train_emb2D',
    'train_emb3D',
    # 'train_view',
    # 'train_occ',
    'faster_logging',
]

############## net configs ##############

groups['carla_bench'] = ['do_carla_bench = True']

groups['train_feat2D'] = [
    'do_feat2D = True',
    'feat2D_dim = 64',
    # 'feat2D_smooth_coeff = 0.01',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 64',
    # 'feat3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    # 'emb_3D_ml_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
    'emb_3D_ce_coeff = 1.0',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 1.0',
    'vq3drgb_num_embeddings = 512', 
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    # 'view_l1_coeff = 0.1',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 0.1',
    # 'occ_smooth_coeff = 0.001',
]
groups['train_vq3drgb'] = [
    'do_vq3drgb = True',
    'vq3drgb_latent_coeff = 2.0',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]


############## datasets ##############

# dims for mem
SIZE = 8
Z = int(SIZE*16)
Y = int(SIZE*2)
X = int(SIZE*16)

ZZ = int(SIZE*3)
ZY = int(SIZE*3)
ZX = int(SIZE*3)

K = 2 # how many objects to consider
N = 8 # how many objects per npz
S = 2
S_test = 8
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

# dataset_location = "/scratch"
# dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data2/carla/processed/npzs"
dataset_location = "/data4/carla/processed/npzs"

groups['carla_narrow_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -1.0', # down (neg is up)
    'YMAX = 1.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
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
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "fags16i3ten"',
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
    'testset = "fags16i3v"',
    'testset_format = "traj"', 
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

