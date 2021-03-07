import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs

from model_base import Model
# from nets.featnet2D import FeatNet2D
from nets.featnet3D import FeatNet3D
from nets.rejectnet3D import RejectNet3D
from nets.embnet3D import EmbNet3D
from nets.occnet import OccNet

from tensorboardX import SummaryWriter
import torch.nn.functional as F

# from utils_basic import *
# import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc
import utils_track
import vox_util

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_REJECT(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaRejectModel()
        if hyp.do_feat3D and hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
            
        if hyp.do_reject3D and hyp.do_freeze_reject3D:
            self.model.rejectnet3D.eval()
            self.set_requires_grad(self.model.rejectnet3D, False)

        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
            
    # override go from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        if hyp.lr > 0:
            params_to_optimize = self.model.parameters()
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=hyp.lr)
        else:
            self.optimizer = None
        
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        print("------ Done loading weights ------")
        # self.start_iter = 0

        set_nums = []
        set_names = []
        set_seqlens = []
        set_batch_sizes = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_names.append(set_name)
                set_seqlens.append(hyp.seqlens[set_name])
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    set_loaders[i] = iter(set_input)

            for (set_num,
                 set_name,
                 set_seqlen,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_names,
                set_seqlens,
                set_batch_sizes,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders
            ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                output_dict = dict()

                if log_this or set_do_backprop:
                    # print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)
                          
                    read_start_time = time.time()

                    feed = next(set_loader)
                    feed_cuda = {}
                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True)
                        except:
                            # some things are not tensors (e.g., filename)
                            feed_cuda[k] = feed[k]
                    read_time = time.time() - read_start_time


                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_batch_size'] = set_batch_size
                    
                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_py = loss.cpu().item()

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0)):
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if hyp.do_emb3D:
                        def update_slow_network(slow_net, fast_net, beta=0.999):
                            param_k = slow_net.state_dict()
                            param_q = fast_net.named_parameters()
                            for n, q in param_q:
                                if n in param_k:
                                    param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                            slow_net.load_state_dict(param_k)
                        update_slow_network(self.model.featnet3D_slow, self.model.featnet3D)
                        
                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_py,
                                                                                        set_name))
                    if log_this:
                        set_writer.flush()
                    
            if hyp.do_save_outputs:
                out_fn = '%s_output_dict.npy' % (hyp.name)
                np.save(out_fn, output_dict)
                print('saved %s' % out_fn)
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

            
class CarlaRejectModel(nn.Module):
    def __init__(self):
        super(CarlaRejectModel, self).__init__()

        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
        if hyp.do_reject3D:
            self.rejectnet3D = RejectNet3D(in_dim=4)
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=4)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
            
    def prepare_common_tensors(self, feed, prep_summ=True):
        results = dict()
        
        if prep_summ:
            self.summ_writer = utils_improc.Summ_writer(
                writer=feed['writer'],
                global_step=feed['global_step'],
                log_freq=feed['set_log_freq'],
                fps=8,
                just_gif=feed['just_gif'],
            )
        else:
            self.summ_writer = None

        self.include_vis = False

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.vox_size_X = (hyp.XMAX-hyp.XMIN)/self.X
        self.vox_size_Y = (hyp.YMAX-hyp.YMIN)/self.Y
        self.vox_size_Z = (hyp.ZMAX-hyp.ZMIN)/self.Z
        
        # self.rgb_camRs = feed["rgb_camRs"]
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0 = __u(utils_geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils_geom.safe_inverse(__p(self.camRs_T_camXs)))
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camX0_T_camR0 = utils_basic.matmul2(self.camX0s_T_camXs[:,0], self.camXs_T_camRs[:,0])
        self.camR0s_T_camXs = utils_basic.matmul2(self.camR0s_T_camRs, self.camRs_T_camXs)

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        if feed['set_name']=='test':
            self.box_camRs = feed["box_traj_camR"]
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            self.lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(self.box_camRs)
            self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            self.lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)
        
        if feed['set_name']=='test':
            # center on an object, so that it does not fall out of bounds
            self.scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            self.vox_util = vox_util.Vox_util(feed['set_name'], scene_centroid=self.scene_centroid, assert_cube=True)
        else:
            # # center randomly
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            # center on a random non-outlier point
            # x_camXs = xyz_camXs[:,:,0].cpu().numpy()
            # y_camXs = xyz_camXs[:,:,1].cpu().numpy()
            # z_camXs = xyz_camXs[:,:,2].cpu().numpy()

            # xyz = xyz_camXs[:,:,0].cpu().numpy()


            all_ok = False
            num_tries = 0
            while not all_ok:
                scene_centroid_x = np.random.uniform(-8.0, 8.0)
                scene_centroid_y = np.random.uniform(-1.5, 3.0)
                scene_centroid_z = np.random.uniform(10.0, 26.0)
                scene_centroid = np.array([scene_centroid_x,
                                           scene_centroid_y,
                                           scene_centroid_z]).reshape([1, 3])
                self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
                num_tries += 1

                # try to vox
                self.vox_util = vox_util.Vox_util(feed['set_name'], scene_centroid=self.scene_centroid, assert_cube=True)
                all_ok = True

                # we want to ensure this gives us a few points inbound for each batch el
                inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X, already_mem=False))
                num_inb = torch.sum(inb.float(), axis=2)
                if torch.min(num_inb) < 100:
                    all_ok = False


                if num_tries > 100:
                    return False
                    

            self.summ_writer.summ_scalar('zoom_sampling/num_tries', num_tries)
            self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
            # print('all set')
            # input()
            
            
            # perm = np.random.permutation(num_exms)
            # # np.var(a)
            
            # scene_centroid_x = np.random.uniform(-8.0, 8.0)
            # scene_centroid_y = np.random.uniform(-1.5, 3.0)
            # scene_centroid_z = np.random.uniform(10.0, 26.0)
            # scene_centroid = np.array([scene_centroid_x,
            #                            scene_centroid_y,
            #                            scene_centroid_z]).reshape([1, 3])
            # self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            

        
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s), self.Z, self.Y, self.X))
        self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        self.occ_memRs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
        # self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))
        self.occ_memX0s_quar = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X4))
        self.occ_memR0s_quar = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s), self.Z4, self.Y4, self.X4))

        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
        self.unp_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)
        self.unp_memR0s = self.vox_util.apply_4x4s_to_voxs(self.camR0s_T_camXs, self.unp_memXs)

        if prep_summ and self.include_vis:
            # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
            # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
            # self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))

            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
            # self.summ_writer.summ_unps('3D_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))

            self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
            # self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
            # self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        return True

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        encode_in_X0 = True

        if hyp.do_feat3D:
            if encode_in_X0:
                feat_memX0s_input = torch.cat([
                    self.occ_memX0s,
                    self.unp_memX0s*self.occ_memX0s,
                ], dim=2)
                feat3D_loss, feat_memX0s_, valid_memX0s_ = self.featnet3D(
                    __p(feat_memX0s_input[:,1:]),
                    self.summ_writer,
                )
                feat_memX0s = __u(feat_memX0s_)
                valid_memX0s = __u(valid_memX0s_)
                total_loss += feat3D_loss
                
                feat_memX0 = utils_basic.reduce_masked_mean(
                    feat_memX0s,
                    valid_memX0s.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
                    dim=1)
                valid_memX0 = torch.sum(valid_memX0s, dim=1).clamp(0, 1)
                self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, valid=valid_memX0, pca=True)
                self.summ_writer.summ_feat('3D_feats/valid_memX0', valid_memX0, pca=False)

                if hyp.do_emb3D:
                    _, altfeat_memX0, altvalid_memX0 = self.featnet3D_slow(feat_memX0s_input[:,0])
                    self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
                    self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)
                elif hyp.do_reject3D:
                    # use the regular (non-slow) net
                    _, altfeat_memX0, altvalid_memX0 = self.featnet3D(feat_memX0s_input[:,0])
                    self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
                    self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)
            else:
                feat_memXs_input = torch.cat([
                    self.occ_memXs,
                    self.unp_memXs*self.occ_memXs,
                ], dim=2)
                feat_memRs_input = torch.cat([
                    self.occ_memRs,
                    self.unp_memRs*self.occ_memRs,
                ], dim=2)

                feat3D_loss, feat_memXs_, valid_memXs_ = self.featnet3D(
                    __p(feat_memXs_input[:,1:]),
                    self.summ_writer,
                )
                feat_memXs = __u(feat_memXs_)
                valid_memXs = __u(valid_memXs_)
                total_loss += feat3D_loss

                # warp things to R
                feat_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], feat_memXs)
                valid_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], valid_memXs)

                feat_memR = utils_basic.reduce_masked_mean(
                    feat_memRs,
                    valid_memRs.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
                    dim=1)
                valid_memR = torch.sum(valid_memRs, dim=1).clamp(0, 1)

                self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)
                self.summ_writer.summ_feat('3D_feats/valid_memR', valid_memR, pca=False)

                if hyp.do_emb3D:
                    _, altfeat_memR, altvalid_memR = self.featnet3D_slow(feat_memRs_input[:,0])
                    self.summ_writer.summ_feat('3D_feats/altfeat_memR', altfeat_memR, pca=True)
                    self.summ_writer.summ_feat('3D_feats/altvalid_memR', altvalid_memR, pca=False)

        if hyp.do_occ:
            _, _, Z_, Y_, X_ = list(feat_memX0.shape)
            if encode_in_X0:
                occ_memX0_sup, free_memX0_sup, _, free_memX0s = self.vox_util.prep_occs_supervision(
                    self.camX0s_T_camXs,
                    self.xyz_camXs,
                    Z_, Y_, X_,
                    agg=True)
                self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memX0_sup)
                self.summ_writer.summ_occ('occ_sup/free_sup', free_memX0_sup)
                self.summ_writer.summ_occs('occ_sup/freeX0s_sup', torch.unbind(free_memX0s, dim=1))
                self.summ_writer.summ_occs('occ_sup/occX0s_sup', torch.unbind(self.occ_memX0s_half, dim=1))
                occ_loss, occ_memX0_pred = self.occnet(
                    altfeat_memX0, 
                    occ_memX0_sup,
                    free_memX0_sup,
                    altvalid_memX0, 
                    self.summ_writer)
            else:
                occ_memR_sup, free_memR_sup, _, free_memRs = self.vox_util.prep_occs_supervision(
                    self.camRs_T_camXs,
                    self.xyz_camXs,
                    Z_, Y_, X_,
                    agg=True)
                self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memR_sup)
                self.summ_writer.summ_occ('occ_sup/free_sup', free_memR_sup)
                self.summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(free_memRs, dim=1))
                self.summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(self.occ_memRs_half, dim=1))

                occ_loss, occ_memR_pred = self.occnet(
                    altfeat_memR, 
                    occ_memR_sup,
                    free_memR_sup,
                    altvalid_memR,
                    self.summ_writer)
                
            total_loss += occ_loss
                    
        if hyp.do_emb3D:
            # compute 3D ML
            
            if encode_in_X0:
                emb_loss_3D = self.embnet3D(
                    feat_memX0,
                    altfeat_memX0,
                    valid_memX0.round(),
                    altvalid_memX0.round(),
                    self.summ_writer)
            else:
                emb_loss_3D = self.embnet3D(
                    feat_memR,
                    altfeat_memR,
                    valid_memR.round(),
                    altvalid_memR.round(),
                    self.summ_writer)
            total_loss += emb_loss_3D

        if hyp.do_reject3D:
            assert(encode_in_X0)
            reject3D_loss, _, _ = self.rejectnet3D(
                feat_memX0.detach(),
                altfeat_memX0.detach(),
                valid_memX0,
                altvalid_memX0,
                self.summ_writer,
            )
            total_loss += reject3D_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.obj_clist_camR0 = utils_geom.get_clist_from_lrtlist(self.lrt_camR0s)
        self.obj_clist_camX0 = utils_geom.get_clist_from_lrtlist(self.lrt_camX0s)

        if self.include_vis:
            visX_g = []
            for s in list(range(self.S)):
                visX_g.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)

        if hyp.do_feat3D:
            feat_memX0s_input = torch.cat([
                self.occ_memX0s,
                self.unp_memX0s*self.occ_memX0s,
            ], dim=2)
            _, feat_memX0s_, valid_memX0s_ = self.featnet3D(
                __p(feat_memX0s_input),
                self.summ_writer,
            )
            feat_memX0s = __u(feat_memX0s_)
            valid_memX0s = __u(valid_memX0s_)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s', torch.unbind(feat_memX0s, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s_input', torch.unbind(feat_memX0s_input, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s_input_', torch.unbind(feat_memX0s_input, dim=1), pca=False)

            if hyp.do_reject3D:
                reject_memX0s = []
                reject_memX0s_dyn = []
                reject_memX0s_sta = []
                for s in list(range(self.S)):
                    _, reject_memX0, synth_reject_memX0 = self.rejectnet3D(
                        feat_memX0s[:,0].detach(),
                        feat_memX0s[:,s].detach(),
                        valid_memX0s[:,0],
                        valid_memX0s[:,s],
                        None, 
                    )
                    reject_memX0s.append(reject_memX0)
                    reject_memX0_full = F.interpolate(reject_memX0, scale_factor=2, mode='nearest').round()
                    # reject_memX0s_occ.append(reject_memX0 * self.occ_memX0s_quar[:,s])
                    reject_memX0s_dyn.append(reject_memX0_full * self.occ_memX0s[:,s])
                    reject_memX0s_sta.append((1.0 - reject_memX0_full) * self.occ_memX0s[:,s])
                self.summ_writer.summ_oneds('reject/reject_memX0s', reject_memX0s, bev=True, norm=False)
                self.summ_writer.summ_occs('reject/reject_memX0s_dyn', reject_memX0s_dyn)
                self.summ_writer.summ_occs('reject/reject_memX0s_sta', reject_memX0s_sta)
            
            _, _, Z_, Y_, X_ = list(feat_memX0s[:,0].shape)
            obj_mask_memX0s = self.vox_util.assemble_padded_obj_masklist(
                self.lrt_camX0s,
                self.score_s,
                Z_, Y_, X_).squeeze(1)
            # only take the occupied voxels
            occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), Z_, Y_, X_))
            obj_mask_memX0s = obj_mask_memX0s * occ_memX0s

            for b in list(range(self.B)):
                if torch.sum(obj_mask_memX0s[b,0]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True
            self.summ_writer.summ_feats('track/obj_mask_memX0s', torch.unbind(obj_mask_memX0s, dim=1), pca=False)

            lrt_camX0s_e, point_counts, ious = utils_track.track_via_inner_products(
                self.lrt_camX0s, obj_mask_memX0s, feat_memX0s, self.vox_util)   

            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))
                
            lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)
            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_e' % s, self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
            
            obj_clist_camX0_e = utils_geom.get_clist_from_lrtlist(lrt_camX0s_e)

            dists = torch.norm(obj_clist_camX0_e - self.obj_clist_camX0, dim=2)
            # this is B x S
            mean_dist = utils_basic.reduce_masked_mean(dists, self.score_s)
            median_dist = utils_basic.reduce_masked_median(dists, self.score_s)
            # this is []
            self.summ_writer.summ_scalar('track/centroid_dist_mean', mean_dist.cpu().item())
            self.summ_writer.summ_scalar('track/centroid_dist_median', median_dist.cpu().item())
            
            self.summ_writer.summ_traj_on_occ('track/traj_e',
                                              obj_clist_camX0_e, 
                                              self.occ_memX0s[:,0],
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)
            self.summ_writer.summ_traj_on_occ('track/traj_g',
                                              self.obj_clist_camX0,
                                              self.occ_memX0s[:,0],
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)
            
            total_loss += mean_dist # we won't backprop, but it's nice to plot and print this anyway
        else:
            
            # self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            
            ious = torch.zeros([self.B, self.S]).float().cuda()
            for s in list(range(self.S)):
                # lrt_camIs_e[:,s] = utils_geom.merge_lrt(obj_length, camI_T_obj)
                ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,0:1], self.lrt_camRs[:,s:s+1]).squeeze(1)
            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, None, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        if set_name=='test':
            just_gif = True
        else:
            just_gif = False
        feed['just_gif'] = just_gif

        if set_name=='train' or set_name=='val':
            ok = self.prepare_common_tensors(feed)
            if not ok:
                total_loss = torch.tensor(0.0).cuda()
                return total_loss, None, True
            return self.run_train(feed)
        elif set_name=='test':
            ok = self.prepare_common_tensors(feed)
            if not ok:
                total_loss = torch.tensor(0.0).cuda()
                return total_loss, None, True
            return self.run_test(feed)
        
        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
