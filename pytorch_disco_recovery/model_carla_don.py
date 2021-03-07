import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs

from model_base import Model
from nets.featnet2D import FeatNet2D
# from nets.featnet3D import FeatNet3D
from nets.trinet2D import TriNet2D
from nets.colnet2D import ColNet2D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
# from nets.viewnet import ViewNet
# from nets.mocnet3D import MocNet3D

from tensorboardX import SummaryWriter
import torch.nn.functional as F

# from utils_moc import MocTrainer
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

class CARLA_DON(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaDonModel()
        if hyp.do_feat2D and hyp.do_freeze_feat2D:
            self.model.featnet2D.eval()
            self.set_requires_grad(self.model.featnet2D, False)
            
        if hyp.do_tri2D:
            # freeze the slow model
            self.model.featnet2D_slow.eval()
            self.set_requires_grad(self.model.featnet2D_slow, False)
            
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

        if hyp.do_test:
            all_ious = np.zeros([hyp.max_iters, hyp.S_test], np.float32)
            test_count = 0

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

                if log_this or set_do_backprop or hyp.do_test:
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

                    if hyp.do_test and (not returned_early):
                        ious = results['ious']
                        ious = ious[0].cpu().numpy()
                        all_ious[test_count] = ious
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        print('mean_ious', mean_ious)
                    
                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0) and
                        (not (loss_py==0.0))):
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                    if hyp.do_tri2D:
                        def update_slow_network(slow_net, fast_net, beta=0.999):
                            param_k = slow_net.state_dict()
                            param_q = fast_net.named_parameters()
                            for n, q in param_q:
                                if n in param_k:
                                    param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                            slow_net.load_state_dict(param_k)
                        update_slow_network(self.model.featnet2D_slow, self.model.featnet2D)
                        
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

        if hyp.do_test:
            mean_ious = np.mean(all_ious[:test_count], axis=0)
            print('mean_ious', mean_ious)
            
class CarlaDonModel(nn.Module):
    def __init__(self):
        super(CarlaDonModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_tri2D:
            self.trinet2D = TriNet2D()
            # make a slow net
            self.featnet2D_slow = FeatNet2D()
            # init slow params with fast params
            self.featnet2D_slow.load_state_dict(self.featnet2D.state_dict())
        if hyp.do_col2D:
            self.colnet2D = ColNet2D()
            
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

        self.include_vis = hyp.do_include_vis

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)
        
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

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N

        self.set_name = feed['set_name']
        
        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        
        if self.set_name=='test':
            self.box_camRs = feed["box_traj_camR"]
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            self.lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(self.box_camRs)
            self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
        
        if self.set_name=='test':
            # center on an object, so that it does not fall out of bounds
            self.scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
        else:
            # center randomly 
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

        self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
            self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.xyz_camXs = feed["xyz_camXs"]
        # self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        # self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # # self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        # self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        # self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # # self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
        # #     __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        # self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)

        # ## projected depth, and inbound mask
        # self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # # we need to go to X0 to see what will be inbounds
        # self.dense_xyz_camX0s_ = utils_geom.apply_4x4(__p(self.camX0s_T_camXs), self.dense_xyz_camXs_)
        # self.inbound_camXs_ = utils_vox.get_inbounds(self.dense_xyz_camX0s_, self.Z, self.Y, self.X).float()
        # self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # self.depth_camXs = __u(self.depth_camXs_)
        # self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)

        if prep_summ and self.include_vis:
            # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
            # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))

            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            # self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
            # self.summ_writer.summ_unps('3D_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))

            self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
            # self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
            # self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_test:
            # return early
            return total_loss, results, True
        
        assert(self.S==2) # i am writing the loss assuming this
        
        if hyp.do_feat2D:
            if hyp.do_tri2D:
                feat2D_loss, feat_camX0 = self.featnet2D(
                    self.rgb_camXs[:,0],
                    self.summ_writer,
                )
                _, feat_camX1 = self.featnet2D_slow(
                    self.rgb_camXs[:,1],
                    self.summ_writer,
                )
            else:
                lab_camXs = __u(utils_improc.rgb2lab(__p(self.rgb_camXs)))
                self.summ_writer.summ_oneds('2D_inputs/rgb_camXs_L', torch.unbind(lab_camXs[:,:,0:1], dim=1))
                self.summ_writer.summ_oneds('2D_inputs/rgb_camXs_A', torch.unbind(lab_camXs[:,:,1:2], dim=1))
                self.summ_writer.summ_oneds('2D_inputs/rgb_camXs_B', torch.unbind(lab_camXs[:,:,2:3], dim=1))
                
                # either drop the A or the B
                A_mask = torch.randint(0, 2, (self.B, self.S, 1, 1, 1)).cuda().float()
                B_mask = 1.0 - A_mask
                lab_masks = torch.cat([torch.ones_like(A_mask), A_mask, B_mask], dim=2)
                # print('lab_masks', lab_masks.shape)
                lab_camXs = lab_camXs * lab_masks
                rgb_camXs = __u(utils_improc.lab2rgb(__p(lab_camXs)))
                self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs_drop', torch.unbind(rgb_camXs, dim=1))
                
                feat2D_loss, feat_camXs_ = self.featnet2D(
                    __p(lab_camXs), # feed the dropped lab data
                    self.summ_writer,
                )
                feat_camXs = __u(feat_camXs_)
                feat_camX0 = feat_camXs[:,0]
                feat_camX1 = feat_camXs[:,1]
                
            _, _, H, W = feat_camX0.shape
            sx = float(W)/float(hyp.W)
            sy = float(H)/float(hyp.H)
            self.summ_writer.summ_feat('feat2D/feat_camX0', feat_camX0, pca=True)
            self.summ_writer.summ_feat('feat2D/feat_camX1', feat_camX1, pca=True)
            featpix_T_cam0 = utils_geom.scale_intrinsics(self.pix_T_cams[:,0], sx, sy)
            featpix_T_cam1 = utils_geom.scale_intrinsics(self.pix_T_cams[:,1], sx, sy)
            
        if hyp.do_tri2D:
            assert(hyp.do_feat2D)
            Z_, Y_, X_ = self.Z, self.Y, self.X
            occ_memX0s, free_memX0s, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                Z_, Y_, X_,
                agg=False)

            # if a point is visible in BOTH frames,
            # i want the embedding at the corresponding pixels to match
            
            # only occ points matter; a point is OK if it is seen in both views
            present_in_both_memX0 = occ_memX0s[:,0] * occ_memX0s[:,1]
            self.summ_writer.summ_occ('rely/present_in_both', present_in_both_memX0)
            self.summ_writer.summ_occ('rely/aggressive_occ', torch.max(occ_memX0s, dim=1)[0])
            
            tri_loss_2D = self.trinet2D(
                feat_camX0,
                feat_camX1,
                present_in_both_memX0,
                featpix_T_cam0,
                featpix_T_cam1,
                self.camXs_T_camX0s[:,1],
                self.vox_util, 
                self.summ_writer)
            total_loss += tri_loss_2D
            
        if hyp.do_col2D:
            assert(hyp.do_feat2D)
            _, _, H, W = feat_camX0.shape
            sx = float(W)/float(hyp.W)
            sy = float(H)/float(hyp.H)
            assert(sx==sy)
            col2D_loss = self.colnet2D(
                F.interpolate(self.rgb_camXs[:,0], scale_factor=sx),
                F.interpolate(self.rgb_camXs[:,1], scale_factor=sx),
                feat_camX0,
                feat_camX1,
                self.summ_writer)
            total_loss += col2D_loss
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.obj_clist_camX0 = utils_geom.get_clist_from_lrtlist(self.lrt_camX0s)
        
        self.original_centroid = self.scene_centroid.clone()

        obj_lengths, cams_T_obj0 = utils_geom.split_lrtlist(self.lrt_camX0s)
        obj_length = obj_lengths[:,0]
        for b in list(range(self.B)):
            if self.score_s[b,0] < 1.0:
                # we need the template to exist
                print('returning early, since score_s[%d,0] = %.1f' % (b, self.score_s[b,0].cpu().numpy()))
                return total_loss, results, True
            # if torch.sum(self.score_s[b]) < (self.S/2):
            if not (torch.sum(self.score_s[b]) == self.S):
                # the full traj should be valid
                print('returning early, since sum(score_s) = %d, while S = %d' % (torch.sum(self.score_s).cpu().numpy(), self.S))
                return total_loss, results, True
            
        # if self.include_vis:
        #     visX_g = []
        #     for s in list(range(self.S)):
        #         visX_g.append(self.summ_writer.summ_lrtlist(
        #             'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
        #             self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
        #     self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)


        # # self.summ_writer.summ_lrtlist('obj/boxlist_g', self.rgb_camRs[:,0], self.lrtlist_camRs[:,0],
        # #                               self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])
        # # boxlist2d = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], self.lrtlist_camRs[:,0], self.H, self.W)
        
        # boxlist2d = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], self.lrt_camXs, self.H, self.W)
        # vis = []
        # for s in range(self.S):
        #     vis.append(self.summ_writer.summ_boxlist2D(
        #         '', self.rgb_camXs[:,s], boxlist2d[:,s:s+1], only_return=True))
        # self.summ_writer.summ_rgbs('track/boxes2D_g', vis)
        

        if hyp.do_feat2D:
            if hyp.do_tri2D:
                _, feat_camXs_ = self.featnet2D(
                    __p(self.rgb_camXs),
                    self.summ_writer,
                )
            else:
                # feed lab data
                lab_camXs = __u(utils_improc.rgb2lab(__p(self.rgb_camXs)))
                _, feat_camXs_ = self.featnet2D(
                    __p(lab_camXs),
                    self.summ_writer,
                )
            feat_camXs = __u(feat_camXs_)
            feat_camXs = feat_camXs.detach()
            self.summ_writer.summ_feats('feat2D/feat_camXs', torch.unbind(feat_camXs, dim=1), pca=True)
            feat_camX0 = feat_camXs[:,0]

            _, _, _, H, W = feat_camXs.shape
            sx = float(W)/float(hyp.W)
            sy = float(H)/float(hyp.H)
            featpix_T_cam = utils_geom.scale_intrinsics(self.pix_T_cams[:,0], sx, sy)
            feat_memX0 = self.vox_util.unproject_rgb_to_mem(
                feat_camX0, self.Z, self.Y, self.X, featpix_T_cam)
            S = self.S
            B, C, Z, Y, X = list(feat_memX0.shape)
            
            obj_mask_memX0s = self.vox_util.assemble_padded_obj_masklist(
                self.lrt_camX0s,
                self.score_s,
                Z, Y, X).squeeze(1)

            self.summ_writer.summ_oneds('track/obj_masks_memX0s', torch.unbind(obj_mask_memX0s, dim=1), bev=True)
            
            # only take the occupied voxels
            occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], Z, Y, X)
            obj_mask_memX0 = obj_mask_memX0s[:,0] * occ_memX0

            for b in list(range(self.B)):
                num_pts = torch.sum(obj_mask_memX0[b]*occ_memX0[b]).cpu().numpy()
                if num_pts <= 8:
                    print('returning early, since there are only %d valid object points' % (num_pts))
                    return total_loss, results, True
            
            feat0_vec = feat_memX0.view(B, hyp.feat2D_dim, -1)
            # this is B x C x huge
            feat0_vec = feat0_vec.permute(0, 2, 1)
            # this is B x huge x C

            obj_mask0_vec = obj_mask_memX0.reshape(B, -1).round()
            # these are B x huge

            orig_xyz = utils_basic.gridcloud3D(B, Z, Y, X)
            # this is B x huge x 3

            obj_lengths, cams_T_obj0 = utils_geom.split_lrtlist(self.lrt_camX0s)
            obj_length = obj_lengths[:,0]
            cam0_T_obj = cams_T_obj0[:,0]
            # this is B x S x 4 x 4

            mem_T_cam = self.vox_util.get_mem_T_ref(B, Z, Y, X)
            cam_T_mem = self.vox_util.get_ref_T_mem(B, Z, Y, X)

            lrt_camIs_g = self.lrt_camX0s.clone()
            lrt_camIs_e = torch.zeros_like(self.lrt_camX0s)
            # we will fill this up

            ious = torch.zeros([B, S]).float().cuda()
            point_counts = np.zeros([B, S])
            inb_counts = np.zeros([B, S])

            for s in range(self.S):
                if not (s==0):
                    # remake the vox util and all the mem data
                    self.scene_centroid = utils_geom.get_clist_from_lrtlist(lrt_camIs_e[:,s-1:s])[:,0]
                    delta = self.scene_centroid - self.original_centroid
                    self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                        self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                    self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
                    self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

                    self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
                        __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
                    self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
                    self.summ_writer.summ_occ('track/reloc_occ_%d' % s, self.occ_memX0s[:,s])
                else:
                    self.summ_writer.summ_occ('track/init_occ_%d' % s, self.occ_memX0s[:,s])
                    delta = torch.zeros([B, 3]).float().cuda()
                # print('scene centroid:', self.scene_centroid.detach().cpu().numpy())

                inb = self.vox_util.get_inbounds(self.xyz_camX0s[:,s], self.Z4, self.Y4, self.X, already_mem=False)
                num_inb = torch.sum(inb.float(), axis=1)
                # print('num_inb', num_inb, num_inb.shape)
                inb_counts[:, s] = num_inb.cpu().numpy()

                # _, feat_camXI = self.featnet2D(
                #     self.rgb_camXs[:,s],
                #     self.summ_writer,
                # )
                # featpix_T_cam = utils_geom.scale_intrinsics(self.pix_T_cams[:,s], 0.5, 0.5)
                # feat_memXI = self.vox_util.unproject_rgb_to_mem(
                #     feat_camXI, self.Z, self.Y, self.X, featpix_T_cam)

                feat_memXI = self.vox_util.unproject_rgb_to_mem(
                    feat_camXs[:,s], self.Z, self.Y, self.X, featpix_T_cam)
                feat_memI = self.vox_util.apply_4x4_to_vox(self.camX0s_T_camXs[:,s], feat_memXI)
                # now we are in the local X0 system
                feat_memI = feat_memI * self.occ_memX0s[:,s]

                self.summ_writer.summ_feat('3D_feats/feat_%d' % s, feat_memI, pca=True)

                feat_vec = feat_memI.view(B, hyp.feat2D_dim, -1)
                # this is B x C x huge
                feat_vec = feat_vec.permute(0, 2, 1)
                # this is B x huge x C

                memI_T_mem0 = utils_geom.eye_4x4(B)
                # we will fill this up

                # to simplify the impl, we will iterate over the batch dim
                for b in list(range(B)):
                    feat_vec_b = feat_vec[b]
                    feat0_vec_b = feat0_vec[b]
                    obj_mask0_vec_b = obj_mask0_vec[b]
                    orig_xyz_b = orig_xyz[b]
                    # these are huge x C

                    # take any points within the mask
                    obj_inds_b = torch.where(obj_mask0_vec_b > 0)
                    obj_vec_b = feat0_vec_b[obj_inds_b]
                    xyz0 = orig_xyz_b[obj_inds_b]
                    # these are med x C

                    # issues arise when "med" is too large
                    # trim down to max_pts
                    num = len(xyz0)
                    max_pts = 2000
                    if num > max_pts:
                        print('have %d pts; taking a random set of %d pts inside' % (num, max_pts))
                        perm = np.random.permutation(num)
                        obj_vec_b = obj_vec_b[perm[:max_pts]]
                        xyz0 = xyz0[perm[:max_pts]]

                    obj_vec_b = obj_vec_b.permute(1, 0)
                    # this is is C x med

                    corr_b = torch.matmul(feat_vec_b, obj_vec_b)
                    # this is huge x med

                    heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z, Y, X)
                    # this is med x 1 x Z4 x Y4 x X4

                    # # for numerical stability, we sub the max, and mult by the resolution
                    # heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    # heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    # heat_b = heat_b - heat_b_max
                    # heat_b = heat_b * float(len(heat_b[0].reshape(-1)))


                    # # for numerical stability, we sub the max, and mult by the resolution
                    # heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    # heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    # heat_b = heat_b - heat_b_max
                    # heat_b = heat_b * float(len(heat_b[0].reshape(-1)))
                    # heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    # # heat_b_min = (torch.min(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    # heat_b_min = (torch.min(heat_b_).values)
                    # free_b = free_memI[b:b+1]
                    # print('free_b', free_b.shape)
                    # print('heat_b', heat_b.shape)
                    # heat_b[free_b > 0.0] = heat_b_min
                    
                    # make the min zero
                    heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    heat_b_min = (torch.min(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    heat_b = heat_b - heat_b_min
                    # # zero out the freespace
                    # heat_b = heat_b * (1.0 - free_memI[b:b+1])
                    # # only take occ
                    # heat_b = heat_b * self.occ_memX0s[b:b+1,s]
                    # make the max zero
                    heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    heat_b = heat_b - heat_b_max
                    # scale up, for numerical stability
                    heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

                    xyzI = utils_basic.argmax3D(heat_b, hard=False, stack=True)
                    # this is med x 3

                    xyzI_cam = self.vox_util.Mem2Ref(xyzI.unsqueeze(1), Z, Y, X)
                    xyzI_cam += delta
                    xyzI = self.vox_util.Ref2Mem(xyzI_cam, Z, Y, X).squeeze(1)

                    memI_T_mem0[b] = utils_track.rigid_transform_3D(xyz0, xyzI)

                    # record #points, since ransac depends on this
                    point_counts[b, s] = len(xyz0)
                # done stepping through batch

                mem0_T_memI = utils_geom.safe_inverse(memI_T_mem0)
                cam0_T_camI = utils_basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

                # eval
                camI_T_obj = utils_basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
                # this is B x 4 x 4
                lrt_camIs_e[:,s] = utils_geom.merge_lrt(obj_length, camI_T_obj)
                ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)
            results['ious'] = ious
            # if ious[0,-1] > 0.5:
            #     print('returning early, since acc is too high')
            #     return total_loss, results, True
                

            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())

            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))
            # self.summ_writer.summ_scalar('track/inb_counts', torch.mean(inb_counts).cpu().item())
            self.summ_writer.summ_scalar('track/inb_counts', np.mean(inb_counts))

            lrt_camX0s_e = lrt_camIs_e.clone()
            lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)

            if self.include_vis:
                visX_e = []
                for s in list(range(self.S)):
                    visX_e.append(self.summ_writer.summ_lrtlist(
                        'track/box_camX%d_e' % s, self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
                        self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
                self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
                visX_g = []
                for s in list(range(self.S)):
                    visX_g.append(self.summ_writer.summ_lrtlist(
                        'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                        self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
                self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)

            obj_clist_camX0_e = utils_geom.get_clist_from_lrtlist(lrt_camX0s_e)

            dists = torch.norm(obj_clist_camX0_e - self.obj_clist_camX0, dim=2)
            # this is B x S
            mean_dist = utils_basic.reduce_masked_mean(dists, self.score_s)
            median_dist = utils_basic.reduce_masked_median(dists, self.score_s)
            # this is []
            self.summ_writer.summ_scalar('track/centroid_dist_mean', mean_dist.cpu().item())
            self.summ_writer.summ_scalar('track/centroid_dist_median', median_dist.cpu().item())

            # if self.include_vis:
            if (True):
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

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        if set_name=='test':
            just_gif = True
        else:
            just_gif = False
        feed['just_gif'] = just_gif
        
        if set_name=='train' or set_name=='val':
            self.prepare_common_tensors(feed)
            return self.run_train(feed)
        elif set_name=='test':
            self.prepare_common_tensors(feed)
            return self.run_test(feed)
        
        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
