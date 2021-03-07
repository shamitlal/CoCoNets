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
from nets.embnet3D import EmbNet3D
from nets.occnet import OccNet
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

class KITTI_ZOOM(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = KittiZoomModel()
        if hyp.do_feat3D and hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

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
                    # if log_this and (not returned_early):
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

            
class KittiZoomModel(nn.Module):
    def __init__(self):
        super(KittiZoomModel, self).__init__()

        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
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

        self.include_vis = hyp.do_include_vis

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        
        self.set_name = feed['set_name']
        # print('set_name', self.set_name)
        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        # print('Z, Y, X = %d, %d, %d' % (self.Z, self.Y, self.X))
        
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camXs = feed["origin_T_camXs"]

        self.cams_T_velos = feed["cams_T_velos"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.xyz_veloXs = feed["xyz_veloXs"]
        self.xyz_camXs = __u(utils_geom.apply_4x4(__p(self.cams_T_velos), __p(self.xyz_veloXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        if self.set_name=='test':
            self.boxlist_camXs = feed["boxlists"]
            self.scorelist_s = feed["scorelists"]
            self.tidlist_s = feed["tidlists"]


            boxlist_camXs_ = __p(self.boxlist_camXs)
            scorelist_s_ = __p(self.scorelist_s)
            tidlist_s_ = __p(self.tidlist_s)
            boxlist_camXs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
                boxlist_camXs_, tidlist_s_, scorelist_s_)
            self.boxlist_camXs = __u(boxlist_camXs_)
            self.scorelist_s = __u(scorelist_s_)
            self.tidlist_s = __u(tidlist_s_)
            
            # self.boxlist_camXs[:,0], self.scorelist_s[:,0], self.tidlist_s[:,0] = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            #     self.boxlist_camXs[:,0], self.tidlist_s[:,0], self.scorelist_s[:,0])
            
            # self.score_s = feed["scorelists"]
            # self.tid_s = torch.ones_like(self.score_s).long()
            # self.lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(self.box_camRs)
            # self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            # self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            # self.lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)


            # boxlist_camXs_ = __p(self.boxlist_camXs)
            # boxlist_camXs_ = __p(self.boxlist_camXs)

            # lrtlist_camXs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camXs))).reshape(
            #     self.B, self.S, self.N, 19)
            
            self.lrtlist_camXs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camXs)))
            
            # print('lrtlist_camXs', lrtlist_camXs.shape)
            # # self.B, self.S, self.N, 19)
            # # lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
            # self.summ_writer.summ_lrtlist('2D_inputs/lrtlist_camX0', self.rgb_camXs[:,0], lrtlist_camXs[:,0],
            #                               self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])
            # self.summ_writer.summ_lrtlist('2D_inputs/lrtlist_camX1', self.rgb_camXs[:,1], lrtlist_camXs[:,1],
            #                               self.scorelist_s[:,1], self.tidlist_s[:,1], self.pix_T_cams[:,1])

            (self.lrt_camXs,
             self.box_camXs,
             self.score_s,
            ) = utils_misc.collect_object_info(self.lrtlist_camXs,
                                               self.boxlist_camXs,
                                               self.tidlist_s,
                                               self.scorelist_s,
                                               1, mod='X',
                                               do_vis=False,
                                               summ_writer=None)
            self.lrt_camXs = self.lrt_camXs.squeeze(0)
            self.score_s = self.score_s.squeeze(0)
            self.tid_s = torch.ones_like(self.score_s).long()

            self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            
            if prep_summ and self.include_vis:
                visX_g = []
                for s in list(range(self.S)):
                    visX_g.append(self.summ_writer.summ_lrtlist(
                        '', self.rgb_camXs[:,s], self.lrtlist_camXs[:,s],
                        self.scorelist_s[:,s], self.tidlist_s[:,s], self.pix_T_cams[:,0], only_return=True))
                self.summ_writer.summ_rgbs('2D_inputs/box_camXs', visX_g)
                # visX_g = []
                # for s in list(range(self.S)):
                #     visX_g.append(self.summ_writer.summ_lrtlist(
                #         'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                #         self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
                # self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)
        
        if self.set_name=='test':
            # center on an object, so that it does not fall out of bounds
            self.scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        else:
            # center randomly
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            # center on a random non-outlier point

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
                self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X,
                                                  self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
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

        
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)

        if prep_summ and self.include_vis:
            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
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

        if hyp.do_feat3D:
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

        if hyp.do_emb3D:
            if hyp.do_feat3D:
                _, _, Z_, Y_, X_ = list(feat_memX0.shape)
            else:
                Z_, Y_, X_ = self.Z2, self.Y2, self.X2
            # Z_, Y_, X_ = self.Z, self.Y, self.X
                
            occ_memX0s, free_memX0s, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                Z_, Y_, X_,
                agg=False)
            
            not_ok = torch.zeros_like(occ_memX0s[:,0])
            # it's not ok for a voxel to be marked occ only once
            not_ok += (torch.sum(occ_memX0s, dim=1)==1.0).float()
            # it's not ok for a voxel to be marked occ AND free
            occ_agg = torch.sum(occ_memX0s, dim=1).clamp(0, 1)
            free_agg = torch.sum(free_memX0s, dim=1).clamp(0, 1)
            have_either = (occ_agg + free_agg).clamp(0, 1)
            have_both = occ_agg*free_agg
            not_ok += have_either * have_both
            # it's not ok for a voxel to be totally unobserved
            not_ok += (have_either==0.0).float()
            not_ok = not_ok.clamp(0, 1)
            self.summ_writer.summ_occ('rely/not_ok', not_ok)
            self.summ_writer.summ_occ('rely/not_ok_occ', not_ok * torch.max(self.occ_memX0s_half, dim=1)[0])
            self.summ_writer.summ_occ('rely/ok_occ', (1.0 - not_ok) * torch.max(self.occ_memX0s_half, dim=1)[0])
            self.summ_writer.summ_occ('rely/aggressive_occ', torch.max(self.occ_memX0s_half, dim=1)[0])

            be_safe = False
            if hyp.do_feat3D and be_safe:
                # update the valid masks
                valid_memX0 = valid_memX0 * (1.0 - not_ok)
                altvalid_memX0 = altvalid_memX0 * (1.0 - not_ok)
                
        if hyp.do_occ:
            _, _, Z_, Y_, X_ = list(feat_memX0.shape)
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
            total_loss += occ_loss
            
        if hyp.do_emb3D:
            # compute 3D ML
            emb_loss_3D = self.embnet3D(
                feat_memX0,
                altfeat_memX0,
                valid_memX0.round(),
                altvalid_memX0.round(),
                self.summ_writer)
            total_loss += emb_loss_3D

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

        if hyp.do_feat3D:

            feat_memX0_input = torch.cat([
                self.occ_memX0s[:,0],
                self.unp_memX0s[:,0]*self.occ_memX0s[:,0],
            ], dim=1)
            _, feat_memX0, valid_memX0 = self.featnet3D(feat_memX0_input)
            B, C, Z, Y, X = list(feat_memX0.shape)
            S = self.S

            obj_mask_memX0s = self.vox_util.assemble_padded_obj_masklist(
                self.lrt_camX0s,
                self.score_s,
                Z, Y, X).squeeze(1)
            # only take the occupied voxels
            occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], Z, Y, X)
            # obj_mask_memX0 = obj_mask_memX0s[:,0] * occ_memX0
            obj_mask_memX0 = obj_mask_memX0s[:,0]

            # discard the known freespace
            _, free_memX0_, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs[:,0:1],
                self.xyz_camXs[:,0:1],
                Z, Y, X, agg=True)
            free_memX0 = free_memX0_.squeeze(1)
            obj_mask_memX0 = obj_mask_memX0 * (1.0 - free_memX0)

            for b in list(range(self.B)):
                if torch.sum(obj_mask_memX0[b]*occ_memX0[b]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True

            # for b in list(range(self.B)):
            #     sum_b = torch.sum(obj_mask_memX0[b])
            #     print('sum_b', sum_b.detach().cpu().numpy())
            #     if sum_b > 1000:
            #         obj_mask_memX0[b] *= occ_memX0[b]
            #         sum_b = torch.sum(obj_mask_memX0[b])
            #         print('reducing this to', sum_b.detach().cpu().numpy())
                    
            feat0_vec = feat_memX0.view(B, hyp.feat3D_dim, -1)
            # this is B x C x huge
            feat0_vec = feat0_vec.permute(0, 2, 1)
            # this is B x huge x C

            obj_mask0_vec = obj_mask_memX0.reshape(B, -1).round()
            occ_mask0_vec = occ_memX0.reshape(B, -1).round()
            free_mask0_vec = free_memX0.reshape(B, -1).round()
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

            feat_vis = []
            occ_vis = []

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
                occ_vis.append(self.summ_writer.summ_occ('', self.occ_memX0s[:,s], only_return=True))

                # inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X, already_mem=False))
                inb = self.vox_util.get_inbounds(self.xyz_camX0s[:,s], self.Z4, self.Y4, self.X, already_mem=False)
                num_inb = torch.sum(inb.float(), axis=1)
                # print('num_inb', num_inb, num_inb.shape)
                inb_counts[:, s] = num_inb.cpu().numpy()

                feat_memI_input = torch.cat([
                    self.occ_memX0s[:,s],
                    self.unp_memX0s[:,s]*self.occ_memX0s[:,s],
                ], dim=1)
                _, feat_memI, valid_memI = self.featnet3D(feat_memI_input)

                self.summ_writer.summ_feat('3D_feats/feat_%d_input' % s, feat_memI_input, pca=True)
                self.summ_writer.summ_feat('3D_feats/feat_%d' % s, feat_memI, pca=True)
                feat_vis.append(self.summ_writer.summ_feat('', feat_memI, pca=True, only_return=True))

                # collect freespace here, to discard bad matches
                _, free_memI_, _, _ = self.vox_util.prep_occs_supervision(
                    self.camX0s_T_camXs[:,s:s+1],
                    self.xyz_camXs[:,s:s+1],
                    Z, Y, X, agg=True)
                free_memI = free_memI_.squeeze(1)
                
                feat_vec = feat_memI.view(B, hyp.feat3D_dim, -1)
                # this is B x C x huge
                feat_vec = feat_vec.permute(0, 2, 1)
                # this is B x huge x C

                memI_T_mem0 = utils_geom.eye_4x4(B)
                # we will fill this up

                # # put these on cpu, to save mem
                # feat0_vec = feat0_vec.detach().cpu()
                # feat_vec = feat_vec.detach().cpu()

                # to simplify the impl, we will iterate over the batch dim
                for b in list(range(B)):
                    feat_vec_b = feat_vec[b]
                    feat0_vec_b = feat0_vec[b]
                    obj_mask0_vec_b = obj_mask0_vec[b]
                    occ_mask0_vec_b = occ_mask0_vec[b]
                    free_mask0_vec_b = free_mask0_vec[b]
                    orig_xyz_b = orig_xyz[b]
                    # these are huge x C

                    careful = False
                    if careful:
                        # start with occ points, since these are definitely observed
                        obj_inds_b = torch.where((occ_mask0_vec_b * obj_mask0_vec_b) > 0)
                        obj_vec_b = feat0_vec_b[obj_inds_b]
                        xyz0 = orig_xyz_b[obj_inds_b]
                        # these are med x C

                        # also take random non-free non-occ points in the mask
                        ok_mask = obj_mask0_vec_b * (1.0 - occ_mask0_vec_b) * (1.0 - free_mask0_vec_b)
                        alt_inds_b = torch.where(ok_mask > 0)
                        alt_vec_b = feat0_vec_b[alt_inds_b]
                        alt_xyz0 = orig_xyz_b[alt_inds_b]
                        # these are med x C

                        # issues arise when "med" is too large
                        num = len(alt_xyz0)
                        max_pts = 2000
                        if num > max_pts:
                            # print('have %d pts; taking a random set of %d pts inside' % (num, max_pts))
                            perm = np.random.permutation(num)
                            alt_vec_b = alt_vec_b[perm[:max_pts]]
                            alt_xyz0 = alt_xyz0[perm[:max_pts]]

                        obj_vec_b = torch.cat([obj_vec_b, alt_vec_b], dim=0)
                        xyz0 = torch.cat([xyz0, alt_xyz0], dim=0)
                        if s==0:
                            print('have %d pts in total' % (len(xyz0)))
                    else:
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
                    # zero out the freespace
                    heat_b = heat_b * (1.0 - free_memI[b:b+1])
                    # make the max zero
                    heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    heat_b = heat_b - heat_b_max
                    # scale up, for numerical stability
                    heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

                    xyzI = utils_basic.argmax3D(heat_b, hard=False, stack=True)
                    # xyzI = utils_basic.argmax3D(heat_b*float(Z*10), hard=False, stack=True)
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
                

            self.summ_writer.summ_rgbs('track/feats', feat_vis)
            self.summ_writer.summ_oneds('track/occs', occ_vis, norm=False)
            
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

        else:
            ious = torch.zeros([self.B, self.S]).float().cuda()
            for s in list(range(self.S)):
                ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camX0s[:,0:1], self.lrt_camX0s[:,s:s+1]).squeeze(1)
            results['ious'] = ious
            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())

            lrt_camX0s_e = self.lrt_camX0s[:,0:1].repeat(1, self.S, 1)
            obj_clist_camX0_e = utils_geom.get_clist_from_lrtlist(lrt_camX0s_e)
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
