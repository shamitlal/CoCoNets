import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import os

from model_base import Model
from nets.featnet3D import FeatNet3D
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.detnet import DetNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

import vox_util
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net
import utils_eval

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_DET(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaDetModel()
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_det:
            self.model.detnet.eval()
            self.set_requires_grad(self.model.detnet, False)

    # def go(self):
    #     self.start_time = time.time()
    #     self.initialize_model()
    #     print("------ Done creating models ------")
    #     if hyp.lr > 0:
    #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
    #         self.start_iter = saverloader.load_weights(self.model, self.optimizer)
    #         print("------ Done loading weights ------")
    #     else:
    #         self.start_iter = 0

    #     set_nums = []
    #     set_names = []
    #     set_inputs = []
    #     set_writers = []
    #     set_log_freqs = []
    #     set_do_backprops = []
    #     set_dicts = []
    #     set_loaders = []

    #     for set_name in hyp.set_names:
    #         if hyp.sets_to_run[set_name]:
    #             set_nums.append(hyp.set_nums[set_name])
    #             set_names.append(set_name)
    #             set_inputs.append(self.all_inputs[set_name])
    #             set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
    #             set_log_freqs.append(hyp.log_freqs[set_name])
    #             set_do_backprops.append(hyp.sets_to_backprop[set_name])
    #             set_dicts.append({})
    #             set_loaders.append(iter(set_inputs[-1]))

    #     for step in list(range(self.start_iter+1, hyp.max_iters+1)):
    #         for i, (set_input) in enumerate(set_inputs):
    #             if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
    #                 set_loaders[i] = iter(set_input)

    #         for (set_num,
    #              set_name,
    #              set_input,
    #              set_writer,
    #              set_log_freq,
    #              set_do_backprop,
    #              set_dict,
    #              set_loader
    #         ) in zip(
    #             set_nums,
    #             set_names,
    #             set_inputs,
    #             set_writers,
    #             set_log_freqs,
    #             set_do_backprops,
    #             set_dicts,
    #             set_loaders
    #         ):   

    #             log_this = np.mod(step, set_log_freq)==0
    #             total_time, read_time, iter_time = 0.0, 0.0, 0.0

    #             if log_this or set_do_backprop:
                          
    #                 read_start_time = time.time()

    #                 feed, _ = next(set_loader)
    #                 feed_cuda = {}
    #                 for k in feed:
    #                     try:
    #                         feed_cuda[k] = feed[k].cuda(non_blocking=True)
    #                     except:
    #                         # some things are not tensors (e.g., filename)
    #                         feed_cuda[k] = feed[k]

    #                 # feed_cuda = next(iter(set_input))
    #                 read_time = time.time() - read_start_time
                    
    #                 feed_cuda['writer'] = set_writer
    #                 feed_cuda['global_step'] = step
    #                 feed_cuda['set_num'] = set_num
    #                 feed_cuda['set_name'] = set_name

    #                 filename = feed_cuda['filename'][0]
    #                 # print('filename = %s' % filename)
    #                 tokens = filename.split('/')
    #                 filename = tokens[-1]
    #                 # print('new filename = %s' % filename)
                    
    #                 iter_start_time = time.time()
    #                 if set_do_backprop:
    #                     self.model.train()
    #                     loss, results, returned_early = self.model(feed_cuda)
    #                 else:
    #                     self.model.eval()
    #                     with torch.no_grad():
    #                         loss, results, returned_early = self.model(feed_cuda)
    #                 loss_py = loss.cpu().item()
                    
    #                 if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
    #                     self.optimizer.zero_grad()
    #                     loss.backward()
    #                     self.optimizer.step()
    #                 iter_time = time.time()-iter_start_time
    #                 total_time = time.time()-self.start_time

    #                 print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
    #                                                                                     step,
    #                                                                                     hyp.max_iters,
    #                                                                                     total_time,
    #                                                                                     read_time,
    #                                                                                     iter_time,
    #                                                                                     loss_py,
    #                                                                                     set_name))
            
    #         if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
    #             saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

    #     for writer in set_writers: #close writers to flush cache into file
    #         writer.close()
            

class CarlaDetModel(nn.Module):
    def __init__(self):
        super(CarlaDetModel, self).__init__()
            
        self.device = torch.device("cuda")

        self.include_image_summs = True
        
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
        if hyp.do_det:
            self.detnet = DetNet()
        
    def prepare_common_tensors(self, feed):

        results = dict()
        
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        # if self.set_name=='test':
        #     self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        # elif self.set_name=='val':
        #     self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        # else:
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(__p(self.camR0s_T_camRs).inverse())
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        
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
            all_ok = True
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
            # we want to ensure this gives us a few points inbound for each batch el
            inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X4, already_mem=False))
            num_inb = torch.sum(inb.float(), axis=2)
            if torch.min(num_inb) < 200:
                all_ok = False
            if num_tries > 100:
                return False
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
        
        _boxlist_camRs = feed["boxlists"]
        _tidlist_s = feed["tidlists"] # coordinate-less and plural
        _scorelist_s = feed["scorelists"] # coordinate-less and plural
        _scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(
            utils_geom.eye_4x4(self.B*self.S),
            __p(_boxlist_camRs),
            __p(_tidlist_s),
            self.Z, self.Y, self.X,
            self.vox_util,
            only_cars=False, pad=2.0))
        boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            __p(_boxlist_camRs), __p(_tidlist_s), __p(_scorelist_s))
        self.boxlist_camRs = __u(boxlist_camRs_)
        self.tidlist_s = __u(tidlist_s_)
        self.scorelist_s = __u(scorelist_s_)

        # for b in list(range(self.B)):
        #     # if torch.sum(scorelist_s[b,0]) == 0:
        #     if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
        #         return 0.0, None, True

        N = self.boxlist_camRs.shape[2]
        origin_T_camRs = self.origin_T_camRs.unsqueeze(2).repeat(1, 1, N, 1, 1)
        lrtlist_camRs_ = utils_misc.parse_boxes(__p(self.boxlist_camRs), __p(origin_T_camRs))
        lrtlist_camXs_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), lrtlist_camRs_)
        lrtlist_camX0s_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), lrtlist_camXs_)
        scorelist_s_ = utils_misc.rescore_lrtlist_with_inbound(
            lrtlist_camX0s_, __p(self.tidlist_s), self.Z, self.Y, self.X, self.vox_util)
        self.lrtlist_camX0s = __u(lrtlist_camX0s_)

        return True # OK
        
    def run_detector(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        results = dict()

        self.rgb_camXs = feed['rgb_camXs']

        if hyp.do_feat3D:
            self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
                __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
            self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
            self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
            self.occ_memX0 = self.occ_memX0s[:,0]
            
            feat_memX0_input = torch.cat([
                self.occ_memX0s[:,0],
                self.rgb_memX0s[:,0]*self.occ_memX0s[:,0],
            ], dim=1)
            feat3D_loss, feat_memX0, valid_memX0 = self.featnet3D(
                feat_memX0_input,
                self.summ_writer,
            )
            total_loss += feat3D_loss
            self.summ_writer.summ_feat('3D_feats/feat_memX0_input', feat_memX0_input, pca=True)
            self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, pca=True)
            
        if hyp.do_det:

            self.occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z, self.Y, self.X)
            self.rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
                self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            
            feat_memX0_input = torch.cat([
                self.occ_memX0,
                self.rgb_memX0*self.occ_memX0,
            ], dim=1)
            
            
            lrtlist_camX = self.lrtlist_camX0s[:, 0]
            axlrtlist_camX = utils_geom.inflate_to_axis_aligned_lrtlist(lrtlist_camX)
            lrtlist_memX = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_camX, self.Z, self.Y, self.X)
            axlrtlist_memX = utils_geom.inflate_to_axis_aligned_lrtlist(lrtlist_memX)
            self.summ_writer.summ_lrtlist_bev(
                'det/boxlist_g',
                self.occ_memX0[0:1],
                lrtlist_memX[0:1],
                torch.ones(1, 50).float().cuda(),  # scores
                torch.ones(1, 50).long().cuda(),  # tids
                self.vox_util, 
                already_mem=True)
            self.summ_writer.summ_lrtlist_bev(
                'det/axboxlist_g',
                self.occ_memX0[0:1],
                axlrtlist_memX[0:1],
                torch.ones(1, 50).float().cuda(),  # scores
                torch.ones(1, 50).long().cuda(),  # tids
                self.vox_util, 
                already_mem=True)

            lrtlist_halfmemX = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_camX, self.Z2, self.Y2, self.X2)
            # print('lrtlist_camR', lrtlist_camR[:, 0])
            # print('lrtlist_camR_check', utils_vox.apply_ref_T_mem_to_lrtlist(lrtlist_halfmemR, self.Z2, self.Y2, self.X2)[:, 0])
            axlrtlist_halfmemX = utils_geom.inflate_to_axis_aligned_lrtlist(lrtlist_halfmemX)
            # print('axlrtlist_halfmem_g', axlrtlist_halfmemR[:, 0])

            # axlrtlist_halfmemX_check = self.vox_util.apply_mem_T_ref_to_lrtlist(axlrtlist_camX, self.Z2, self.Y2,
            #                                                                     self.X2)
            # print('axlrtlist_halfmemR_check', axlrtlist_halfmemR_check[:, 0])

            # feat_memX0 = torch.mean(feat_memX0s, dim=1)
            

            detect_loss, boxlist_halfmemX_e, scorelist_e, tidlist_e, pred_objectness, sco, ove = self.detnet(
                axlrtlist_halfmemX,
                self.scorelist_s[:, 0],
                feat_memX0_input,
                self.summ_writer)
            lrtlist_halfmemX_e = utils_geom.convert_boxlist_to_lrtlist(boxlist_halfmemX_e)
            # print('lenlist_halfmem_e', utils_geom.get_lenlist_from_lrtlist(lrtlist_halfmemR_e))
            lrtlist_camX_e = self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_halfmemX_e, self.Z2, self.Y2, self.X2)
            # print('lenlist_cam_e', utils_geom.get_lenlist_from_lrtlist(lrtlist_camR_e))
            total_loss += detect_loss

            lrtlist_e = lrtlist_camX_e[0:1]
            lrtlist_g = lrtlist_camX[0:1]
            scorelist_e = scorelist_e[0:1]
            scorelist_g = self.scorelist_s[0:1, 0]
            lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                lrtlist_e, lrtlist_g, scorelist_e, scorelist_g)

            lenlist_e, _ = utils_geom.split_lrtlist(lrtlist_e)
            clist_e = utils_geom.get_clist_from_lrtlist(lrtlist_e)
            lenlist_g, _ = utils_geom.split_lrtlist(lrtlist_g)
            clist_g = utils_geom.get_clist_from_lrtlist(lrtlist_g)
            axlenlist_g, _ = utils_geom.split_lrtlist(axlrtlist_camX[0:1])
            axclist_g = utils_geom.get_clist_from_lrtlist(axlrtlist_camX[0:1])

            if self.include_image_summs:
                self.summ_writer.summ_lrtlist('obj/boxlist_e', self.rgb_camXs[:1, 0],
                                              lrtlist_e,
                                              scorelist_e,
                                              tidlist_e, self.pix_T_cams[:1, 0])
                _, Ne, _ = list(lrtlist_e.shape)
                _, Ng, _ = list(lrtlist_g.shape)
                # there may be no prediction or gt in the scene.
                if Ne > 0 and Ng > 0:
                    lrtlist_e_ = lrtlist_e.unsqueeze(2).repeat(1, 1, Ng, 1).reshape(1, Ne * Ng, -1)
                    lrtlist_g_ = lrtlist_g.unsqueeze(1).repeat(1, Ne, 1, 1).reshape(1, Ne * Ng, -1)
                    ious = utils_geom.get_iou_from_corresponded_lrtlists(lrtlist_e_, lrtlist_g_)
                    ious = ious.reshape(1, Ne, Ng)
                    ious_e = torch.max(ious, dim=2)[0]
                    self.summ_writer.summ_lrtlist('obj/boxlist', self.rgb_camXs[0:1, 0],
                                                  torch.cat((lrtlist_e, lrtlist_g), dim=1),
                                                  torch.cat((ious_e, ious_e.new_ones(1, Ng)), dim=1),
                                                  torch.cat((ious_e.new_ones(1, Ne).long(),
                                                             ious_e.new_ones(1, Ng).long() * 2), dim=1),
                                                  self.pix_T_cams[0:1, 0])

                    self.summ_writer.summ_lrtlist_bev('det/boxlist_e', self.occ_memX0[:1],
                                                      lrtlist_e,
                                                      scorelist_e,
                                                      tidlist_e,
                                                      self.vox_util, 
                                                      already_mem=False)
                    # visualize the gt and prediction on occ
                    self.summ_writer.summ_lrtlist_bev('det/boxlist', self.occ_memX0[0:1],
                                                      torch.cat((lrtlist_e, lrtlist_g), dim=1),
                                                      torch.cat((ious_e, ious_e.new_ones(1, Ng)), dim=1),
                                                      torch.cat((ious_e.new_ones(1, Ne).long(),
                                                                 ious_e.new_ones(1, Ng).long() * 2), dim=1),
                                                      self.vox_util, 
                                                      already_mem=False)

                ious = [0.3, 0.4, 0.5, 0.6, 0.7]
                maps = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, ious)
                for ind, overlap in enumerate(ious):
                    self.summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])


            
            # axboxlist_camRs = __u(utils_geom.inflate_to_axis_aligned_boxlist(__p(self.boxlist_camRs)))
            # axlrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(axboxlist_camRs)))
            # if self.include_image_summs:
            #     self.summ_writer.summ_lrtlist('obj/axboxlist', self.rgb_camRs[:,0], axlrtlist_camRs[:,0],
            #                              self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])

            # boxlist_memR = utils_vox.convert_boxlist_camR_to_memR(self.boxlist_camRs[:,0], self.Z2, self.Y2, self.X2)
            # axboxlist_memR = utils_geom.inflate_to_axis_aligned_boxlist(boxlist_memR)

            # featRs = utils_vox.apply_4x4s_to_voxs(self.camRs_T_camXs, feat_memXs)
            # featR = torch.mean(featRs, dim=1)

            # detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
            #     axboxlist_memR,
            #     self.scorelist_s[:,0],
            #     featR,
            #     self.summ_writer)
            # total_loss += detect_loss

            # boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, self.Z2, self.Y2, self.X2)
            # lrtlist_camR_e = utils_geom.convert_boxlist_to_lrtlist(boxlist_camR_e)
            # if self.include_image_summs:
            #     self.summ_writer.summ_lrtlist('obj/boxlist_e', self.rgb_camRs[:,0], lrtlist_camR_e,
            #                              scorelist_e, tidlist_e, self.pix_T_cams[:,0])

            # boxlist_e = boxlist_camR_e[0:1].detach().cpu().numpy()
            # boxlist_g = self.boxlist_camRs[0:1,0].detach().cpu().numpy()
            # scorelist_e = scorelist_e[0:1].detach().cpu().numpy()
            # scorelist_g = self.scorelist_s[0:1,0].detach().cpu().numpy()
            # boxlist_e, boxlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_boxes(
            #     boxlist_e, boxlist_g, scorelist_e, scorelist_g)

            # ious = [0.3, 0.4, 0.5, 0.6, 0.7]
            # maps = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
            # for ind, overlap in enumerate(ious):
            #     self.summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
        
        
    def forward(self, feed):
        self.prepare_common_tensors(feed)
        return self.run_detector(feed)
        

    
