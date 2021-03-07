import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import os

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.detnet import DetNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

import utils_vox
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

class INTPHYS_DET(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = IntphysDetModel()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        # if hyp.do_freeze_view:
        #     self.model.viewnet.eval()
        #     self.set_requires_grad(self.model.viewnet, False)
        # if hyp.do_freeze_preocc:
        #     self.model.preoccnet.eval()
        #     self.set_requires_grad(self.model.preoccnet, False)
        # if hyp.do_freeze_emb2D:
        #     self.model.embnet2D.eval()
        #     self.set_requires_grad(self.model.embnet2D, False)

    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")
        if hyp.lr > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
            self.start_iter = saverloader.load_weights(self.model, self.optimizer)
            print("------ Done loading weights ------")
        else:
            self.start_iter = 0

        set_nums = []
        set_names = []
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
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
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
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_names,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders
            ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                if log_this or set_do_backprop:
                          
                    read_start_time = time.time()

                    feed = next(set_loader)
                    feed_cuda = {}
                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True)
                        except:
                            # some things are not tensors (e.g., filename)
                            feed_cuda[k] = feed[k]

                    # feed_cuda = next(iter(set_input))
                    read_time = time.time() - read_start_time
                    
                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_name'] = set_name


                    # filename = feed_cuda['filename'][0]
                    # # print('filename = %s' % filename)
                    # tokens = filename.split('/')
                    # filename = tokens[-1]
                    # # print('new filename = %s' % filename)
                    
                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_py = loss.cpu().item()
                    
                    if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    if not returned_early:
                        print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                            step,
                                                                                            hyp.max_iters,
                                                                                            total_time,
                                                                                            read_time,
                                                                                            iter_time,
                                                                                            loss_py,
                                                                                            set_name))
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: # close writers, to flush cache into the file
            writer.close()
            
class IntphysDetModel(nn.Module):
    def __init__(self):
        super(IntphysDetModel, self).__init__()

        # self.feat_net = frozen_flow_net.FrozenFeatNet(
        #     '/projects/katefgroup/cvpr2020_share/frozen_flow_net/feats_model.pb')
        # self.flow_net = frozen_flow_net.FrozenFlowNet(
        #     '/projects/katefgroup/cvpr2020_share/frozen_flow_net/flow_model_no_dep.pb')
            
        self.device = torch.device("cuda")

        self.include_image_summs = True
        
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_det:
            self.detnet = DetNet()
        
        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=True)
        global_step = feed['global_step']
        
        total_loss = torch.tensor(0.0).cuda()

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        assert(S==1) # in this mode, we assume single-view, single-timestep data
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9

        # we immediately drop X/R notation, since we will only use X
        rgb_cam = feed["rgb_camXs"][:,0]
        pix_T_cam = feed["pix_T_cams"][:,0]
        xyz_cam = feed["xyz_camXs"][:,0]
        boxlist_cam = feed["boxlist_camXs"][:,0]
        vislist = feed["vislist_s"][:,0]
        tidlist = feed["tidlist_s"][:,0]
        inboundlist = utils_misc.rescore_boxlist_with_inbound(
            utils_geom.eye_4x4(B), boxlist_cam, tidlist,
            Z, Y, X, only_cars=False, pad=0.0)
        vislist = vislist * inboundlist

        for b in list(range(B)):
            if torch.sum(vislist[b]) == 0:
                return total_loss, results, True
            
        lrtlist_cam = utils_geom.convert_boxlist_to_lrtlist(boxlist_cam).reshape(B, N, 19)
        occ_mem = utils_vox.voxelize_xyz(xyz_cam, Z, Y, X)

        ## projected depth, and inbound mask
        depth_cam, valid_cam = utils_geom.create_depth_image(pix_T_cam, xyz_cam, H, W)
        dense_xyz_cam = utils_geom.depth2pointcloud(depth_cam, pix_T_cam)
        inbound_cam = utils_vox.get_inbounds(dense_xyz_cam, Z, Y, X).float()
        inbound_cam = torch.reshape(inbound_cam, [B, 1, H, W])
        valid_cam = valid_cam * inbound_cam

        if self.include_image_summs and summ_writer.save_this:
            depth_vis = summ_writer.summ_oned('', depth_cam, logvis=True, maxval=20, norm=False, only_return=True)
            depth_vis = utils_improc.preprocess_color(depth_vis)
            summ_writer.summ_lrtlist('obj/lrtlist_cam', rgb_cam, lrtlist_cam,
                                     vislist, tidlist, pix_T_cam)
            summ_writer.summ_lrtlist('obj/lrtlist_cam_on_depth', depth_vis[0:1], lrtlist_cam[0:1],
                                     vislist[0:1], tidlist[0:1], pix_T_cam[0:1])
        
            summ_writer.summ_rgb('2D_inputs/rgb_cam', rgb_cam)
            summ_writer.summ_oned('2D_inputs/depth_cam', depth_cam, logvis=True, maxval=20, norm=False)
            summ_writer.summ_oned('2D_inputs/valid_cam', valid_cam)
            
            summ_writer.summ_occ('3D_inputs/occ_mem', occ_mem)
            # summ_writer.summ_unps('3D_inputs/unp_mems', torch.unbind(unp_mems, dim=1), torch.unbind(occ_mems, dim=1))

        if hyp.do_feat:
            # occ_mem is B x 1 x H x W x D

            feat_mem_input = occ_mem
            feat_mem, valid_mem, feat_loss = self.featnet(
                feat_mem_input,
                summ_writer,
                comp_mask=occ_mem,
            )
            total_loss += feat_loss

            summ_writer.summ_feat('3D_feats/feat_mem_input', feat_mem_input, pca=False)
            summ_writer.summ_feat('3D_feats/feat_mem_output', feat_mem, valid_mem, pca=True)
            
        if hyp.do_occ:
            occ_mem_sup = utils_vox.voxelize_xyz(xyz_cam, Z2, Y2, X2)
            free_mem_sup = utils_vox.get_freespace(xyz_cam, occ_mem_sup)
            summ_writer.summ_occ('occ_sup/occ_mem', occ_mem_sup)
            summ_writer.summ_occ('occ_sup/free_mem', free_mem_sup)
                
            occ_loss, occ_mem_pred = self.occnet(
                feat_mem,
                occ_mem_sup,
                free_mem_sup,
                valid_mem, 
                summ_writer)
            total_loss += occ_loss
                
        if hyp.do_det:
            
            boxlist_mem = utils_vox.convert_boxlist_camR_to_memR(boxlist_cam, Z2, Y2, X2)
            axboxlist_mem = utils_geom.inflate_to_axis_aligned_boxlist(boxlist_mem)

            detect_loss, boxlist_mem_e, scorelist_e, tidlist_e, _, _ = self.detnet(
                axboxlist_mem,
                vislist,
                feat_mem,
                summ_writer)
            total_loss += detect_loss

            if boxlist_mem_e is not None:
                # note the returned data has batchsize=1
                
                boxlist_cam_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_mem_e, Z2, Y2, X2)
                
                axboxlist_cam = utils_geom.inflate_to_axis_aligned_boxlist(boxlist_cam)
                
                if self.include_image_summs:
                    axlrtlist_cam = utils_geom.convert_boxlist_to_lrtlist(axboxlist_cam)
                    lrtlist_cam_e = utils_geom.convert_boxlist_to_lrtlist(boxlist_cam_e)
                    summ_writer.summ_lrtlist('det/axboxlist_cam_e', rgb_cam[0:1], lrtlist_cam_e,
                                             scorelist_e, tidlist_e, pix_T_cam[0:1])
                    
                    summ_writer.summ_lrtlist('det/axlrtlist_cam_g', rgb_cam, axlrtlist_cam,
                                             vislist, tidlist, pix_T_cam)

                boxlist_e = boxlist_cam_e.detach().cpu().numpy()
                boxlist_g = axboxlist_cam[0:1].detach().cpu().numpy()
                scorelist_e = scorelist_e.detach().cpu().numpy()
                scorelist_g = vislist[0:1].detach().cpu().numpy()
                boxlist_e, boxlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_boxes(
                    boxlist_e, boxlist_g, scorelist_e, scorelist_g)

                ious = [0.3, 0.4, 0.5, 0.6, 0.7]
                maps = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
                for ind, overlap in enumerate(ious):
                    summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])
            
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False


