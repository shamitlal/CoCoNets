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
# from nets.prinet import PriNet
# from nets.rponet import RpoNet
from nets.forecastnet import ForecastNet
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
import utils_py

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class INTPHYS_FORECAST(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = IntphysForecastModel()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)

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
            
class IntphysForecastModel(nn.Module):
    def __init__(self):
        super(IntphysForecastModel, self).__init__()
            
        self.device = torch.device("cuda")

        self.include_image_summs = True
        
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_pri:
            self.prinet = PriNet()
        if hyp.do_rpo:
            self.rponet = RpoNet()
        if hyp.do_forecast:
            self.forecastnet = ForecastNet()
        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=True)
        global_step = feed['global_step']
        
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        ZZ, ZY, ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9

        pix_T_cams = feed["pix_T_cams"]
        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camXs = feed["origin_T_camXs"]
        xyz_camXs = feed["xyz_camXs"]
        
        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        
        boxlist_camXs = feed["boxlist_camXs"]
        scorelist_s = feed["validlist_s"] # coordinate-less and plural
        vislist_s = feed["vislist_s"] # coordinate-less and plural
        tidlist_s = feed["tidlist_s"] # coordinate-less and plural

        # postproc the boxes:
        scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(
            __p(camXs_T_camRs), __p(boxlist_camXs), __p(tidlist_s),
            Z, Y, X, only_cars=False, pad=0.0))
        
        for b in list(range(B)):
            if torch.sum(scorelist_s[b,0]) == 0:
                return total_loss, results, True

        # rescore again, to keep invisible stuff
        scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(
            __p(camXs_T_camRs), __p(boxlist_camXs), __p(tidlist_s),
            Z, Y, X, only_cars=False, pad=0.0))
            
        lrtlist_camXs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camXs))).reshape(B, S, N, 19)
        # lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))

        (obj_lrtlist_camXs,
         obj_scorelist_s,
         obj_vislist_s,
        ) = utils_misc.collect_object_info(lrtlist_camXs,
                                           tidlist_s,
                                           scorelist_s,
                                           vislist_s,
                                           mod='X',
                                           do_vis=False,
                                           summ_writer=summ_writer)
        # obj_lrtlist_camXs is N x B x S x 19

        
        # xy = torch.randint(low=0, high=H, size=(B, 3, 2)).type(torch.FloatTensor).cuda()
        # heats = utils_improc.draw_circles_at_xy(xy, H, W)
        # heat = torch.max(heats, dim=1, keepdim=True)[0]
        # summ_writer.summ_oned('heat', heat, norm=False)
        

        rgb_camXs = feed["rgb_camXs"]
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        occ_memXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occ_memRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        occ_memXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        unp_memXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unp_memXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))

        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        if self.include_image_summs and summ_writer.save_this:
            for k in list(range(K)):
                obj_lrtlist_camX = obj_lrtlist_camXs[k]
                # this is B x S x 19
                obj_scorelist = obj_scorelist_s[k]
                # this is B x S
                obj_clist = utils_geom.get_clist_from_lrtlist(obj_lrtlist_camX).squeeze(2)
                # this is B x S x 3
                obj_clist = obj_clist*obj_scorelist.unsqueeze(2)
                summ_writer.summ_traj_on_occ('traj/obj%d_clist' % k,
                                             obj_clist, occ_memXs[:,0], already_mem=False)
                
            lrt_vis = []
            for s in list(range(S)):
                depth_vis = summ_writer.summ_oned('', depth_camXs[:,s],
                                                  logvis=True, maxval=20, norm=False, only_return=True)
                depth_vis = utils_improc.preprocess_color(depth_vis)
                o = summ_writer.summ_lrtlist('', depth_vis[0:1], lrtlist_camXs[0:1,s],
                                             scorelist_s[0:1,s], tidlist_s[0:1,s], pix_T_cams[0:1,s], only_return=True)
                lrt_vis.append(o)
            summ_writer.summ_rgbs('obj/lrtlist_camXs_on_depthXs', lrt_vis)
                
            summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
            summ_writer.summ_oneds('2D_inputs/depth_camXs',
                                   torch.unbind(depth_camXs, dim=1), logvis=True, maxval=20, norm=False)
            summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
            
            summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(occ_memXs, dim=1))
            summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(occ_memRs, dim=1), reduce_axes=[2,3])

        if hyp.do_feat:
            # occ_memXs is B x S x 1 x H x W x D
            # unp_memXs is B x S x 3 x H x W x D

            # occ_memXs_input = F.dropout(occ_memXs, p=0.75, training=(feed['set_name']=='train'))
            occ_memXs_input = occ_memXs[:,:2]
            feat_memXs_input = torch.cat([occ_memXs_input, occ_memXs_input*unp_memXs[:,:2]], dim=2)
            feat_memXs_input_ = __p(feat_memXs_input)

            # feat_memXs_input = torch.cat([occ_memXs, occ_memXs*unp_memXs], dim=2)
            # feat_memXs_input = torch.cat([occ_memXs, occ_memXs*unp_memXs], dim=2)
            # feat_memXs_input_ = __p(feat_memXs_input)
            
            # it is useful to keep track of what was visible from each viewpoint
            free_memXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occ_memXs_half))
            free_memXs = __u(free_memXs_)
            vis_memXs = torch.clamp(occ_memXs_half+free_memXs, 0.0, 1.0)

            feat_memXs_, valid_memXs_, feat_loss = self.featnet(
                feat_memXs_input_,
                summ_writer,
                comp_mask=__p(occ_memXs_input),
            )
            total_loss += feat_loss

            feat_memXs = __u(feat_memXs_)
            valid_memXs = __u(valid_memXs_)
            
            summ_writer.summ_feats('3D_feats/feat_memXs_input', torch.unbind(feat_memXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/feat_memXs_output', torch.unbind(feat_memXs, dim=1),
                                   valids=torch.unbind(valid_memXs, dim=1), pca=True)
            
        if hyp.do_occ:
            # note we assume each cam is along the same view
            occ_memXs_sup, free_memXs_sup, _, _ = utils_vox.prep_occs_supervision(
                utils_geom.eye_4x4s(B, S),
                xyz_camXs,
                Z2, Y2, X2, 
                agg=False)
            summ_writer.summ_occs('occ_sup/free_memXs_sup', torch.unbind(free_memXs_sup, dim=1))
            summ_writer.summ_occs('occ_sup/occ_memXs_sup', torch.unbind(occ_memXs_sup, dim=1))

            occ_loss, occ_memXs_pred_ = self.occnet(
                __p(feat_memXs),
                __p(occ_memXs_sup),
                __p(free_memXs_sup),
                __p(valid_memXs), 
                summ_writer,
            )
            occ_memXs_pred = __u(occ_memXs_pred_)
            total_loss += occ_loss
            
        if hyp.do_pri:
            pri_input_memX = torch.cat([feat_memXs[:,0], feat_memXs[:,1]], dim=1)
            # this is B x hyp.feat_dim*2 x Z x Y x X
            
            pri_loss = self.prinet(
                pri_input_memX,
                obj_lrtlist_camXs,
                obj_scorelist_s,
                summ_writer)
            total_loss += pri_loss
            
        if hyp.do_rpo:
            rpo_input_memX = torch.cat([feat_memXs[:,0], feat_memXs[:,1]], dim=1)
            # this is B x hyp.feat_dim*2 x Z x Y x X
            
            rpo_loss = self.rponet(
                rpo_input_memX,
                obj_lrtlist_camXs,
                obj_scorelist_s,
                summ_writer)
            total_loss += rpo_loss
            
        if hyp.do_forecast:

            if hyp.do_feat:
                forecast_input_memX = torch.cat([feat_memXs[:,0], feat_memXs[:,1]], dim=1)
                # this is B x hyp.feat_dim*2 x Z x Y x X
            else:
                # pre_input = torch.cat([occ_memXs, occ_memXs*unp_memXs], dim=2)
                pre_input = torch.cat([occ_memXs_half, occ_memXs_half*unp_memXs_half], dim=2)
                forecast_input_memX = torch.cat([pre_input[:,0], pre_input[:,1]], dim=1)
            
            # obj_lrtlist_cams is N x B x S x 19
            obj_lrtlist_camXs_ = obj_lrtlist_camXs.reshape(N*B, S, 19)
            obj_clist_camXs_ = utils_geom.get_clist_from_lrtlist(obj_lrtlist_camXs_)
            obj_clist_camXs = obj_clist_camXs_.reshape(N, B, S, 3)

            obj0_clist_camXs = obj_clist_camXs[0]
            # this is B x S x 3
            
            # for ind in range(3):
            #     print('obj%d_clist' % ind, np.squeeze(obj_clist_camXs[ind,0,:,0].cpu().detach().numpy()))
            #     print('obj%d_scorelist' % ind, np.squeeze(obj_scorelist_s[ind,0].cpu().detach().numpy()))

            summ_writer.summ_traj_on_occ('forecast/true_traj',
                                         obj0_clist_camXs*obj_scorelist_s[0].unsqueeze(2),
                                         torch.max(occ_memXs, dim=1)[0], 
                                         already_mem=False,
                                         sigma=2)
            
            normalize_yaw = True
            if normalize_yaw:            
                # i only need to rotate two things: obj_clist_camXs[0], and xyz_camXs

                rot0 = utils_geom.eye_3x3(B)
                
                xyz0 = obj0_clist_camXs[:,0]
                xyz1 = obj0_clist_camXs[:,1]
                # these are B x 3

                delta = xyz1-xyz0
                delta = delta.detach().cpu().numpy()
                dx = delta[:,0]
                dy = delta[:,1]
                dz = delta[:,2]

                yaw = np.arctan2(dz, dx)
                rot = np.stack([utils_py.eul2rotm(0,y,0) for y in yaw])
                rot = torch.from_numpy(rot).float().cuda()
                # rot is B x 3 x 3
                t = -xyz0
                # t is B x 3
                t0 = torch.zeros_like(t)
                # noyaw_T_yaw = utils_geom.merge_rt(rot, t)

                zero_T_camX = utils_geom.merge_rt(rot0, t)
                noyaw_T_zero = utils_geom.merge_rt(rot, t0)
                
                mid_x = (hyp.XMAX + hyp.XMIN)/2.0
                mid_y = (hyp.YMAX + hyp.YMIN)/2.0
                mid_z = (hyp.ZMAX + hyp.ZMIN)/2.0
                mid_xyz = np.array([mid_x, mid_y, mid_z]).reshape(1, 3)
                tra = torch.from_numpy(mid_xyz).float().cuda().repeat(B, 1)
                center_T_noyaw = utils_geom.merge_rt(rot0, tra)

                noyaw_T_camX = utils_basic.matmul3(center_T_noyaw, noyaw_T_zero, zero_T_camX)
                
                noyaw_T_camXs = noyaw_T_camX.unsqueeze(1).repeat(1, S, 1, 1)
                
                noyaw_obj0_clist_camXs = utils_geom.apply_4x4(noyaw_T_camX, obj0_clist_camXs)

                # summ_writer.summ_traj_on_occ('forecast/noyaw_traj',
                #                              obj0_clist_camXs*scorelist_s[0],
                #                              torch.max(occ_memXs, dim=1)[0], 
                #                              already_mem=False,
                #                              sigma=2)

                # new_trajs = np.zeros_like(all_trajs)
                # for n in list(range(N)):
                #     traj = all_trajs[n]
                #     traj = np.dot(rot[n], traj.T).T
                #     new_trajs[n] = traj
                
                # traj = all_trajs[n]
                # traj = np.dot(rot[n], traj.T).T
                # new_trajs[n] = traj

                noyaw_xyz_camXs = __u(utils_geom.apply_4x4(__p(noyaw_T_camXs), __p(xyz_camXs)))
                
                noyaw_obj0_occXs = __u(utils_vox.voxelize_near_xyz(__p(noyaw_xyz_camXs), __p(noyaw_obj0_clist_camXs), ZZ, ZY, ZX))
                summ_writer.summ_occs('3D_inputs/noyaw_obj0_occXs', torch.unbind(noyaw_obj0_occXs, dim=1))

                noyaw_occ_memXs = __u(utils_vox.voxelize_xyz(__p(noyaw_xyz_camXs), ZZ, ZY, ZX))

                forecast_loss = self.forecastnet(
                    noyaw_obj0_occXs, 
                    noyaw_obj0_clist_camXs,
                    obj_scorelist_s[0],
                    obj_vislist_s[0],
                    noyaw_occ_memXs, 
                    summ_writer)
                total_loss += forecast_loss
            else:
                obj0_occXs = __u(utils_vox.voxelize_near_xyz(__p(xyz_camXs), __p(obj_clist_camXs[0]), ZZ, ZY, ZX))
                summ_writer.summ_occs('3D_inputs/obj0_occXs', torch.unbind(obj0_occXs, dim=1))
                    
                forecast_loss = self.forecastnet(
                    obj0_occXs, 
                    obj_clist_camXs[0],
                    obj_scorelist_s[0],
                    obj_vislist_s[0],
                    occ_memXs,
                    summ_writer)
                total_loss += forecast_loss
            
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False


