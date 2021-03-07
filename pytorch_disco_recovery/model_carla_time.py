import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from tensorboardX import SummaryWriter
from backend import saverloader, inputs

from model_base import Model
from nets.featnet3D import FeatNet3D
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.locnet import LocNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

from utils_basic import *
import vox_util
import search_util
import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net

np.set_printoptions(precision=2)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_TIME(Model):
    # take over __init__() from base
    def __init__(self, checkpoint_dir, log_dir):

        print('------ CREATING NEW MODEL ------')
        print(hyp.name)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.all_inputs = inputs.get_inputs()
        print("------ Done getting inputs ------")
        
        self.device = torch.device("cuda")
        
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaTimeModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

class CarlaTimeModel(nn.Module):
    def __init__(self):
        super(CarlaTimeModel, self).__init__()
        self.featnet3D = FeatNet3D(in_dim=4)

        self.B, self.H, self.W, self.V, self.N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.ZZ, self.ZY, self.ZX = hyp.ZY, hyp.ZY, hyp.ZX
        self.vox_size_X = (hyp.XMAX-hyp.XMIN)/self.X
        self.vox_size_Y = (hyp.YMAX-hyp.YMIN)/self.Y
        self.vox_size_Z = (hyp.ZMAX-hyp.ZMIN)/self.Z
        
        self.locnet = LocNet(int(self.Z/4),
                             int(self.Y/4),
                             int(self.X/4),
                             int(self.ZZ/4),
                             int(self.ZY/4),
                             int(self.ZX/4))
        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

    def prepare_common_tensors(self, feed):
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.vox_util = vox_util.Vox_util(feed['set_name'], delta=0.0, assert_cube=True)
        self.search_util = search_util.Search_util(feed['set_name'])

        self.search_X = int(self.ZX * self.search_util.search_size_factor)
        self.search_Y = int(self.ZY * self.search_util.search_size_factor)
        self.search_Z = int(self.ZZ * self.search_util.search_size_factor)

        self.S = feed["set_seqlen"]
        
        __p = lambda x: pack_seqdim(x, self.B)
        __u = lambda x: unpack_seqdim(x, self.B)

        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        self.camXs_T_camX0 = __u(__p(self.camX0_T_camXs).inverse())
        # self.camX0_T_camR = torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))

        
        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))

        self.occ_memX0s = __u(utils_vox.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memX0s_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))
        self.occ_memXs_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))

        unp_memXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unp_memX0s = utils_vox.apply_4x4s_to_voxs(self.camX0_T_camXs, unp_memXs)

        
        self.input_memX0s = torch.cat([self.occ_memX0s, self.occ_memX0s*self.unp_memX0s], dim=2)

        self.box_camRs = feed["box_traj_camR"]
        score_s = feed["score_traj"]
        tid_s = torch.ones_like(score_s).long()

        # let's immediately adjust the boxes, so that they are uniform size
        obj_cxs,obj_cys,obj_czs,obj_lxs,obj_lys,obj_lzs,obj_rxs,obj_rys,obj_rzs = torch.unbind(
            self.box_camRs, dim=2)
        lx = (self.ZZ*self.vox_size_Z)
        ly = (self.ZY*self.vox_size_Y)
        lz = (self.ZX*self.vox_size_X)
        obj_len = torch.from_numpy(np.reshape(np.array(
            [lx, ly, lz], np.float32), [1, 3])).cuda().repeat([self.B, 1])
        print('lx, ly, lz', lx, ly, lz)
        self.box_camRs = torch.stack([
            obj_cxs, 
            obj_cys, 
            obj_czs, 
            torch.ones_like(obj_lxs)*lx, 
            torch.ones_like(obj_lys)*ly, 
            torch.ones_like(obj_lzs)*lz, 
            obj_rxs, 
            obj_rys, 
            obj_rzs], dim=2)
        # box_camRs is B x S x 9
        lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(self.box_camRs)
        lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs)
        self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0_T_camXs, lrt_camXs)
        
        if self.include_image_summs:
            visX_g = []
            for s in list(range(self.S)):
                visX_g.append(self.summ_writer.summ_lrtlist(
                    'obj/box_camX%d_g' % s, self.rgb_camXs[:,s], lrt_camXs[:,s:s+1],
                    score_s[:,s:s+1], tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('obj/box_camXs_g', visX_g)
        
        #####################
        ## visualize what we got
        #####################
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        self.summ_writer.summ_occ('3D_inputs/occ_memX0s_ind0', self.occ_memX0s[:,0])
        
    def run_train(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0).cuda()
        self.K = 8
        __p = lambda x: pack_seqdim(x, self.B)
        __u = lambda x: unpack_seqdim(x, self.B)

        # idea here is:
        # extract a crop in this frame
        # feed it to locnet, which spits out the absolute RT of the object
        # do the warp to get the new crop
        # repeat

        # here, i need to be careful
        # i want the crop to be the same resolution as the original thing

        sce_inputs = self.input_memX0s
        sce_inputs_ = __p(sce_inputs)
        _, sce_feats_ = self.featnet3D(sce_inputs_)
        sce_feats = __u(sce_feats_)
        # this is B x S x C x Z2 x Y2 x X2

        if hyp.do_loc:
            starting_obj_lrt = self.lrt_camX0s[:,0]

            # search_lrt_camR = self.search_util.convert_box_to_search_lrt(self.box_camRs[:,0])
            # search_lrt_camX0 = utils_geom.apply_4x4_to_lrtlist(
            #     self.camXs_T_camRs[:,0], search_lrt_camR.unsqueeze(1)).squeeze(1)
            # starting_search_lrt = search_lrt_camX0
            
            obj_len, _ = utils_geom.split_lrt(starting_obj_lrt)
            print('new obj_len', obj_len.detach().cpu().numpy())

            if self.summ_writer.save_this:
                sce_input_vis = []
                obj_input_vis = []
                sce_feat_vis = []
                obj_feat_vis = []
                obj_xyzs_e = []
                obj_xyzs_g = []
                obj_lrts_e = []
                obj_lrts_g = []
            
            loc_feat_losses = []

            fw_seq = list(range(0, self.S, 1))
            bw_seq = list(range(self.S-2, -1, -1))
            cycle_seq = fw_seq + bw_seq
            
            num_cycle_steps = len(cycle_seq)
            print('cycle_seq:', cycle_seq)

            # crop the object out right away
            sce_input = sce_inputs[:,0]
            obj_input = utils_vox.crop_zoom_from_mem(sce_input, starting_obj_lrt, self.ZZ, self.ZY, self.ZX)

            # search_input = utils_vox.crop_zoom_from_mem(sce_input,
            #                                             starting_search_lrt,
            #                                             self.search_Z,
            #                                             self.search_Y,
            #                                             self.search_X)
            # self.summ_writer.summ_feat('search/obj', obj_input, pca=True)
            # self.summ_writer.summ_feat('search/search', search_input, pca=True)


            # now, suppose i find the object in here
            # that is, suppose i output a box in this coordinate system
            # then, i need to find my way back to cam coords, and sample from the mem in the next timestep

            # search_T_cam = utils_vox.get_zoom_T_ref(starting_search_lrt,
            #                                         self.search_Z,
            #                                         self.search_Y,
            #                                         self.search_X)
            # cam_T_search = utils_vox.get_ref_T_zoom(starting_search_lrt,
            #                                         self.search_Z,
            #                                         self.search_Y,
            #                                         self.search_X)
            
            # now if i obtain a box in search coords, i can get to cam

            # the rt in lrt_e is camX0_T_obj


            # but right now, the location

            # hm...

            # using mem coords made things much easier, since the mem system is tied precisely to the cam system

            # i need estimate the box parameters in a local system, and then map back to the refcam
            #


            # i can start this in the current mem thing
            
            

            

            # we will include step0 in this loop, which means just locating the object in the tensor where we cropped it
            for step in list(range(num_cycle_steps)):
                s = cycle_seq[step]
                suffix = '_step%d' % step
                    
                # featurize the object crop
                _, obj_feat = self.featnet3D(obj_input)
                # this is B x C x ZZ2 x ZY2 x ZX2

                # we will try to find the object in the current timestep's scene
                sce_input = sce_inputs[:,s]
                sce_feat = sce_feats[:,s]
                # the answer is here:
                obj_lrt_g = self.lrt_camX0s[:,s]

                # right now, 
                
                if self.summ_writer.save_this:
                    sce_input_vis.append(self.summ_writer.summ_feat('', sce_input, pca=True, only_return=True))
                    obj_input_vis.append(self.summ_writer.summ_feat('', obj_input, pca=True, only_return=True))
                    sce_feat_vis.append(self.summ_writer.summ_feat('', sce_feat, pca=True, only_return=True))
                    obj_feat_vis.append(self.summ_writer.summ_feat('', obj_feat, pca=True, only_return=True))

                cam_T_sce = utils_vox.get_ref_T_mem(self.B, self.Z, self.Y, self.X)
                
                obj_lrt_e = self.locnet(
                    obj_feat, 
                    sce_feat,
                    obj_len,
                    cam_T_sce,
                    self.Z, self.Y, self.X,
                    summ_writer=None,
                    suffix=suffix)

                obj_len_e, _ = utils_geom.split_lrt(obj_lrt_e)
                print('obj_len_e', obj_len_e.detach().cpu().numpy())

                loc_feat_losses.append(self.locnet.compute_feat_loss(
                    obj_feat, sce_feat, obj_lrt_e, self.summ_writer, suffix=suffix))

                if self.summ_writer.save_this:
                    obj_xyzs_e.append(utils_geom.get_clist_from_lrtlist(obj_lrt_e.unsqueeze(1)).squeeze(1))
                    obj_xyzs_g.append(utils_geom.get_clist_from_lrtlist(obj_lrt_g.unsqueeze(1)).squeeze(1))
                    obj_lrts_e.append(obj_lrt_e)
                    obj_lrts_g.append(obj_lrt_g)

                # to prep for the next step in the loop, let's crop out the object we located
                obj_input = utils_vox.crop_zoom_from_mem(sce_input, obj_lrt_e, self.ZZ, self.ZY, self.ZX)
                
            loc_feat_loss = torch.mean(torch.stack(loc_feat_losses)) # note the coeff is already accounted for
            total_loss = utils_misc.add_loss('loc/feat_loss', total_loss, loc_feat_loss, 1.0, self.summ_writer)

            final_obj_lrt = obj_lrt_e.clone()
            loc_loss = self.locnet.compute_samp_loss(final_obj_lrt, starting_obj_lrt, self.summ_writer)
            total_loss += loc_loss

            if self.summ_writer.save_this:
                obj_start = utils_vox.crop_zoom_from_mem(sce_inputs[:,0], starting_obj_lrt, self.ZZ, self.ZY, self.ZX)
                obj_final = utils_vox.crop_zoom_from_mem(sce_inputs[:,0], final_obj_lrt, self.ZZ, self.ZY, self.ZX)
                self.summ_writer.summ_feat('loc/obj_start', obj_start, pca=True)
                self.summ_writer.summ_feat('loc/obj_final', obj_final, pca=True)
                
                # show trajectories
                self.summ_writer.summ_traj_on_occ('loc/traj_e', torch.stack(obj_xyzs_e, dim=1), self.occ_memX0s[:,0],
                                                  already_mem=False, sigma=2, only_return=False)
                self.summ_writer.summ_traj_on_occ('loc/traj_g', torch.stack(obj_xyzs_g, dim=1), self.occ_memX0s[:,0],
                                                  already_mem=False, sigma=2, only_return=False)

                # show features
                self.summ_writer.summ_rgbs('loc/sce_inputs', sce_input_vis)
                self.summ_writer.summ_rgbs('loc/obj_inputs', obj_input_vis)
                self.summ_writer.summ_rgbs('loc/sce_feats', sce_feat_vis)
                self.summ_writer.summ_rgbs('loc/obj_feats', obj_feat_vis)

                # show boxes
                bev_vis_e = []
                bev_vis_g = []
                cam_vis_e = []
                cam_vis_g = []

                for step in list(range(num_cycle_steps)):
                    s = cycle_seq[step]
                    # for s in list(range(self.S)):
                    scores = torch.ones([self.B, 1]).float()
                    tids = torch.ones([self.B, 1]).long()

                    bev_vis_e.append(self.summ_writer.summ_lrtlist_bev(
                        '', self.unp_memX0s[:,s], self.occ_memX0s[:,s], obj_lrts_e[step].unsqueeze(1),
                        scores, tids, only_return=True))
                    bev_vis_g.append(self.summ_writer.summ_lrtlist_bev(
                        '', self.unp_memX0s[:,s], self.occ_memX0s[:,s], obj_lrts_g[step].unsqueeze(1),
                        scores, tids, only_return=True))
                    # for cam vis, we need to convert to X coords (from X0)
                    obj_lrt_e = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camX0[:,s], obj_lrts_e[step].unsqueeze(1))
                    obj_lrt_g = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camX0[:,s], obj_lrts_g[step].unsqueeze(1))
                    cam_vis_e.append(self.summ_writer.summ_lrtlist(
                        '', self.rgb_camXs[:,s], obj_lrt_e,
                        scores, tids, self.pix_T_cams[:,s], only_return=True))
                    cam_vis_g.append(self.summ_writer.summ_lrtlist(
                        '', self.rgb_camXs[:,s], obj_lrt_g,
                        scores, tids, self.pix_T_cams[:,s], only_return=True))
                self.summ_writer.summ_rgbs('loc/box_bev_e', bev_vis_e)
                self.summ_writer.summ_rgbs('loc/box_bev_g', bev_vis_g)
                self.summ_writer.summ_rgbs('loc/box_cam_e', cam_vis_e)
                self.summ_writer.summ_rgbs('loc/box_cam_g', cam_vis_g)

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        self.prepare_common_tensors(feed)
        
        set_name = feed['set_name']
        
        if set_name=='train':
            return self.run_train(feed)
        elif set_name=='val':
            return self.run_train(feed)
        
        # if set_data_format=='seq':
        #     return self.run_flow(feed)
        # elif set_data_format=='traj':
        #     return self.run_tracker(feed)
            
        print('weird set_name:', set_name)
        assert(False)
        
        
