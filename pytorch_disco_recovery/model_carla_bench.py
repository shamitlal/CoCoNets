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
from nets.featnet3D import FeatNet3D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet

from nets.occnet import OccNet

import sklearn

# from nets.mocnet2D import MocNet2D
# from nets.mocnet3D import MocNet3D

from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

from tensorboardX import SummaryWriter
import torch.nn.functional as F

from utils_moc import MocTrainer
# from utils_basic import *
# import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc
import vox_util
import utils_track

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

class CARLA_BENCH(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaBenchModel()
        if hyp.do_freeze_feat2D:
            self.model.featnet2D.eval()
            self.set_requires_grad(self.model.featnet2D, False)
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)

class CarlaBenchModel(nn.Module):
    def __init__(self):
        super(CarlaBenchModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            
        if hyp.do_view:
            self.viewnet = ViewNet()

        if hyp.do_occ:
            self.occnet = OccNet()
            
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

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        
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
            scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            # print('scene_centroid', scene_centroid.detach().cpu().numpy())

            # scene_centroid [[-2.75  0.78 15.34]]
            # scene_centroid [[-6.   -0.9  20.81]]
            # scene_centroid [[-6.   -0.9  20.81]]
            # scene_centroid [[6.98 1.46 6.89]]
            # scene_centroid [[ 0.3   1.48 23.1 ]]
            # scene_centroid [[ 3.24  0.94 12.4 ]]
            # scene_centroid [[-2.29  2.58 11.56]]
            # scene_centroid [[ 3.95  0.01 14.16]]
            # scene_centroid [[ 3.93 -1.33 21.13]]
            

            # ok, how bout i allow centroids in -1.5, 3.0
            
            # scene_centroid_x = 0.0
            # scene_centroid_y = 1.0
            # scene_centroid_z = 18.0
            # scene_centroid = np.array([scene_centroid_x,
            #                            scene_centroid_y,
            #                            scene_centroid_z]).reshape([1, 3])
            # scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        else:
            # center randomly 
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

        self.vox_util = vox_util.Vox_util(feed['set_name'], scene_centroid=scene_centroid, assert_cube=True)
        
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.ZZ, self.ZY, self.ZX = hyp.ZY, hyp.ZY, hyp.ZX
        self.vox_size_X = (hyp.XMAX-hyp.XMIN)/self.X
        self.vox_size_Y = (hyp.YMAX-hyp.YMIN)/self.Y
        self.vox_size_Z = (hyp.ZMAX-hyp.ZMIN)/self.Z
        
        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

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

        if prep_summ:
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
        
    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_feat2D:
            feat2D_loss, feat_camX0 = self.featnet2D(
                self.rgb_camXs[:,0],
                self.summ_writer,
            )
            total_loss += feat2D_loss
            
        if hyp.do_feat3D:
            feat_memXs_input = torch.cat([
                self.occ_memXs,
                self.unp_memXs*self.occ_memXs,
            ], dim=2)
            feat_memRs_input = torch.cat([
                self.occ_memRs,
                self.unp_memRs*self.occ_memRs,
            ], dim=2)
            
            feat3D_loss, feat_memXs_,_ = self.featnet3D(
                __p(feat_memXs_input[:,1:]),
                self.summ_writer,
            )
            feat_memXs = __u(feat_memXs_)
            valid_memXs = torch.ones_like(feat_memXs[:,:,0:1])
            total_loss += feat3D_loss

            # warp things to R
            feat_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], feat_memXs)
            valid_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], valid_memXs)

            feat_memR = utils_basic.reduce_masked_mean(
                feat_memRs,
                valid_memRs.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
                dim=1)
            valid_memR = torch.sum(valid_memRs, dim=1).clamp(0, 1)

            _, altfeat_memR,_ = self.featnet3D(feat_memRs_input[:,0])
            
            self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)
            self.summ_writer.summ_feat('3D_feats/altfeat_memR', altfeat_memR, pca=True)

            # feat_memX0s_input = torch.cat([
            #     self.occ_memX0s,
            #     self.unp_memX0s*self.occ_memX0s,
            # ], dim=2)
            # feat3D_loss, feat_memX0s_ = self.featnet3D(
            #     __p(feat_memX0s_input),
            #     self.summ_writer,
            # )
            # feat_memX0s = __u(feat_memX0s_)
            # valid_memX0s = torch.ones_like(feat_memX0s[:,:,0:1])
            # total_loss += feat3D_loss
            
            # # warp things to R
            # feat_memX0 = torch.mean(feat_memX0s[:,1:], dim=1)
            # valid_memX0 = valid_memX0s[:,0]
            # altfeat_memX0 = feat_memX0s[:,0]
            # self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, valid=valid_memX0, pca=True)
            # self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, pca=True)


        if hyp.do_occ:
            occ_memR_sup, free_memR_sup, _, free_memRs = self.vox_util.prep_occs_supervision(
                self.camRs_T_camXs,
                self.xyz_camXs,
                self.Z4, self.Y4, self.X4, 
                agg=True)
        
            self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memR_sup)
            self.summ_writer.summ_occ('occ_sup/free_sup', free_memR_sup)
            self.summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(free_memRs, dim=1))
            self.summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(self.occ_memRs_half, dim=1))
                
            occ_loss, occ_memR_pred = self.occnet(
                altfeat_memR, 
                occ_memR_sup,
                free_memR_sup,
                torch.ones_like(valid_memR), 
                self.summ_writer)
            total_loss += occ_loss

        if hyp.do_view:
            assert(hyp.do_feat3D)
            # decode the feat volume into an image
            view_loss, rgb_e, view_camX0 = self.viewnet(
                self.pix_T_cams[:,0],
                self.camX0_T_camR0,
                altfeat_memR,
                self.rgb_camXs[:,0], 
                summ_writer=self.summ_writer)
            total_loss += view_loss
            
        if hyp.do_emb2D:
            assert(hyp.do_view)
            # create an embedding image, representing the bottom-up 2D feature tensor
            emb_loss_2D, _ = self.embnet2D(
                view_camX0,
                feat_camX0,
                torch.ones_like(view_camX0[:,0:1]),
                self.summ_writer)
            total_loss += emb_loss_2D
            
        if hyp.do_emb3D:
            # compute 3D ML
            emb_loss_3D = self.embnet3D(
                # feat_memX0,
                # altfeat_memX0,
                # valid_memX0.round(),
                # torch.ones_like(valid_memX0),
                feat_memR,
                altfeat_memR,
                valid_memR.round(),
                torch.ones_like(valid_memR),
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

        self.obj_clist_camR0 = utils_geom.get_clist_from_lrtlist(self.lrt_camR0s)
        self.obj_clist_camX0 = utils_geom.get_clist_from_lrtlist(self.lrt_camX0s)

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
            _, feat_memX0s_,_ = self.featnet3D(
                __p(feat_memX0s_input),
                self.summ_writer,
            )
            feat_memX0s = __u(feat_memX0s_)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s', torch.unbind(feat_memX0s, dim=1), pca=True)

            obj_mask_memX0s = self.vox_util.assemble_padded_obj_masklist(
                self.lrt_camX0s,
                self.score_s,
                self.Z2, self.Y2, self.X2).squeeze(1)
            # only take the occupied voxels
            obj_mask_memX0s = obj_mask_memX0s * self.occ_memX0s_half

            for b in list(range(self.B)):
                if torch.sum(obj_mask_memX0s[b,0]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True

            self.summ_writer.summ_feats('track/obj_mask_memX0s', torch.unbind(obj_mask_memX0s, dim=1), pca=False)

            lrt_camX0s_e, point_counts, ious = utils_track.track_via_inner_products(
                self.lrt_camX0s, obj_mask_memX0s, feat_memX0s, self.vox_util)   
            
            ious_remask = utils_track.remask_via_inner_products(
                self.lrt_camX0s, obj_mask_memX0s, feat_memX0s, self.vox_util, self.summ_writer)   

            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
                self.summ_writer.summ_scalar('track/mean_iou_remask_%02d' % s, torch.mean(ious_remask[:,s]).cpu().item())

            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))

            # lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0, lrt_camR0s_e)
            # lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs_e)
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
            self.prepare_common_tensors(feed)
            return self.run_train(feed)
        elif set_name=='test':
            self.prepare_common_tensors(feed)
            return self.run_test(feed)
        # else:

        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
