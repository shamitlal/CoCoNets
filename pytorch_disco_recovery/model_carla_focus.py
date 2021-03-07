import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_FOCUS(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaFocusNet()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_preocc:
            self.model.preoccnet.eval()
            self.set_requires_grad(self.model.preoccnet, False)
        if hyp.do_freeze_emb2D:
            self.model.embnet2D.eval()
            self.set_requires_grad(self.model.embnet2D, False)

class CarlaFocusNet(nn.Module):
    def __init__(self):
        super(CarlaFocusNet, self).__init__()
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
        self.device = torch.device("cuda")
        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=False)
        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9

        rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        cam_T_velos = feed["cam_T_velos"]

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camXs = feed["origin_T_camXs"]

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))
                            
        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))

        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)
        
        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))

        PH, PW = hyp.PH, hyp.PW
        sy = float(PH)/float(hyp.H)
        sx = float(PW)/float(hyp.W)
        assert(sx==0.5) # else we need a fancier downsampler
        assert(sy==0.5)
        projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

        unp_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
            projpix_T_cams[:,0], camX0_T_camXs[:,0], unpXs[:,0],
            hyp.view_depth, PH, PW)
        occ_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
            projpix_T_cams[:,0], camX0_T_camXs[:,0], occXs[:,0],
            hyp.view_depth, PH, PW)
        occ_projX00 = occ_projX00.repeat(1, 3, 1, 1, 1)
        
        unp_projX01 = utils_vox.apply_pixX_T_memR_to_voxR(
            projpix_T_cams[:,0], camX0_T_camXs[:,1], unpXs[:,1],
            hyp.view_depth, PH, PW)
        occ_projX01 = utils_vox.apply_pixX_T_memR_to_voxR(
            projpix_T_cams[:,0], camX0_T_camXs[:,1], occXs[:,1],
            hyp.view_depth, PH, PW)
        occ_projX01 = occ_projX01.repeat(1, 3, 1, 1, 1)

        # these are B x C x Z x Y x X

        projX00_vis = utils_basic.reduce_masked_mean(
            unp_projX00, occ_projX00, dim=2)
        projX01_vis = utils_basic.reduce_masked_mean(
            unp_projX01, occ_projX01, dim=2)
        
        summ_writer.summ_rgb('proj/projX00', projX00_vis)
        summ_writer.summ_rgb('proj/projX01', projX01_vis)
        
        rgb_X0 = utils_basic.downsample(rgb_camXs[:,0], 2)
        # rgb_X01 = utils_basic.downsample(rgb_camXs[:,1], 2)

        summ_writer.summ_rgb('proj/rgbX0', rgb_X0)
        # summ_writer.summ_rgb('proj/rgbX1', projX01_vis)

        # box2D = np.reshape(np.array([0.5, 0.25, 1.0, 0.75], np.float32), (1, 4))
        # # box2D = np.reshape(np.array([0.0, 0.0, 1.0, 1.0], np.float32), (1, 4))
        # box2D = torch.from_numpy(box2D).to(self.device)
        # # this is B x 4
        # boxlist2D = box2D.unsqueeze(1)
        # # this is B x 1 x 4
        # summ_writer.summ_boxlist2D('boxes/fake_boxlist2D', rgb_camXs[:,0], boxlist2D)#, scores=None, tids=None, only_return=False)

        # objpix_T_cam, _ = utils_geom.convert_box2D_to_intrinsics(box2D, pix_T_cams[:,0], H, W, use_image_aspect_ratio=False)
        # objpix_T_cam = utils_geom.scale_intrinsics(objpix_T_cam, sx, sy)
        # unp_objX00 = utils_vox.apply_pixX_T_memR_to_voxR(
        #     objpix_T_cam, camX0_T_camXs[:,0], unpXs[:,0],
        #     hyp.view_depth, PH, PW)
        # occ_objX00 = utils_vox.apply_pixX_T_memR_to_voxR(
        #     objpix_T_cam, camX0_T_camXs[:,0], occXs[:,0],
        #     hyp.view_depth, PH, PW)
        # occ_objX00 = occ_objX00.repeat(1, 3, 1, 1, 1)
        # objX00_vis = utils_basic.reduce_masked_mean(
        #     unp_objX00, occ_objX00, dim=2)
        # summ_writer.summ_rgb('proj/objX00', objX00_vis)

        # box2D_crop = utils_samp.crop_and_resize_box2D(rgb_camXs[:,0], box2D, PH, PW)
        # summ_writer.summ_rgb('proj/obj_crop0_fake', box2D_crop)
        


        
        boxlist_camRs = feed["boxes3D"]
        tidlist_s = feed["tids"] # coordinate-less and plural
        scorelist_s = feed["scores"] # coordinate-less and plural
        # # postproc the boxes:
        # scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(boxlist_camRs), __p(tidlist_s), Z, Y, X))
        boxlist_camRs_, tidlist_s_, scorelist_s_ = __p(boxlist_camRs), __p(tidlist_s), __p(scorelist_s)
        boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            boxlist_camRs_, tidlist_s_, scorelist_s_)
        boxlist_camRs = __u(boxlist_camRs_)
        tidlist_s = __u(tidlist_s_)
        scorelist_s = __u(scorelist_s_)

        lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))) #))).reshape(B, S, N, 19)
        lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
        xyzlist_camXs = __u(utils_geom.get_xyzlist_from_lrtlist(__p(lrtlist_camXs)))
        # this is B x S x N x 8 x 3
        xyzlist_camXs__ = xyzlist_camXs.reshape(B*S, N*8, 3)
        xylist_camXs__ = utils_geom.apply_pix_T_cam(__p(pix_T_cams), xyzlist_camXs__)
        # this is B*S x N*8 x 2
        xylist_camXs = xylist_camXs__.reshape(B, S, N, 8, 2)
        
        # print('scorelist_s', scorelist_s[:,0,:3])
        # print('xyzlist_camXs', xyzlist_camXs[:,0,:3])
        # print('xylist_camXs', xylist_camXs[:,0,:3])
        
        # xylist_camXs is B x S x N x 8 x 2
        xminlists = torch.min(xylist_camXs[:,:,:,:,0], dim=3)[0]
        yminlists = torch.min(xylist_camXs[:,:,:,:,1], dim=3)[0]
        xmaxlists = torch.max(xylist_camXs[:,:,:,:,0], dim=3)[0]
        ymaxlists = torch.max(xylist_camXs[:,:,:,:,1], dim=3)[0]
        # these are B x S x N
        boxlists = torch.stack([yminlists, xminlists, ymaxlists, xmaxlists], dim=3)
        boxlists = __u(utils_geom.normalize_boxlist2D(__p(boxlists), H, W))
        # boxlists is B x S x N x 4
        # print('boxlists', boxlists[:,0,:3])
        
        summ_writer.summ_boxlist2D('boxes/boxlist2D_g', rgb_camXs[:,0], boxlists[:,0])

        
        if hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)

            # it is useful to keep track of what was visible from each viewpoint
            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)

            featXs_, validXs_, feat_loss = self.featnet(
                featXs_input_,
                summ_writer,
                comp_mask=None,
            )
            total_loss += feat_loss
            
            validXs = __u(validXs_)
            _validX00 = validXs[:,0:1]
            _validX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], validXs[:,1:])
            validX0s = torch.cat([_validX00, _validX01], dim=1)
            
            _visX00 = visXs[:,0:1]
            _visX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], visXs[:,1:])
            visX0s = torch.cat([_visX00, _visX01], dim=1)
            
            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) # context
            vis3D_e = torch.max(validX0s[:,1:], dim=1)[0]*torch.max(visX0s[:,1:], dim=1)[0]
            emb3D_g = featX0s[:,0] # obs
            vis3D_g = validX0s[:,0]*visX0s[:,0] # obs

            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(validX0s, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validX0s', torch.unbind(validX0s, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e', vis3D_e, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g', vis3D_g, pca=False)
            
        if hyp.do_occ:
            occX0_sup, freeX0_sup, _, freeXs = utils_vox.prep_occs_supervision(
                camX0_T_camXs,
                xyz_camXs,
                Z2, Y2, X2, 
                agg=True)
        
            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))
                
            occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                 occX0_sup,
                                                 freeX0_sup,
                                                 torch.max(validX0s[:,1:], dim=1)[0],
                                                 summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss

        if hyp.do_view:
            assert(hyp.do_feat)

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            assert(S==2) # else we should warp each feat in 1:
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                hyp.view_depth, PH, PW)
            rgb_X00 = utils_basic.downsample(rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(valid_camXs[:,0], 2)

            # decode the perspective volume into an image
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                summ_writer)
            total_loss += view_loss

            objpix_T_cam, box2D_new = utils_geom.convert_box2D_to_intrinsics(
                boxlists[:,0,0], pix_T_cams[:,0], H, W, use_image_aspect_ratio=False, mult_padding=1.2)
            summ_writer.summ_boxlist2D('boxes/boxlist2D_g_new', rgb_camXs[:,0], box2D_new.unsqueeze(1))
            # we will render at half res
            objpix_T_cam = utils_geom.scale_intrinsics(objpix_T_cam, sx, sy)

            feat_objX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                objpix_T_cam, camX0_T_camXs[:,0], featXs[:,1],
                hyp.view_depth, PH, PW)
            summ_writer.summ_feat('proj/featX00', torch.mean(feat_objX00, dim=2))

            # extract a half-res crop, to use as gt for viewnet 
            rgb_objX00 = utils_samp.crop_and_resize_box2D(rgb_camXs[:,0], boxlists[:,0,0], PH, PW)
            valid_objX00 = utils_samp.crop_and_resize_box2D(valid_camXs[:,0], boxlists[:,0,0], PH, PW)
            valid_objX00 = torch.round(valid_objX00)
            
            # also compute loss at the object
            # decode the perspective volume into an image
            obj_view_loss, _, emb2D_obj_e = self.viewnet(
                feat_objX00,
                rgb_objX00,
                valid_objX00,
                summ_writer,
                suffix='_obj')
            total_loss += obj_view_loss
            
        if hyp.do_emb2D:
            assert(hyp.do_view)
            # create an embedding image, representing the bottom-up 2D feature tensor
            emb_loss_2D, emb2D_g = self.embnet2D(
                rgb_camXs[:,0],
                emb2D_e,
                valid_camXs[:,0],
                summ_writer)
            total_loss += emb_loss_2D

            # extract a full-res crop, to use as input to embnet 
            rgb_objX00 = utils_samp.crop_and_resize_box2D(rgb_camXs[:,0], boxlists[:,0,0], H, W)
            valid_objX00 = utils_samp.crop_and_resize_box2D(valid_camXs[:,0], boxlists[:,0,0], H, W)
            valid_objX00 = torch.round(valid_objX00)
            obj_emb_loss_2D, emb2D_obj_g = self.embnet2D(
                rgb_objX00,
                emb2D_obj_e,
                valid_objX00,
                summ_writer,
                suffix='_obj')
            total_loss += obj_emb_loss_2D

        if hyp.do_emb3D:
            # compute 3D ML
            emb_loss_3D = self.embnet3D(
                emb3D_e,
                emb3D_g,
                vis3D_e,
                vis3D_g,
                summ_writer)
            total_loss += emb_loss_3D
            
        
        
        
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results


