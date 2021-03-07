import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.featnet import FeatNet
from nets.feat2net import Feat2Net
from nets.occnet import OccNet
from nets.vq3drgbnet import Vq3drgbNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet
from nets.gen3dvqnet import Gen3dvqNet
from nets.sigen3dnet import Sigen3dNet

import torch.nn.functional as F

# from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py

np.set_printoptions(precision=2)
np.random.seed(0)

class CLEVR_VQ3DRGB(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = ClevrVq3drgbModel()
        if hyp.do_freeze_feat2:
            self.model.feat2net.eval()
            self.set_requires_grad(self.model.feat2net, False)
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_vq3drgb:
            self.model.vq3drgbnet.eval()
            self.set_requires_grad(self.model.vq3drgbnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_render:
            self.model.rendernet.eval()
            self.set_requires_grad(self.model.rendernet, False)

class ClevrVq3drgbModel(nn.Module):
    def __init__(self):
        super(ClevrVq3drgbModel, self).__init__()

        if hyp.do_feat2:
            self.feat2net = Feat2Net(in_dim=3)
        if hyp.do_feat:
            self.featnet = FeatNet(in_dim=(4 + hyp.feat2_dim))
        if hyp.do_occ:
            self.occnet = OccNet()
        
        if hyp.do_vq3drgb:
            self.vq3drgbnet = Vq3drgbNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
            
        if hyp.do_view:
            self.viewnet = ViewNet()
            
        if hyp.do_render:
            self.rendernet = RenderNet()
            
        if hyp.do_gen3dvq:
            self.gen3dvqnet = Gen3dvqNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
        if hyp.do_sigen3d:
            self.sigen3dnet = Sigen3dNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()

    def prepare_common_tensors(self, feed):
        results = dict()
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            set_name=feed['set_name'],
            fps=8,
            # just_gif=True,
        )

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.B, self.H, self.W, self.V, self.S, self.N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.Z8, self.Y8, self.X8 = int(self.Z/8), int(self.Y/8), int(self.X/8)
        self.D = 9

        self.rgb_camXs = feed["rgb_camXs_raw"]
        self.pix_T_cams = feed["pix_T_cams_raw"]
        self.origin_T_camXs = feed["origin_T_camXs_raw"]
        self.camRs_T_origin = feed['camR_T_origin_raw']
        self.origin_T_camRs = __u(utils_geom.safe_inverse(__p(self.camRs_T_origin)))

        self.camX0_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camX1_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=1)
        self.camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils_geom.safe_inverse(__p(self.camRs_T_camXs)))

        self.xyz_camXs = feed["xyz_camXs_raw"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))
                            
        self.occ_memXs = __u(utils_vox.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occ_memX0s = __u(utils_vox.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memRs = __u(utils_vox.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        # self.occ_memXs_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        # self.occ_memX0s_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unp_memXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unp_memRs = utils_vox.apply_4x4s_to_voxs(self.camX0_T_camXs, self.unp_memXs)

        ## projected depth, and inbound mask
        self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = utils_vox.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        # #####################
        # ## visualize what we got
        # #####################
        # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1), maxval=20.0)
        # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
        self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_feat2:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D

            assert(self.S > 1)
            
            feat2_camXs_input = self.rgb_camXs
            feat2_camXs_ = self.feat2net(
                __p(feat2_camXs_input[:,1:]),
                self.summ_writer,
            )
            feat2_camXs = __u(feat2_camXs_)

            if global_step==1:
                print('rgb_camXs', self.rgb_camXs.shape)
                print('feat2_camXs', feat2_camXs.shape)

            self.summ_writer.summ_feats('2D_inputs/feat2_camXs', torch.unbind(feat2_camXs, dim=1), pca=True)
            _, _, _, H2, W2 = list(feat2_camXs.shape)
            sx = float(W2)/hyp.W
            sy = float(H2)/hyp.H
            feat_memXs = __u(utils_vox.unproject_rgb_to_mem(
                __p(feat2_camXs), self.Z, self.Y, self.X,
                utils_geom.scale_intrinsics(__p(self.pix_T_cams[:,1:]), sx, sy)))
            feat_memXs = torch.cat([
                self.occ_memXs[:,1:],
                feat_memXs*self.occ_memXs[:,1:],
                self.unp_memXs[:,1:]*self.occ_memXs[:,1:],
            ], dim=2)
            
        if hyp.do_feat:
            feat_memXs_, valid_memXs_, feat_loss = self.featnet(
                __p(feat_memXs),
                self.summ_writer,
            )
            feat_memXs = __u(feat_memXs_)
            valid_memXs = __u(valid_memXs_)
            total_loss += feat_loss
            # warp things to X1
            feat_memX1s = utils_vox.apply_4x4s_to_voxs(self.camX1_T_camXs[:, 1:], feat_memXs)
            valid_memX1s = utils_vox.apply_4x4s_to_voxs(self.camX1_T_camXs[:, 1:], valid_memXs)
            feat_memX1 = utils_basic.reduce_masked_mean(feat_memX1s,
                                                        valid_memX1s.repeat(1, 1, hyp.feat_dim, 1, 1, 1),
                                                        dim=1)
            valid_memX1 = torch.max(valid_memX1s, dim=1)[0]
            self.summ_writer.summ_feat('3D_feats/feat_memX1', feat_memX1, valid=valid_memX1, pca=True)
            
        if hyp.do_vq3drgb:
            vq3drgb_loss, feat_memX1, _ = self.vq3drgbnet(
                feat_memX1,
                self.summ_writer,
            )
            total_loss += vq3drgb_loss

        if hyp.do_gen3dvq:
            gen3dvq_loss = self.gen3dvqnet(
                ind_memX,
                self.summ_writer,
            )
            total_loss += gen3dvq_loss

        if hyp.do_sigen3d:
            sigen3d_loss = self.sigen3dnet(
                ind_image,
                self.summ_writer,
                is_train=is_train,
            )
            total_loss += sigen3d_loss
            
        if hyp.do_view:
            assert(hyp.do_feat)

            # decode the feat volume into an image
            view_loss, rgb_e = self.viewnet(
                self.pix_T_cams[:,0],
                self.camX0_T_camXs[:,1],
                feat_memX1,
                self.rgb_camXs[:,0], 
                self.valid_camXs[:,0], 
                self.summ_writer)
            total_loss += view_loss

        if hyp.do_occ:
            occ_memX1_sup, free_memX1_sup, _, free_memXs = utils_vox.prep_occs_supervision(
                self.camX1_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2, 
                agg=True)
        
            self.summ_writer.summ_occ('occ_sup/occ_memX1_sup', occ_memX1_sup)
            self.summ_writer.summ_occ('occ_sup/free_memX1_sup', free_memX1_sup)
            self.summ_writer.summ_occs('occ_sup/free_memXs_sup', torch.unbind(free_memXs, dim=1))
                
            occ_loss, occ_memX1 = self.occnet(
                feat_memX1,
                occ_memX1_sup,
                free_memX1_sup,
                valid_memX1,
                self.summ_writer)
            total_loss += occ_loss

        if hyp.do_render:
            assert(hyp.do_feat)

            # projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            
            # feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
            #     projpix_T_cams[:,0], self.camX0_T_camXs[:,1], self.featXs[:,1], # use feat1 to predict rgb0
            #     hyp.view_depth, PH, PW)
            
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,1], feat_memX1,
                hyp.view_depth, PH, PW)
            occ_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,1], occ_memX1, #*valid_memX1,
                hyp.view_depth, PH, PW)
            # occ_projX00 is B x 1 x hyp.view_depth x PH x PW
            
            rgb_X00 = utils_basic.downsample(self.rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(self.valid_camXs[:,0], 2)

            # decode the perspective volume into an image
            render_loss, rgb_e = self.rendernet(
                feat_projX00,
                occ_projX00,
                # self.rgb_camXs[:,0],
                # self.valid_camXs[:,0],
                rgb_X00,
                valid_X00,
                self.summ_writer)
            total_loss += render_loss

            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # print('test not ready')
        
        ind_vox = self.gen3dvqnet.generate_sample(1, self.Z8, self.Y8, self.X8)
        self.summ_writer.summ_oned('vqrgbnet/sampled_ind_vox', ind_vox.unsqueeze(1)/512.0, bev=True, norm=False)
        quant = self.vq3drgbnet.convert_inds_to_embeds(ind_vox)
        self.summ_writer.summ_feat('vq3drgbnet/sampled_quant', quant, pca=True)
        # rgb_e = self.vqrgbnet._decoder(quant)

        # utils_py.print_stats('rgb_e', rgb_e.detach().cpu().numpy())

        # self.summ_writer.summ_rgb('vqrgbnet/sampled_rgb_e', rgb_e.clamp(-0.5, 0.5))
        
        return total_loss, None, False
            
    def forward(self, feed):
        self.prepare_common_tensors(feed)

        # print(feed['filename'])
        set_name = feed['set_name']
        if set_name=='train':
            return self.run_train(feed)
        elif set_name=='val':
            return self.run_train(feed)
        elif set_name=='test':
            return self.run_test(feed)
        else:
            print('weird set_name:', set_name)
            assert(False)
