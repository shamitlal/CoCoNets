import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.vq3drgbnet import Vq3drgbNet
from nets.viewnet import ViewNet
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


# the purpose of this mode is to train gen3dvq, after feat/vq/view have been trained and frozen
# i am separating it out because here i want to aggregate lots of views, and do everything in memR (except rendering)

class CLEVR_GEN3DVQ(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = ClevrGen3dvqModel()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_vq3drgb:
            self.model.vq3drgbnet.eval()
            self.set_requires_grad(self.model.vq3drgbnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)

class ClevrGen3dvqModel(nn.Module):
    def __init__(self):
        super(ClevrGen3dvqModel, self).__init__()

        if hyp.do_feat:
            self.featnet = FeatNet(in_dim=4)
        if hyp.do_occ:
            self.occnet = OccNet()
        
        if hyp.do_vq3drgb:
            self.vq3drgbnet = Vq3drgbNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
            
        if hyp.do_view:
            self.viewnet = ViewNet()
            
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
                            
        self.occ_memRs = __u(utils_vox.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))

        self.unp_memXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unp_memRs = utils_vox.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)
        # self.unp_memRs = __u(utils_vox.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z, self.Y, self.X, utils_basic.matmul2(
        #         __p(self.pix_T_cams), utils_geom.safe_inverse(__p(self.camRs_T_camXs)))))

        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
        self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        self.summ_writer.summ_unps('3D_inputs/unp_memRs', torch.unbind(self.unp_memRs, dim=1), torch.unbind(self.occ_memRs, dim=1))

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_feat:
            # simplify this, to only featurize ind1 if S==2

            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D

            feat_memRs_input = torch.cat([self.occ_memRs, self.occ_memRs*self.unp_memRs], dim=2)
            feat_memRs_, valid_memRs_, feat_loss = self.featnet(
                __p(feat_memRs_input[:,1:]),
                self.summ_writer,
            )
            total_loss += feat_loss
            
            feat_memRs = __u(feat_memRs_)
            valid_memRs = __u(valid_memRs_)

            feat_memR = torch.mean(feat_memRs, dim=1)
            valid_memR = torch.max(valid_memRs, dim=1)[0]
            self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)

        if hyp.do_vq3drgb:
            vq3drgb_loss, feat_memR, ind_memR = self.vq3drgbnet(
                feat_memR,
                self.summ_writer,
            )
            total_loss += vq3drgb_loss

        if hyp.do_gen3dvq:
            gen3dvq_loss = self.gen3dvqnet(
                ind_memR,
                self.summ_writer,
            )
            total_loss += gen3dvq_loss

        # if hyp.do_sigen3d:
        #     sigen3d_loss = self.sigen3dnet(
        #         ind_image,
        #         self.summ_writer,
        #         is_train=is_train,
        #     )
        #     total_loss += sigen3d_loss
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)
        
        # ind_vox = self.gen3dvqnet.generate_sample(1, self.Z8, self.Y8, self.X8)
        # self.summ_writer.summ_oned('sample/ind_vox', ind_vox.unsqueeze(1)/512.0, bev=True, norm=False)
        # quant_vox, quant_vox_up = self.vq3drgbnet.convert_inds_to_embeds(ind_vox)
        # self.summ_writer.summ_feat('sample/quant', quant_vox, pca=True)
        # # quant_vox = self.vq3drgbnet._post_vq_conv(quant_vox)
        # # quant_vox = self.vq3drgbnet._post_vq_unpack(quant_vox)
        # self.summ_writer.summ_feat('sample/quant_up', quant_vox_up, pca=True)
        # feat_memR = quant_vox_up.clone()
        
        ind_vox = self.gen3dvqnet.generate_sample(1, self.Z4, self.Y4, self.X4)
        self.summ_writer.summ_oned('sample/ind_vox', ind_vox.unsqueeze(1)/512.0, bev=True, norm=False)
        quant_vox = self.vq3drgbnet.convert_inds_to_embeds(ind_vox)
        self.summ_writer.summ_feat('sample/quant', quant_vox, pca=True)
        feat_memR = quant_vox.clone()

        if hyp.do_view:
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            rgbs = []
            for k in list(range(self.S)):
                print('generating view %d' % k)

                print(projpix_T_cams[:,k].shape)
                print(self.camXs_T_camRs[:,k].shape)
                print(feat_memR.shape)
                
                feat_projXk = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[0:1,k], self.camXs_T_camRs[0:1,k], feat_memR,
                    hyp.view_depth, PH, PW)
                # decode the perspective volume into an image
                _, rgb_e, _ = self.viewnet(
                    feat_projXk,
                    None,
                    None,
                    self.summ_writer,
                    test=True)
                rgbs.append(rgb_e.clamp(-0.5, 0.5))
            self.summ_writer.summ_rgbs('sample/rgbs', rgbs)
        
        return total_loss, None, False
            
    def forward(self, feed):
        self.prepare_common_tensors(feed)
        
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
