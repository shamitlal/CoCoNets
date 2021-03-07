import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.gengraynet import GengrayNet

import torch.nn.functional as F

# from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_GENGRAY(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaGengrayModel()
        if hyp.do_freeze_gengray:
            self.model.gengraynet.eval()
            self.set_requires_grad(self.model.gengraynet, False)

class CarlaGengrayModel(nn.Module):
    def __init__(self):
        super(CarlaGengrayModel, self).__init__()
        if hyp.do_gengray:
            self.gengraynet = GengrayNet(
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
            just_gif=True,
        )

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.B, self.H, self.W, self.V, self.S, self.N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.D = 9

        self.rgb_camRs = feed["rgb_camRs"]
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils_geom.safe_inverse(__p(self.camRs_T_camXs)))

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))
                            
        self.occXs = __u(utils_vox.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occXs_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        self.occX0s_half = __u(utils_vox.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))

        ## projected depth, and inbound mask
        self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = utils_vox.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        #####################
        ## visualize what we got
        #####################
        self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
        self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(self.occXs, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(self.unpXs, dim=1), torch.unbind(self.occXs, dim=1))

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_gengray:

            rgb_input = self.rgb_camXs[:,0]
            gray_input = torch.mean(rgb_input, dim=1, keepdim=True)
            gray_input = F.interpolate(gray_input, scale_factor=0.5, mode='bilinear')
            self.summ_writer.summ_oned('gengray/gray_input', gray_input)
            gengray_loss = self.gengraynet(
                gray_input,
                self.summ_writer,
            )
            total_loss += gengray_loss
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_val(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_gengray:

            # sample = self.gengraynet.generate_sample(1, int(self.H/2), int(self.W/2))
            sample = self.gengraynet.generate_sample(1, int(self.H/2), int(self.W/2))
            print('sample', sample.shape)
            self.summ_writer.summ_oned('gengray/image_sample', sample.float()/255.0 - 0.5)
            # self.summ_writer.summ_feat('gengraynet/sample_occ', sample.float(), pca=False)
            # self.summ_writer.summ_occ('gengraynet/sample_feat', sample.float())

        # self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        print('test not ready')
        
        return total_loss, None, False
            
    def forward(self, feed):
        self.prepare_common_tensors(feed)
        
        set_name = feed['set_name']
        if set_name=='train':
            return self.run_train(feed)
        elif set_name=='val':
            return self.run_val(feed)
        elif set_name=='test':
            return self.run_test(feed)
        else:
            print('weird set_name:', set_name)
            assert(False)
