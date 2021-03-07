import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.genoccnet import GenoccNet

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

class CARLA_GENOCC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaGenoccModel()
        if hyp.do_freeze_genocc:
            self.model.genoccnet.eval()
            self.set_requires_grad(self.model.genoccnet, False)

class CarlaGenoccModel(nn.Module):
    def __init__(self):
        super(CarlaGenoccModel, self).__init__()
        if hyp.do_genocc:
            self.genoccnet = GenoccNet(
                input_dim=2,
                num_layers=2,
            ).cuda()

    def prepare_common_tensors(self, feed):
        results = dict()
        self.summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)

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
        self.summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(self.occXs, dim=1))
        self.summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(self.unpXs, dim=1), torch.unbind(self.occXs, dim=1))

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        occX0_sup, freeX0_sup, _, freeXs = utils_vox.prep_occs_supervision(
            self.camX0_T_camXs,
            self.xyz_camXs,
            self.Z, self.Y, self.X, 
            agg=True)
        self.summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
        self.summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
        self.summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
        self.summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(self.occXs_half, dim=1))

        if hyp.do_genocc:
            occX0_input = F.dropout(self.occXs[:,0], p=0.75, training=True).float()

            dropout_mask = torch.randint(0, 2, (self.B, 1, self.Z, self.Y, self.X)).cuda().float()
            occX0_input = self.occXs[:,0] * dropout_mask

            utils_basic.print_stats_py('occXs[:,0]', self.occXs[:,0].detach().cpu().numpy())
            utils_basic.print_stats_py('occX0_input', occX0_input.detach().cpu().numpy())
            _, genocc_loss = self.genoccnet(
                occX0_input,
                # self.occXs[:,0],
                occX0_sup,
                freeX0_sup,
                self.summ_writer,
            )
            total_loss += genocc_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_val(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_genocc:

            # sample = self.genoccnet.generate_uncond_sample(1, self.Z2, self.Y2, self.X2, self.occXs[:,0])
            
            occXs_sup, freeXs_sup, _, _ = utils_vox.prep_occs_supervision(
                self.camX0_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2, 
                agg=False)
            
            
            sample = self.genoccnet.generate_sample(
                # self.occXs_half[0:1,0].long(),
                self.occXs_half[0:1,0].long(),
                occXs_sup[0:1,0],
                freeXs_sup[0:1,0],
            )
            print('sample', sample.shape)
            self.summ_writer.summ_feat('genoccnet/sample_occ', sample.float(), pca=False)
            self.summ_writer.summ_occ('genoccnet/sample_feat', sample.float())

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
