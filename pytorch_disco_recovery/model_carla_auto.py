import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs

from model_base import Model
from nets.linclassnet import LinClassNet
from nets.featnet2D import FeatNet2D
from nets.featnet3D import FeatNet3D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet

from nets.vq3drgbnet import Vq3drgbNet
from nets.occnet import OccNet


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

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

class CARLA_AUTO(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaAutoModel()
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

class CarlaAutoModel(nn.Module):
    def __init__(self):
        super(CarlaAutoModel, self).__init__()

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

        if hyp.do_vq3drgb:
            self.vq3drgbnet = Vq3drgbNet()
            self.labelpools = [utils_misc.SimplePool(100) for i in list(range(hyp.vq3drgb_num_embeddings))]
            print('declared labelpools')

        if hyp.do_linclass:
            self.linclassnet = LinClassNet(hyp.feat3D_dim)

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
                just_gif=False,
            )
        else:
            self.summ_writer = None
            
        self.vox_util = vox_util.Vox_util(feed['set_name'], delta=0.0, assert_cube=False)

        
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
        self.camX0_T_camR = utils_basic.matmul2(self.camX0_T_camXs[:,0], self.camXs_T_camRs[:,0])
        
        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))

        # some of the raw boxes may be out of bounds
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
        self.lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))
        
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        self.occ_memRs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
        # self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        # self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        self.unp_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)

        # ## projected depth, and inbound mask
        # self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
        # self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        # self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        
        # self.depth_camXs = __u(self.depth_camXs_)
        # self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        if prep_summ:
            # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
            # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
            # self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))

            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
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
            feat_memR_input = torch.cat([
                self.occ_memRs[:,0],
                self.unp_memRs[:,0]*self.occ_memRs[:,0],
            ], dim=1)
            # feat_memX_input = torch.cat([
            #     self.occ_memXs[:,0],
            #     self.unp_memXs[:,0]*self.occ_memXs[:,0],
            # ], dim=1)
            feat3D_loss, feat_memR = self.featnet3D(
                feat_memR_input,
                self.summ_writer,
            )
            valid_memR = torch.ones_like(feat_memR[:,0:1])
            total_loss += feat3D_loss
            
            # self.summ_writer.summ_feat('3D_feats/feat_memX', feat_memX, valid=valid_memX, pca=True)
            self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)
            # self.summ_writer.summ_feat('3D_feats/altfeat_memR', altfeat_memR, pca=True)

        if hyp.do_vq3drgb:
            # overwrite altfeat_memR with its quantized version
            vq3drgb_loss, feat_memR, _ = self.vq3drgbnet(
                feat_memR,
                self.summ_writer,
            )
            total_loss += vq3drgb_loss

        if hyp.do_occ:
            occ_memR_sup, free_memR_sup, _, free_memRs = self.vox_util.prep_occs_supervision(
                self.camRs_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2, 
                agg=True)
        
            self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memR_sup)
            self.summ_writer.summ_occ('occ_sup/free_sup', free_memR_sup)
            self.summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(free_memRs, dim=1))
            self.summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(self.occ_memRs_half, dim=1))
                
            occ_loss, occ_memR_pred = self.occnet(
                feat_memR, 
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
                self.camX0_T_camR,
                feat_memR,
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
                feat_memR,
                altfeat_memR,
                valid_memR.round(),
                torch.ones_like(valid_memR),
                self.summ_writer)
            total_loss += emb_loss_3D

        if hyp.do_linclass:
            
            masklist_memR = self.vox_util.assemble_padded_obj_masklist(
                self.lrtlist_camRs[:,0], self.scorelist_s[:,0], self.Z2, self.Y2, self.X2, coeff=1.2)
            mask_memR = torch.sum(masklist_memR, dim=1)
            self.summ_writer.summ_oned('obj/mask_memR', mask_memR.clamp(0, 1), bev=True)

            occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
            occ_memR = torch.sum(occ_memRs, dim=1).clamp(0,1)
            
            obj_memR = (mask_memR * occ_memR).view(-1)
            bkg_memR = ((1.0 - mask_memR) * occ_memR).view(-1)
            obj_inds = torch.nonzero(obj_memR, as_tuple=False)
            # obj_inds = obj_inds.detach().cpu().numpy()
            bkg_inds = torch.nonzero(bkg_memR, as_tuple=False)
            # bkg_inds = bkg_inds.detach().cpu().numpy()

            # print('altfeat_memR', altfeat_memR.shape)
            # print('mask_memR', mask_memR.shape)
            # print('occ_memR', occ_memR.shape)
            # print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))
            # input()
            
            code_vec = feat_memR.detach().permute(0,2,3,4,1).reshape(-1, hyp.feat3D_dim)
            obj_inds = obj_inds.reshape([-1])
            bkg_inds = bkg_inds.reshape([-1])

            if len(obj_inds) and len(bkg_inds):

                # print('obj_inds', obj_inds.shape)
                # print('bkg_inds', bkg_inds.shape)
                # print('codes_flat', codes_flat.shape)

                linclass_loss = self.linclassnet(
                    code_vec, obj_inds, bkg_inds, self.summ_writer)

                # print('feat_memR', feat_memR.shape)
                # print('mask_memR', mask_memR.shape)
                
                total_loss += linclass_loss
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        test_linmatch = True

        if test_linmatch:

            # use inputs pre-registered to R
            feat_memRs = torch.cat([
                self.occ_memRs,
                self.unp_memRs*self.occ_memRs,
            ], dim=2)
            _, feat_memRs_ = self.featnet3D(
                __p(feat_memRs),
                self.summ_writer,
            )
            feat_memRs = __u(feat_memRs_)
            feat_memR = torch.mean(feat_memRs, dim=1)
            self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, pca=True)
            # overwrite feat_memR with its quantized version
            _, feat_memR, code_vox = self.vq3drgbnet(
                feat_memR,
                None,
            )

            self.summ_writer.summ_lrtlist('obj/boxlist_g', self.rgb_camRs[:,0], self.lrtlist_camRs[:,0],
                                          self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])
            masklist_memR = self.vox_util.assemble_padded_obj_masklist(
                self.lrtlist_camRs[:,0], self.scorelist_s[:,0], self.Z4, self.Y4, self.X4, coeff=1.2)
            mask_memR = torch.sum(masklist_memR, dim=1)
            print('mask_memR', mask_memR.shape)
            self.summ_writer.summ_oned('obj/mask_memR', mask_memR.clamp(0, 1), bev=True)

            occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z4, self.Y4, self.X4))
            occ_memR = torch.sum(occ_memRs, dim=1).clamp(0,1)

            obj_memR = (mask_memR * occ_memR).view(-1)
            bkg_memR = ((1.0 - mask_memR) * occ_memR).view(-1)
            obj_inds = torch.nonzero(obj_memR, as_tuple=False)
            obj_inds = obj_inds.detach().cpu().numpy()
            bkg_inds = torch.nonzero(bkg_memR, as_tuple=False)
            bkg_inds = bkg_inds.detach().cpu().numpy()
            print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))
            # input()

            # print('feat_memR', feat_memR.shape)
            # print('mask_memR', mask_memR.shape)
            
            codes_flat = np.reshape(code_vox.detach().cpu().numpy(), [-1])
            mean_acc, mean_pool_size, num_codes_w_20 = utils_eval.linmatch(self.labelpools, obj_inds, bkg_inds, codes_flat)
            utils_misc.add_loss('vq3drgbnet/mean_acc', 0.0, mean_acc, 0.0, self.summ_writer)
            utils_misc.add_loss('vq3drgbnet/mean_pool_size', 0.0, mean_pool_size, 0.0, self.summ_writer)
            utils_misc.add_loss('vq3drgbnet/num_codes_w_20', 0.0, num_codes_w_20, 0.0, self.summ_writer)
        return total_loss, None, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        
        # if set_name=='moc2D_init':
        #     self.prepare_common_tensors(feed, prep_summ=False)
        #     return self.prep_neg_emb2D(feed)
        
        # if set_name=='moc3D_init':
        #     self.prepare_common_tensors(feed, prep_summ=False)
        #     return self.prep_neg_emb3D(feed)
        
        if set_name=='train':
            self.prepare_common_tensors(feed)
            return self.run_train(feed)
            # elif set_name=='val':
            #     return self.run_train(feed)
        elif set_name=='test':
            self.prepare_common_tensors(feed)
            return self.run_test(feed)
        # else:

        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
