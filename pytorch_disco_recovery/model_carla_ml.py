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

class CARLA_ML(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaMlModel()
        if hyp.do_freeze_feat2D:
            self.model.featnet2D.eval()
            self.set_requires_grad(self.model.featnet2D, False)
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)

class CarlaMlModel(nn.Module):
    def __init__(self):
        super(CarlaMlModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
        # if hyp.do_emb3D:
        #     self.embnet3D = EmbNet3D()
        
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=4)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
            
            
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
            
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        # if self.set_name=='test':
        #     self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        # elif self.set_name=='val':
        #     self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        # else:
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(__p(self.camR0s_T_camRs).inverse())
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        
        all_ok = False
        num_tries = 0
        while not all_ok:
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            num_tries += 1
            all_ok = True
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
            # we want to ensure this gives us a few points inbound for each batch el
            inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X4, already_mem=False))
            num_inb = torch.sum(inb.float(), axis=2)
            if torch.min(num_inb) < 200:
                all_ok = False
            if num_tries > 100:
                return False
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
        
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

        # for b in list(range(self.B)):
        #     # if torch.sum(scorelist_s[b,0]) == 0:
        #     if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
        #         return 0.0, None, True

        self.lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))
        self.lrtlist_camR0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        
            
        
        # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)

        # # self.deglist = [-6, 0, 6]
        # # self.trim = 5
        # # self.radlist = [utils_geom.deg2rad(deg) for deg in self.deglist]


        # visX_e = []
        # for s in list(range(0, self.S, 2)):
        #     visX_e.append(self.summ_writer.summ_lrtlist(
        #         '', self.rgb_camXs[:,s],
        #         self.lrtlist_camXs[:,s],
        #         self.scorelist_s[:,s],
        #         self.tidlist_s[:,s],
        #         # torch.cat([self.lrtlist_camXs[:,s:s+1]], dim=1),
        #         # torch.cat([self.lrtlist_camXs[:,s:s+1]], dim=1),
        #         # torch.cat([torch.ones([self.B,1]).float().cuda()], dim=1),
        #         # torch.arange(1,3).reshape(self.B, 2).long().cuda(),
        #         self.pix_T_cams[:,s], only_return=True))
        # self.summ_writer.summ_rgbs('obj/box_camXs_g', visX_e)

        # print('set_name', self.set_name)
        # print('vox_size_X', self.vox_size_X)
        # print('vox_size_Y', self.vox_size_Y)
        # print('vox_size_Z', self.vox_size_Z)
        
        return True # OK
    
    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)
        
        self.rgb_camXs = feed["rgb_camXs"]
        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
            
        if hyp.do_feat3D:
            feat_memX0s_input = torch.cat([
                self.occ_memX0s,
                self.unp_memX0s*self.occ_memX0s,
            ], dim=2)
            feat3D_loss, feat_memX0s_, valid_memX0s_ = self.featnet3D(
                __p(feat_memX0s_input[:,1:]),
                self.summ_writer,
            )
            feat_memX0s = __u(feat_memX0s_)
            valid_memX0s = __u(valid_memX0s_)
            total_loss += feat3D_loss

            # print('feat_memX0s', feat_memX0s.shape)
            # input()

            feat_memX0 = utils_basic.reduce_masked_mean(
                feat_memX0s,
                valid_memX0s.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
                dim=1)
            valid_memX0 = torch.sum(valid_memX0s, dim=1).clamp(0, 1)
            self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, valid=valid_memX0, pca=True)
            # self.summ_writer.summ_feat('3D_feats/valid_memX0', valid_memX0, pca=False)

            if hyp.do_emb3D:
                _, altfeat_memX0, altvalid_memX0 = self.featnet3D_slow(feat_memX0s_input[:,0])
                self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
                # self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)

        if hyp.do_emb3D:
            # compute 3D ML
            
            emb_loss_3D = self.embnet3D(
                feat_memX0,
                altfeat_memX0,
                valid_memX0.round(),
                altvalid_memX0.round(),
                self.summ_writer)
            total_loss += emb_loss_3D
                
            
        #     feat_memXs_input = torch.cat([
        #         self.occ_memXs,
        #         self.unp_memXs*self.occ_memXs,
        #     ], dim=2)
        #     feat_memRs_input = torch.cat([
        #         self.occ_memRs,
        #         self.unp_memRs*self.occ_memRs,
        #     ], dim=2)
            
        #     feat3D_loss, feat_memXs_ = self.featnet3D(
        #         __p(feat_memXs_input[:,1:]),
        #         self.summ_writer,
        #     )
        #     feat_memXs = __u(feat_memXs_)
        #     valid_memXs = torch.ones_like(feat_memXs[:,:,0:1])
        #     total_loss += feat3D_loss
            
        #     # warp things to R
        #     feat_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], feat_memXs)
        #     valid_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], valid_memXs)

        #     feat_memR = utils_basic.reduce_masked_mean(
        #         feat_memRs,
        #         valid_memRs.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
        #         dim=1)
        #     valid_memR = torch.sum(valid_memRs, dim=1).clamp(0, 1)

        #     _, altfeat_memR = self.featnet3D(feat_memRs_input[:,0])
            
        #     self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)
        #     self.summ_writer.summ_feat('3D_feats/altfeat_memR', altfeat_memR, pca=True)

        # if hyp.do_vq3drgb:
        #     # overwrite altfeat_memR with its quantized version
        #     vq3drgb_loss, altfeat_memR, _ = self.vq3drgbnet(
        #         altfeat_memR,
        #         self.summ_writer,
        #     )
        #     total_loss += vq3drgb_loss

        if hyp.do_occ:
            _, _, Z_, Y_, X_ = list(feat_memX0.shape)
            occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                Z_, Y_, X_,
                agg=True)
            self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memX0_sup)
            self.summ_writer.summ_occ('occ_sup/free_sup', free_memX0_sup)
            occ_loss, occ_memX0_pred = self.occnet(
                altfeat_memX0, 
                occ_memX0_sup,
                free_memX0_sup,
                altvalid_memX0, 
                self.summ_writer)
            total_loss += occ_loss

        # if hyp.do_view:
        #     assert(hyp.do_feat3D)
        #     # decode the feat volume into an image
        #     view_loss, rgb_e, view_camX0 = self.viewnet(
        #         self.pix_T_cams[:,0],
        #         self.camX0_T_camR,
        #         altfeat_memR,
        #         self.rgb_camXs[:,0], 
        #         summ_writer=self.summ_writer)
        #     total_loss += view_loss
            
        # if hyp.do_emb2D:
        #     assert(hyp.do_view)
        #     # create an embedding image, representing the bottom-up 2D feature tensor
        #     emb_loss_2D, _ = self.embnet2D(
        #         view_camX0,
        #         feat_camX0,
        #         torch.ones_like(view_camX0[:,0:1]),
        #         self.summ_writer)
        #     total_loss += emb_loss_2D
            
        # if hyp.do_emb3D:
        #     # compute 3D ML
        #     emb_loss_3D = self.embnet3D(
        #         feat_memR,
        #         altfeat_memR,
        #         valid_memR.round(),
        #         torch.ones_like(valid_memR),
        #         self.summ_writer)
        #     total_loss += emb_loss_3D

        # if hyp.do_linclass:
            
        #     masklist_memR = self.vox_util.assemble_padded_obj_masklist(
        #         self.lrtlist_camRs[:,0], self.scorelist_s[:,0], self.Z2, self.Y2, self.X2, coeff=1.2)
        #     mask_memR = torch.sum(masklist_memR, dim=1)
        #     self.summ_writer.summ_oned('obj/mask_memR', mask_memR.clamp(0, 1), bev=True)

        #     occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
        #     occ_memR = torch.sum(occ_memRs, dim=1).clamp(0,1)
            
        #     obj_memR = (mask_memR * occ_memR).view(-1)
        #     bkg_memR = ((1.0 - mask_memR) * occ_memR).view(-1)
        #     obj_inds = torch.nonzero(obj_memR, as_tuple=False)
        #     # obj_inds = obj_inds.detach().cpu().numpy()
        #     bkg_inds = torch.nonzero(bkg_memR, as_tuple=False)
        #     # bkg_inds = bkg_inds.detach().cpu().numpy()

        #     # print('altfeat_memR', altfeat_memR.shape)
        #     # print('mask_memR', mask_memR.shape)
        #     # print('occ_memR', occ_memR.shape)
        #     # print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))
        #     # input()
            
        #     code_vec = altfeat_memR.detach().permute(0,2,3,4,1).reshape(-1, hyp.feat3D_dim)
        #     obj_inds = obj_inds.reshape([-1])
        #     bkg_inds = bkg_inds.reshape([-1])

        #     if len(obj_inds) and len(bkg_inds):

        #         # print('obj_inds', obj_inds.shape)
        #         # print('bkg_inds', bkg_inds.shape)
        #         # print('codes_flat', codes_flat.shape)

        #         linclass_loss = self.linclassnet(
        #             code_vec, obj_inds, bkg_inds, self.summ_writer)

        #         # print('feat_memR', feat_memR.shape)
        #         # print('mask_memR', mask_memR.shape)
                
        #         total_loss += linclass_loss
            
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

        ok = self.prepare_common_tensors(feed)
        if not ok:
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if set_name=='train':
                return self.run_train(feed)
            elif set_name=='test':
                return self.run_test(feed)

        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)
