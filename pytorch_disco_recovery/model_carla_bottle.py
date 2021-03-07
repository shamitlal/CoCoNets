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
from nets.upnet3D import UpNet3D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet

from nets.vq3dnet import Vq3dNet
from nets.occnet import OccNet
from nets.centernet import CenterNet
from nets.segnet import SegNet


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

class CARLA_BOTTLE(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaBottleModel()
        if hyp.do_freeze_feat2D:
            self.model.featnet2D.eval()
            self.set_requires_grad(self.model.featnet2D, False)
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_up3D:
            self.model.upnet3D.eval()
            self.set_requires_grad(self.model.upnet3D, False)
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
        if hyp.do_freeze_center:
            self.model.centernet.eval()
            self.set_requires_grad(self.model.centernet, False)
        if hyp.do_freeze_seg:
            self.model.segnet.eval()
            self.set_requires_grad(self.model.segnet, False)
        if hyp.do_freeze_vq3d:
            self.model.vq3dnet.eval()
            self.set_requires_grad(self.model.vq3dnet, False)

class CarlaBottleModel(nn.Module):
    def __init__(self):
        super(CarlaBottleModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)
            
        if hyp.do_up3D:
            self.upnet3D = UpNet3D()
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

        if hyp.do_vq3d:
            self.vq3dnet = Vq3dNet()
            self.labelpools = [utils_misc.SimplePool(100) for i in list(range(hyp.vq3d_num_embeddings))]
            print('declared labelpools')

        if hyp.do_linclass:
            self.linclassnet = LinClassNet(hyp.feat3D_dim)

        if hyp.do_occ:
            self.occnet = OccNet()
            
        if hyp.do_center:
            self.centernet = CenterNet()
            
        if hyp.do_seg:
            self.num_seg_labels = 13 # note label0 is "none"
            # we will predict all 12 valid of these, plus one "air" class
            self.segnet = SegNet(self.num_seg_labels)
            
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
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
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
            # inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X4, already_mem=False, padding=12.0))
            inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z, self.Y, self.X, already_mem=False, padding=28.0))
            # this is B x S x N
            num_inb = torch.sum(inb.float(), axis=2)
            # this is B x S
            if torch.min(num_inb) < 300:
                all_ok = False
            if num_tries > 100:
                return False
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
        
        # _boxlist_camRs = feed["boxlists"]
        # _tidlist_s = feed["tidlists"] # coordinate-less and plural
        # _scorelist_s = feed["scorelists"] # coordinate-less and plural
        # _scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(
        #     utils_geom.eye_4x4(self.B*self.S),
        #     __p(_boxlist_camRs),
        #     __p(_tidlist_s),
        #     self.Z, self.Y, self.X,
        #     self.vox_util,
        #     only_cars=False, pad=2.0))
        # boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
        #     __p(_boxlist_camRs), __p(_tidlist_s), __p(_scorelist_s))
        # self.boxlist_camRs = __u(boxlist_camRs_)
        # self.tidlist_s = __u(tidlist_s_)
        # self.scorelist_s = __u(scorelist_s_)

        # for b in list(range(self.B)):
        #     # if torch.sum(scorelist_s[b,0]) == 0:
        #     if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
        #         return 0.0, None, True

        # lrtlist_camRs_, obj_lens_ = utils_misc.parse_boxes(__p(feed["boxlists"]), __p(self.origin_T_camRs))
        origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
        boxlists = feed["boxlists"]
        self.scorelist_s = feed["scorelists"]
        self.tidlist_s = feed["tidlists"]
        print('boxlists', boxlists.shape)
        boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
        lrtlist_camRs_, _ = utils_misc.parse_boxes(boxlists_, origin_T_camRs_)
        self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        
        # origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4)
        # self.lrtlist_camRs = utils_misc.parse_boxes(box_camRs, origin_T_camRs)
        # self.lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))
        self.lrtlist_camR0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        # self.crop_guess = (40,40,40)
        # self.crop_guess = (20,20,20)
        # self.crop_guess = (52,52,52)
        self.crop_guess = (50,50,50)
        lrtlist = self.lrtlist_camX0s[:,0]
        clist = utils_geom.get_clist_from_lrtlist(lrtlist)
        # this is B x N x 3
        mask = self.vox_util.xyz2circles(clist, self.Z, self.Y, self.X, radius=1.0, soft=True, already_mem=False)
        mask = mask[:,:,
                    self.crop_guess[0]:-self.crop_guess[0],
                    self.crop_guess[1]:-self.crop_guess[2],
                    self.crop_guess[2]:-self.crop_guess[2]]
        self.center_mask = torch.max(mask, dim=1, keepdim=True)[0]

        mask_max = torch.max(self.center_mask.reshape(self.B, -1), dim=1)[0]
        # print('mask_max', mask_max.detach().cpu().numpy())
        if torch.min(mask_max) < 1.0:
            # print('returning early!!!')
            # at least one ex has no objects in the crop; let's return early
            return False
        # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)


        self.rgb_camXs = feed['rgb_camXs']
        visX_e = []
        for s in list(range(0, self.S, 2)):
            visX_e.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,s],
                self.lrtlist_camXs[:,s],
                self.scorelist_s[:,s],
                self.tidlist_s[:,s],
                self.pix_T_cams[:,s], only_return=True))
        self.summ_writer.summ_rgbs('obj/box_camXs_g', visX_e)

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
        
        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

        if hyp.do_feat3D:
            feat_memX0s_input = torch.cat([
                self.occ_memX0s,
                self.rgb_memX0s*self.occ_memX0s,
            ], dim=2)
            feat_memX0_input = utils_basic.reduce_masked_mean(
                feat_memX0s_input,
                self.occ_memX0s.repeat(1, 1, 4, 1, 1, 1),
                dim=1)
            
            feat3D_loss, feat_memX0, _ = self.featnet3D(
                feat_memX0_input,
                self.summ_writer,
            )
            total_loss += feat3D_loss
            self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, pca=True)
            print('feat total_loss', total_loss.detach().cpu().numpy())

        if hyp.do_vq3d:
            # overwrite feat_memX0 with its quantized version
            vq3d_loss, feat_memX0, ind_memX0 = self.vq3dnet(feat_memX0, self.summ_writer)
            total_loss += vq3d_loss
            self.summ_writer.summ_feat('3D_feats/feat_memX0_quantized', feat_memX0, pca=True)
            print('vq total_loss', total_loss.detach().cpu().numpy())
            
        if hyp.do_up3D:
            up3D_loss, feat_memX0 = self.upnet3D(feat_memX0, self.summ_writer)
            total_loss += up3D_loss
            print('up total_loss', total_loss.detach().cpu().numpy())

            valid_memX0 = torch.ones_like(feat_memX0[:,0:1])

            _, _, Z2, Y2, X2 = list(feat_memX0.shape)
            Z_crop = int((self.Z - Z2)/2)
            Y_crop = int((self.Y - Y2)/2)
            X_crop = int((self.X - X2)/2)
            crop = (Z_crop, Y_crop, X_crop)
            if not (crop==self.crop_guess):
                print('crop', crop)
            assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
            self.summ_writer.summ_feat('3D_feats/feat_memX0_up', feat_memX0, pca=True)

            # if hyp.do_emb3D:
            #     _, altfeat_memX0, altvalid_memX0, _ = self.featnet3D_slow(feat_memX0s_input[:,0])
            #     self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
            #     self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)

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

        if hyp.do_occ:
            occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z, self.Y, self.X,
                # self.Z2, self.Y2, self.X2,
                agg=True)
            occ_memX0_sup = occ_memX0_sup[:,:,
                                          crop[0]:-crop[0],
                                          crop[1]:-crop[2],
                                          crop[2]:-crop[2]]
            free_memX0_sup = free_memX0_sup[:,:,
                                            crop[0]:-crop[0],
                                            crop[1]:-crop[2],
                                            crop[2]:-crop[2]]
            
            occ_loss, occ_memX0_pred = self.occnet(
                # altfeat_memX0, 
                feat_memX0, 
                occ_memX0_sup,
                free_memX0_sup,
                # altvalid_memX0, 
                valid_memX0, 
                self.summ_writer)
            total_loss += occ_loss
            print('occ total_loss', total_loss.detach().cpu().numpy())

        if hyp.do_center:
            # this net achieves the following:
            # objectness: put 1 at each object center and 0 everywhere else
            # orientation: at the object centers, classify the orientation into a rough bin
            # size: at the object centers, regress to the object size

            lrtlist_camX0 = self.lrtlist_camX0s[:,0]
            lrtlist_memX0 = self.vox_util.apply_mem_T_ref_to_lrtlist(
                lrtlist_camX0, self.Z, self.Y, self.X)
            scorelist = self.scorelist_s[:,0]
            
            center_loss, lrtlist_camX0_e, scorelist_e = self.centernet(
                feat_memX0, 
                crop,
                self.vox_util, 
                self.center_mask,
                lrtlist_camX0,
                lrtlist_memX0,
                scorelist, 
                self.summ_writer)
            total_loss += center_loss
            print('cen total_loss', total_loss.detach().cpu().numpy())

            if lrtlist_camX0_e is not None:
                # lrtlist_camX_e = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camX0[:,0], lrtlist_camX0_e)
                # lrtlist_camR_e = utils_geom.apply_4x4_to_lrtlist(self.camRs_T_camXs[:,0], lrtlist_camXs_e)
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_e',
                    self.rgb_camXs[0:1,0],
                    lrtlist_camX0_e[0:1], 
                    scorelist_e[0:1],
                    torch.arange(50).reshape(1, 50).long().cuda(), # tids
                    self.pix_T_cams[0:1,0])
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_g',
                    self.rgb_camXs[0:1,0],
                    self.lrtlist_camXs[0:1,0],
                    self.scorelist_s[0:1,0],
                    self.tidlist_s[0:1,0],
                    self.pix_T_cams[0:1,0])
            
            
        if hyp.do_seg:
            seg_camXs = feed['seg_camXs']
            self.summ_writer.summ_seg('seg/seg_camX0', seg_camXs[:,0])
            seg_memX0 = utils_misc.parse_seg_into_mem(
                seg_camXs, self.num_seg_labels, self.occ_memX0s,
                self.pix_T_cams, self.camX0s_T_camXs, self.vox_util)
            seg_memX0 = seg_memX0[:,
                                  crop[0]:-crop[0],
                                  crop[1]:-crop[2],
                                  crop[2]:-crop[2]]
            seg_memX0_vis = torch.max(seg_memX0, dim=2)[0]
            # self.summ_writer.summ_seg('seg/seg_memX0_vis', seg_memX0_vis)

            # occ_memX0 = torch.max(self.occ_memXs, dim=1)[0]
            # occ_memX0 = occ_memX0[:,:,
            #                       crop[0]:-crop[0],
            #                       crop[1]:-crop[2],
            #                       crop[2]:-crop[2]]
            
            seg_loss, seg_memX0_pred = self.segnet(
                feat_memX0, 
                seg_memX0,
                # occ_memX0,
                occ_memX0_sup,
                free_memX0_sup,
                self.summ_writer)
            total_loss += seg_loss
            
            print('seg total_loss', total_loss.detach().cpu().numpy())
            
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

    def run_export(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        
        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.rgb_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.rgb_memXs)
        self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        feat_memRs_input = torch.cat([
            self.occ_memRs,
            self.rgb_memRs*self.occ_memRs,
        ], dim=2)
        feat_memR0_input = utils_basic.reduce_masked_mean(
            feat_memRs_input,
            self.occ_memRs.repeat(1, 1, 4, 1, 1, 1),
            dim=1)
        _, feat_memR0, _ = self.featnet3D(feat_memR0_input, self.summ_writer)
        self.summ_writer.summ_feat('3D_feats/feat_memR0', feat_memR0, pca=True)
        # overwrite feat_memR0 with its quantized version
        _, feat_memR0, ind_memR0 = self.vq3dnet(feat_memR0)
        # print('ind_memR0', ind_memR0.shape)
        # ind_memR0 = ind_memR0.detach().cpu().numpy()[0]
        # this is 8 x 8 x 8
        results['ind_memR0'] = ind_memR0
        
        # i think the right way to do this is:
        # save the entire dataset of these 8x8x8 grids into one npz
        
        return total_loss, results, False

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
                return self.run_export(feed)

        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)
