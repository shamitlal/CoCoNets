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
import utils_hard_eval
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc
import vox_util
import utils_vox
np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

class CARLA_RET(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaRetModel()
        # if hyp.do_freeze_feat2D:
        #     self.model.featnet2D.eval()
        #     self.set_requires_grad(self.model.featnet2D False)
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_feat3D:
            self.model.vq3drgbnet.eval()
            self.set_requires_grad(self.model.vq3drgbnet, False)
        # if hyp.do_freeze_view:
        #     self.model.viewnet.eval()
        #     self.set_requires_grad(self.model.viewnet, False)
        # if hyp.do_freeze_occ:
        #     self.model.occnet.eval()
        #     self.set_requires_grad(self.model.occnet, False)

class CarlaRetModel(nn.Module):
    def __init__(self):
        super(CarlaRetModel, self).__init__()

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
        self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        self.unp_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)
        self.unp_memRs_half = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs_half)

        ## projected depth, and inbound mask
        self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
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
            self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=30.0)
            self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        
    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_feat3D:
            feat_memR_input_e = torch.cat([
                self.occ_memRs[:,0],
                self.unp_memRs[:,0]*self.occ_memRs[:,0],
            ], dim=1)
            feat3D_loss, feat_memR_e = self.featnet3D(
                feat_memR_input_e,
                self.summ_writer,
            )

            total_loss += feat3D_loss

            feat_memR_input_g = torch.cat([
                self.occ_memRs[:,1],
                self.unp_memRs[:,1]*self.occ_memRs[:,1],
            ], dim=1)
            
            feat3D_loss, feat_memR_g = self.featnet3D(
                feat_memR_input_g,
                self.summ_writer,
            )

            total_loss += feat3D_loss

            self.summ_writer.summ_feat('3D_feats/feat_memR_e', feat_memR_e, pca=True)
        if hyp.do_vq3drgb:
            # overwrite feat_memR with its quantized version
            vq3drgb_loss, feat_memR_e, ind_memR = self.vq3drgbnet(
                feat_memR_e,
                self.summ_writer,
            )
            total_loss += vq3drgb_loss

            vq3drgb_loss, feat_memR_g, ind_memR = self.vq3drgbnet(
                feat_memR_g,
                self.summ_writer,
            )
            total_loss += vq3drgb_loss            
        if hyp.do_hard_vis:
            Z_SIZE,Y_SIZE,X_SIZE = (hyp.Z//2, hyp.Y//2, hyp.X//2)
            __pb = lambda x: utils_hard_eval.pack_boxdim(x, hyp.N)
            __ub = lambda x: utils_hard_eval.unpack_boxdim(x, hyp.N)            
            if hyp.use_random_boxes:
                bboxes_random_camR = utils_hard_eval.gen_random_boxes().cuda().to(torch.float32)
                scores = torch.ones([hyp.B,hyp.N])
            else:
                bboxes_random_camR = self.boxlist_camRs[:,0]
                scores = self.scorelist_s[:,0]
            bboxesCorner_random_camR = utils_geom.transform_boxes_to_corners(bboxes_random_camR, legacy_format=True)
            bboxesCorner_random_Mem = __ub(utils_vox.Ref2Mem(__pb(bboxesCorner_random_camR),Z_SIZE,Y_SIZE,X_SIZE))
            bboxesEnd_random_Mem = utils_hard_eval.get_ends_of_corner(bboxesCorner_random_Mem)

            bboxesCorner_random_pixR = __ub(utils_geom.apply_pix_T_cam(self.pix_T_cams[:,0], __pb(bboxesCorner_random_camR)))

            emb_e,emb_g = utils_hard_eval.create_object_tensors([feat_memR_e,feat_memR_g],bboxesEnd_random_Mem,scores)

            object_rgb = utils_hard_eval.create_object_rgbs(self.rgb_camRs[:,0],bboxesCorner_random_pixR,scores)

            results['emb3D_e'] = emb_e
            results['emb3D_g'] = emb_g
            results['rgb'] = object_rgb
        # for z in list(range(5, self.Z2 - 4, 8)):
        #     # for y in list(range(0, self.Y2 - 1, 2)):
        #     for y in [0,1]:
        #         for x in list(range(1, self.X2 - 4, 8)):
        #             feat_crop = feat_memR[:, # batch
        #                                   :, # chans
        #                                   z:z+5,
        #                                   y:y+5,
        #                                   x:x+5]
        #             print('feat_crop', feat_crop.shape)
        #             unp_crop = self.unp_memRs_half[:, # batch
        #                                            0, # seq
        #                                            :, # chans
        #                                            z:z+5,
        #                                            y:y+5,
        #                                            x:x+5]
        #             print('unp_crop', unp_crop.shape)
        #             occ_crop = self.occ_memRs_half[:, # batch
        #                                            0, # seq
        #                                            :, # chans
        #                                            z:z+5,
        #                                            y:y+5,
        #                                            x:x+5]
        #             print('occ_crop', occ_crop.shape)

        #             if torch.sum(occ_crop) > 3:

        #                 xs = np.array([x+5, x+5, x, x, x+5, x+5, x, x]).astype(np.float32)
        #                 ys = np.array([y+5, y+5, y+5, y+5, y, y, y, y]).astype(np.float32)
        #                 zs = np.array([z+5, z, z, z+5, z+5, z, z, z+5]).astype(np.float32)
        #                 xs = np.reshape(xs, (1, 8))
        #                 ys = np.reshape(ys, (1, 8))
        #                 zs = np.reshape(zs, (1, 8))
        #                 xyz = np.stack([xs, ys, zs], axis=2)
        #                 xyz = torch.from_numpy(xyz).float().cuda()
        #                 # this is 1 x 8 x 3
        #                 # print('xyz', xyz.shape)
        #                 corners_cam = self.vox_util.Mem2Ref(xyz, self.Z2, self.Y2, self.X2)
        #                 corners_cam = corners_cam.unsqueeze(1)
        #                 # this is 1 x 1 x 8 x 3
        #                 scores = torch.ones_like(corners_cam[:,:,0,0])
        #                 tids = torch.ones_like(corners_cam[:,:,0,0]).long()

        #                 print('boxes/%d_%d_%d' % (z, y, x))
        #                 self.summ_writer.summ_box_by_corners('boxes/box_%d_%d_%d' % (z, y, x), self.rgb_camRs[:,0],
        #                                                      corners_cam, scores, tids, self.pix_T_cams[:,0],
        #                                                      only_return=False)
        #                 self.summ_writer.summ_unp('boxes/unp_%d_%d_%d' % (z, y, x), unp_crop, occ_crop)
        #                 self.summ_writer.summ_occ('boxes/occ_%d_%d_%d' % (z, y, x), occ_crop)
        #                 self.summ_writer.summ_feat('boxes/feat_%d_%d_%d' % (z, y, x), feat_crop, pca=True)
                    
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        
        if set_name=='train':
            self.prepare_common_tensors(feed)
            return self.run_train(feed)
        elif set_name=='test':
            self.prepare_common_tensors(feed)
            return self.run_train(feed)
        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
