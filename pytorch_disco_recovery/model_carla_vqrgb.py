import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.linclassnet import LinClassNet
from nets.vqrgbnet import VqrgbNet
from nets.gen2dvqnet import Gen2dvqNet
from nets.sigen2dnet import Sigen2dNet

import torch.nn.functional as F

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

class CARLA_VQRGB(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaVqrgbModel()
        if hyp.do_freeze_vqrgb:
            self.model.vqrgbnet.eval()
            self.set_requires_grad(self.model.vqrgbnet, False)

class CarlaVqrgbModel(nn.Module):
    def __init__(self):
        super(CarlaVqrgbModel, self).__init__()
        if hyp.do_vqrgb:
            self.vqrgbnet = VqrgbNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
        if hyp.do_linclass:
            self.linclassnet = LinClassNet(64) # hardcoded emb dim in vqrgbnet
        if hyp.do_gen2dvq:
            self.gen2dvqnet = Gen2dvqNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
        if hyp.do_sigen2d:
            self.sigen2dnet = Sigen2dNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()

        self.labelpools = [utils_misc.SimplePool(1000) for i in list(range(hyp.vqrgb_num_embeddings))]
        print('declared labelpools')
        # input()
            
    def prepare_common_tensors(self, feed):
        results = dict()
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True,
        )

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
                            
        self.occXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        self.occX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unpXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.unpXs_half = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))

        ## projected depth, and inbound mask
        self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        #####################
        ## visualize what we got
        #####################
        # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
        # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(self.occXs, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(self.unpXs, dim=1), torch.unbind(self.occXs, dim=1))

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        set_name = feed['set_name']
        if 'train' in set_name:
            is_train = True
        else:
            is_train = False
            
        for b in list(range(self.B)):
            for s in list(range(self.S)):
                if torch.sum(self.scorelist_s[b,s]) == 0.0:
                    # return early
                    return total_loss, None, True
        # print('ok, scorelist is this:')
        # print(self.scorelist_s)
        # print(self.lrtlist_camRs)
        # print(self.tidlist_s)

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_vqrgb:
            rgb_input = self.rgb_camXs[:,0]
            vqrgb_loss, _, ind_image = self.vqrgbnet(
                rgb_input,
                self.summ_writer,
            )
            total_loss += vqrgb_loss

            # ind_image is B x H/8 x W/8

            quant = self.vqrgbnet.convert_inds_to_embeds(ind_image)
            # print('quant', quant.shape)
            
        if hyp.do_linclass:

            self.summ_writer.summ_lrtlist('obj/boxlist_g', self.rgb_camRs[:,0], self.lrtlist_camRs[:,0],
                                          self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])
            boxlist2d = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], self.lrtlist_camRs[:,0], self.H, self.W)

            boxlist2d_small = boxlist2d.clone()
            # for b in list(range(self.B)):
            for n in list(range(self.N)):
                mult_padding = 0.75
                box2D = boxlist2d[:,n]
                y, x = utils_geom.get_centroid_from_box2D(box2D)
                h, w = utils_geom.get_size_from_box2D(box2D)
                box2D = utils_geom.get_box2D_from_centroid_and_size(
                    y, x, h*mult_padding, w*mult_padding, clip=True)
                boxlist2d_small[:,n] = box2D
            
            self.summ_writer.summ_boxlist2D('obj/boxes2d', self.rgb_camRs[:,0], boxlist2d)

            boxlist2d = boxlist2d[0:1] # make B==1
            boxlist2d_small = boxlist2d_small[0:1] # make B==1
            
            # obj_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d_small, self.H, self.W)
            # bkg_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d, self.H, self.W)
            obj_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d_small, self.H//8, self.W//8)
            bkg_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d, self.H//8, self.W//8)
            # mask = torch.sum(masklist, dim=1, keepdim=True)
            # self.summ_writer.summ_oned('obj/mask', mask.clamp(0, 1))
            obj_mask = torch.sum(obj_masklist, dim=1, keepdim=True).clamp(0, 1)
            bkg_mask = 1.0 - torch.sum(bkg_masklist, dim=1, keepdim=True).clamp(0, 1)
            self.summ_writer.summ_oned('obj/obj_mask', obj_mask)
            self.summ_writer.summ_oned('obj/bkg_mask', bkg_mask)

            valid = F.interpolate(self.valid_camXs[0:1,0], scale_factor=0.125, mode='nearest')
            obj_mask = obj_mask * valid
            bkg_mask = bkg_mask * valid

            obj_mask = obj_mask.view(-1)
            bkg_mask = bkg_mask.view(-1)
            obj_inds = torch.nonzero(obj_mask, as_tuple=False)
            bkg_inds = torch.nonzero(bkg_mask, as_tuple=False)
            # print(obj_inds.shape)
            # print(bkg_inds.shape)
            # print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))

            code_vec = quant[0:1].detach().permute(0,2,3,1).reshape(-1, 64) # hardcoded emb dim

            if len(obj_inds) and len(bkg_inds):

                # print('obj_inds', obj_inds.shape)
                # print('bkg_inds', bkg_inds.shape)
                # print('code_vec', code_vec.shape)

                linclass_loss = self.linclassnet(
                    code_vec, obj_inds, bkg_inds, self.summ_writer)

                # print('feat_memR', feat_memR.shape)
                # print('mask_memR', mask_memR.shape)
                
                total_loss += linclass_loss
            
            
        if hyp.do_gen2dvq:
            # rgb_input = self.rgb_camXs[:,0]
            # gray_input = torch.mean(rgb_input, dim=1, keepdim=True)
            # gray_input = F.interpolate(gray_input, scale_factor=0.5, mode='bilinear')
            gen2dvq_loss = self.gen2dvqnet(
                ind_image,
                self.summ_writer,
            )
            total_loss += gen2dvq_loss
            
            # ind_map = self.gen2dvqnet.generate_sample(1, int(self.H/8), int(self.W/8))
            # print('got ind_map', ind_map.shape)
            
        if hyp.do_sigen2d:
            # rgb_input = self.rgb_camXs[:,0]
            # gray_input = torch.mean(rgb_input, dim=1, keepdim=True)
            # gray_input = F.interpolate(gray_input, scale_factor=0.5, mode='bilinear')
            sigen2d_loss = self.sigen2dnet(
                ind_image,
                self.summ_writer,
                is_train=is_train,
            )
            total_loss += sigen2d_loss
            
            # ind_map = self.sigen2dnet.generate_sample(1, int(self.H/8), int(self.W/8))
            # print('got ind_map', ind_map.shape)
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # print('test not ready')
        
        if hyp.do_gen2dvq:
            ind_map = self.gen2dvqnet.generate_sample(1, int(self.H/8), int(self.W/8))
        elif hyp.do_sigen2d:
            # ind_map = self.sigen2dnet.generate_sample(1, int(self.H/8), int(self.W/8))

            # we need to condition at least the topleft pixel
            rgb_g = self.rgb_camXs[0:1,0]
            z = self.vqrgbnet._encoder(rgb_g)
            z = self.vqrgbnet._pre_vq_conv(z)
            _, quantized, _, _, inds = self.vqrgbnet._vq_vae(z)
            _, _, H, W = list(quantized.shape)
            cond_mask = torch.zeros([1, 1, H, W]).cuda().float()
            cond_mask[:,:,0,0] = 1.0
            self.summ_writer.summ_oned('vqrgbnet/uncond_mask', cond_mask, norm=False)
            inds = inds.reshape(1, H, W)
            # mask out
            inds = (inds.float() * cond_mask[:,0]).long()
            # ind_map = self.sigen2dnet.generate_cond_sample(inds, cond_mask, summ_writer=self.summ_writer, mod='un')
            ind_map = self.sigen2dnet.generate_uncond_sample(inds, cond_mask, summ_writer=self.summ_writer, mod='un')
            quant = self.vqrgbnet.convert_inds_to_embeds(ind_map)
            rgb_e = self.vqrgbnet._decoder(quant)
        # else:
        #     assert(False) # you need either gen2dvq or sigen2d
            
        if hyp.do_gen2dvq or hyp.do_sigen2d:
            ind_map = self.gen2dvqnet.generate_sample(1, int(self.H/8), int(self.W/8))
            
            # self.summ_writer.summ_oned('vqrgbnet/sampled_ind_map', ind_map.unsqueeze(1)/512.0, norm=False)
            # quant = self.vqrgbnet.convert_inds_to_embeds(ind_map)
            # rgb_e = self.vqrgbnet._decoder(quant)
            # utils_py.print_stats('rgb_e', rgb_e.detach().cpu().numpy())
            self.summ_writer.summ_oned('vqrgbnet/uncond_sampled_ind_map', ind_map.unsqueeze(1)/512.0, norm=False)
            self.summ_writer.summ_rgb('vqrgbnet/uncond_sampled_rgb_e', rgb_e.clamp(-0.5, 0.5))
        
        if hyp.do_sigen2d:
            # let's also generate a conditional sample

            rgb_g = self.rgb_camXs[0:1,0]
            z = self.vqrgbnet._encoder(rgb_g)
            z = self.vqrgbnet._pre_vq_conv(z)
            _, quantized, _, _, inds = self.vqrgbnet._vq_vae(z)
            _, _, H, W = list(quantized.shape)
            cond_mask = (torch.rand([1, 1, H, W]).cuda() > 0.9).float() 
            utils_py.print_stats('cond_mask', cond_mask.detach().cpu().numpy())
            self.summ_writer.summ_oned('vqrgbnet/cond_mask', cond_mask, norm=False)
            inds = inds.reshape(1, H, W)
            
            # mask out
            inds = (inds.float() * cond_mask[:,0]).long()
            ind_map = self.sigen2dnet.generate_cond_sample(inds, cond_mask, summ_writer=self.summ_writer)
            # self.summ_writer.summ_oned('vqrgbnet/cond_sampled_ind_map', ind_map.unsqueeze(1)/512.0, norm=False)
            quant = self.vqrgbnet.convert_inds_to_embeds(ind_map)
            rgb_e = self.vqrgbnet._decoder(quant)
            self.summ_writer.summ_rgb('vqrgbnet/cond_sampled_rgb_e', rgb_e.clamp(-0.5, 0.5))

            recon_loss = F.mse_loss(rgb_g, rgb_e)
            utils_misc.add_loss('vqrgbnet/cond_recon', 0.0, recon_loss, 0.0, self.summ_writer)

            rgb_e = self.vqrgbnet._decoder(quantized)
            self.summ_writer.summ_rgb('vqrgbnet/cond_sampled_rgb_g', rgb_e.clamp(-0.5, 0.5))

            # generate another, to see if it is multimodal
            ind_map = self.sigen2dnet.generate_cond_sample(inds, cond_mask)
            quant = self.vqrgbnet.convert_inds_to_embeds(ind_map)
            rgb_e = self.vqrgbnet._decoder(quant)
            self.summ_writer.summ_rgb('vqrgbnet/cond_sampled_rgb_e_v2', rgb_e.clamp(-0.5, 0.5))

        # print('ok done testing')

        test_linmatch = True

        if test_linmatch:
            # let's start with axis-aligned 2d boxes
            
            self.summ_writer.summ_lrtlist('obj/boxlist_g', self.rgb_camRs[:,0], self.lrtlist_camRs[:,0],
                                          self.scorelist_s[:,0], self.tidlist_s[:,0], self.pix_T_cams[:,0])

            boxlist2d = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], self.lrtlist_camRs[:,0], self.H, self.W)


            # boxlist = []
            
            boxlist2d_small = boxlist2d.clone()
            # for b in list(range(self.B)):
            for n in list(range(self.N)):
                mult_padding = 0.75
                box2D = boxlist2d[:,n]
                y, x = utils_geom.get_centroid_from_box2D(box2D)
                h, w = utils_geom.get_size_from_box2D(box2D)
                box2D = utils_geom.get_box2D_from_centroid_and_size(
                    y, x, h*mult_padding, w*mult_padding, clip=True)
                boxlist2d_small[:,n] = box2D
            
            self.summ_writer.summ_boxlist2D('obj/boxes2d', self.rgb_camRs[:,0], boxlist2d)

            boxlist2d = boxlist2d[0:1] # make B==1
            boxlist2d_small = boxlist2d_small[0:1] # make B==1
            
            # obj_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d_small, self.H, self.W)
            # bkg_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d, self.H, self.W)
            obj_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d_small, self.H//8, self.W//8)
            bkg_masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d, self.H//8, self.W//8)
            # mask = torch.sum(masklist, dim=1, keepdim=True)
            # self.summ_writer.summ_oned('obj/mask', mask.clamp(0, 1))
            obj_mask = torch.sum(obj_masklist, dim=1, keepdim=True).clamp(0, 1)
            bkg_mask = 1.0 - torch.sum(bkg_masklist, dim=1, keepdim=True).clamp(0, 1)
            self.summ_writer.summ_oned('obj/obj_mask', obj_mask)
            self.summ_writer.summ_oned('obj/bkg_mask', bkg_mask)

            # masklist = utils_improc.convert_boxlist2D_to_masklist(boxlist2d,
            #                                                       self.H//8,
            #                                                       self.W//8)
            # mask = torch.sum(masklist, dim=1, keepdim=True)
            # self.summ_writer.summ_oned('obj/mini_mask', mask.clamp(0, 1))

            obj_mask = obj_mask.view(-1)
            bkg_mask = bkg_mask.view(-1)
            obj_inds = torch.nonzero(obj_mask, as_tuple=False)
            obj_inds = obj_inds.detach().cpu().numpy()
            bkg_inds = torch.nonzero(bkg_mask, as_tuple=False)
            bkg_inds = bkg_inds.detach().cpu().numpy()
            print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))
            # input()

            rgb_input = self.rgb_camXs[0:1,0]
            _, _, code_map = self.vqrgbnet(
                rgb_input,
                None,
            )
            
            codes_flat = np.reshape(code_map.detach().cpu().numpy(), [-1])
            mean_acc, mean_pool_size, num_codes_w_20 = utils_eval.linmatch(self.labelpools, obj_inds, bkg_inds, codes_flat)
            utils_misc.add_loss('vqrgbnet/num_codes_w_20', 0.0, num_codes_w_20, 0.0, self.summ_writer)
            utils_misc.add_loss('vqrgbnet/mean_acc', 0.0, mean_acc, 0.0, self.summ_writer)
            utils_misc.add_loss('vqrgbnet/mean_pool_size', 0.0, mean_pool_size, 0.0, self.summ_writer)

            
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
