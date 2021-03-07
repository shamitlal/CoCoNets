import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.preoccnet import PreoccNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

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

class CARLA_STATIC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaStaticModel()
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

class CarlaStaticModel(nn.Module):
    def __init__(self):
        super(CarlaStaticModel, self).__init__()
        if hyp.do_feat:
            self.featnet = FeatNet(in_dim=4)
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_preocc:
            self.preoccnet = PreoccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_render:
            self.rendernet = RenderNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()

        if hyp.feat_cluster:
            self.KMeans = KMeans(n_clusters=hyp.feat_quantize_dictsize,
                                 max_iter=500,
                                 n_jobs=10,
                                 verbose=True)
            self.kmeans_objs = list()
            self.kmeans_obj_num_samples = 150
            self.cluster_vis_done = False

    def prepare_common_tensors(self, feed):
        results = dict()
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
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

        #####################
        ## run the nets
        #####################

        if hyp.do_preocc:
            # pre-occ is a pre-estimate of occupancy
            # as another mnemonic, it marks the voxels we will preoccupy ourselves with

            # if not hyp.do_freeze_preocc:
            unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, utils_basic.matmul2(
                    __p(self.pix_T_cams), utils_geom.safe_inverse(__p(self.camRs_T_camXs)))))
            occR0_sup, freeR0_sup, occRs, freeRs = utils_vox.prep_occs_supervision(
                self.camRs_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2, 
                agg=True)
            self.summ_writer.summ_occ('occ_sup/occR0_sup', self.occR0_sup)
            self.summ_writer.summ_occ('occ_sup/freeR0_sup', self.freeR0_sup)
            self.summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(self.freeRs, dim=1))
            self.summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(self.occRs, dim=1))
            
            preoccR0_input = torch.cat([
                occRs[:,0],
                freeRs[:,0],
                occRs[:,0]*unpRs[:,0]
            ], dim=1)
            
            preocc_loss, compR0 = self.preoccnet(
                preoccR0_input,
                occR0_sup,
                freeR0_sup,
                self.summ_writer,
            )
            total_loss += preocc_loss

            compRs = compR0.unsqueeze(1).repeat(1, hyp.S, 1, 1, 1, 1)
            compXs = utils_vox.apply_4x4s_to_voxs(self.camXs_T_camRs, compRs)
            self.summ_writer.summ_occs('preocc/compXs', torch.unbind(compXs, dim=1))
            amount_compXs = torch.mean(compXs)
            self.summ_writer.summ_scalar('preocc/amount_compXs', amount_compXs.cpu().item())
        else:
            compXs = torch.ones_like(self.occXs)

        if hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            featXs_input = torch.cat([self.occXs, self.occXs*self.unpXs], dim=2)
            featXs_input_ = __p(featXs_input)

            # it is useful to keep track of what was visible from each viewpoint
            freeXs_ = utils_vox.get_freespace(__p(self.xyz_camXs), __p(self.occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(self.occXs_half+freeXs, 0.0, 1.0)

            comp_maskXs_ = F.interpolate(__p(compXs), scale_factor=2)
            
            featXs_, validXs_, feat_loss = self.featnet(
                featXs_input_,
                self.summ_writer,
                comp_mask=comp_maskXs_,
            )
                # comp_mask=__p(occXs))
            total_loss += feat_loss
            
            validXs = __u(validXs_)
            _validX00 = validXs[:,0:1]
            _validX01 = utils_vox.apply_4x4s_to_voxs(self.camX0_T_camXs[:,1:], validXs[:,1:])
            validX0s = torch.cat([_validX00, _validX01], dim=1)
            
            _visX00 = visXs[:,0:1]
            _visX01 = utils_vox.apply_4x4s_to_voxs(self.camX0_T_camXs[:,1:], visXs[:,1:])
            visX0s = torch.cat([_visX00, _visX01], dim=1)
            
            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4s_to_voxs(self.camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            # if hyp.feat_cluster:
            #     featXs_unnorm = __u(featXs_unnorm_)
            #     _featX00_unnorm = featXs_unnorm[:,0:1]
            #     _featX01_unnorm = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], 
            #                                                    featXs_unnorm[:,1:])
            #     featX0s_unnorm = torch.cat([_featX00_unnorm,_featX01_unnorm],dim=1)
            #     del featXs_unnorm,featXs_unnorm_,_featX00_unnorm,_featX01_unnorm
            # else:
            #     del featXs_unnorm_

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) # context
            vis3D_e = torch.max(validX0s[:,1:], dim=1)[0]*torch.max(visX0s[:,1:], dim=1)[0]
            emb3D_g = featX0s[:,0] # obs
            vis3D_g = validX0s[:,0]*visX0s[:,0] # obs

            if hyp.do_eval_recall:
                results['emb3D_e'] = emb3D_e
                results['emb3D_g'] = emb3D_g
            if hyp.do_save_vis:
                # np.save('%s_rgb_%06d.npy' % (hyp.name, global_step), rgb_camRs[:,0].detach().cpu().numpy())
                imageio.imwrite('%s_rgb_%06d.png' % (hyp.name, global_step), np.transpose(utils_improc.back2color(self.rgb_camRs)[0,0].detach().cpu().numpy(), axes=[1, 2, 0]))
                np.save('%s_emb3D_g_%06d.npy' % (hyp.name, global_step), emb3D_e.detach().cpu().numpy())

            self.summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(validX0s, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/validX0s', torch.unbind(validX0s, dim=1), pca=False)
            self.summ_writer.summ_feat('3D_feats/vis3D_e', vis3D_e, pca=False)
            self.summ_writer.summ_feat('3D_feats/vis3D_g', vis3D_g, pca=False)
            
        if hyp.do_occ:
            occX0_sup, freeX0_sup, _, freeXs = utils_vox.prep_occs_supervision(
                self.camX0_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2, 
                agg=True)
        
            self.summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            self.summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            self.summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            self.summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(self.occXs_half, dim=1))
                
            occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                 occX0_sup,
                                                 freeX0_sup,
                                                 torch.max(validX0s[:,1:], dim=1)[0],
                                                 self.summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss

            if hyp.feat_cluster and feed['set_name'] == 'train': # and not self.cluster_vis_done:
                KNS = self.kmeans_obj_num_samples
                KNOV = hyp.feat_cluster_num_objs_views
                featX0s_unnorm = featX0s_unnorm.cpu().detach().numpy()
                occX0_sup = occX0_sup.cpu().detach().numpy()
                for b in range(hyp.B):
                    for s in range(hyp.S):
                        featX0s_unnorm_b_s = featX0s_unnorm[b,s] # [C,Z2,Y2,X2]
                        occX0_sup_b = occX0_sup[b,0]>0 # [Z2,Y2,X2]
                        feat_obj = featX0s_unnorm_b_s[:,occX0_sup_b] # [C,?]
                        feat_obj = np.transpose(feat_obj) # [?,C]
                        np.random.shuffle(feat_obj)
                        feat_obj = feat_obj[:KNS]
                        self.kmeans_objs.append(feat_obj)
                print(f'num kmeans objs: {len(self.kmeans_objs):5d} / {KNOV:5d}')
                if len(self.kmeans_objs) >= KNOV:
                    voxels = np.concatenate(self.kmeans_objs,axis=0) # [KNS*KNOV,C]
                    self.KMeans.fit(voxels)
                    np.save('vqvae/kmeans_cluster_centers.npy',
                            {'cluster_centers':self.KMeans.cluster_centers_,
                             'voxels':voxels,
                             'labels':self.KMeans.labels_})
                    summ_writer.summ_embeddings('kmeans_clusters',
                                                voxels,
                                                self.KMeans.labels_)
                    summ_writer.summ_histogram('kmeans_clusters',self.KMeans.labels_)
                    feed['writer'].close()
                    del voxels
                    import sys; sys.exit()

        if hyp.do_view:
            assert(hyp.do_feat)
            # we warped the features into the canonical view
            # now we resample to the target view and decode

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            assert(self.S==2) # else we should warp each feat in 1:
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                hyp.view_depth, PH, PW)
            
            rgb_X00 = utils_basic.downsample(self.rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(self.valid_camXs[:,0], 2)

            # decode the perspective volume into an image
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                self.summ_writer)
            
            total_loss += view_loss

        if hyp.do_render:
            assert(hyp.do_feat)
            # we warped the feaself.tures into the canonical view
            # now we resample to the target view anode

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            assert(S==2) # else we should warp each feat in 1:
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,1], self.featXs[:,1], # use feat1 to predict rgb0
                hyp.view_depth, PH, PW)
            # feat_projX00 is B x hyp.feat_dim x hyp.view_depth x PH x PW
            occ_pred_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,0], self.occX0s_pred[:,0]*torch.max(self.validX0s[:,1:], dim=1)[0], # note occX0s already comes from feat1
                hyp.view_depth, PH, PW)
            occ_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camX0_T_camXs[:,0], self.occX0s_half[:,0],
                hyp.view_depth, PH, PW)
            # occ_projX00 is B x 1 x hyp.view_depth x PH x PW
            
            rgb_X00 = utils_basic.downsample(self.rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(self.valid_camXs[:,0], 2)

            # decode the perspective volume into an image
            render_loss, rgb_e, emb2D_e = self.rendernet(
                feat_projX00,
                occ_pred_projX00,
                rgb_X00,
                valid_X00,
                self.summ_writer)
            total_loss += render_loss
            
        if hyp.do_emb2D:
            assert(hyp.do_view)
            # create an embedding image, representing the bottom-up 2D feature tensor
            emb_loss_2D, emb2D_g = self.embnet2D(
                self.rgb_camXs[:,0],
                emb2D_e,
                self.valid_camXs[:,0],
                self.summ_writer)
            total_loss += emb_loss_2D

        if hyp.do_emb3D:
            # compute 3D ML
            emb_loss_3D = self.embnet3D(
                emb3D_e,
                emb3D_g,
                vis3D_e,
                vis3D_g,
                self.summ_writer)
            total_loss += emb_loss_3D

        if hyp.do_eval_recall:
            results['emb2D_e'] = None
            results['emb2D_g'] = None
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # get the boxes
        boxlist_camRs = feed["boxlists"]
        tidlist_s = feed["tidlists"] # coordinate-less and plural
        scorelist_s = feed["scorelists"] # coordinate-less and plural
        
        lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))).reshape(self.B, self.S, self.N, 19)
        lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(lrtlist_camRs)))
        # these are is B x S x N x 19

        self.summ_writer.summ_lrtlist('obj/lrtlist_camX0', self.rgb_camXs[:,0], lrtlist_camXs[:,0],
                                      scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        self.summ_writer.summ_lrtlist('obj/lrtlist_camR0', self.rgb_camRs[:,0], lrtlist_camRs[:,0],
                                      scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        # mask_memX0 = utils_vox.assemble_padded_obj_masklist(
        #     lrtlist_camXs[:,0], scorelist_s[:,0], self.Z2, self.Y2, self.X2, coeff=1.0)
        # mask_memX0 = torch.sum(mask_memX0, dim=1).clamp(0, 1) 
        # self.summ_writer.summ_oned('obj/mask_memX0', mask_memX0, bev=True)

        mask_memXs = __u(utils_vox.assemble_padded_obj_masklist(
            __p(lrtlist_camXs), __p(scorelist_s), self.Z2, self.Y2, self.X2, coeff=1.0))
        mask_memXs = torch.sum(mask_memXs, dim=2).clamp(0, 1)
        self.summ_writer.summ_oneds('obj/mask_memXs', torch.unbind(mask_memXs, dim=1), bev=True)

        for b in list(range(self.B)):
            for s in list(range(self.S)):
                mask = mask_memXs[b,s]
                if torch.sum(mask) < 2.0:
                    # return early
                    return total_loss, None, True
                
        # next: i want to treat features differently if they are in obj masks vs not
        # in particular, i want a different kind of retrieval metric
        
        if hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            feat_memXs_input = torch.cat([self.occXs, self.occXs*self.unpXs], dim=2)
            feat_memXs_input_ = __p(feat_memXs_input)

            feat_memXs_, _, _ = self.featnet(
                feat_memXs_input_,
                self.summ_writer,
                comp_mask=None,
            )
            feat_memXs = __u(feat_memXs_)
                                    
            self.summ_writer.summ_feats('3D_feats/feat_memXs_input', torch.unbind(feat_memXs_input, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/feat_memXs_output', torch.unbind(feat_memXs, dim=1), pca=True)

            mv_precision = utils_eval.measure_semantic_retrieval_precision(feat_memXs[0], mask_memXs[0])
            self.summ_writer.summ_scalar('semantic_retrieval/multiview_precision', mv_precision)
            ms_precision = utils_eval.measure_semantic_retrieval_precision(feat_memXs[:,0], mask_memXs[:,0])
            self.summ_writer.summ_scalar('semantic_retrieval/multiscene_precision', ms_precision)
            
        return total_loss, None, False
            
    def forward(self, feed):
        self.prepare_common_tensors(feed)
        
        set_name = feed['set_name']
        if set_name=='train':
            return self.run_train(feed)
        elif set_name=='test':
            return self.run_test(feed)
        else:
            print('weird set_name:', set_name)
            assert(False)
