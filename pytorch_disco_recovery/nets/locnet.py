import torch
import torch.nn as nn
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler
import numpy as np

# import sys
# sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
import utils_basic
import utils_improc
import utils_misc
import utils_samp
import utils_vox
import utils_geom
import math

class LocNet(nn.Module):
    def __init__(self, sce_Z, sce_Y, sce_X, obj_Z, obj_Y, obj_X):
        super(LocNet, self).__init__()

        print('LocNet...')

        self.debug = False
        # # self.debug = True

        self.C = hyp.feat3D_dim
        self.sce_Z = sce_Z
        self.sce_Y = sce_Y
        self.sce_X = sce_X

        self.obj_Z = obj_Z
        self.obj_Y = obj_Y
        self.obj_X = obj_X
        
        
        # self.heatmap_size = hyp.loc_heatmap_size
        # # self.scales = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
        # # self.scales = [1.0]
        # # self.scales = [0.25, 0.5, 1.0]
        # # self.scales = [0.125, 0.25, 0.5, 0.75, 1.0]
        # self.scales = [0.25, 0.5, 0.75, 1.0]
        # self.num_scales = len(self.scales)

        # self.compress_dim = 16
        # self.compressor = nn.Sequential(
        #     nn.Conv3d(in_channels=hyp.feat_dim, out_channels=self.compress_dim, kernel_size=1, stride=1, padding=0),
        # )

        # self.correlation_sampler = SpatialCorrelationSampler(
        #     kernel_size=1,
        #     patch_size=self.heatmap_size,
        #     stride=1,
        #     padding=0,
        #     dilation_patch=1,
        # ).cuda()


        chans = 32
        
        self.compressor = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Conv3d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.LeakyReLU(negative_slope=0.1),
            # nn.Conv3d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        ).cuda()
        
        vec_len = (self.sce_Z-6)*(self.sce_Y-6)*(self.sce_X-6)*chans
        # print('vec_len = %d' % vec_len)
        
        self.locator = nn.Sequential(
            nn.Linear(vec_len, 9, bias=True),
        ).cuda()
        
        print(self.compressor)
        print(self.locator)

        self.mse = torch.nn.MSELoss(reduction='mean')
        
    def compute_corr_softmax(self, obj_feat, sce_feat):
        # obj_feat is B x C x Z x Y x X
        # scene_feat is B x C x Z2 x Y2 x X2

        B, C, _, _, _ = list(obj_feat.shape)

        obj_feat = obj_feat.reshape(B, C, -1)
        sce_feat = sce_feat.reshape(B, C, -1)
        sce_feat = sce_feat.permute(0, 2, 1) # B x -1 x C

        corr = torch.matmul(sce_feat, obj_feat)
        # this is B x sce_Z*sce_Y*sce_X x obj_Z*obj_Y*obj_X
        corr = F.softmax(corr, dim=1)

        corr = corr.reshape(B, self.sce_Z, self.sce_Y, self.sce_X,
                            self.obj_Z*self.obj_Y*self.obj_X)
        corr = corr.permute(0, 4, 1, 2, 3)
        return corr

    def compute_samp_loss(self, obj_lrt_e, obj_lrt_g, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        coords_e = utils_vox.convert_lrt_to_sampling_coords(
            obj_lrt_e,
            self.sce_Z, self.sce_Y, self.sce_X,
            self.obj_Z, self.obj_Y, self.obj_X,
            additive_pad=0.0)
        coords_g = utils_vox.convert_lrt_to_sampling_coords(
            obj_lrt_g,
            self.sce_Z, self.sce_Y, self.sce_X,
            self.obj_Z, self.obj_Y, self.obj_X,
            additive_pad=0.0)

        # normalize these by resolution
        coords_e = coords_e/float(self.sce_Z)
        coords_g = coords_g/float(self.sce_Z)

        samp_loss = self.mse(coords_e, coords_g)

        total_loss = utils_misc.add_loss('loc/samp_loss', total_loss, samp_loss, hyp.loc_samp_coeff, summ_writer)

        if summ_writer is not None:
            summ_writer.summ_histogram('coords_e', coords_e)
            summ_writer.summ_histogram('coords_g', coords_g)
        
        return total_loss

    def convert_params_to_lrt(self,
                              obj_len,
                              obj_xyz_sce,
                              obj_rot,
                              cam_T_sce):
        # this borrows from utils_geom.convert_box_to_ref_T_obj
        B = list(obj_xyz_sce.shape)[0]

        obj_xyz_cam = utils_geom.apply_4x4(cam_T_sce, obj_xyz_sce.unsqueeze(1)).squeeze(1)
        # # compose with lengths
        # lrt = utils_geom.merge_lrt(obj_len, cam_T_obj)
        # return lrt

        rot0 = utils_geom.eye_3x3(B)
        # tra = torch.stack([x, y, z], axis=1)
        center_T_ref = utils_geom.merge_rt(rot0, -obj_xyz_cam)
        # center_T_ref is B x 4 x 4

        t0 = torch.zeros([B, 3])
        obj_T_center = utils_geom.merge_rt(obj_rot, t0)
        # this is B x 4 x 4

        # we want obj_T_ref
        # first we to translate to center,
        # and then rotate around the origin
        obj_T_ref = utils_basic.matmul2(obj_T_center, center_T_ref)

        # return the inverse of this, so that we can transform obj corners into cam coords
        ref_T_obj = utils_geom.safe_inverse(obj_T_ref)
        # return ref_T_obj

        # compose with lengths
        lrt = utils_geom.merge_lrt(obj_len, ref_T_obj)
        return lrt
        
        # cam_T_obj = utils_basic.matmul2(cam_T_sce, sce_T_obj)
        # # obj_rot and obj_xyz specify the found object's pose in sce coords

        # # obj_xyz specifies the object's centroid wrt the scene
        # rot0 = utils_geom.eye_3x3(B)
        # centroid_T_sce = utils_geom.merge_rt(rot0, -obj_xyz)
        # # this is B x 4 x 4

        # # obj_rot specifies the object's rotation wrt the object's centroid (in sce coords)
        # t0 = torch.zeros([B, 3])
        # obj_rot = torch.transpose(obj_rot, 1, 2) # other dir
        # obj_T_centroid = utils_geom.merge_rt(obj_rot, t0)
        # # this is B x 4 x 4

        # # we want obj_T_sce
        # # first we to translate to center,
        # # and then rotate around the origin
        # obj_T_sce = utils_basic.matmul2(obj_T_centroid, centroid_T_sce)

        # # get the inverse of this
        # sce_T_obj = utils_geom.safe_inverse(obj_T_sce)

        # # get to cam coords
        # cam_T_obj = utils_basic.matmul2(cam_T_sce, sce_T_obj)

        # # compose with lengths
        # lrt = utils_geom.merge_lrt(obj_len, cam_T_obj)
        # return lrt
    
    def compute_feat_loss(self,
                          obj_feat,
                          sce_feat,
                          obj_lrt_e,
                          summ_writer=None,
                          suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        obj_crop = utils_vox.crop_zoom_from_mem(
            sce_feat, obj_lrt_e,
            self.obj_Z, self.obj_Y, self.obj_X,
            additive_pad=0.0)
        
        # whatever we match to, we want the feat distance to be small
        # we'll use the negative frobenius inner prod, normalized for resolution
        feat_loss = -torch.mean(obj_feat*obj_crop)
        # summ_writer.summ_feat('loc/obj_feat%s' % suffix, obj_feat, pca=True)
        # summ_writer.summ_feat('loc/obj_crop%s' % suffix, obj_crop, pca=True)
        total_loss = utils_misc.add_loss('loc/feat_loss%s' % suffix, total_loss, feat_loss, hyp.loc_feat_coeff, summ_writer)
        return total_loss
    
    def forward(self, obj_feat, sce_feat, obj_len, cam_T_sce,
                big_Z, big_Y, big_X,
                summ_writer=None, suffix=''):
        # in the forward we'll just locate the object
        # we'll compute loss later

        # B, C, Z, Y, X = list(obj_feat.shape)
        
        B, C, Z, Y, X = list(sce_feat.shape)
        assert(Z==self.sce_Z)
        assert(Y==self.sce_Y)
        assert(X==self.sce_X)
        _, _, Z2, Y2, X2 = list(obj_feat.shape)
        assert(Z2==self.obj_Z)
        assert(Y2==self.obj_Y)
        assert(X2==self.obj_X)

        corr = self.compute_corr_softmax(obj_feat, sce_feat)
        # print('corr', corr.shape)

        comp = self.compressor(corr)
        # print('comp', comp.shape)

        if summ_writer is not None:
            summ_writer.summ_feat('loc/corr%s' % suffix, corr, pca=True)
            summ_writer.summ_feat('loc/comp%s' % suffix, comp, pca=True)

        comp = comp.reshape(B, -1)
        loc_e = self.locator(comp)
        normalized_xyz_e = loc_e[:,:3]
        sin_e = loc_e[:,3:6]
        cos_e = 1.0+loc_e[:,6:9]
        sin_e, cos_e = utils_geom.sincos_norm(sin_e, cos_e)
        # let's say the sines and cosines are in xyz order
        rot_e = utils_geom.sincos2rotm(
            sin_e[:,2], # z
            sin_e[:,1], # y
            sin_e[:,0], # x
            cos_e[:,2], # z
            cos_e[:,1], # y
            cos_e[:,0]) # x
        rot_e = utils_geom.eye_3x3(B)

        print('normalized_xyz_e', normalized_xyz_e.detach().cpu().numpy())

        # def normalize_obj_xyz(xyz, Z, Y, X):
        #     xyz = utils_vox.Ref2Mem(xyz.unsqueeze(1), Z, Y, X).squeeze(1)
        #     normalizer = torch.from_numpy(np.reshape(
        #         np.array([1./X, 1./Y, 1./Z], np.float32), [1, 3])).cuda()
        #     xyz = xyz*normalizer - 0.5 # now it's in [-0.5, 0.5]
        #     return xyz
        
        # def unnormalize_obj_xyz(xyz, Z, Y, X):
        #     normalizer = torch.from_numpy(np.reshape(
        #         np.array([1./X, 1./Y, 1./Z], np.float32), [1, 3])).cuda()
        #     xyz = (xyz + 0.5) / normalizer
        #     xyz = utils_vox.Mem2Ref(xyz.unsqueeze(1), Z, Y, X).squeeze(1)
        #     return xyz

        # def unnormalize_obj_xyz(xyz, Z, Y, X):
        #     normalizer = torch.from_numpy(np.reshape(
        #         np.array([1./X, 1./Y, 1./Z], np.float32), [1, 3])).cuda()
        #     xyz = (xyz + 0.5) / normalizer
        #     return xyz

        def unnormalize_obj_xyz(xyz, Z, Y, X):
            scale = torch.from_numpy(np.reshape(
                np.array([float(X), float(Y), float(Z)], np.float32), [1, 3])).cuda()
            bias = torch.from_numpy(np.reshape(
                np.array([X/2., Y//2., Z/2.], np.float32), [1, 3])).cuda()
            xyz = xyz*scale + bias
            return xyz
        
        obj_xyz_e = unnormalize_obj_xyz(normalized_xyz_e, big_Z, big_Y, big_X)
        print('obj_xyz_e', obj_xyz_e.detach().cpu().numpy())
        
        obj_lrt_e = self.convert_params_to_lrt(obj_len,
                                               obj_xyz_e,
                                               rot_e,
                                               cam_T_sce)
        # obj_lrt_e = utils_geom.merge_lrt(obj_len, utils_geom.merge_rt(rot_e, obj_xyz_e))

        return obj_lrt_e
        
