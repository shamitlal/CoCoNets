import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("..")

import hyperparams as hyp
import utils_basic
import utils_geom
import utils_vox
import utils_misc
import utils_track
import utils_py

EPS = 1e-4
class RigidNet(nn.Module):
    def __init__(self):
        super(RigidNet, self).__init__()

        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='mean')
        
        # self.med = int(hyp.ZZ/2*hyp.ZY/2*hyp.ZX/2)
        if hyp.rigid_use_cubes:
            # # rule19
            # self.hidden_dim = 1024
            # self.predictor1 = nn.Sequential(
            #     nn.Conv3d(in_channels=8, out_channels=int(self.hidden_dim/16), kernel_size=4, stride=4, padding=0),
            #     nn.LeakyReLU(negative_slope=0.1),
            #     # nn.Conv3d(in_channels=int(self.hidden_dim/16), out_channels=int(self.hidden_dim/8), kernel_size=4, stride=2, padding=0),
            #     nn.Conv3d(in_channels=int(self.hidden_dim/16), out_channels=self.hidden_dim, kernel_size=4, stride=2, padding=0),
            #     nn.LeakyReLU(negative_slope=0.1),
            # ).cuda()
            # self.predictor2 = nn.Sequential(
            #     # nn.Linear(3456, self.hidden_dim),
            #     nn.Linear(self.hidden_dim, self.hidden_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(self.hidden_dim, self.hidden_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(self.hidden_dim, 9),
            # ).cuda()

            # rule17
            self.hidden_dim = 1024
            self.predictor1 = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=int(self.hidden_dim/16), kernel_size=4, stride=4, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv3d(in_channels=int(self.hidden_dim/16), out_channels=int(self.hidden_dim/8), kernel_size=4, stride=2, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
            ).cuda()
            self.predictor2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.Linear(3456, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 9),
            ).cuda()

        else:
            self.med = 1728
            self.hidden_dim = 1024
            self.predictor1 = nn.Sequential(
                nn.Conv3d(in_channels=self.med, out_channels=self.hidden_dim, kernel_size=4, stride=4, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv3d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=4, stride=4, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
            ).cuda()
            self.predictor2 = nn.Sequential(
                nn.Linear(1024, self.hidden_dim*2),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim*2, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 9),
            ).cuda()

        
    def forward(self, template, template_mask, search_region, xyz, r_delta, sampled_corners, sampled_centers, vox_util=None, summ_writer=None):
        # template is the thing we are searching for; it is B x C x ZZ x ZY x ZX
        # template_mask marks the voxels of the object we care about, within the template; it is B x C x ZZ x ZY x ZX
        # search_region is the featuremap where we are searching; it is B x C x Z x Y x X
        # xyz is the location of the answer in the search region; it is B x 3
        total_loss = torch.tensor(0.0).cuda()
        
        B, C, ZZ, ZY, ZX = list(template.shape)
        _, _, Z, Y, X = list(search_region.shape)
        _, D = list(xyz.shape)
        assert(D==3)

        if hyp.rigid_use_cubes:

            R = hyp.rigid_repeats
            template = template.repeat(R, 1, 1, 1, 1)
            template_mask = template_mask.repeat(R, 1, 1, 1, 1)
            search_region = search_region.repeat(R, 1, 1, 1, 1)

            if summ_writer is not None:
                box_vox = vox_util.voxelize_xyz(torch.cat([sampled_corners, sampled_centers], dim=1), ZZ, ZY, ZX, already_mem=True)
                # summ_writer.summ_scalar('rigid/num_tries', num_tries)
                summ_writer.summ_occ('rigid/box', box_vox, reduce_axes=[2,3,4])
                summ_writer.summ_oned('rigid/mask', template_mask, bev=True, norm=False)


            ## NOTE: YOU GOTTA HANDLE THE CENTROID OFFSET, FOR THE TRANSLATION TASK TO BE WELL FORMED
            # > to simplify this problem at the start, let's use a fixed centroid
            # >> ok done

            # now i need to sample features from these locations
            # i think this is fast enough:

            template_vec = torch.zeros([B*R, C, 8]).float().cuda()
            for b in list(range(B*R)):
                corners_b = sampled_corners[b].long()
                for ci, corner in enumerate(corners_b):
                    template_vec[b,:,ci] = template[b,:,corner[2],corner[1],corner[0]]

            search_vec = search_region.view(B*R, C, -1)
            # this is B x C x huge
            search_vec = search_vec.permute(0, 2, 1)
            # this is B x huge x C

            corr_vec = torch.matmul(search_vec, template_vec)
            # this is B x huge x med

            # print('corr_vec', corr_vec.shape)

            corr = corr_vec.reshape(B*R, Z, Y, X, 8)
            corr = corr.permute(0, 4, 1, 2, 3)
            # corr is B x 8 x Z x Y x X

            # print('corr', corr.shape)
            # next step is:
            # a network should do quick work of this and turn it into an output

            # corr = corr.reshape(B, -1)
            # rigid = self.predictor(corr)
            # rigid = self.predictor2(feat)

            feat = self.predictor1(corr)
            # # rule19:
            # # print('feat', feat.shape)
            # feat = torch.mean(feat, dim=[2,3,4])

            # rule17:
            feat = feat.reshape(B*R, -1)
            # print('feat', feat.shape)

            rigid = self.predictor2(feat)

            # rigid is B*R x 9

            rigid = rigid.reshape(B, R, 9)

            normal_center = np.reshape(np.array([ZX/2, ZY/2, ZZ/2]), [1, 1, 3])
            normal_centers = torch.from_numpy(normal_center).float().cuda().repeat(B*R, 1, 1)
            rigid[:,:,:3] = rigid[:,:,:3] - (sampled_centers.reshape(B, R, 3) - normal_centers.reshape(B, R, 3))
            # rigid[:,:,:3] = rigid[:,:,:3] + (normal_centers.reshape(B, R, 3) - sampled_centers.reshape(B, R, 3))
            # rigid[:,:,:3] = rigid[:,:,:3] + (sampled_centers.reshape(B, R, 3) - normal_centers.reshape(B, R, 3))

            rigid = torch.mean(rigid, dim=1)
            
            # 
            # xyz_e is the location of the object in the search region, assuming we used normal center
            # but we didn't
            # xyz_e = rigid[:,:3]

        else:
            # ok, i want to corr each voxel of the template with the search region.
            # this is a giant matmul

            search_vec = search_region.view(B, C, -1)
            # this is B x C x huge
            search_vec = search_vec.permute(0, 2, 1)
            # this is B x huge x C

            template_vec = template.view(B, C, -1)
            # this is B x C x med

            corr_vec = torch.matmul(search_vec, template_vec)
            # this is B x huge x med

            # print('corr_vec', corr_vec.shape)

            corr = corr_vec.reshape(B, Z, Y, X, ZZ*ZY*ZX)
            corr = corr.permute(0, 4, 1, 2, 3)
            # corr is B x med x Z x Y x X

            # next step is:
            # a network should do quick work of this and turn it into an output

            # rigid = self.predictor(corr)
            # # this is B x 3 x 1 x 1 x 1
            # rigid = rigid.view(B, 3)
            # # now, i basically want this to be the answer

            feat = self.predictor1(corr)
            # print('feat', feat.shape)
            feat = feat.reshape(B, -1)
            # print('feat', feat.shape)
            rigid = self.predictor2(feat)
            
        xyz_e = rigid[:,:3]
        # center = np.reshape(np.array([ZX/2, ZY/2, ZZ/2]), [1, 3])
        # xyz_e = xyz_e -
        # center = np.reshape(np.array([ZX/2, ZY/2, ZZ/2]), [1, 3])

        sin_e = rigid[:,3:6]
        cos_e = 1.0+rigid[:,6:9]
        sin_e, cos_e = utils_geom.sincos_norm(sin_e, cos_e)
        # let's say the sines and cosines are in xyz order
        rot_e = utils_geom.sincos2rotm(
            sin_e[:,2], # z
            sin_e[:,1], # y
            sin_e[:,0], # x
            cos_e[:,2], # z
            cos_e[:,1], # y
            cos_e[:,0]) # x
        rx_e, ry_e, rz_e = utils_geom.rotm2eul(rot_e)

        rad_e = torch.stack([rx_e, ry_e, rz_e], dim=1)
        deg_e = utils_geom.rad2deg(rad_e)
        deg_g = utils_geom.rad2deg(r_delta)
        # rad_g = torch.stack([rx_e, ry_e, rz_e], dim=1)
        
        if summ_writer is not None:
            rx_e, ry_e, rz_e = torch.unbind(deg_e, dim=1)
            summ_writer.summ_histogram('rx_e', rx_e)
            summ_writer.summ_histogram('ry_e', ry_e)
            summ_writer.summ_histogram('rz_e', rz_e)
            rx_g, ry_g, rz_g = torch.unbind(deg_e, dim=1)
            summ_writer.summ_histogram('rx_g', rx_g)
            summ_writer.summ_histogram('ry_g', ry_g)
            summ_writer.summ_histogram('rz_g', rz_g)

        
        # # let's be in degrees, for the loss
        # rx_e = utils_geom.rad2deg(rx_e)
        # ry_e = utils_geom.rad2deg(ry_e)
        # rz_e = utils_geom.rad2deg(rz_e)
        # rx_g = utils_geom.rad2deg(rx_g)
        # ry_g = utils_geom.rad2deg(ry_g)
        # rz_g = utils_geom.rad2deg(rz_g)

        # r_loss = torch.mean(torch.norm(deg_e - deg_g, dim=1))
        # t_loss = torch.mean(torch.norm(xyz_e - xyz, dim=1))
        r_loss = self.smoothl1(deg_e, deg_g)
        t_loss = self.smoothl1(xyz_e, xyz)
        
        # now, i basically want this to be the answer
        
        # rigid_loss = torch.mean(torch.norm(rigid - xyz, dim=1))
        total_loss = utils_misc.add_loss('rigid/r_loss', total_loss, r_loss, hyp.rigid_r_coeff, summ_writer)
        total_loss = utils_misc.add_loss('rigid/t_loss', total_loss, t_loss, hyp.rigid_t_coeff, summ_writer)
        # print('r_loss', r_loss.detach().cpu().numpy())
        # print('t_loss', t_loss.detach().cpu().numpy())
        
        if summ_writer is not None:
            # inputs
            summ_writer.summ_feat('rigid/input_template', template, pca=False)
            summ_writer.summ_feat('rigid/input_search_region', search_region, pca=False)

        return xyz_e, rad_e, total_loss

