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

EPS = 1e-4
class RobustNet(nn.Module):
    def __init__(self):
        super(RobustNet, self).__init__()

        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='mean')

        # num points to sample for svd
        # self.num_pts = 512
        # self.num_pts = 256
        # self.num_pts = 256
        self.num_pts = 128
        # self.num_pts = 105
        
    def forward(self,
                template_feat, search_feat,
                template_mask, 
                template_lrt, search_lrt,
                vox_util,
                lrt_cam0s,
                summ_writer=None):
        # template_feat is the thing we are searching for; it is B x C x ZZ x ZY x ZX
        # search_feat is the featuremap where we are searching; it is B x C x Z x Y x X
        total_loss = torch.tensor(0.0).cuda()
        
        B, C, ZZ, ZY, ZX = list(template_feat.shape)
        _, _, Z, Y, X = list(search_feat.shape)

        xyz0_template = utils_basic.gridcloud3D(B, ZZ, ZY, ZX)
        # this is B x med x 3
        xyz0_cam = vox_util.Zoom2Ref(
            xyz0_template, template_lrt,
            ZZ, ZY, ZX)
        # ok, next, after i relocate the object in search coords,
        # i need to transform those coords into cam, and then do svd on that


        # print('template_feat', template_feat.shape)
        # print('search_feat', search_feat.shape)
        
        search_feat = search_feat.view(B, C, -1)
        # this is B x C x huge
        template_feat = template_feat.view(B, C, -1)
        # this is B x C x med
        template_mask = template_mask.view(B, -1)
        # this is B x med

        # next i need to sample
        # i would like to take N random samples within the mask
        
        cam1_T_cam0_e = utils_geom.eye_4x4(B)

        # to simplify the impl, we will iterate over the batch dim
        for b in list(range(B)):
            template_feat_b = template_feat[b]
            template_mask_b = template_mask[b]
            search_feat_b = search_feat[b]
            xyz0_cam_b = xyz0_cam[b]


            # print('xyz0_cam_b', xyz0_cam_b.shape)
            # print('template_mask_b', template_mask_b.shape)
            # print('template_mask_b sum', torch.sum(template_mask_b).cpu().numpy())
            
            # take any points within the mask
            inds = torch.where(template_mask_b > 0)

            # gather up
            template_feat_b = template_feat_b.permute(1, 0)
            # this is C x med
            template_feat_b = template_feat_b[inds]
            xyz0_cam_b = xyz0_cam_b[inds]
            # these are self.num_pts x C

            # print('inds', inds)
            # not sure why this is a tuple
            # inds = inds[0] 
            
            # trim down to self.num_pts
            # inds = inds.squeeze()
            assert(len(xyz0_cam_b) > 8) # otw we should have returned early

            # i want to have self.num_pts pts every time
            if len(xyz0_cam_b) < self.num_pts:
                reps = int(self.num_pts/len(xyz0_cam_b)) + 1
                print('only have %d pts; repeating %d times...' % (len(xyz0_cam_b), reps))
                xyz0_cam_b = xyz0_cam_b.repeat(reps, 1)
                template_feat_b = template_feat_b.repeat(reps, 1)
            assert(len(xyz0_cam_b) >= self.num_pts)
            # now trim down
            perm = np.random.permutation(len(xyz0_cam_b))
            # print('perm', perm[:10])
            xyz0_cam_b = xyz0_cam_b[perm[:self.num_pts]]
            template_feat_b = template_feat_b[perm[:self.num_pts]]

            heat_b = torch.matmul(template_feat_b, search_feat_b)
            # this is self.num_pts x huge
            # it represents each point's heatmap in the search region

            # make the min zero
            heat_b = heat_b - (torch.min(heat_b, dim=1).values).unsqueeze(1)
            # scale up, for numerical stability
            heat_b = heat_b * float(len(heat_b[0].reshape(-1)))
            
            heat_b = heat_b.reshape(self.num_pts, 1, Z, Y, X)
            xyz1_search_b = utils_basic.argmax3D(heat_b, hard=False, stack=True)
            # this is self.num_pts x 3

            # i need to get to cam coords
            xyz1_cam_b = vox_util.Zoom2Ref(
                xyz1_search_b.unsqueeze(0), search_lrt[b:b+1],
                Z, Y, X).squeeze(0)

            # print('xyz0, xyz1', xyz0_cam_b.shape, xyz1_cam_b.shape)
            # cam1_T_cam0_e[b] = utils_track.rigid_transform_3D(xyz0_cam_b, xyz1_cam_b)

            # cam1_T_cam0_e[b] = utils_track.differentiable_rigid_transform_3D(xyz0_cam_b, xyz1_cam_b)
            cam1_T_cam0_e[b] = utils_track.rigid_transform_3D(xyz0_cam_b, xyz1_cam_b)

        _, rt_cam0_g = utils_geom.split_lrt(lrt_cam0s[:,0])
        _, rt_cam1_g = utils_geom.split_lrt(lrt_cam0s[:,1])
        # these represent ref_T_obj
        cam1_T_cam0_g = torch.matmul(rt_cam1_g, rt_cam0_g.inverse())
        
        # cam1_T_cam0_e = cam1_T_cam0_g
        lrt_cam1_e = utils_geom.apply_4x4_to_lrtlist(cam1_T_cam0_e, lrt_cam0s[:,0:1]).squeeze(1)
        # lrt_cam1_g = lrt_cam0s[:,1]

        # _, rt_cam1_e = utils_geom.split_lrt(lrt_cam1_e)
        # _, rt_cam1_g = utils_geom.split_lrt(lrt_cam1_g)
        
        # let's try the cube loss
        lx, ly, lz = 1.0, 1.0, 1.0
        x = np.array([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.])
        y = np.array([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.])
        z = np.array([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.])
        xyz = np.stack([x,y,z], axis=1)
        # this is 8 x 3
        xyz = torch.from_numpy(xyz).float().cuda()
        xyz = xyz.reshape(1, 8, 3)
        # this is B x 8 x 3

        # xyz_e = utils_geom.apply_4x4(rt_cam1_e, xyz)
        # xyz_g = utils_geom.apply_4x4(rt_cam1_g, xyz)
        xyz_e = utils_geom.apply_4x4(cam1_T_cam0_e, xyz)
        xyz_g = utils_geom.apply_4x4(cam1_T_cam0_g, xyz)

        # print('xyz_e', xyz_e.detach().cpu().numpy())
        # print('xyz_g', xyz_g.detach().cpu().numpy())

        corner_loss = self.smoothl1(xyz_e, xyz_g)
        total_loss = utils_misc.add_loss('robust/corner_loss', total_loss, corner_loss, hyp.robust_corner_coeff, summ_writer)

        # rot_e, t_e = utils_geom.split_rt(rt_cam1_e)
        # rot_g, t_g = utils_geom.split_rt(rt_cam1_g)
        rot_e, t_e = utils_geom.split_rt(cam1_T_cam0_e)
        rot_g, t_g = utils_geom.split_rt(cam1_T_cam0_g)
        rx_e, ry_e, rz_e = utils_geom.rotm2eul(rot_e)
        rx_g, ry_g, rz_g = utils_geom.rotm2eul(rot_g)

        rad_e = torch.stack([rx_e, ry_e, rz_e], dim=1)
        rad_g = torch.stack([rx_g, ry_g, rz_g], dim=1)
        deg_e = utils_geom.rad2deg(rad_e)
        deg_g = utils_geom.rad2deg(rad_g)
        
        r_loss = self.smoothl1(deg_e, deg_g)
        t_loss = self.smoothl1(t_e, t_g)
        
        total_loss = utils_misc.add_loss('robust/r_loss', total_loss, r_loss, hyp.robust_r_coeff, summ_writer)
        total_loss = utils_misc.add_loss('robust/t_loss', total_loss, t_loss, hyp.robust_t_coeff, summ_writer)
        # print('r_loss', r_loss.detach().cpu().numpy())
        # print('t_loss', t_loss.detach().cpu().numpy())
        
        
        
        
        return lrt_cam1_e, total_loss

