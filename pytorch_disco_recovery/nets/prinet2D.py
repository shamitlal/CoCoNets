import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import archs.sparse_invar_encoder2D
import hyperparams as hyp
# from utils_basic import *
import utils_improc
import utils_misc
import utils_vox
import utils_geom
import utils_basic
import utils_samp

class PriNet2D(nn.Module):
    def __init__(self):
        super(PriNet2D, self).__init__()

        print('PriNet2D...')

        # self.net = archs.encoder2D.Net2D(in_chans=16, mid_chans=32, out_chans=1).cuda()
        # self.net = archs.sparse_invar_encoder2D.CustomCNN(16, 1, 3).cuda()
        self.net = archs.sparse_invar_encoder2D.CustomCNN(18, 1, 3).cuda()
        # self.net = self.net.to('cuda:0')
        # self.conv3d = nn.Conv3d(in_channels=(hyp.feat_dim*2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        # print('conv3D, [in_channels={}, out_channels={}, ksize={}]'.format(hyp.feat_dim, 1, 1))

    def forward(self, feat_mem, clist_cam, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat_mem.shape)
        B2, S, D = list(clist_cam.shape)
        assert(B==B2)
        assert(D==3)

        clist_mem = utils_vox.Ref2Mem(clist_cam, Z, Y, X)
        # this is (still) B x S x 3

        feat_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, C*Y, Z, X)
        mask_ = 1.0 - (feat_==0).all(dim=1, keepdim=True).float().cuda()
        grid_ = utils_basic.meshgrid2D(B, Z, X, stack=True, norm=True).permute(0, 3, 1, 2)
        halfgrid_ = utils_basic.meshgrid2D(B, int(Z/2), int(X/2), stack=True, norm=True).permute(0, 3, 1, 2)
        feat_ = torch.cat([feat_, grid_], dim=1)
        energy_map, mask = self.net(feat_, mask_, halfgrid_)
        # energy_map = self.net(feat_)
        # energy_map is B x 1 x Z x X
        # don't do this: # energy_map = energy_map + (1.0-mask) * (torch.min(torch.min(energy_map, dim=2)[0], dim=2)[0]).reshape(B, 1, 1, 1)
        summ_writer.summ_feat('pri/energy_input', feat_)
        summ_writer.summ_oned('pri/energy_map', energy_map)
        summ_writer.summ_oned('pri/mask', mask, norm=False)
        summ_writer.summ_histogram('pri/energy_map_hist', energy_map)

        loglike_per_traj = utils_misc.get_traj_loglike(clist_mem*0.5, energy_map) # 0.5 since it's half res
        # loglike_per_traj = self.get_traj_loglike(clist_mem*0.25, energy_map) # 0.25 since it's quarter res
        # this is B x K
        ce_loss = -1.0*torch.mean(loglike_per_traj)
        # this is []
        
        total_loss = utils_misc.add_loss('pri/ce_loss', total_loss, ce_loss, hyp.pri2D_ce_coeff, summ_writer)
        
        reg_loss = torch.sum(torch.abs(energy_map))
        total_loss = utils_misc.add_loss('pri/reg_loss', total_loss, reg_loss, hyp.pri2D_reg_coeff, summ_writer)

        # smooth loss
        dz, dx = utils_basic.gradient2D(energy_map, absolute=True)
        smooth_vox = torch.mean(dz+dx, dim=1, keepdims=True)
        summ_writer.summ_oned('pri/smooth_loss', smooth_vox)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('pri/smooth_loss', total_loss, smooth_loss, hyp.pri2D_smooth_coeff, summ_writer)

        return total_loss, energy_map

