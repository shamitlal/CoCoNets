import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_vox
import utils_geom
import utils_basic
import utils_samp

class PriNet3D(nn.Module):
    def __init__(self):
        super(PriNet3D, self).__init__()

        print('PriNet3D...')

        self.conv3d = nn.Conv3d(in_channels=(hyp.feat_dim*2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        print('conv3D, [in_channels={}, out_channels={}, ksize={}]'.format(hyp.feat_dim, 1, 1))

    def forward(self, feat, obj_lrtlist_cams, obj_scorelist_s, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)
        K, B2, S, D = list(obj_lrtlist_cams.shape)
        assert(B==B2)

        # obj_scorelist_s is K x B x S

        # __p = lambda x: utils_basic.pack_seqdim(x, B)
        # __u = lambda x: utils_basic.unpack_seqdim(x, B)
        
        # obj_lrtlist_cams is K x B x S x 19
        obj_lrtlist_cams_ = obj_lrtlist_cams.reshape(K*B, S, 19)
        obj_clist_cam_ = utils_geom.get_clist_from_lrtlist(obj_lrtlist_cams_)
        obj_clist_cam = obj_clist_cam_.reshape(K, B, S, 1, 3)
        # obj_clist_cam is K x B x S x 1 x 3
        obj_clist_cam = obj_clist_cam.squeeze(3)
        # obj_clist_cam is K x B x S x 3
        clist_cam = obj_clist_cam.reshape(K*B, S, 3)
        clist_mem = utils_vox.Ref2Mem(clist_cam, Z, Y, X)
        # this is K*B x S x 3
        clist_mem = clist_mem.reshape(K, B, S, 3)
        
        energy_vol = self.conv3d(feat)
        # energy_vol is B x 1 x Z x Y x X
        summ_writer.summ_oned('pri/energy_vol', torch.mean(energy_vol, dim=3))
        summ_writer.summ_histogram('pri/energy_vol_hist', energy_vol)

        # for k in range(K):
        # let's start with the first object
        # loglike_per_traj = self.get_traj_loglike(clist_mem[0], energy_vol)
        # # this is B
        # ce_loss = -1.0*torch.mean(loglike_per_traj)
        # # this is []
        
        loglike_per_traj = self.get_trajs_loglike(clist_mem, obj_scorelist_s, energy_vol)
        # this is B x K
        valid = torch.max(obj_scorelist_s.permute(1,0,2), dim=2)[0]
        ce_loss = -1.0*utils_basic.reduce_masked_mean(loglike_per_traj, valid)
        # this is []
        
        total_loss = utils_misc.add_loss('pri/ce_loss', total_loss, ce_loss, hyp.pri_ce_coeff, summ_writer)
        
        reg_loss = torch.sum(torch.abs(energy_vol))
        total_loss = utils_misc.add_loss('pri/reg_loss', total_loss, reg_loss, hyp.pri_reg_coeff, summ_writer)

        # smooth loss
        dz, dy, dx = gradient3D(energy_vol, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        summ_writer.summ_oned('pri/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('pri/smooth_loss', total_loss, smooth_loss, hyp.pri_smooth_coeff, summ_writer)
    
        # pri_e = F.sigmoid(energy_vol)
        # energy_volbinary = torch.round(pri_e)

        # # collect some accuracy stats 
        # pri_match = pri_g*torch.eq(energy_volbinary, pri_g).float()
        # free_match = free_g*torch.eq(1.0-energy_volbinary, free_g).float()
        # either_match = torch.clamp(pri_match+free_match, 0.0, 1.0)
        # either_have = torch.clamp(pri_g+free_g, 0.0, 1.0)
        # acc_pri = reduce_masked_mean(pri_match, pri_g*valid)
        # acc_free = reduce_masked_mean(free_match, free_g*valid)
        # acc_total = reduce_masked_mean(either_match, either_have*valid)

        # summ_writer.summ_scalar('pri/acc_pri%s' % suffix, acc_pri.cpu().item())
        # summ_writer.summ_scalar('pri/acc_free%s' % suffix, acc_free.cpu().item())
        # summ_writer.summ_scalar('pri/acc_total%s' % suffix, acc_total.cpu().item())

        # # vis
        # summ_writer.summ_pri('pri/pri_g%s' % suffix, pri_g, reduce_axes=[2,3])
        # summ_writer.summ_pri('pri/free_g%s' % suffix, free_g, reduce_axes=[2,3]) 
        # summ_writer.summ_pri('pri/pri_e%s' % suffix, pri_e, reduce_axes=[2,3])
        # summ_writer.summ_pri('pri/valid%s' % suffix, valid, reduce_axes=[2,3])
        
        # prob_loss = self.compute_loss(energy_vol, pri_g, free_g, valid, summ_writer)
        # total_loss = utils_misc.add_loss('pri/prob_loss%s' % suffix, total_loss, prob_loss, hyp.pri_coeff, summ_writer)

        return total_loss#, pri_e

    def compute_loss(self, pred, pri, free, valid, summ_writer):
        pos = pri.clone()
        neg = free.clone()

        # pri is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('pri/prob_loss', loss_vis, summ_writer)

        pos_loss = reduce_masked_mean(loss, pos*valid)
        neg_loss = reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    def get_traj_loglike(self, xyz, energy_vol):
        # energy_vol is B x 1 x Z x Y x X; it is not normalized in any way
        # xyz is B x T x 3; it specifies vox coordinates of the traj 

        B, T, D = list(xyz.shape)
        _, _, Z, Y, X = list(energy_vol.shape)
        assert(D==3)

        x, y, z = torch.unbind(xyz, dim=2)
        energy_per_timestep = utils_samp.bilinear_sample3D(energy_vol, x, y, z)
        energy_per_timestep = energy_per_timestep.reshape(B, T) # get rid of the trailing channel dim
        # this is B x T

        # to construct the probability-based loss, i need the log-sum-exp over the spatial dims
        energy_vec = energy_vol.reshape(B, Z*Y*X)
        logpartition_function = torch.logsumexp(energy_vec, 1, keepdim=True)
        # this is B x 1

        loglike_per_timestep = energy_per_timestep - logpartition_function
        # this is B x T

        loglike_per_traj = torch.sum(loglike_per_timestep, dim=1)
        # this is B

        return loglike_per_traj

    def get_trajs_loglike(self, xyzs, scores, energy_vol):
        # energy_vol is B x 1 x Z x Y x X; it is not normalized in any way
        # xyzs is K x B x T x 3; it specifies vox coordinates of the traj
        # scores is K x B x T

        K, B, T, D = list(xyzs.shape)
        _, _, Z, Y, X = list(energy_vol.shape)
        assert(D==3)

        xyzs_ = xyzs.permute(1, 0, 2, 3).reshape(B, K*T, 3)
        scores_ = scores.permute(1, 0, 2).reshape(B, K*T)
        x, y, z = torch.unbind(xyzs_, dim=2)
        energy_per_timestep_ = utils_samp.bilinear_sample3D(energy_vol, x, y, z)
        energy_per_timestep = energy_per_timestep_.reshape(B, K, T)
        
        scores = scores.permute(1, 0, 2)
        # scores is B x K x T

        # to construct the probability-based loss, i need the log-sum-exp over the spatial dims
        energy_vec = energy_vol.reshape(B, Z*Y*X, 1)
        logpartition_function = torch.logsumexp(energy_vec, 1, keepdim=True)
        # this is B x 1 x 1

        loglike_per_timestep = energy_per_timestep - logpartition_function
        # this is B x K x T

        loglike_per_traj = torch.sum(loglike_per_timestep*scores, dim=2)
        # this is B x K 

        return loglike_per_traj

    
