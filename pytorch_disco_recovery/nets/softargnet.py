import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic
import utils.geom
import utils.samp
import numpy as np

def _topk(objectness, K=10):
    B, C, Z, Y, X = list(objectness.shape)
    assert(C==1)
      
    scorelist, indlist = torch.topk(objectness.view(B, C, -1), K)

    indlist_z = indlist // (Y*X)
    indlist_y = (indlist % (Y*X)) // X
    indlist_x = (indlist % (Y*X)) % X

    scorelist = scorelist.reshape(B, K)
    indlist_z = indlist_z.reshape(B, K)
    indlist_y = indlist_y.reshape(B, K)
    indlist_x = indlist_x.reshape(B, K)

    xyzlist = torch.stack([indlist_x, indlist_y, indlist_z], dim=2).float()
    return scorelist, xyzlist

def _nms(heat, kernel=15):
    pad = (kernel - 1) // 2

    hmax = F.max_pool3d(
        heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def clamped_sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x whatever)
      gt_regr (batch x c x whatever)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  # loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  # num_pos  = pos_inds.float().sum()
  # pos_loss = pos_loss.sum()
  # neg_loss = neg_loss.sum()
  
  # print('num_pos', num_pos.detach().cpu().numpy())
  # print('pos_loss', pos_loss.detach().cpu().numpy())
  # print('neg_loss', neg_loss.detach().cpu().numpy())
  # print('neg_weights', neg_weights.detach().cpu().numpy())

  # if num_pos == 0:
  #   loss = loss - neg_loss
  # else:
  #   loss = loss - (pos_loss + neg_loss) / num_pos

  pos_loss = pos_loss.mean()
  neg_loss = neg_loss.mean()
  
  # loss = loss - (pos_loss + neg_loss)
  loss = - (pos_loss + neg_loss)
  # loss = - (torch.mean(pos_loss) + torch.mean(neg_loss))

  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class SoftargNet(nn.Module):
    def __init__(self):
        super(SoftargNet, self).__init__()

        print('SoftargNet...')


        self.K = 1
        
        # # for rotation, i have decided to be heading-unaware
        # # so, 0 and 180 are treated as equivalent
        # # self.num_rot_bins = 8
        # self.num_rot_bins = 1
        # bin_angles = np.linspace(0, np.pi, self.num_rot_bins, endpoint=False)
        # bin_complements = bin_angles + np.pi
        # all_bins = np.concatenate([bin_angles, bin_complements], axis=0)
        # all_inds = np.concatenate([np.arange(self.num_rot_bins), np.arange(self.num_rot_bins)], axis=0)
        # self.bin_angles = torch.from_numpy(bin_angles).float().cuda()
        # self.all_bins = torch.from_numpy(all_bins).float().cuda()
        # self.all_inds = torch.from_numpy(all_inds).long().cuda()

        # # a thing missing here is: one set of bins, maybe rz, should not be fully determined
        # # for now, let's remove rz. objects don't roll anyway

        obj_channels = self.K
        size_channels = 0
        self.total_channels = obj_channels + size_channels
        
        self.conv3d = nn.Conv3d(
            in_channels=128+128+64,
            out_channels=self.total_channels,
            kernel_size=1,
            stride=1,
            padding=0).cuda()
        print(self.conv3d)

        self.mse = torch.nn.MSELoss(reduction='none')
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        
    def forward(self, feat, crop, vox_util, softarg_g=None, lrtlist_cam_g=None, lrtlist_mem_g=None, scorelist_g=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)

        corr = self.conv3d(feat)
        xyz_offset = torch.zeros([B, 3], dtype=torch.float32).cuda()
        xyz_mem = utils.track.convert_corr_to_xyz(corr, xyz_offset, hard=False)

        # box = torch.zeros((B, 9), dtype=torch.float32).cuda()
        # rs = torch.zeros((B, 3), dtype=torch.float32).cuda()
        # ls = 3.0*torch.ones((B, 3), dtype=torch.float32).cuda()
        # box_mem = torch.cat([xyz_mem, ls, rs], dim=1)
        # lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(box_mem.unsqueeze(1))
        
        # if summ_writer is not None:
        #     summ_writer.summ_lrtlist_bev(
        #         'softarg/boxes_original_e',
        #         utils.basic.normalize(torch.mean(torch.abs(feat[0:1]), dim=1, keepdim=True)),
        #         lrtlist_mem[0:1],
        #         torch.ones(1,self.K).float().cuda(), # scores
        #         torch.ones(1,self.K).long().cuda(), # tids
        #         vox_util,
        #         already_mem=True)
        
        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(corr, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        # total_loss = utils.misc.add_loss('softarg/smooth_loss', total_loss, smooth_loss, hyp.softarg_coeff, summ_writer)
        total_loss = utils.misc.add_loss('softarg/smooth_loss', total_loss, smooth_loss, 0.0, summ_writer)
        
        Z_crop, Y_crop, X_crop = crop
        Z_mem = Z + Z_crop*2
        Y_mem = Y + Y_crop*2
        X_mem = X + X_crop*2
        feat_pad = F.pad(feat, (Z_crop, Z_crop, Y_crop, Y_crop, X_crop, X_crop), 'constant', 0)
        feat_pad = torch.mean(torch.abs(feat_pad), dim=1, keepdim=True)
        crop_vec = torch.from_numpy(np.reshape(np.array([X_crop, Y_crop, Z_crop]), (1, 1, 3))).float().cuda()

        xyz_cam = vox_util.Mem2Ref(xyz_mem.unsqueeze(1) + crop_vec, Z_mem, Y_mem, X_mem).squeeze(1)
        
        if summ_writer is not None:
            summ_writer.summ_oned('softarg/corr', corr, bev=True)

        # utils.basic.print_stats('xyz', xyz)

        box = torch.zeros((B, 9), dtype=torch.float32).cuda()
        rs = torch.zeros((B, 3), dtype=torch.float32).cuda()
        ls = 3.0*torch.ones((B, 3), dtype=torch.float32).cuda()
        ls[:,0] = 2.5
        ls[:,1] = 1.5
        ls[:,2] = 5.0
        box_cam = torch.cat([xyz_cam, ls, rs], dim=1)
        lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(box_cam.unsqueeze(1))
        lrtlist_mem = vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam, Z_mem, Y_mem, X_mem)

        if summ_writer is not None:
            summ_writer.summ_lrtlist_bev(
                'softarg/boxes_e',
                utils.basic.normalize(feat_pad[0:1]),
                lrtlist_mem[0:1],
                torch.ones(1,self.K).float().cuda(), # scores
                torch.ones(1,self.K).long().cuda(), # tids
                vox_util,
                already_mem=True)
            
        return total_loss, lrtlist_cam
