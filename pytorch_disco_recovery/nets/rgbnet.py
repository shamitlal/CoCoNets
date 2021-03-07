import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import ipdb
st = ipdb.set_trace
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic

class RgbNet(nn.Module):
    def __init__(self):
        super(RgbNet, self).__init__()

        print('RgbNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        self.conv3d = nn.Conv3d(in_channels=hyp.feat3d_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()
        
    def compute_loss(self, pred, rgb, free, valid, summ_writer):
        pos = rgb.clone()
        neg = free.clone()

        # rgb is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('rgb/prob_loss', loss_vis)

        pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss, loss

    def forward(self, feat, rgb_g=None, valid=None, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()
        feat = self.conv3d(feat)
        # st()

        rgb_e = F.sigmoid(feat) - 0.5
        # rgb_e is B x 3 x Z x Y x X
        
        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(rgb_e, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        if valid is not None:
            smooth_loss = utils.basic.reduce_masked_mean(smooth_vox, valid)
        else:
            smooth_loss = torch.mean(smooth_vox)
        total_loss = utils.misc.add_loss('rgb/smooth_loss%s' % suffix, total_loss, smooth_loss, hyp.rgb_smooth_coeff, summ_writer)
    
        if rgb_g is not None:
            loss_im = utils.basic.l1_on_axis(rgb_e-rgb_g, 1, keepdim=True)
            if valid is not None:
                rgb_loss = utils.basic.reduce_masked_mean(loss_im, valid)
            total_loss = utils.misc.add_loss('rgb/rgb_l1_loss', total_loss, rgb_loss, hyp.rgb_l1_coeff, summ_writer)
        else:
            total_loss = None

        if summ_writer is not None:
            summ_writer.summ_oned('rgb/smooth_loss%s' % suffix, torch.mean(smooth_vox, dim=3))

        return total_loss, rgb_e

