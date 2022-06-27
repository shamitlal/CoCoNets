import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_basic

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        print('SegNet...')

        self.num_classes = num_classes

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        self.conv3d = nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0).cuda()
        
    def compute_loss(self, pred, seg, free, summ_writer):
        # pred is B x C x Z x Y x X
        # seg is B x Z x Y x X

        # ensure the "free" voxels are labelled zero
        seg = seg * (1-free)

        
        # seg_bak = seg.clone()

        # note that label0 in seg is invalid, so i need to ignore these
        # but all "free" voxels count as the air class
        # seg = (seg-1).clamp(min=0)
        # seg = (seg-1).clamp(min=0)
        # # ignore_mask = (seg[seg==-1]).float()
        # seg[seg==-1] = 0
        loss = F.cross_entropy(pred, seg, reduction='none')
        # loss is B x Z x Y x X

        loss_any = ((seg>0).float() + (free>0).float()).clamp(0,1)
        loss_vis = torch.mean(loss*loss_any, dim=2).unsqueeze(1)
        summ_writer.summ_oned('seg/prob_loss', loss_vis)

        loss = loss.reshape(-1)
        # seg_bak = seg_bak.reshape(-1)
        seg = seg.reshape(-1)
        free = free.reshape(-1)

        # print('loss', loss.shape)
        # print('seg_bak', seg_bak.shape)

        losses = []
        # total_loss = 0.0
        
        # next, i want to gather up the loss for each valid class, and balance these into a total
        for cls in list(range(self.num_classes)):
            if cls==0:
                mask = free.clone()
            else:
                # mask = (seg_bak==cls).float()
                mask = (seg==cls).float()
                
            # print('mask', mask.shape)
            # print('loss', loss.shape)
            cls_loss = utils_basic.reduce_masked_mean(loss, mask)
            print('cls %d sum' % cls, torch.sum(mask).detach().cpu().numpy(), 'loss', cls_loss.detach().cpu().numpy())
            
            # print('cls_loss', cls_loss.shape)
            # print('cls %d loss' % cls, cls_loss.detach().cpu().numpy())
            # total_loss = total_loss + cls_loss
            if torch.sum(mask) >= 1:
                losses.append(cls_loss)
            # print('mask', mask.shape)
            # loss_ = loss[seg_bak==cls]
            # print('loss_', loss_.shape)
            # loss_ = loss[seg_bak==cls]
        total_loss = torch.mean(torch.stack(losses))
        return total_loss
    
    def forward(self, feat, seg_g=None, occ_g=None, free_g=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        seg_e = self.conv3d(feat)

        # smooth loss
        dz, dy, dx = gradient3D(seg_e, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('seg/smooth_loss', total_loss, smooth_loss, hyp.seg_smooth_coeff, summ_writer)

        if summ_writer is not None:
            summ_writer.summ_oned('seg/smooth_loss', torch.mean(smooth_vox, dim=3))
            seg_e_vis = torch.max(torch.max(seg_e, dim=1)[1], dim=2)[0]
            summ_writer.summ_seg('seg/seg_e', seg_e_vis)

        if seg_g is not None:
            prob_loss = self.compute_loss(seg_e, seg_g, free_g.long().squeeze(1), summ_writer)
            total_loss = utils_misc.add_loss('seg/prob_loss', total_loss, prob_loss, hyp.seg_prob_coeff, summ_writer)
            seg_g_vis = torch.max(seg_g, dim=2)[0]
            if summ_writer is not None:
                summ_writer.summ_seg('seg/seg_g', seg_g_vis)
        
        return total_loss, seg_e


