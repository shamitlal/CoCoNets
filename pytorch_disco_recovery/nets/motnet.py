import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import archs.pixelshuffle3d
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_basic

class MotNet(nn.Module):
    def __init__(self, num_classes):
        super(MotNet, self).__init__()

        print('MotNet...')

        self.num_classes = num_classes

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        # self.conv3d = nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0).cuda()

        
        self.hidden_dim = 64
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.BatchNorm3d(self.hidden_dim),
            nn.Conv3d(in_channels=self.hidden_dim, out_channels=self.num_classes, kernel_size=1, padding=0)
        ).cuda()

        # self.net = archs.encoder3D.Net3D(in_channel=4, pred_dim=1).cuda()
        
        # self.net = archs.encoder3D.EncoderDecoder3D(in_dim=4, out_dim=1).cuda()
        
    def compute_loss(self, pred, occ, free, valid, summ_writer):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('mot/prob_loss', loss_vis)

        pos_loss = reduce_masked_mean(loss, pos*valid)
        neg_loss = reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    # def compute_loss(self, pred, mot, val, summ_writer):
    #     # pred is B x C x Z x Y x X
    #     # mot is B x Z x Y x X and long
    #     # val is B x Z x Y x X and float

    #     loss = F.cross_entropy(pred, mot, reduction='none')
    #     # loss is B x Z x Y x X

    #     loss_vis = torch.mean(loss*val, dim=2).unsqueeze(1)
    #     summ_writer.summ_oned('mot/prob_loss', loss_vis)

    #     loss = loss.reshape(-1)
    #     mot = mot.reshape(-1)
    #     val = val.reshape(-1)

    #     losses = []
        
    #     # next, i want to gather up the loss for each class, and balance these into a total
    #     for cls in list(range(self.num_classes)):
    #         mask = (mot==cls).float()*val
    #         cls_loss = utils_basic.reduce_masked_mean(loss, mask)
    #         # print('cls %d sum' % cls, torch.sum(mask).detach().cpu().numpy(), 'loss', cls_loss.detach().cpu().numpy())
    #         if torch.sum(mask) >= 1:
    #             losses.append(cls_loss)
    #     if len(losses):
    #         total_loss = torch.mean(torch.stack(losses))
    #     else:
    #         total_loss = 0.0
    #     return total_loss
    
    def forward(self, feat, obj_g=None, bkg_g=None, valid_g=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        mot_e_ = self.net(feat)
        mot_e = F.sigmoid(mot_e_)

        # smooth loss
        dz, dy, dx = gradient3D(mot_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('mot/smooth_loss', total_loss, smooth_loss, hyp.mot_smooth_coeff, summ_writer)

        if summ_writer is not None:
            summ_writer.summ_occ('mot/mot_e', mot_e)
            
            summ_writer.summ_oned('mot/smooth_loss', torch.mean(smooth_vox, dim=3))
            # mot_e_vis = torch.max(torch.max(mot_e, dim=1)[1], dim=2)[0]
            # summ_writer.summ_seg('mot/seg_mot_e', mot_e_vis)

            # for cls in list(range(self.num_classes)):
            #     cls_mask_e = mot_e[:,cls:cls+1]
            #     cls_mask_e = F.sigmoid(cls_mask_e)
            #     summ_writer.summ_oned('mot/mot_e_cls%d' % cls, cls_mask_e, bev=True, norm=False)
            #     cls_mask_g = (valid_g*(mot_g==cls).float()).unsqueeze(1)
            #     summ_writer.summ_oned('mot/mot_g_cls%d' % cls, cls_mask_g, bev=True, norm=False)

        if obj_g is not None:
            prob_loss = self.compute_loss(mot_e_, obj_g, bkg_g, valid_g, summ_writer)
            total_loss = utils_misc.add_loss('mot/prob_loss', total_loss, prob_loss, hyp.mot_prob_coeff, summ_writer)
            # mot_g_vis = torch.max(mot_g, dim=2)[0]
            # if summ_writer is not None:
            #     summ_writer.summ_seg('mot/seg_mot_g', mot_g_vis)
            #     summ_writer.summ_oned('mot/valid_g', valid_g.unsqueeze(1), bev=True, norm=False)
        
        return total_loss, mot_e


