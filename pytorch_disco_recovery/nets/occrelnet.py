import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils_basic
import utils_improc
import utils_misc
import utils_basic

class OccrelNet(nn.Module):
    def __init__(self):
        super(OccrelNet, self).__init__()

        print('OccrelNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        # self.conv3d = nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()

        self.hidden_dim = 64
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1, padding=0)
        ).cuda()


        
        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()
        
    def compute_loss(self, pred, pos, neg, occ, free, valid, summ_writer):
        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('occrel/prob_loss', loss_vis)

        # pos_loss = reduce_masked_mean(loss, pos*valid)
        # neg_loss = reduce_masked_mean(loss, neg*valid)
        # balanced_loss = pos_loss + neg_loss

        pos_occ_loss = utils_basic.reduce_masked_mean(loss, pos*valid*occ)
        pos_free_loss = utils_basic.reduce_masked_mean(loss, pos*valid*free)
        neg_occ_loss = utils_basic.reduce_masked_mean(loss, neg*valid*occ)
        neg_free_loss = utils_basic.reduce_masked_mean(loss, neg*valid*free)
        balanced_loss = pos_occ_loss + pos_free_loss + neg_occ_loss + neg_free_loss

        return balanced_loss

    def forward(self, feat, occrel_g=None, occ_g=None, free_g=None, valid=None, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()
        
        occrel_e_ = self.net(feat)
        occrel_e = F.sigmoid(occrel_e_)
        occrel_e_binary = torch.round(occrel_e)

        if occrel_g is not None:
            # assume free_g and valid are also not None
            
            # collect some accuracy stats 
            pos_match = occrel_g*torch.eq(occrel_e_binary, occrel_g).float()
            neg_match = (1.0 - occrel_g)*torch.eq(1.0-occrel_e_binary, 1.0 - occrel_g).float()
            either_match = torch.clamp(pos_match+neg_match, 0.0, 1.0)
            either_have = occrel_g.clone()
            acc_pos = utils_basic.reduce_masked_mean(pos_match, occrel_g*valid)
            acc_neg = utils_basic.reduce_masked_mean(neg_match, (1.0-occrel_g)*valid)
            acc_total = utils_basic.reduce_masked_mean(either_match, either_have*valid)
            acc_bal = (acc_pos + acc_neg)*0.5

            summ_writer.summ_scalar('unscaled_occrel/acc_pos', acc_pos.cpu().item())
            summ_writer.summ_scalar('unscaled_occrel/acc_neg', acc_neg.cpu().item())
            summ_writer.summ_scalar('unscaled_occrel/acc_total', acc_total.cpu().item())
            summ_writer.summ_scalar('unscaled_occrel/acc_bal', acc_bal.cpu().item())
            
            prob_loss = self.compute_loss(occrel_e_, occrel_g, (1.0 - occrel_g), occ_g, free_g, valid, summ_writer)
            total_loss = utils_misc.add_loss('occrel/prob_loss', total_loss, prob_loss, hyp.occrel_coeff, summ_writer)

        if summ_writer is not None:
            if occrel_g is not None:
                summ_writer.summ_occ('occrel/occrel_g', occrel_g)
                summ_writer.summ_oned('occrel/occrel_g_', occrel_g, bev=True, norm=False)
            summ_writer.summ_occ('occrel/occrel_e', occrel_e)
            summ_writer.summ_oned('occrel/occrel_e', occrel_e, bev=True, norm=False)
        return total_loss, occrel_e
