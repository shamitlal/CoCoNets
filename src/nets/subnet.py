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

class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()

        print('SubNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        # self.conv3d = nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()

        self.hidden_dim = 64
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=hyp.feat3D_dim*2, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1, padding=0)
        ).cuda()


        
        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()
        
    def compute_loss(self, pred, pos, neg, valid, summ_writer):
        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('sub/prob_loss', loss_vis)

        # pos_loss = reduce_masked_mean(loss, pos*valid)
        # neg_loss = reduce_masked_mean(loss, neg*valid)
        # balanced_loss = pos_loss + neg_loss

        # pos_occ_loss = utils_basic.reduce_masked_mean(loss, pos*valid*occ)
        # pos_free_loss = utils_basic.reduce_masked_mean(loss, pos*valid*free)
        # neg_occ_loss = utils_basic.reduce_masked_mean(loss, neg*valid*occ)
        # neg_free_loss = utils_basic.reduce_masked_mean(loss, neg*valid*free)
        # balanced_loss = pos_occ_loss + pos_free_loss + neg_occ_loss + neg_free_loss

        pos_loss = utils_basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils_basic.reduce_masked_mean(loss, neg*valid)
        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    # def forward(self, feat, sub_g=None, valid=None, summ_writer=None, suffix=''):
    #     total_loss = torch.tensor(0.0).cuda()
    def forward(self, feat, obj_g=None, bkg_g=None, valid_g=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        
        sub_e_ = self.net(feat)
        sub_e = F.sigmoid(sub_e_)
        sub_e_binary = torch.round(sub_e)

        # smooth loss
        dz, dy, dx = utils_basic.gradient3D(sub_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('sub/smooth_loss', total_loss, smooth_loss, hyp.sub_smooth_coeff, summ_writer)

        if obj_g is not None:
            
            # # collect some accuracy stats 
            # pos_match = sub_g*torch.eq(sub_e_binary, sub_g).float()
            # neg_match = (1.0 - sub_g)*torch.eq(1.0-sub_e_binary, 1.0 - sub_g).float()
            # either_match = torch.clamp(pos_match+neg_match, 0.0, 1.0)
            # either_have = sub_g.clone()
            # acc_pos = utils_basic.reduce_masked_mean(pos_match, sub_g*valid)
            # acc_neg = utils_basic.reduce_masked_mean(neg_match, (1.0-sub_g)*valid)
            # acc_total = utils_basic.reduce_masked_mean(either_match, either_have*valid)
            # acc_bal = (acc_pos + acc_neg)*0.5

            # summ_writer.summ_scalar('unscaled_sub/acc_pos', acc_pos.cpu().item())
            # summ_writer.summ_scalar('unscaled_sub/acc_neg', acc_neg.cpu().item())
            # summ_writer.summ_scalar('unscaled_sub/acc_total', acc_total.cpu().item())
            # summ_writer.summ_scalar('unscaled_sub/acc_bal', acc_bal.cpu().item())
            
            prob_loss = self.compute_loss(sub_e_, obj_g, bkg_g, valid_g, summ_writer)
            # prob_loss = self.compute_loss(sub_e_, sub_g, (1.0 - sub_g), valid, summ_writer)
            total_loss = utils_misc.add_loss('sub/prob_loss', total_loss, prob_loss, hyp.sub_coeff, summ_writer)

        # if summ_writer is not None:
        #     if sub_g is not None:
        #         summ_writer.summ_occ('sub/sub_g', sub_g)
        #         summ_writer.summ_oned('sub/sub_g_', sub_g, bev=True, norm=False)
        #     summ_writer.summ_occ('sub/sub_e', sub_e)
        #     summ_writer.summ_oned('sub/sub_e', sub_e, bev=True, norm=False)
        return total_loss, sub_e
