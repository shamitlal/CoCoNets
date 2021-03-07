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

class ResolveNet(nn.Module):
    def __init__(self):
        super(ResolveNet, self).__init__()

        print('ResolveNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        # self.conv3d = nn.Conv3d(in_channels=hyp.feat3D_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()

        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()

        self.hidden_dim = 256
        self.predictor = nn.Sequential(
            nn.Conv1d(in_channels=324, out_channels=int(self.hidden_dim), kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Conv1d(in_channels=self.hidden_dim, out_channels=hyp.feat3d_dim, kernel_size=1, bias=True),
            nn.Conv1d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1, bias=True),
        ).cuda()
        
    def compute_loss(self, pred, occ, free, summ_writer):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x N

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()

        pos_loss = utils.basic.reduce_masked_mean(loss, pos)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss, loss

    def forward(self, feat, occ_g=None, free_g=None, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        print('feat', feat.shape)
        occ_e_ = self.predictor(feat)
        print('occ_e_', occ_e_.shape)
        
        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = occ_e.round()

        if occ_g is not None:
            # collect some accuracy stats 
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g)
            acc_free = utils.basic.reduce_masked_mean(free_match, free_g)
            acc_total = utils.basic.reduce_masked_mean(either_match, either_have)
            acc_bal = (acc_occ + acc_free)*0.5

            summ_writer.summ_scalar('unscaled_resolve/acc_occ%s' % suffix, acc_occ.cpu().item())
            summ_writer.summ_scalar('unscaled_resolve/acc_free%s' % suffix, acc_free.cpu().item())
            summ_writer.summ_scalar('unscaled_resolve/acc_total%s' % suffix, acc_total.cpu().item())
            summ_writer.summ_scalar('unscaled_resolve/acc_bal%s' % suffix, acc_bal.cpu().item())
            
            prob_loss, full_loss = self.compute_loss(occ_e_, occ_g, free_g, summ_writer)
            total_loss = utils.misc.add_loss('resolve/prob_loss%s' % suffix, total_loss, prob_loss, hyp.resolve_coeff, summ_writer)

        return total_loss, occ_e_

