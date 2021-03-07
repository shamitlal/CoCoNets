import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

# import archs.sparse_invar_encoder3D
import hyperparams as hyp
import archs.encoder3D
import utils_improc
import utils_misc
import utils_basic

class PreoccNet(nn.Module):
    def __init__(self):
        super(PreoccNet, self).__init__()

        print('PreoccNet...')
        self.net = archs.encoder3D.EncoderDecoder3D(in_dim=4, out_dim=1).cuda()
        print(self.net)
        
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
        summ_writer.summ_oned('preocc/prob_loss', loss_vis)

        pos_loss = utils_basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils_basic.reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss, loss
    
    def forward(self, feat, occ_g=None, free_g=None, valid=None, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)
        # feat is B x C x Z x Y x X
        # occ_g is B x 1 x Z x Y x X
        
        occ_e_ = self.net(feat)
        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        # smooth loss
        dz, dy, dx = utils_basic.gradient3D(occ_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        if valid is not None:
            smooth_loss = utils_basic.reduce_masked_mean(smooth_vox, valid)
        else:
            smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('preocc/smooth_loss%s' % suffix, total_loss, smooth_loss, hyp.preocc_smooth_coeff, summ_writer)

        if summ_writer is not None:
            summ_writer.summ_feat('preocc/input', feat, pca=(C>3))
        
    
        if occ_g is not None:
            # assume free_g and valid are also not None
            
            # collect some accuracy stats 
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils_basic.reduce_masked_mean(occ_match, occ_g*valid)
            acc_free = utils_basic.reduce_masked_mean(free_match, free_g*valid)
            acc_total = utils_basic.reduce_masked_mean(either_match, either_have*valid)
            acc_bal = (acc_occ + acc_free)*0.5

            summ_writer.summ_scalar('unscaled_preocc/acc_occ%s' % suffix, acc_occ.cpu().item())
            summ_writer.summ_scalar('unscaled_preocc/acc_free%s' % suffix, acc_free.cpu().item())
            summ_writer.summ_scalar('unscaled_preocc/acc_total%s' % suffix, acc_total.cpu().item())
            summ_writer.summ_scalar('unscaled_preocc/acc_bal%s' % suffix, acc_bal.cpu().item())
            
            prob_loss, full_loss = self.compute_loss(occ_e_, occ_g, free_g, valid, summ_writer)
            total_loss = utils_misc.add_loss('preocc/prob_loss%s' % suffix, total_loss, prob_loss, hyp.preocc_coeff, summ_writer)
        else:
            full_loss, either_match = None, None

        if summ_writer is not None:
            summ_writer.summ_oned('preocc/smooth_loss%s' % suffix, torch.mean(smooth_vox, dim=3))
            if occ_g is not None:
                summ_writer.summ_occ('preocc/occ_g%s' % suffix, occ_g)
                summ_writer.summ_occ('preocc/free_g%s' % suffix, free_g)
            summ_writer.summ_occ('preocc/occ_e%s' % suffix, occ_e)
            # summ_writer.summ_occ('preocc/valid%s' % suffix, valid)

        return total_loss, occ_e


