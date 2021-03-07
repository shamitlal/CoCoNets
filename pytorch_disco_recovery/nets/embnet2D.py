import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder2D as encoder2D
import hyperparams as hyp
from utils_basic import *
import utils_basic
import utils_misc
import utils_improc

class EmbNet2D(nn.Module):
    def __init__(self):
        super(EmbNet2D, self).__init__()

        print('EmbNet2D...')
        self.batch_k = 2
        self.num_samples = hyp.emb_2D_num_samples
        assert(self.num_samples > 0)
        self.sampler = utils_misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False)
        self.criterion = utils_misc.MarginLoss() #margin=args.margin,nu=args.nu)
        self.beta = 1.2

    def sample_embs(self, emb0, emb1, valid, B, Y, X, mod='', do_vis=False, summ_writer=None):
        if hyp.emb_2D_mindist == 0.0:
            # pure random
            perm = torch.randperm(B*Y*X)
            emb0 = emb0.reshape(B*Y*X, -1)
            emb1 = emb1.reshape(B*Y*X, -1)
            valid = valid.reshape(B*Y*X, -1)
            emb0 = emb0[perm[:self.num_samples*B]]
            emb1 = emb1[perm[:self.num_samples*B]]
            valid = valid[perm[:self.num_samples*B]]
            return emb0, emb1, valid
        else:
            emb0_all = []
            emb1_all = []
            valid_all = []
            for b in list(range(B)):
                sample_indices, sample_locs, sample_valids = utils_misc.get_safe_samples(
                    valid[b], (Y, X), self.num_samples, mode='2D', tol=hyp.emb_2D_mindist)
                emb0_s_ = emb0[b, sample_indices]
                emb1_s_ = emb1[b, sample_indices]
                # these are N x D
                emb0_all.append(emb0_s_)
                emb1_all.append(emb1_s_)
                valid_all.append(sample_valids)

            if do_vis and (summ_writer is not None):
                sample_mask = utils_improc.xy2mask_single(sample_locs, Y, X)
                summ_writer.summ_oned('emb2D/samples_%s/sample_mask' % mod, torch.unsqueeze(sample_mask, dim=0))
                summ_writer.summ_oned('emb2D/samples_%s/valid' % mod, torch.reshape(valid, [B, 1, Y, X]))
            emb0_all = torch.cat(emb0_all, axis=0)
            emb1_all = torch.cat(emb1_all, axis=0)
            valid_all = torch.cat(valid_all, axis=0)
            return emb0_all, emb1_all, valid_all

    def compute_margin_loss(self, B, C, Y, X, emb0_vec, emb1_vec, valid_vec, mod='', do_vis=False, summ_writer=None):
        emb0_vec, emb1_vec, valid_vec = self.sample_embs(
            emb0_vec,
            emb1_vec,
            valid_vec,
            B, Y, X,
            mod=mod,
            do_vis=do_vis,
            summ_writer=summ_writer)
        
        emb_vec = torch.stack((emb0_vec, emb1_vec), dim=1).view(B*self.num_samples*self.batch_k,C)
        # this tensor goes e,g,e,g,... on dim 0
        # note this means 2 samples per class; batch_k=2
        y = torch.stack([torch.arange(0,self.num_samples*B), torch.arange(0,self.num_samples*B)], dim=1).view(self.num_samples*B*self.batch_k)
        # this tensor goes 0,0,1,1,2,2,...

        a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)
        margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
        return margin_loss

    def forward(self, emb0, emb1, valid, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, H, W = list(emb0.shape)

        emb0_vec = emb0.permute(0,2,3,1).reshape(B, H*W, C)
        emb1_vec = emb1.permute(0,2,3,1).reshape(B, H*W, C)
        valid_vec = valid.permute(0,2,3,1).reshape(B, H*W, 1)
        assert(self.num_samples < (B*H*W))
        # we will take num_samples from each one
        margin_loss = self.compute_margin_loss(B, C, H, W, emb0_vec, emb1_vec, valid_vec, 'all', True, summ_writer)
        total_loss = utils_misc.add_loss('emb2D/emb_2D_ml_loss%s' % suffix, total_loss,
                                         margin_loss, hyp.emb_2D_ml_coeff, summ_writer)

        l2_loss_im = sql2_on_axis(emb0-emb1, 1, keepdim=True)
        summ_writer.summ_oned('emb2D/emb_2D_l2_loss%s' % suffix, l2_loss_im)
        emb_l2_loss = reduce_masked_mean(l2_loss_im, valid)
        total_loss = utils_misc.add_loss('emb2D/emb_2D_l2_loss%s' % suffix, total_loss,
                                         emb_l2_loss, hyp.emb_2D_l2_coeff, summ_writer)

        summ_writer.summ_feats('emb2D/embs_2D%s' % suffix, [emb0, emb1], pca=True)
        return total_loss, emb1

    
