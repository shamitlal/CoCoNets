import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_vox
import utils_basic

class MocNet2D(nn.Module):
    def __init__(self, moc_trainer):
        super(MocNet2D, self).__init__()

        self.moc_trainer = moc_trainer

        print('MocNet2D...')

        self.num_samples = hyp.moc2D_num_samples
        assert(self.num_samples > 0)

    def sample_embs(self, emb0, emb1):
        B, C, H, W = list(emb0.shape)
        
        # pure random sampling
        perm = torch.randperm(B*H*W)
        
        # put channels at the end
        emb0 = emb0.permute(0,2,3,1).reshape(B*H*W, C)
        emb1 = emb1.permute(0,2,3,1).reshape(B*H*W, C)
        
        emb0 = emb0[perm[:self.num_samples*B]]
        emb1 = emb1[perm[:self.num_samples*B]]

        return emb0, emb1

    # def sample_embs_simple(self, emb0):
    #     B, C, H, W = emb0.shape
    #     # pure random sampling
    #     perm = torch.randperm(B*H*W)
    #     emb0 = emb0.reshape(B*H*W, C)
    #     emb0 = emb0[perm[:self.num_samples*B]]
    #     return emb0
    
    def forward(self, emb0, emb1, summ_writer, mod=''):
        total_loss = torch.tensor(0.0).cuda()

        if torch.isnan(emb0).any() or torch.isnan(emb1).any():
            assert(False)

        B, C, H, W = list(emb0.shape)
        
        # we will take num_samples across the batch
        assert(self.num_samples < (B*H*W))
    
        emb0_vec, emb1_vec = self.sample_embs(emb0, emb1)
        
        moc_loss = self.moc_trainer.forward(
            emb0_vec, 
            emb1_vec.detach()
        )
        self.moc_trainer.enqueue(emb1_vec)

        total_loss = utils_misc.add_loss('moc2D/moc2D_loss%s' % mod, total_loss, moc_loss, hyp.moc2D_coeff, summ_writer)
        summ_writer.summ_feats('moc2D/embs%s' % mod, [emb0, emb1], pca=True)
        return total_loss

