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

class MocNet3D(nn.Module):
    def __init__(self, moc_trainer):
        super(MocNet3D, self).__init__()

        self.moc_trainer = moc_trainer

        print('MocNet3D...')
        
        self.num_samples = hyp.moc3D_num_samples
        assert(self.num_samples > 0)

    def sample_embs(self, emb0, emb1, valid):
        B, C, Z, Y, X = list(emb0.shape)

        # pure random sampling
        perm = torch.randperm(B*Z*Y*X)

        # put channels at the end
        emb0 = emb0.permute(0,2,3,4,1).reshape(B*Z*Y*X, C)
        emb1 = emb1.permute(0,2,3,4,1).reshape(B*Z*Y*X, C)
        valid = valid.permute(0,2,3,4,1).reshape(B*Z*Y*X, 1)
        emb0 = emb0[perm[:self.num_samples*B]]
        emb1 = emb1[perm[:self.num_samples*B]]
        valid = valid[perm[:self.num_samples*B]]
        return emb0, emb1, valid
            
    def forward(self, emb0, emb1, valid0, valid1, summ_writer):
        total_loss = torch.tensor(0.0).cuda()

        if torch.isnan(emb0).any() or torch.isnan(emb1).any():
            assert(False)

        B, C, Z, Y, X = list(emb0.shape)

        # we will take num_samples across the batch
        assert(self.num_samples < (B*Z*Y*X))

        emb0_vec, emb1_vec, _ = self.sample_embs(emb0, emb1, valid0*valid1)
        
        moc_loss = self.moc_trainer.forward(
            emb0_vec, 
            emb1_vec.detach()
        )
        self.moc_trainer.enqueue(emb1_vec)

        total_loss = utils_misc.add_loss('moc3D/moc3D_loss', total_loss, moc_loss, hyp.moc3D_coeff, summ_writer)
        summ_writer.summ_feats('moc3D/embs', [emb0, emb1], valids=[valid0, valid1], pca=True)
        return total_loss

