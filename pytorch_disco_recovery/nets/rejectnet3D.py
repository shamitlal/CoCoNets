import torch
import torch.nn as nn
import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3D as encoder3D
import archs.v2v
import archs.DCCA_sparse_networks_3d
import archs.sparse_encoder3D 
import utils_geom
# import utils_vox
import utils_misc
import utils_basic
import numpy as np
import torch.nn.functional as F

EPS = 1e-4
class RejectNet3D(nn.Module):
    def __init__(self, in_dim=1):
        super(RejectNet3D, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels=hyp.feat3D_dim,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(True),
            nn.Conv3d(
                in_channels=128,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
        ).cuda()
        
        print(self.net)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, feat0, feat1, valid0, valid1, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat0.shape)

        neg_input = torch.cat([feat1-feat0], dim=1)

        feat1_flat = feat1.reshape(B, C, -1)
        valid1_flat = valid1.reshape(B, 1, -1)
        perm = np.random.permutation(D*H*W)
        feat1_flat_shuf = feat1_flat[:,:,perm]
        feat1_shuf = feat1_flat_shuf.reshape(B, C, D, H, W)
        valid1_flat_shuf = valid1_flat[:,:,perm]
        valid1_shuf = valid1_flat_shuf.reshape(B, 1, D, H, W)
        # pos_input = torch.cat([feat0, feat1_shuf-feat0], dim=1)
        pos_input = torch.cat([feat1_shuf-feat0], dim=1)

        # noncorresps should be rejected 
        pos_output = self.net(pos_input)
        # corresps should NOT be rejected
        neg_output = self.net(neg_input)

        pos_sig = F.sigmoid(pos_output)
        neg_sig = F.sigmoid(neg_output)

        if summ_writer is not None:
            summ_writer.summ_feat('reject/pos_input', pos_input, pca=True)
            summ_writer.summ_feat('reject/neg_input', neg_input, pca=True)
            summ_writer.summ_oned('reject/pos_sig', pos_sig, bev=True, norm=False)
            summ_writer.summ_oned('reject/neg_sig', neg_sig, bev=True, norm=False)

        pos_output_vec = pos_output.reshape(B, D*H*W)
        neg_output_vec = neg_output.reshape(B, D*H*W)
        pos_target_vec = torch.ones([B, D*H*W]).float().cuda()
        neg_target_vec = torch.zeros([B, D*H*W]).float().cuda()

        # if feat1_shuf is valid, then it is practically guranateed to mismatch feat0
        pos_valid_vec = valid1_shuf.reshape(B, D*H*W)
        # both have to be valid to not reject
        neg_valid_vec = (valid0*valid1).reshape(B, D*H*W)

        pos_loss_vec = self.criterion(pos_output_vec, pos_target_vec)
        neg_loss_vec = self.criterion(neg_output_vec, neg_target_vec)
        
        pos_loss = utils_basic.reduce_masked_mean(pos_loss_vec, pos_valid_vec)
        neg_loss = utils_basic.reduce_masked_mean(neg_loss_vec, pos_valid_vec)

        ce_loss = pos_loss + neg_loss
        utils_misc.add_loss('reject3D/ce_loss_pos', 0, pos_loss, 0, summ_writer)
        utils_misc.add_loss('reject3D/ce_loss_neg', 0, neg_loss, 0, summ_writer)
        total_loss = utils_misc.add_loss('reject3D/ce_loss', total_loss, ce_loss, hyp.reject3D_ce_coeff, summ_writer)
            
        return total_loss, neg_sig, pos_sig

