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
import torch.nn.functional as F

EPS = 1e-4
class FeatNet3D(nn.Module):
    def __init__(self, in_dim=1):
        super(FeatNet3D, self).__init__()
        # if hyp.feat3D_skip:
        #     self.crop = crop

        if hyp.feat3D_skip:
            self.net = encoder3D.Net3D(in_channel=in_dim, pred_dim=hyp.feat3D_dim).cuda()
        else:
            self.net = encoder3D.EncoderDecoder3D(in_dim=in_dim, out_dim=hyp.feat3D_dim).cuda()
            # self.net = encoder3D.ResNet3D(in_channel=in_dim, pred_dim=hyp.feat3D_dim, padding=0).cuda()
            # self.net = encoder3D.Encoder3D(in_dim=in_dim, out_dim=hyp.feat3D_dim).cuda()
            # self.net = archs.sparse_encoder3D.SparseResNet3D(in_channel=in_dim, pred_dim=hyp.feat3D_dim).cuda()
            # self.net = archs.DCCA_sparse_networks_3d.Encoder3D(in_dim=in_dim, out_dim=hyp.feat3D_dim).cuda()
            
        print(self.net)

    def forward(self, feat, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat.shape)

        mask = (feat[:,0:1] > 0.0).float()
        # if summ_writer is not None:
        #     summ_writer.summ_feat('feat3D/feat_mask', mask, pca=False)
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat3D/feat_input', feat, pca=(C>3))

        feat = self.net(feat)
        mask = torch.ones_like(feat[:,0:1])

        # smooth loss
        dz, dy, dx = utils_basic.gradient3D(feat, absolute=True)
        smooth_vox = torch.mean(dz+dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('feat3D/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('feat3D/smooth_loss', total_loss, smooth_loss, hyp.feat3D_smooth_coeff, summ_writer)
            
        feat = utils_basic.l2_normalize(feat, dim=1)
        if hyp.feat3D_sparse:
            feat = feat * mask
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat3D/feat_output', feat, pca=True)
            # summ_writer.summ_feat('feat3D/feat_mask', mask, pca=False)
            
        # if hyp.feat3D_skip:
        #     feat = feat[:,:,
        #                 self.crop[0]:-self.crop[0],
        #                 self.crop[1]:-self.crop[1],
        #                 self.crop[2]:-self.crop[2]]
        #     mask = mask[:,:,
        #                 self.crop[0]:-self.crop[0],
        #                 self.crop[1]:-self.crop[1],
        #                 self.crop[2]:-self.crop[2]]
            
        return total_loss, feat, mask

