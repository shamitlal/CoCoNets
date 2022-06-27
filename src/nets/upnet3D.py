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

# import archs.pixelshuffle3d


EPS = 1e-4
class UpNet3D(nn.Module):
    def __init__(self):
        super(UpNet3D, self).__init__()
        
        # self.net = encoder3D.Up3D(in_dim=128, out_dim=hyp.feat3D_dim, scale=4).cuda()
        # self.net = encoder3D.Up3D(in_dim=128, out_dim=hyp.feat3D_dim, scale=2).cuda()
        # self.net = encoder3D.Up3D(in_dim=hyp.feat3D_dim*2, out_dim=hyp.feat3D_dim, scale=4).cuda()
        # self.net = encoder3D.Up3D(in_dim=hyp.feat3D_dim*2, out_dim=hyp.feat3D_dim, scale=2).cuda()
        # self.net = encoder3D.Up3D(in_dim=hyp.feat3D_dim*4, out_dim=hyp.feat3D_dim, scale=4).cuda()
        # self.net = encoder3D.Up3D(in_dim=hyp.feat3D_dim*4, out_dim=4, scale=4).cuda()
        
        self.net = encoder3D.Up3D(in_dim=128, chans=128, out_dim=4, scale=4).cuda()
        # self.net = encoder3D.Up3D(in_dim=128, chans=128, out_dim=32, scale=4).cuda()
        print(self.net)

    def forward(self, feat, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        # B, C, Z, Y, X = list(feat.shape)

        feat = self.net(feat)

        # smooth loss
        dz, dy, dx = utils_basic.gradient3D(feat, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        summ_writer.summ_oned('up3D/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('up3D/smooth_loss', total_loss, smooth_loss, hyp.up3D_smooth_coeff, summ_writer)

        # feat = utils_basic.l2_normalize(feat, dim=1)
        # print('feat', feat.shape)
        
        if summ_writer is not None:
            summ_writer.summ_feat('up3D/feat_output', feat, pca=True)
        return total_loss, feat

