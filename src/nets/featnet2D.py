import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.v2v2d 
import archs.encoder2D 
import utils_geom
import utils_vox
import utils_misc
import utils_basic

EPS = 1e-4
class FeatNet2D(nn.Module):
    def __init__(self, in_dim=3):
        super(FeatNet2D, self).__init__()
        
        # self.net = archs.v2v2d.V2VModel(in_dim, hyp.feat2D_dim).cuda()
        self.net = archs.encoder2D.Net2D(in_dim, 64, hyp.feat2D_dim).cuda()
        
        print(self.net)

    def forward(self, rgb, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, H, W = list(rgb.shape)

        if summ_writer is not None:
            summ_writer.summ_rgb('feat2D/rgb', rgb)
        
        feat = self.net(rgb)

        # smooth loss
        dy, dx = utils_basic.gradient2D(feat, absolute=True)
        smooth_im = torch.mean(dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('feat2D/smooth_loss', smooth_im)
        smooth_loss = torch.mean(smooth_im)
        total_loss = utils_misc.add_loss('feat2D/smooth_loss', total_loss, smooth_loss, hyp.feat2D_smooth_coeff, summ_writer)

        feat = utils_basic.l2_normalize(feat, dim=1)
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat2D/feat_output', feat, pca=True)
        
        return total_loss, feat

