import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.v2v2d 
import utils_geom
import utils_vox
import utils_misc

EPS = 1e-4
class Feat2Net(nn.Module):
    def __init__(self, in_dim=1):
        super(Feat2Net, self).__init__()
        
        self.net = archs.v2v2d.V2VModel(in_dim, hyp.feat2_dim).cuda()
        print(self.net)

    def forward(self, feat, summ_writer=None, comp_mask=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, H, W = list(feat.shape)

        if summ_writer is not None:
            summ_writer.summ_feat('feat2/feat_input', feat, pca=False)
        
        feat = self.net(feat)
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat2/feat_output', feat)
        
        return feat

