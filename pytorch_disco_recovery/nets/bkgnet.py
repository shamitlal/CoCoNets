import torch
import torch.nn as nn
import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3d
import archs.bottle3d
import utils.geom
import utils.misc
import utils.basic
import torch.nn.functional as F
import numpy as np

EPS = 1e-4
class BkgNet(nn.Module):
    def __init__(self, bkg_k=1, resolution=64):
        super(BkgNet, self).__init__()

        print('BkgNet...')

        self.epsilon = hyp.bkg_epsilon
        assert(self.epsilon >= 0.0 and self.epsilon < 1.0)
        
        self.K = bkg_k
        self.net = archs.bottle3d.ResNetBottle3d(
            in_dim=4,
            mid_dim=64,
            out_dim=self.K,
            resolution=resolution,
        ).cuda()

        # self.bkg_dict = torch.randn([1, self.K, 4, int(hyp.Z), int(hyp.Y), int(hyp.X)]).float().cuda()
        # self.bkg_dict = torch.autograd.Variable(self.bkg_dict, requires_grad=True)
        # self.bkg_dict = torch.Tensor(self.bkg_dict, requires_grad=True)
        
        print(self.net)
        # print(self.bkg_dict)

    def forward(self, feat_input, bkg_dict, target_occ, target_vis, is_train=True, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat_input.shape)

        val = np.random.uniform()
        if not is_train:
            # force prediction
            val = 1.0
            
        if val < self.epsilon:
            action = np.random.randint(0, self.K)
            soft = np.zeros((B, self.K), np.float32)
            soft[:,action] = 1.0
            soft = torch.from_numpy(soft).float().cuda()
        else:
            logits = self.net(feat_input)
            print('logits', logits.shape)
            # this is B x K

            noise = torch.from_numpy(np.random.uniform(low=-4.0, high=4.0, size=[B, self.K])).float().cuda()
            logits = logits + noise # this helps

            # soft = F.softmax(logits/0.1, dim=1) # this hurts
            soft = F.softmax(logits, dim=1)

        soft = soft.reshape(1, self.K, 1, 1, 1, 1)
        feat = torch.sum(bkg_dict * soft, dim=1)

        if summ_writer is not None:
            summ_writer.summ_feat('bkg/feat_input', feat_input, pca=(C>3))
            # summ_writer.summ_feat('bkg/feat_output', feat, pca=True)
            summ_writer.summ_feat('bkg/feat_output', feat, pca=False)

            summ_writer.summ_feats('bkg/bkg_dict', torch.unbind(bkg_dict, dim=1), pca=False)
            for k in list(range(self.K)):
                # summ_writer.summ_feat('bkg/bkg_%d' % k, bkg_dict[:,k], pca=True)
                summ_writer.summ_feat('bkg/bkg_%d' % k, bkg_dict[:,k], pca=False)
    
        return total_loss, feat

