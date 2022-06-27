import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("..")

import hyperparams as hyp
import utils_basic
import utils_geom
import utils_vox
import utils_misc
import utils_track

EPS = 1e-4
class TranslationNet(nn.Module):
    def __init__(self):
        super(TranslationNet, self).__init__()
        self.med = int(hyp.ZZ/2*hyp.ZY/2*hyp.ZX/2)
        self.low = int(self.med/2/2/2)
        # print('med, low', self.med, self.low)

        # self.predictor = nn.Sequential(
        #     nn.Conv3d(in_channels=self.med, out_channels=1024, kernel_size=4, stride=4, padding=0),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=4, stride=4, padding=0),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv3d(in_channels=512, out_channels=3, kernel_size=2, stride=1, padding=0),
        # ).cuda()
        
        # self.predictor = nn.Sequential(
        #     nn.Conv3d(in_channels=self.med, out_channels=1024, kernel_size=4, stride=4, padding=0),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv3d(in_channels=1024, out_channels=3, kernel_size=4, stride=4, padding=0),
        # ).cuda()

        self.hidden_dim = 1024
        self.predictor1 = nn.Sequential(
            # nn.Conv3d(in_channels=self.med, out_channels=self.hidden_dim, kernel_size=4, stride=2, padding=0),
            # nn.LeakyReLU(negative_slope=0.1),
            # nn.Conv3d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=4, stride=2, padding=0),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(in_channels=self.med, out_channels=self.hidden_dim, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
        ).cuda()
        self.predictor2 = nn.Sequential(
            nn.Linear(self.hidden_dim*self.low, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 3),
        ).cuda()

    def forward(self, template, search_region, xyz, use_window=False, summ_writer=None):
        # template is the thing we are searching for; it is B x C x ZZ x ZY x ZX
        # search_region is the featuremap where we are searching; it is B x C x Z x Y x X
        # xyz is the location of the answer in the search region; it is B x 3
        total_loss = torch.tensor(0.0).cuda()
        
        B, C, ZZ, ZY, ZX = list(template.shape)
        _, _, Z, Y, X = list(search_region.shape)
        _, D = list(xyz.shape)
        assert(D==3)

        # ok, i want to corr each voxel of the template with the search region.
        # this is a giant matmul

        search_vec = search_region.view(B, C, -1)
        # this is B x C x huge
        search_vec = search_vec.permute(0, 2, 1)
        # this is B x huge x C
        
        template_vec = template.view(B, C, -1)
        # this is B x C x med
        
        corr_vec = torch.matmul(search_vec, template_vec)
        # this is B x huge x med

        corr = corr_vec.reshape(B, Z, Y, X, ZZ*ZY*ZX)
        corr = corr.permute(0, 4, 1, 2, 3)
        # corr is B x med x Z x Y x X
        
        # next step is:
        # a network should do quick work of this and turn it into an output
        
        # translation = self.predictor(corr)
        # # this is B x 3 x 1 x 1 x 1
        # # print('translation', translation.shape)
        # translation = torch.mean(translation, dim=[2, 3, 4])
        # # translation = translation.view(B, 3)
        # # now, i basically want this to be the answer

        feat = self.predictor1(corr)
        # print('feat', feat.shape)
        feat = feat.reshape(B, -1)
        # print('feat', feat.shape)
        translation = self.predictor2(feat)
        # print('tr', translation.shape)
        
        # now, i basically want this to be the answer
        
        translation_loss = torch.mean(torch.norm(translation - xyz, dim=1))
        total_loss = utils_misc.add_loss('translation/translation_loss', total_loss, translation_loss, hyp.translation_coeff, summ_writer)
        
        # if summ_writer is not None:
        #     # inputs
        #     summ_writer.summ_feat('translation/input_template', template, pca=False)
        #     summ_writer.summ_feat('translation/input_search_region', search_region, pca=False)
        #     # outputs
        #     summ_writer.summ_oned('translation/corr', torch.mean(corr, dim=3)) # reduce the vertical dim

        return translation, total_loss

