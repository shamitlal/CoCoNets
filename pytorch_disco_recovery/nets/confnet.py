import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import archs.bottle3D

import sys
sys.path.append("..")

import hyperparams as hyp
import utils_basic
import utils_geom
import utils_vox
import utils_misc
import utils_track
import utils_samp

EPS = 1e-4
class ConfNet(nn.Module):
    def __init__(self):
        super(ConfNet, self).__init__()
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='none')
        self.num_rots = 3

        # self.num_replicas = 4
        self.num_replicas = 8

        self.bottle_chans = 16
        self.bottle = nn.Sequential(
            nn.Conv3d(
                in_channels=self.num_rots+1,
                out_channels=self.bottle_chans*self.num_replicas,
                kernel_size=4,
                stride=2,
                padding=0),
            nn.LeakyReLU(),
        ).cuda()
        print('bottle', self.bottle)
        
        # self.hidden_dim = 256
        # self.linear_layers = nn.Sequential(
        #     nn.Linear(self.bottle_chans*4*4*4, self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        # ).cuda()
        # print('linear', self.linear_layers)


        self.hidden_dim = 128
        F = 4

        self.pred0 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred1 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred2 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred3 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred4 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred5 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred6 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        self.pred7 = nn.Sequential(nn.Linear(self.bottle_chans*F*F*F, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)).cuda()
        
        # self.pred1 = nn.Sequential(
        #     nn.Linear(self.bottle_chans*4*4*4, self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        # ).cuda()
        # self.pred2 = nn.Sequential(
        #     nn.Linear(self.bottle_chans*4*4*4, self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        # ).cuda()
        # self.pred3 = nn.Sequential(
        #     nn.Linear(self.bottle_chans*4*4*4, self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        # ).cuda()
        print('pred0', self.pred0)
        print('pred1', self.pred1)
        print('pred2', self.pred2)
        print('pred3', self.pred3)
        print('pred4', self.pred4)
        print('pred5', self.pred5)
        print('pred6', self.pred6)
        print('pred7', self.pred7)

        # self.preds = [self.pred0, self.pred1, self.pred2, self.pred3]
        self.preds = [self.pred0,
                      self.pred1,
                      self.pred2,
                      self.pred3,
                      self.pred4,
                      self.pred5,
                      self.pred6,
                      self.pred7]
        assert(len(self.preds)==self.num_replicas)
        
    def forward(self, corrs, occ, iou=None, summ_writer=None):
        # corrs is the set of corr heatmaps output by the matcher
        total_loss = torch.tensor(0.0).cuda()
        
        B, C, Z, Y, X = list(corrs.shape)

        corrs = corrs.detach() # do not backprop into the corrs
        # print('N', N)
        # print('corrs', corrs.shape)

        occ = torch.mean(occ, dim=[2,3,4], keepdims=True).repeat(1, 1, Z, Y, X)
        corrs = torch.cat([corrs, occ], dim=1)
        
        feat = self.bottle(corrs)
        # this is B x C*num_replicas x 4 x 4 x 4
        # print('feat', feat.shape)
        
        feats = feat.reshape(B, self.bottle_chans, self.num_replicas, -1)
        
        # print('feats', feats.shape)
        
        # feat_vec = feat.reshape(B, -1)
        # # print('feat_vec', feat_vec.shape)

        # if no bottle:
        # feat_vec = corrs.reshape(B, -1)
        
        ious = []
        for ind, pred in enumerate(self.preds):
            feat_vec = feats[:,:,ind].reshape(B, -1)
            # print('feat_vec', feat_vec.shape)
            iou_e = pred(feat_vec).reshape(B)

            if iou is not None:
                # print('iou', iou.shape)
                l2_loss = torch.mean(self.mse(iou_e, iou))
                l2_norm = torch.mean(torch.abs(iou_e-iou))
                # print('l2_loss', l2_loss.detach().cpu().numpy())
                # print('l2_norm', l2_norm.detach().cpu().numpy())
                total_loss = utils_misc.add_loss('conf/l2_loss_%d' % ind, total_loss, l2_loss, hyp.conf_coeff, summ_writer)
                utils_misc.add_loss('conf/l2_norm_%d' % ind, 0, l2_norm, 0, summ_writer)

            iou_e = iou_e.clamp(0.0, 0.999) # not 1.0, to distinguish this from hardcoded true/false gt scores
            ious.append(iou_e)
        ious = torch.stack(ious, dim=1)
        # this is B x num_replicas4
        
        return ious, total_loss


        # corrs = corrs.reshape(B, N, Z2, Y2, X2)
        # corrlist = torch.unbind(corrs, dim=1)
        
        # corr = corrlist[0].unsqueeze(1)
        # # xyz_e = utils_track.convert_corr_to_xyz(corr, xyz_offset, hard=False)
        # # conf_loss = self.smoothl1(xyz_e, xyz_g)
        # # total_loss = utils_misc.add_loss('conf/conf_loss', total_loss, conf_loss, hyp.conf_coeff, summ_writer)
        # # rad_e = torch.zeros_like(xyz_e[:,0])

        # rad_e, xyz_e = utils_track.convert_corrlist_to_xyzr(corrlist, radlist, xyz_offset, hard=False)
        # # print('rad_e', rad_e.shape, rad_e.detach().cpu().numpy())
        # # print('rad_g', rad_g.shape, rad_g.detach().cpu().numpy())

        # corrs_ = torch.nn.functional.softmax(corrs.reshape(B, -1)).reshape(B, N, Z2, Y2, X2)
        # conf = utils_samp.bilinear_sample3D(corrs_, (xyz_e - xyz_offset).unsqueeze(1))
        # max_conf = torch.max(conf, dim=1)[0]
        
        # deg_e = utils_geom.rad2deg(rad_e)
        # deg_g = utils_geom.rad2deg(rad_g)

        # # print('deg_e', deg_e.shape, deg_e.detach().cpu().numpy())
        # # print('deg_g', deg_g.shape, deg_g.detach().cpu().numpy())
        
        # # print('rad_g', rad_g.shape)
        # conf_loss = self.smoothl1(xyz_e, xyz_g)
        # conf_loss_r = self.smoothl1(deg_e, deg_g)
        # total_loss = utils_misc.add_loss('conf/conf_loss', total_loss, conf_loss, hyp.conf_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('conf/conf_loss_r', total_loss, conf_loss_r, hyp.conf_r_coeff, summ_writer)
        # # total_loss = utils_misc.add_loss('conf/conf_loss_r', total_loss, conf_loss_r, 0.0, summ_writer)
        
        # # # no rot really
        # # assert(N==1)
        # # template = templatelist[0]
        # # corr, xyz_offset = utils_track.cross_corr_with_template(search_region, template)
        # # xyz_e = utils_track.convert_corr_to_xyz(corr, xyz_offset, hard=False)
        # # conf_loss = self.smoothl1(xyz_e, xyz_g)
        # # total_loss = utils_misc.add_loss('conf/conf_loss', total_loss, conf_loss, hyp.conf_coeff, summ_writer)
        # # rad_e = torch.zeros_like(xyz_e[:,0])
        
        # # print('xyz_e:', xyz_e.detach().cpu().numpy())
        
        # # print('corrlist[0]', corrlist[0].shape)
        # if summ_writer is not None:
        #     summ_writer.summ_feat('conf/input_search_region', search_region, pca=True)
            
        #     if N > 1:
        #         for n in list(range(N)):
        #             summ_writer.summ_feat('conf/input_template_%d' % n, templatelist[n], pca=True)
        #             # summ_writer.summ_oned('conf/corr_%d', torch.mean(corr[:,n:n+1], dim=3)) # reduce the vertical dim
        #             summ_writer.summ_oned('conf/corr_%d' % n, torch.mean(corrlist[n].unsqueeze(1), dim=3)) # reduce the vertical dim
        #     else:
        #         template = templatelist[0]
        #         summ_writer.summ_feat('conf/input_template', template, pca=True)
        #         summ_writer.summ_oned('conf/corr', torch.mean(corr, dim=3)) # reduce the vertical dim

        #     # corrlist_ = [torch.mean(corr, dim=3) for corr in corrlist] # reduce the vertical dim
        #     # summ_writer.summ_oneds('conf/corrlist', corrlist_)
        #     corrlist_ = [corr.unsqueeze(1) for corr in corrlist]
        #     summ_writer.summ_oneds('conf/corrlist', corrlist_, bev=True)

                    

        # if use_window:
        #     z_window = np.reshape(np.hanning(corr.shape[2]), [1, 1, corr.shape[2], 1, 1])
        #     y_window = np.reshape(np.hanning(corr.shape[3]), [1, 1, 1, corr.shape[3], 1])
        #     x_window = np.reshape(np.hanning(corr.shape[4]), [1, 1, 1, 1, corr.shape[4]])
        #     z_window = torch.from_numpy(z_window).float().cuda()
        #     y_window = torch.from_numpy(y_window).float().cuda()
        #     x_window = torch.from_numpy(x_window).float().cuda()
        #     window_weight = 0.25
        #     corr = corr*(1.0-window_weight) + corr*z_window*window_weight
        #     corr = corr*(1.0-window_weight) + corr*y_window*window_weight
        #     corr = corr*(1.0-window_weight) + corr*x_window*window_weight
        # # normalize each corr map, mostly for vis purposes
        # corr = utils_basic.normalize(corr)
            
        # if summ_writer is not None:
        #     summ_writer.summ_oned('conf/corr_windowed', torch.mean(corr, dim=3)) # reduce the vertical dim
            
        # return corrs, rad_e, xyz_e, max_conf, total_loss

