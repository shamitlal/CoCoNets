import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D2D as encoder3D2D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_basic
import utils_misc
import utils_geom


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)
    
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens=128, num_residual_layers=3, num_residual_hiddens=64):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        # self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
        #                                         out_channels=num_hiddens,
        #                                         kernel_size=4, 
        #                                         stride=2, padding=1)
        
        # self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens, 
        #                                         out_channels=num_hiddens//2,
        #                                         kernel_size=4, 
        #                                         stride=2, padding=1)
        
        # self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
        #                                         out_channels=out_channels,
        #                                         kernel_size=4, 
        #                                         stride=2, padding=1)

        # self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens, 
        #                                         out_channels=out_channels,
        #                                         kernel_size=4, 
        #                                         stride=2, padding=1)

        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=out_channels,
                                 kernel_size=3, 
                                 stride=1, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        # print(x.shape)
        # x = self._conv_trans_2(x)
        # x = F.relu(x)

        # x = self._conv_trans_3(x)

        x = self._conv_2(x)
        
        return x

class ViewNet(nn.Module):
    def __init__(self):
        super(ViewNet, self).__init__()

        print('ViewNet...')

        self.net = encoder3D2D.Net3D2D(hyp.feat3D_dim, 64, 32, hyp.view_depth, depth_pool=8).cuda()

        self.rgb_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        ).cuda()
        self.emb_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, hyp.feat2D_dim, kernel_size=1, stride=1, padding=0),
        ).cuda()
        print(self.net)

        # # this layer acts as a buffer between quantization and projection
        # self.prep_layer = nn.Sequential(
        #     nn.Conv3d(hyp.feat_dim, (hyp.feat_dim//2), kernel_size=1, stride=1, padding=0),
        #     # nn.ConvTranspose3d(hyp.feat_dim, (hyp.feat_dim//2), kernel_size=2, stride=2, padding=0, output_padding=0),
        #     nn.LeakyReLU(),
        #     # nn.BatchNorm3d(num_features=(hyp.feat_dim//2)),
        # ).cuda()
        
        # # one maxpool along depth
        # self.depth_pool = 8
        # self.pool_layer = nn.MaxPool3d(
        #     [self.depth_pool, 1, 1],
        #     stride=[self.depth_pool, 1, 1],
        #     padding=0,
        #     dilation=1,
        # ).cuda()

        # # (after this, we reshape)

        # # decode this into a picture
        # self.num_residual_layers = 6
        # self.decoder = Decoder(
        #     (hyp.feat_dim//2)*(hyp.view_depth//self.depth_pool),
        #     3,
        #     num_residual_layers=self.num_residual_layers,
        # ).cuda()
        
        # print(self.prep_layer)
        # print(self.decoder)

    def forward(self, pix_T_cam0, cam0_T_cam1, feat_mem1, rgb_g, vox_util, valid=None, summ_writer=None, test=False, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, H, W = list(rgb_g.shape)

        PH, PW = hyp.PH, hyp.PW
        if (PH < H) or (PW < W):
            # print('H, W', H, W)
            # print('PH, PW', PH, PW)
            sy = float(PH)/float(H)
            sx = float(PW)/float(W)
            pix_T_cam0 = utils_geom.scale_intrinsics(pix_T_cam0, sx, sy)

            if valid is not None:
                valid = F.interpolate(valid, scale_factor=0.5, mode='nearest')
            rgb_g = F.interpolate(rgb_g, scale_factor=0.5, mode='bilinear')
            
        # feat_prep = self.prep_layer(feat_mem1)
        # feat_proj = utils_vox.apply_pixX_T_memR_to_voxR(
        #     pix_T_cam0, cam0_T_cam1, feat_prep,
        #     hyp.view_depth, PH, PW)
        feat_proj = vox_util.apply_pixX_T_memR_to_voxR(
            pix_T_cam0, cam0_T_cam1, feat_mem1,
            hyp.view_depth, PH, PW)
            # logspace_slices=(hyp.dataset_name=='carla'))

        # def flatten_depth(feat_3d):
        #     B, C, Z, Y, X = list(feat_3d.shape)
        #     feat_2d = feat_3d.view(B, C*Z, Y, X)
        #     return feat_2d
            
        # feat_pool = self.pool_layer(feat_proj)
        # feat_im = flatten_depth(feat_pool)
        # rgb_e = self.decoder(feat_im)

        feat = self.net(feat_proj)
        rgb = self.rgb_layer(feat)
        emb = self.emb_layer(feat)
        emb = utils_basic.l2_normalize(emb, dim=1)
        
        # feat_im = self.net(feat_proj)
        # if hyp.do_emb2D:
        #     emb_e = self.emb_layer(feat)
        #     # postproc
        #     emb_e = l2_normalize(emb_e, dim=1)
        # else:
        #     emb_e = None

        if test:
            return None, rgb, None
        
        # loss_im = torch.mean(F.mse_loss(rgb, rgb_g, reduction='none'), dim=1, keepdim=True)
        loss_im = utils_basic.l1_on_axis(rgb-rgb_g, 1, keepdim=True)
        if valid is not None:
            rgb_loss = utils_basic.reduce_masked_mean(loss_im, valid)
        else:
            rgb_loss = torch.mean(loss_im)

        total_loss = utils_misc.add_loss('view/rgb_l1_loss', total_loss, rgb_loss, hyp.view_l1_coeff, summ_writer)


        # smooth loss
        dy, dx = utils_basic.gradient2D(rgb, absolute=True)
        smooth_im = torch.mean(dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('view/smooth_loss', smooth_im)
        smooth_loss = torch.mean(smooth_im)
        total_loss = utils_misc.add_loss('view/smooth_loss', total_loss, smooth_loss, hyp.view_smooth_coeff, summ_writer)
            
        

        # vis
        if summ_writer is not None:
            summ_writer.summ_oned('view/rgb_loss', loss_im)
            summ_writer.summ_rgbs('view/rgb', [rgb.clamp(-0.5, 0.5), rgb_g])
            summ_writer.summ_rgb('view/rgb_e', rgb.clamp(-0.5, 0.5))
            summ_writer.summ_rgb('view/rgb_g', rgb_g.clamp(-0.5, 0.5))
            summ_writer.summ_feat('view/emb', emb, pca=True)
            if valid is not None:
                summ_writer.summ_rgb('view/rgb_e_valid', valid*rgb.clamp(-0.5, 0.5))
                summ_writer.summ_rgb('view/rgb_g_valid', valid*rgb_g.clamp(-0.5, 0.5))

        return total_loss, rgb, emb
