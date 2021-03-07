import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
st = ipdb.set_trace

import sys
sys.path.append("..")

import archs.encoder3D2D as encoder3D2D
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic

def raw2outputs(raw_feat, raw_occ, z_vals, raw_noise_std=0.1, white_bkg=False):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      raw: [num_rays, num_samples along ray, 4]. Prediction from model.
      z_vals: [num_rays, num_samples along ray]. Integration time.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw_occ, dists):
        return 1.0 - torch.exp(-F.relu(raw_occ) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    # The 'distance' from the last integration time is infinity.
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10*torch.ones_like(dists[:,0:1])], dim=1)
    # this is N_rays x N_samples
    # print('z_vals[0]', z_vals[0].detach().cpu().numpy())
    # print('dists[0]', dists[0,:-1].detach().cpu().numpy())

    # # Multiply each distance by the norm of its corresponding direction ray
    # # to convert to real world distance (accounts for non-unit directions).
    # dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    # # Extract RGB of each sample position along each ray.
    # rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    # feat = F.sigmoid(raw_feat[..., :3]) # [N_rays, N_samples, 3]
    feat = raw_feat

    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_occ.shape).float().cuda() * raw_noise_std

    # print('raw_occ', raw_occ.shape)
    # print('noise', noise.shape)
    
    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha = raw2alpha(raw_occ + noise, dists)  # [N_rays, N_samples]
    # print('alpha[0]', alpha[0])

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # [N_rays, N_samples]
    weights = torch.cumprod(1.-alpha + 1e-10, dim=-1)
    # make it exclusive, and mult by alpha
    weights = alpha * torch.cat([torch.ones_like(weights[:,0:1]), weights[:,:-1]], dim=1)
    # print('weights[0]', weights[0])

    # Computed weighted color of each sample along each ray.
    feat_map = torch.sum(
        weights[..., None].detach() * feat, dim=-2)  # [N_rays, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, axis=-1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error
    acc_map = torch.sum(weights, -1).clamp(0, 1) # clamping added by adam

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkg:
        feat_map = feat_map + (1.-acc_map[..., None])

    return feat_map, acc_map, weights, depth_map


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

        
        # self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
        #                                         out_channels=out_channels,
        #                                         kernel_size=4, 
        #                                         stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=out_channels,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_2(x)
        
        # x = self._conv_trans_1(x)
        # x = F.relu(x)
        
        # x = self._conv_trans_2(x)
        # x = F.relu(x)

        # x = self._conv_trans_3(x)
        return x

class RenderNet(nn.Module):
    def __init__(self):
        super(RenderNet, self).__init__()

        print('RenderNet...')

        # self.prep_layer = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0).cuda()
        # self.resnet = ResidualStack(
        #     in_channels=64,
        #     num_hiddens=64,
        #     num_residual_layers=3,
        #     num_residual_hiddens=64).cuda()
        
        # self.rgb_layer = nn.Conv2d(in_channels=hyp.feat3d_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

        # print(self.prep_layer)
        # print(self.resnet)
        # print(self.rgb_layer)
        
        
        # self.rgb_layer = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

        # num_hiddens = 128
        # num_residual_layers = 3
        # num_residual_hiddens = 64
        # embedding_dim = 64
        # num_embeddings = 512
        # commitment_cost = 0.25
        
        out_channels = 3
        
        # self.decoder = Decoder(
        #     hyp.feat_dim,
        #     out_channels,
        # ).cuda()
            # num_hiddens, 
            # num_residual_layers, 
            # num_residual_hiddens,
            # out_channels)
        
    def accu_render(self, feat, occ):
        B, C, D, H, W = list(feat.shape)
        output = torch.zeros(B, C, H, W).cuda()
        alpha = torch.zeros(B, 1, H, W).cuda()
        for d in list(range(D)):
            contrib = (alpha + occ[:,:,d,:,:]).clamp(max=1.) - alpha
            output += contrib*feat[:,:,d,:,:]
            alpha += contrib
        return output

    def forward(self, feat, occ, z_dist, rgb_g=None, depth_g=None, valid=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat.shape)

        # print('rendernet forward!')
        # print('feat', feat.shape)
        # print('occ', occ.shape)
        # print('z_dist', z_dist.shape)
        # print('rgb_g', rgb_g.shape)
        # print('valid', valid.shape)
        
        # raw = torch.cat([feat, occ], dim=1)
        # # this is B x C+1 x Z x Y x X
        # # i want this to be -1 x Z x C+1
        # raw = raw.permute(0, 3, 4, 2, 1).reshape(B*Y*X, Z, C+1)


        feat = feat.permute(0, 3, 4, 2, 1).reshape(B*Y*X, Z, C)
        occ = occ.permute(0, 3, 4, 2, 1).reshape(B*Y*X, Z)
        
        z_dist = z_dist.reshape(1, -1).repeat(B*Y*X, 1)
        feat_map, acc_map, weights, depth_map = raw2outputs(feat, occ, z_dist)
        # feat_map is B*Y*X x C
        feat_e = feat_map.reshape(B, Y, X, C).permute(0, 3, 1, 2)
        depth_e = depth_map.reshape(B, Y, X, 1).permute(0, 3, 1, 2)
        acc_e = acc_map.reshape(B, Y, X, 1).permute(0, 3, 1, 2)
        # st()
        if summ_writer is not None:
            summ_writer.summ_feat('render/feat_e', feat_e, pca=True)
        
        # feat_e = self.prep_layer(feat_e)
        # feat_e = self.resnet(feat_e)
        # rgb_e = self.rgb_layer(feat_e)
        # rgb_e = F.sigmoid(rgb_e) - 0.5

        # v1, v2:
        rgb_e = F.sigmoid(feat_e) - 0.5

        # # v3: 
        # rgb_e = self.rgb_layer(feat_e)
        # rgb_e = F.sigmoid(rgb_e) - 0.5

        
        # summ_writer.summ_histogram('depth_e', depth_e)
        # utils.basic.print_stats('depth_e', depth_e)

        # summ_writer.summ_histogram('acc_e', acc_e)
        # utils.basic.print_stats('acc_e', acc_e)
        
        
        # rgb_e = self.accu_render(feat, occ)
        # rgb_e = torch.nn.functional.tanh(rgb_e)*0.5

        if rgb_g is not None:
            rgb_loss_im = utils.basic.l1_on_axis(rgb_e-rgb_g, 1, keepdim=True)
            # depth_loss_im = utils.basic.l1_on_axis(depth_e-depth_g, 1, keepdim=True)
            # depth_loss_im = utils.basic.sql2_on_axis(depth_e-depth_g, 1, keepdim=True)
            if valid is not None:
                rgb_loss = utils.basic.reduce_masked_mean(rgb_loss_im, valid)
                # depth_loss = utils.basic.reduce_masked_mean(depth_loss_im, valid)
            total_loss = utils.misc.add_loss('render/rgb_l1_loss', total_loss, rgb_loss, hyp.render_rgb_coeff, summ_writer)
            # total_loss = utils.misc.add_loss('render/depth_l1_loss', total_loss, depth_loss, hyp.render_depth_coeff, summ_writer)
            # total_loss = utils.misc.add_loss('render/rgb_l2_loss', total_loss, rgb_loss, hyp.render_l2_coeff, summ_writer)

            dy, dx = utils.basic.gradient2d(depth_e, absolute=True)
            smooth_im = torch.mean(dy+dx, dim=1, keepdims=True)
            smooth_loss = torch.mean(smooth_im)
            total_loss = utils.misc.add_loss('render/depth_smooth_loss', total_loss, smooth_loss, hyp.render_smooth_coeff, summ_writer)

        # vis
        if summ_writer is not None:
            # occ is B x 1 x hyp.view_depth x PH x PW
            # summ_writer.summ_occs('render/occ', F.sigmoid(occ).unsqueeze(0), reduce_axes=[2])
            rgb_e_ = rgb_e.clamp(-0.5, 0.5)

            # weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
            # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
            weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
            valid_dil = F.conv2d(valid, weights, padding=1).clamp(0, 1)
            
            if rgb_g is not None:
                summ_writer.summ_oned('render/rgb_loss', rgb_loss_im)#*valid)
                summ_writer.summ_oned('render/rgb_loss_valid', rgb_loss_im*valid)
                # summ_writer.summ_oned('render/depth_loss', depth_loss_im)#*valid)
                rgb_g_ = rgb_g.clamp(-0.5, 0.5)
                summ_writer.summ_rgbs('render/rgb', [rgb_e_*valid_dil, rgb_g_*valid_dil])
                # summ_writer.summ_rgbs('render/rgb', [rgb_e_, rgb_g_])
                summ_writer.summ_rgb('render/rgb_g', rgb_g_)
            # summ_writer.summ_rgb('render/rgb_e', rgb_e_)

            summ_writer.summ_rgb('render/rgb_e', rgb_e_*valid_dil)
            # summ_writer.summ_oned('render/depth_e', depth_e*valid, maxval=32.0)
            summ_writer.summ_oned('render/depth_e', depth_e, maxval=32.0)
            summ_writer.summ_oned('render/acc_e', acc_e*valid, norm=False)
        
        return total_loss, rgb_e, feat_e, acc_e

