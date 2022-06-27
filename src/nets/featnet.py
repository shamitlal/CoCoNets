import torch
import torch.nn as nn
import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3D as encoder3D
# if hyp.feat_do_sparse_conv:
#     import archs.sparse_encoder3D as sparse_encoder3D
# if hyp.feat_do_sparse_invar:
#     import archs.sparse_invar_encoder3D as sparse_invar_encoder3D
from utils_basic import *
import archs.v2v 
import utils_geom
# import utils_vox
import utils_misc

EPS = 1e-4
class FeatNet(nn.Module):
    def __init__(self, in_dim=1):
        super(FeatNet, self).__init__()
        
        # if hyp.dataset_name=="intphys":
        #     in_dim = 1 # depth only
        # else:
        #     in_dim = 4
        # in_dim = 1 # depth/occ only
        
        # if hyp.feat_do_sparse_conv:
        #     if hyp.feat_do_resnet:
        #         self.net = sparse_encoder3D.SparseResNet3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        #     else:
        #         self.net = sparse_encoder3D.SparseNet3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        # else:
        #     if hyp.feat_do_resnet:
        #         self.net = encoder3D.ResNet3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        #     elif hyp.feat_do_sparse_invar:
        #         # self.net = sparse_invar_encoder3D.ResNet3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        #         self.net = sparse_invar_encoder3D.Custom3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        #         # self.net = sparse_invar_encoder3D.Net3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        #     else:
        #         self.net = encoder3D.Net3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        # # print(self.net.named_parameters)
        # print(self.net)
        
        self.net = encoder3D.Net3D(in_channel=in_dim, pred_dim=hyp.feat_dim).cuda()
        # self.net = archs.v2v.V2VModel(in_dim, hyp.feat_dim).cuda()
        print(self.net)
        # input()

    def forward(self, feat, summ_writer=None, comp_mask=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)

        if summ_writer is not None:
            summ_writer.summ_feat('feat/feat0_input', feat, pca=False)
        if comp_mask is not None:
            if summ_writer is not None:
                summ_writer.summ_feat('feat/mask_input', comp_mask, pca=False)
        
        if hyp.feat_do_rt:
            # apply a random rt to the feat
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=5.0, t_amount=8.0).cuda()
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=1.0, t_amount=8.0).cuda()
            Y_T_X = utils_geom.get_random_rt(B, r_amount=1.0, t_amount=4.0).cuda()
            feat = utils_vox.apply_4x4_to_vox(Y_T_X, feat)
            if comp_mask is not None:
                comp_mask = utils_vox.apply_4x4_to_vox(Y_T_X, comp_mask)
            if summ_writer is not None:
                summ_writer.summ_feat('feat/feat1_rt', feat, pca=False)

        if hyp.feat_do_flip:
            # randomly flip the input
            flip0 = torch.rand(1)
            flip1 = torch.rand(1)
            flip2 = torch.rand(1)
            if flip0 > 0.5:
                # transpose width/depth (rotate 90deg)
                feat = feat.permute(0,1,4,3,2)
                if comp_mask is not None:
                    comp_mask = comp_mask.permute(0,1,4,3,2)
            if flip1 > 0.5:
                # flip depth
                feat = feat.flip(2)
                if comp_mask is not None:
                    comp_mask = comp_mask.flip(2)
            if flip2 > 0.5:
                # flip width
                feat = feat.flip(4)
                if comp_mask is not None:
                    comp_mask = comp_mask.flip(4)
            if summ_writer is not None:
                summ_writer.summ_feat('feat/feat2_flip', feat, pca=False)
        
        if hyp.feat_do_sparse_conv:
            feat, comp_mask = self.net(feat, comp_mask)
            if summ_writer is not None:
                summ_writer.summ_feat('feat/mask_output', comp_mask, pca=False)
        elif hyp.feat_do_sparse_invar:
            feat, comp_mask = self.net(feat, comp_mask)
        else:
            feat = self.net(feat)

        # smooth loss
        dz, dy, dx = gradient3D(feat, absolute=True)
        smooth_vox = torch.mean(dz+dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('feat/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('feat/smooth_loss', total_loss, smooth_loss, hyp.feat_smooth_coeff, summ_writer)
            
        # feat = l2_normalize(feat, dim=1)
        if summ_writer is not None:
            summ_writer.summ_feat('feat/feat3_out', feat)
        
        if hyp.feat_do_flip:
            if flip2 > 0.5:
                # unflip width
                feat = feat.flip(4)
            if flip1 > 0.5:
                # unflip depth
                feat = feat.flip(2)
            if flip0 > 0.5:
                # untranspose width/depth
                feat = feat.permute(0,1,4,3,2)
            if summ_writer is not None:
                summ_writer.summ_feat('feat/feat4_unflip', feat)
                
        if hyp.feat_do_rt:
            # undo the random rt
            X_T_Y = utils_geom.safe_inverse(Y_T_X)
            feat = utils_vox.apply_4x4_to_vox(X_T_Y, feat)
            if summ_writer is not None:
                summ_writer.summ_feat('feat/feat5_unrt', feat)

        valid_mask = 1.0 - (feat==0).all(dim=1, keepdim=True).float()
        if hyp.feat_do_sparse_conv and (comp_mask is not None):
            valid_mask = valid_mask * comp_mask
        if summ_writer is not None:
            summ_writer.summ_feat('feat/valid_mask', valid_mask, pca=False)
        return feat, valid_mask, total_loss

