import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
# import archs.encoder3D as encoder3D
# import archs.sparse_invar_encoder3D
# import archs.DCCA_sparse_networks_3d
import archs.encoder3d
# import archs.sparse_encoder3D
# import archs.sparse_invar_encoder3D
import utils.geom
import utils.improc
# import utils.vox
import utils.misc
import utils.basic
import utils.py
import numpy as np
import torch.nn.functional as F

EPS = 1e-4
    
class Sigen3dNet(nn.Module):
    def __init__(self):#, input_dim=2, num_layers=15):
        super().__init__()

        self.input_dim = 4
        self.output_dim = 1

        self.num_choices = 64
        chans = 128

        # self.use_grid = False
        # if self.use_grid:
        #     # since we are using a grid as additional input, we add 3 to the input_dim
        #     self.net = archs.DCCA_sparse_networks_3d.SimpleCNN(self.input_dim+3, self.output_dim, chans=chans)
        # else:
        #     self.net = archs.DCCA_sparse_networks_3d.SimpleCNN(self.input_dim, self.output_dim, chans=chans)

        # self.net = archs.sparse_encoder3D.SASparseResNet3D(self.input_dim, self.output_dim, chans=chans)
        # self.net = archs.sparse_invar_encoder3D.Custom3D(self.input_dim, self.output_dim, chans=chans)
        # self.net = archs.sparse_invar_encoder3D.Sharp3D(self.input_dim, self.output_dim, chans=chans)
        # self.net = archs.sparse_invar_encoder3D.Simple3d(self.input_dim, self.output_dim, chans=chans)
        # self.net = archs.sparse_encoder3D.SparseResNet3D(self.input_dim, self.output_dim, chans=chans)
        # self.net = archs.sparse_encoder3D.Simple3d(self.input_dim, chans, self.output_dim)
        self.net = archs.encoder3d.EncoderDecoder3D(in_dim=self.input_dim, out_dim=self.output_dim).cuda()
        print(self.net)

    def embed(self, discrete_vox):
        B, Z, Y, X = list(discrete_vox.shape)
        # utils.py.print_stats('discrete_vox', discrete_vox.cpu().detach().numpy())
        emb = self.embed_dict(discrete_vox.view(-1)).view(B, Z, Y, X, self.emb_dim)
        emb = emb.permute(0, 4, 1, 2, 3)
        if self.use_grid:
            grid = utils.basic.meshgrid3d(B, Z, Y, X, stack=True, norm=True).permute(0, 4, 1, 2, 3).cuda()
            emb = torch.cat([emb, grid], dim=1)
        return emb
    
    def generate_confidence(self, sparse_vox):
        B, C, Z, Y, X = list(sparse_vox.shape)
        assert(C==1)
        
        num_layers = 7
        
        weights = torch.ones(torch.Size([1, 1, 3, 3, 3])).cuda()
        for layer in list(range(num_layers)):
            sparse_vox = F.conv3d(
                sparse_vox,
                weights,
                bias=None,
                stride=1,
                padding=1,
                dilation=1)
        return sparse_vox
        
    def generate_computation_mask(self, sparse_vox):
        B, C, Z, Y, X = list(sparse_vox.shape)
        assert(C==1)
        
        num_layers = 4
        
        weights = torch.ones(torch.Size([1, 1, 3, 3, 3])).cuda()
        for layer in list(range(num_layers)):
            sparse_vox = F.conv3d(
                sparse_vox,
                weights,
                bias=None,
                stride=1,
                padding=1,
                dilation=1)
        sparse_vox = sparse_vox.clamp(0,1)
        return sparse_vox
        
    # def generate_sample(self, B, H, W):
    #     # if sample==None:
    #     sample = torch.zeros(B, H, W).long().cuda()
    #     sample_mask = torch.zeros(B, 1, H, W).float().cuda()
    #     for i in range(H):
    #         for j in range(W):
    #             emb = self.embed(sample)
    #             out, _ = self.net(emb, sample_mask)
    #             probs = F.softmax(out[:, :, i, j]).data
    #             sample[:, i, j] = torch.multinomial(probs, 1).squeeze().data
    #             sample_mask[:, :, i, j] = 1.0
    #     return sample
    
    # def generate_uncond_sample(self, sample, sample_mask, summ_writer=None, mod=''):
    #     # assume the topleft corner is done already
    #     B, H, W = list(sample.shape)
    #     B2, C, H2, W2 = list(sample_mask.shape)
    #     sample = sample.clone()
    #     sample_mask = sample_mask.clone()
    #     assert(B==1)
    #     assert(C==1)
    #     assert(B==B2)
    #     assert(H==H2)
    #     assert(W==W2)
    #     count = 0
        
    #     if summ_writer is not None:
    #         sample_masks = []
    #         sample_masks.append(sample_mask.clone())
            
    #     for i in range(H):
    #         for j in range(W):
    #             if not (i==0 and j==0):
    #                 emb = self.embed(sample)
    #                 out, _ = self.net(emb, sample_mask)
    #                 probs = F.softmax(out[:, :, i, j]).data
    #                 sample[:, i, j] = torch.multinomial(probs, 1).squeeze().data
    #                 sample_mask[:, :, i, j] = 1.0

    #                 if np.mod(count, 80)==0.0 and (summ_writer is not None):
    #                     sample_masks.append(sample_mask.clone())
    #                 count += 1
                    
    #     if summ_writer is not None:
    #         sample_masks.append(sample_mask.clone())
    #         print('saving %d masks as a gif' % len(sample_masks))
    #         summ_writer.summ_oneds('sigen3d/%scond_masks' % mod, sample_masks, bev=True, norm=False)
                    
    #     return sample
    
    def generate_cond_sample(self, sample, sample_mask, stop_after=0, summ_writer=None, speed=1, mod=''):
        B, C, Z, Y, X = list(sample.shape)
        B2, C, Z2, Y2, X2 = list(sample_mask.shape)
        sample = sample.clone()
        sample_mask = sample_mask.clone()
        assert(B==1)
        assert(C==1)
        assert(B==B2)
        assert(Z==Z2)
        assert(Y==Y2)
        assert(X==X2)
        assert(speed>0)

        count = 0

        if summ_writer is not None:
            sample_masks = []
            sample_masks.append(sample_mask.clone())

        if stop_after:
            goal = (torch.sum(sample_mask) + stop_after).clamp(0, Z*Y*X)
        else:
            goal = Z*Y*X
            
        while torch.sum(sample_mask) < goal:
            out, _ = self.net(sample, sample_mask)

            # find the conv confidence in each pixel
            conf = self.generate_confidence(sample_mask)
            
            for l in list(range(speed)):
                # get another confident voxel while we are here

                # do not choose a pixel we've already decided on
                conf = conf * (1.0 - sample_mask)
                # choose the most confident of the rest
                z, y, x = utils.basic.hard_argmax3d(conf)
                z = z[0]
                y = y[0]
                x = x[0]

                # this is "if" is helpful at the very end; 
                # if sample[:,z,y,x]==0.0: # otw what are we doing
                # but let's skip it
                # probs = F.softmax(out[:, :, z, y, x]).data
                # sample[:, :, z, y, x] = torch.multinomial(probs, 1).squeeze().data

                # probs = F.sigmoid(out[:, :, z, y, x]).data
                # sample[:, :, z, y, x] = probs.round()

                # bino = torch.distributions.binomial(logits=out)
                # bern = torch.distributions.bernoulli.Bernoulli(logits=out)
                # sample[:, :, z, y, x] = bern.sample().data
                prob = F.sigmoid(out[:, :, z, y, x]).data
                sample[:, :, z, y, x] = torch.bernoulli(prob).squeeze().data
                sample_mask[:, :, z, y, x] = 1.0

            if np.mod(count, 200)==0.0 and (summ_writer is not None):
                sample_masks.append(sample_mask.clone())
                print('up to %d masks' % len(sample_masks))
            count += 1
                    
        # summ_writer.summ_oned('sigen3d/conf_iter_%06d' % count, conf, norm=True)
        if summ_writer is not None:
            sample_masks.append(sample_mask.clone())
            print('saving %d masks as a gif' % len(sample_masks))
            summ_writer.summ_oneds('sigen3d/%scond_masks' % mod, sample_masks, bev=True, norm=False)
                
        return sample, sample_mask

    def compute_loss(self, pred, occ, free, valid, summ_writer=None):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        if summ_writer is not None:
            summ_writer.summ_oned('sigen3d/prob_loss', loss_vis)

        pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss*0.3 + neg_loss*0.7

        return balanced_loss, loss
    
    def forward(self, vox_input, vox_vis, occ_g=None, free_g=None, drop=False, summ_writer=None):
        B, C, Z, Y, X = list(vox_input.shape)
        total_loss = torch.tensor(0.0).cuda()

        if drop:
            # a random amount should be visible
            coeff = np.random.uniform(0.05, 0.95)
            rand_vis = (torch.rand((B, 1, Z, Y, X)).cuda() > coeff).float()
            vox_vis = vox_vis * rand_vis
            vox_input = vox_input * vox_vis
            
        computation_mask = self.generate_computation_mask(vox_vis)
        vox_output_logits, _ = self.net(vox_input, computation_mask)
        vox_output = F.sigmoid(vox_output_logits)
        # vox_output = vox_output * computation_mask

        if summ_writer is not None:
            summ_writer.summ_feat('sigen3d/vox_input', vox_input, pca=True)
            summ_writer.summ_oned('sigen3d/computation_mask', computation_mask, bev=True, norm=False)
            summ_writer.summ_occ('sigen3d/vox_output', vox_output, reduce_axes=[2,3])
            summ_writer.summ_occ('sigen3d/occ_g', occ_g, reduce_axes=[2,3])
            summ_writer.summ_occ('sigen3d/free_g', free_g)

        if occ_g is not None:
            # print('vox_output', vox_output.shape)
            # print('occ_g', occ_g.shape)
            # computation_mask = F.interpolate(computation_mask, scale_factor=0.5, mode='trilinear')
            label_vis = (occ_g+free_g).clamp(0,1)
            ce_loss, _ = self.compute_loss(vox_output_logits, occ_g, free_g, label_vis*computation_mask, summ_writer)
            total_loss = utils.misc.add_loss('sigen3d/ce_loss', total_loss, ce_loss, hyp.sigen3d_coeff, summ_writer)

            total_loss = utils.misc.add_loss('sigen3d/reg_loss', total_loss, torch.mean(torch.abs(vox_output)), hyp.sigen3d_reg_coeff, summ_writer)
            
            occ_e_binary = vox_output.round()
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g)
            acc_free = utils.basic.reduce_masked_mean(free_match, free_g)
            acc_total = utils.basic.reduce_masked_mean(either_match, either_have)
            acc_bal = (acc_occ + acc_free)*0.5
            if summ_writer is not None:
                summ_writer.summ_scalar('unscaled_sigen3d/acc_occ', acc_occ.cpu().item())
                summ_writer.summ_scalar('unscaled_sigen3d/acc_free', acc_free.cpu().item())
                summ_writer.summ_scalar('unscaled_sigen3d/acc_total', acc_total.cpu().item())
                summ_writer.summ_scalar('unscaled_sigen3d/acc_bal', acc_bal.cpu().item())
        
        return total_loss, vox_output_logits, vox_output
