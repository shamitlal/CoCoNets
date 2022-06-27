import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import hyperparams as hyp
import utils_improc
import utils_misc
import utils_vox
import utils_basic
import utils_geom
import utils_samp
import numpy as np

class TriNet2D(nn.Module):
    def __init__(self):
        super(TriNet2D, self).__init__()

        print('TriNet2D...')
        self.batch_k = 2
        self.num_samples = hyp.tri_2D_num_samples
        assert(self.num_samples > 0)
        self.sampler = utils_misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False)
        self.criterion = utils_misc.MarginLoss() #margin=args.margin,nu=args.nu)
        self.beta = 1.2

        self.dict_len = 20000
        self.neg_pool = utils_misc.SimplePool(self.dict_len)
        self.ce = torch.nn.CrossEntropyLoss()

    def sample_embs(self, emb0, emb1, valid, B, Z, Y, X, mod='', do_vis=False, summ_writer=None):
        if hyp.emb_3D_mindist == 0.0:
            # pure random
            perm = torch.randperm(B*Z*Y*X)
            emb0 = emb0.reshape(B*Z*Y*X, -1)
            emb1 = emb1.reshape(B*Z*Y*X, -1)
            valid = valid.reshape(B*Z*Y*X, -1)
            emb0 = emb0[perm[:self.num_samples*B]]
            emb1 = emb1[perm[:self.num_samples*B]]
            valid = valid[perm[:self.num_samples*B]]
            return emb0, emb1, valid
        else:
            emb0_all = []
            emb1_all = []
            valid_all = []
            for b in list(range(B)):
                sample_indices, sample_locs, sample_valids = utils_misc.get_safe_samples(
                    valid[b], (Z, Y, X), self.num_samples, mode='3D', tol=hyp.emb_3D_mindist)
                emb0_s_ = emb0[b, sample_indices]
                emb1_s_ = emb1[b, sample_indices]
                # these are N x D
                emb0_all.append(emb0_s_)
                emb1_all.append(emb1_s_)
                valid_all.append(sample_valids)

            if do_vis and (summ_writer is not None):
                sample_occ = utils_vox.voxelize_xyz(torch.unsqueeze(sample_locs, dim=0), Z, Y, X, already_mem=True)
                summ_writer.summ_occ('emb3D/samples_%s/sample_occ' % mod, sample_occ, reduce_axes=[2,3])
                summ_writer.summ_occ('emb3D/samples_%s/valid' % mod, torch.reshape(valid, [B, 1, Z, Y, X]), reduce_axes=[2,3])

            emb0_all = torch.cat(emb0_all, axis=0)
            emb1_all = torch.cat(emb1_all, axis=0)
            valid_all = torch.cat(valid_all, axis=0)
            return emb0_all, emb1_all, valid_all
        
    def compute_margin_loss(self, B, C, Z, Y, X, emb_e_vec, emb_g_vec, valid_vec, mod='', do_vis=False, summ_writer=None):
        emb_e_vec, emb_g_vec, valid_vec = self.sample_embs(emb_e_vec,
                                                           emb_g_vec,
                                                           valid_vec,
                                                           B, Z, Y, X,
                                                           mod=mod,
                                                           do_vis=do_vis,
                                                           summ_writer=summ_writer)
        emb_vec = torch.stack((emb_e_vec, emb_g_vec), dim=1).view(B*self.num_samples*self.batch_k, C)
        # this tensor goes e,g,e,g,... on dim 0
        # note this means 2 samples per class; batch_k=2
        y = torch.stack([torch.arange(0,self.num_samples*B), torch.arange(0,self.num_samples*B)], dim=1).view(self.num_samples*B*self.batch_k)
        # this tensor goes 0,0,1,1,2,2,...

        a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)
        margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
        return margin_loss
        
    def compute_ce_loss(self, emb_e_all, emb_g_all):
        N, D = list(emb_e_all.shape)
        perm = np.random.permutation(N)

        emb_e_vec = emb_e_all[perm[:self.num_samples]]
        emb_g_vec = emb_g_all[perm[:self.num_samples]]
        emb_n_vec = emb_g_all[perm[-self.num_samples:]]

        # print('emb_e_vec', emb_e_vec.shape)        
        # print('emb_n_vec', emb_n_vec.shape)
        
        self.neg_pool.update(emb_n_vec.cpu())
        # print('neg_pool len:', len(self.neg_pool))
        emb_n = self.neg_pool.fetch().cuda()

        # N2, C2 = list(negs.shape)
        # assert (C2 == C)
        # l_negs = torch.mm(q.view(N, C), negs.view(C, N2)) # this is N x N2

        emb_q = emb_e_vec.clone()
        emb_k = emb_g_vec.clone()
        # print('emb_q', emb_q.shape)
        # print('emb_k', emb_k.shape)
        # print('emb_n', emb_n.shape)
        N = emb_q.shape[0]
        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))

        # print('l_pos', l_pos.shape)
        l_neg = torch.mm(emb_q, emb_n.T)
        # print('l_neg', l_neg.shape)
        
        l_pos = l_pos.view(N, 1)
        # print('l_pos', l_pos.shape)
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long).cuda()

        temp = 0.07
        emb_loss = self.ce(logits/temp, labels)
        return emb_loss
            
    def forward(self, feat_cam0, feat_cam1, mask_mem0, pix_T_cam0, pix_T_cam1, cam1_T_cam0, vox_util, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(mask_mem0.shape)
        assert(C==1)

        B2, C, H, W = list(feat_cam0.shape)
        assert(B==B2)

        go_slow = True
        go_slow = False
        if go_slow:
            xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
            mask_mem0 = mask_mem0.reshape(B, Z*Y*X)
            vec0_list = []
            vec1_list = []
            for b in list(range(B)):
                xyz_mem0_b = xyz_mem0[b]
                mask_mem0_b = mask_mem0[b]
                xyz_mem0_b = xyz_mem0_b[torch.where(mask_mem0_b > 0)]
                # this is N x 3

                N, D = list(xyz_mem0_b.shape)
                if N > self.num_samples:
                    # to not waste time, i will subsample right here
                    perm = np.random.permutation(N)
                    xyz_mem0_b = xyz_mem0_b[perm[:self.num_samples]]
                    # this is num_samples x 3 (smaller than before)

                xyz_cam0_b = vox_util.Mem2Ref(xyz_mem0_b.unsqueeze(0), Z, Y, X)
                xyz_cam1_b = utils_geom.apply_4x4(cam1_T_cam0[b:b+1], xyz_cam0_b)
                # these are N x 3
                # now, i need to project both of these, and sample from the feats

                xy_cam0_b = utils_geom.apply_pix_T_cam(pix_T_cam0[b:b+1], xyz_cam0_b).squeeze(0)
                xy_cam1_b = utils_geom.apply_pix_T_cam(pix_T_cam1[b:b+1], xyz_cam1_b).squeeze(0)
                # these are N x 2

                vec0 = utils_samp.bilinear_sample_single(feat_cam0[b], xy_cam0_b[:,0], xy_cam0_b[:,1])
                vec1 = utils_samp.bilinear_sample_single(feat_cam1[b], xy_cam1_b[:,0], xy_cam1_b[:,1])
                # these are C x N

                x_pix0 = xy_cam0_b[:,0]
                y_pix0 = xy_cam0_b[:,1]
                x_pix1 = xy_cam1_b[:,0]
                y_pix1 = xy_cam1_b[:,1]
                y_pix0, x_pix0 = utils_basic.normalize_grid2D(y_pix0, x_pix0, H, W)
                y_pix1, x_pix1 = utils_basic.normalize_grid2D(y_pix1, x_pix1, H, W)
                xy_pix0 = torch.stack([x_pix0, y_pix0], axis=1).unsqueeze(0)
                xy_pix1 = torch.stack([x_pix1, y_pix1], axis=1).unsqueeze(0)
                # these are 1 x N x 2
                print('xy_pix0', xy_pix0.shape)
                
                vec0 = F.grid_sample(feat_cam0[b:b+1], xy_pix0)
                vec1 = F.grid_sample(feat_cam1[b:b+1], xy_pix1)
                print('vec0', vec0.shape)

                vec0_list.append(vec0)
                vec1_list.append(vec1)

            vec0 = torch.cat(vec0_list, dim=1).permute(1, 0)
            vec1 = torch.cat(vec1_list, dim=1).permute(1, 0)
        else:
            xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
            mask_mem0 = mask_mem0.reshape(B, Z*Y*X)

            valid_batches = 0
            sampling_coords_mem0 = torch.zeros(B, self.num_samples, 3).float().cuda()
            valid_feat_cam0 = torch.zeros_like(feat_cam0)
            valid_feat_cam1 = torch.zeros_like(feat_cam1)
            valid_pix_T_cam0 = torch.zeros_like(pix_T_cam0)
            valid_pix_T_cam1 = torch.zeros_like(pix_T_cam1)
            valid_cam1_T_cam0 = torch.zeros_like(cam1_T_cam0)
            
            # sampling_coords_mem1 = torch.zeros(B, self.num_samples, 3).float().cuda()
            for b in list(range(B)):
                xyz_mem0_b = xyz_mem0[b]
                mask_mem0_b = mask_mem0[b]
                xyz_mem0_b = xyz_mem0_b[torch.where(mask_mem0_b > 0)]
                # this is N x 3

                N, D = list(xyz_mem0_b.shape)
                if N >= self.num_samples:
                    perm = np.random.permutation(N)
                    xyz_mem0_b = xyz_mem0_b[perm[:self.num_samples]]
                    # this is num_samples x 3 (smaller than before)
                                        
                    valid_batches += 1
                    # sampling_coords_mem0[valid_batches] = xyz_mem0_b
                    
                    sampling_coords_mem0[b] = xyz_mem0_b
                    valid_feat_cam0[b] = feat_cam0[b]
                    valid_feat_cam1[b] = feat_cam1[b]
                    valid_pix_T_cam0[b] = pix_T_cam0[b]
                    valid_pix_T_cam1[b] = pix_T_cam1[b]
                    valid_cam1_T_cam0[b] = cam1_T_cam0[b]

            print('valid_batches:', valid_batches)
            if valid_batches == 0:
                # return early
                return total_loss

            # trim down
            sampling_coords_mem0 = sampling_coords_mem0[:valid_batches]
            feat_cam0 = valid_feat_cam0[:valid_batches]
            feat_cam1 = valid_feat_cam1[:valid_batches]
            pix_T_cam0 = valid_pix_T_cam0[:valid_batches]
            pix_T_cam1 = valid_pix_T_cam1[:valid_batches]
            cam1_T_cam0 = valid_cam1_T_cam0[:valid_batches]
            
            xyz_cam0 = vox_util.Mem2Ref(sampling_coords_mem0, Z, Y, X)
            xyz_cam1 = utils_geom.apply_4x4(cam1_T_cam0, xyz_cam0)
            # these are B x N x 3
            # now, i need to project both of these, and sample from the feats

            xy_cam0 = utils_geom.apply_pix_T_cam(pix_T_cam0, xyz_cam0)
            xy_cam1 = utils_geom.apply_pix_T_cam(pix_T_cam1, xyz_cam1)
            # these are B x N x 2

            vec0 = utils_samp.bilinear_sample2D(feat_cam0, xy_cam0[:,:,0], xy_cam0[:,:,1])
            vec1 = utils_samp.bilinear_sample2D(feat_cam1, xy_cam1[:,:,0], xy_cam1[:,:,1])
            # these are B x C x N

            vec0 = vec0.permute(0, 2, 1).view(valid_batches * self.num_samples, C)
            vec1 = vec1.permute(0, 2, 1).view(valid_batches * self.num_samples, C)
    
        print('vec0', vec0.shape)
        print('vec1', vec1.shape)
        # these are N x C

        # # where g is valid, we use it as reference and pull up e
        # margin_loss = self.compute_margin_loss(B, C, D, H, W, emb_e_vec, emb_g_vec.detach(), vis_g_vec, 'g', True, summ_writer)
        # l2_loss = reduce_masked_mean(sql2_on_axis(emb_e-emb_g.detach(), 1, keepdim=True), vis_g)
        # total_loss = utils_misc.add_loss('emb3D/emb_3D_ml_loss', total_loss, margin_loss, hyp.emb_3D_ml_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('emb3D/emb_3D_l2_loss', total_loss, l2_loss, hyp.emb_3D_l2_coeff, summ_writer)

        ce_loss = self.compute_ce_loss(vec0, vec1.detach())
        total_loss = utils_misc.add_loss('tri2D/emb_ce_loss', total_loss, ce_loss, hyp.tri_2D_ce_coeff, summ_writer)

        # l2_loss_im = torch.mean(sql2_on_axis(emb_e-emb_g, 1, keepdim=True), dim=3)
        # if summ_writer is not None:
        #     summ_writer.summ_oned('emb3D/emb_3D_l2_loss', l2_loss_im)
        #     summ_writer.summ_feats('emb3D/embs_3D', [emb_e, emb_g], pca=True)
        return total_loss

