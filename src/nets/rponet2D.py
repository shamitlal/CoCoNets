import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_vox
import utils_geom
import utils_basic
import utils_samp

# class GRU(torch.nn.Module):
#     # reset-able gru cell
#     def __init__(self, input_size, hidden_size):
#         super(GRU, self).__init__()
#         self.gru = nn.GRUCell(input_size, hidden_size)
        
#     def forward(self, x, hx, cx):
#         # x, (hx, cx) = inputs
#         x = x.view(x.size(0), -1)
#         # hx, cx = self.gru(x, (hx, cx))
#         # hx, cx = self.gru(x, (hx, cx))
#         x = hx
#         return x, (hx, cx)
                                                                        
class RpoNet2D(nn.Module):
    def __init__(self):
        super(RpoNet2D, self).__init__()

        print('RpoNet2D...')

        self.T_past = 3
        self.T_futu = hyp.S - self.T_past
        self.rnn_feat_dim = 32

        print('T_past, T_futu = %d, %d' % (self.T_past, self.T_futu))

        self.feat_dim = 32
        # self.compressor = archs.sparse_invar_encoder2D.CustomCNN(16, self.feat_dim, 3).cuda()
        self.compressor = archs.sparse_invar_encoder2D.CustomCNN(16, self.feat_dim, 2).cuda()
        self.conv2d = nn.Conv2d(in_channels=self.feat_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        # mlp_pred_dim = 3 + 3*3 # 2 for mu, 2x2 for sig
        mlp_pred_dim = 2 + 2*2 # 2 for mu, 2x2 for sig
        self.mlp = nn.Sequential(
            nn.Linear(2112, 64), # 2112 is ZX/8 plus something small i think
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, mlp_pred_dim),
        ).cuda()
        
        # self.rnn_cell = nn.GRUCell(self.T_past*3+self.T_past*1, self.rnn_feat_dim)
        self.rnn_cell = nn.GRUCell(self.T_past*2+self.T_past*1, self.rnn_feat_dim)
        # self.past_encoder_layer = nn.Linear(self.T_past*3, 32).cuda()
        self.past_encoder_layer = nn.Linear(self.T_past*2, 32).cuda()

    def add_fake_y(self, xz):
        x, z = torch.unbind(xz, dim=-1)
        y = torch.ones_like(x)
        xyz = torch.stack([x,y,z], dim=-1)
        return xyz
        
    def expm_softclip(self, sig_raw, L=5):
        B, _, _ = list(sig_raw.shape)

        sig_raw_norm = torch.norm(sig_raw.reshape(B, -1), dim=1)
        ones = torch.ones_like(sig_raw_norm)

        # Choice of L imposes a range of values of Sigma = expm(A + A^T)
        # Choose the right base scale and L... (L=1,b=4e-2 is fine), (L=1, b=1 is bad)
        # Can validate choice by training MLE generator and observing if it can overfit as much as desired.
        # If L is too small, Sigma won't be able to produce values very close to zero.
        # If L is too large, search space harder? (<- validate this claim with more experiments)

        denom = torch.logsumexp(torch.stack([sig_raw_norm / L, ones], dim=-1), dim=-1)
        sig_raw = sig_raw / denom.reshape(B, 1, 1)
        return sig_raw

    def forward_log_det_jacobian_from_sigmas_raw(self, sigmas_raw):
        # sigmas_raw is B x T x 2 x 2

        traces = torch.zeros_like(sigmas_raw[:,:,0,0])
        B, T, _, _ = list(sigmas_raw.shape)
        for b in list(range(B)):
            for t in list(range(T)):
                traces[b,t] = torch.trace(sigmas_raw[b,t])

        log_det_sigmas = 2.0 * traces
        # B x T; log det sigma_t = log det expm(A_t + A_t^T) = tr(A_t + A_t^T) = 2 tr(A)

        log_det_J = torch.sum(log_det_sigmas, dim=1)
        # B; log det \prod_{t=1}^T expm(A+A^T)_t = \sum_t log det \sigma_t

        return log_det_J, log_det_sigmas
    
    def compute_inverse_mapping(self, feat_map, pred_map, traj_past, traj_futu):
        # Compute the z (noise-like) corresponding to x (traj)

        # noise is B x 2 (z in r2p2)
        # traj_past is B x T_past x 2
        # traj_futu is B x T_futu x 2
        # T indicates how many timesteps to generate

        B, T_past, D = list(traj_past.shape)
        B, T_futu, D = list(traj_futu.shape)

        # reset the rnn hidden state
        hx = torch.autograd.Variable(torch.zeros(B, self.rnn_feat_dim)).cuda()

        # prep traj_all to make history easily avail
        traj_all = torch.cat([traj_past, traj_futu], dim=1)

        # get the static feat
        past_encoding = self.past_encoder(traj_past)
        sta_feat = self.prep_sta_feat(feat_map, past_encoding)

        # to get element t in traj_futu from traj_all, we offset by T_past
        to_idx = lambda t: (t+T_past)

        z = []
        sigs_raw = []

        for t_idx in range(T_futu):
            # we want to invert each el of traj_futu
            # print('working on t_idx %d' % t_idx)

            x_tm2 = traj_all[:, to_idx(t_idx-2)] # t minus 2
            x_tm1 = traj_all[:, to_idx(t_idx-1)] # t minus 1 
            x_t = traj_all[:, to_idx(t_idx-0)] # t itself

            # we want to generate the mu and sig for tm1,
            # as if we are going to estimate x_t
            past_start = to_idx(t_idx-T_past)
            past_end = to_idx(t_idx)
            # print('indexing in _all from %d to %d' % (past_start, past_end))
            traj_past_ = traj_all[:, past_start:past_end]
            # print('traj_past_', traj_past.shape)

            dyn_feat = self.prep_dyn_feat(pred_map, traj_past_)
            # this is B x T_past*3 (1 from pred_map, 2 from each xz)
            hx = self.rnn_cell(dyn_feat, hx)
            
            joint_feat = torch.cat([hx, sta_feat], dim=1)
            mu_tm1, _, sig_tm1_inv, sig_raw = self.get_mu_sigma(joint_feat)
            sigs_raw.append(sig_raw)

            # use the real x_t (and momentum terms) to obtain the corresponding z_t
            # z_t = tf.einsum('bij,bj->bi', sig_tm1_inv, (x_t - 2.*x_tm1 + x_tm2 - mu_tm1))
            z_t = torch.sum(sig_tm1_inv * (x_t - 2.*x_tm1 + x_tm2 - mu_tm1).unsqueeze(1), dim=2)
            
            z.append(z_t)

        noise_futu = torch.stack(z, dim=1)
        # noise_futu is B x T_futu x 2

        sigs_raw = torch.stack(sigs_raw, dim=1)
        # sigs_raw is B x T_futu x 2 x 2

        return noise_futu, sigs_raw
    
    def log_prob(self, feat_map, pred_map, traj_past, traj_futu):
        # traj_past is B x T+1 x 2
        # traj_futu is B x T x 2
        # returns log prob of each traj_futu, shaped B

        # Compute z = f^{-1}(x)
        z, sigs_raw = self.compute_inverse_mapping(feat_map, pred_map, traj_past, traj_futu)
        # sigs_raw is B x T x 2 x 2; it's each S in matmul(S,S^T)

        base_distribution = torch.distributions.normal.Normal(loc=0, scale=1.)
        base_log_probs = base_distribution.log_prob(z)
        # this is B x T x 2
        base_log_probs = torch.sum(base_log_probs, [1,2])
        # this is B

        log_det_J, log_det_sigmas = self.forward_log_det_jacobian_from_sigmas_raw(sigs_raw)
        # these are B and B x T

        # log q(x) = log (N(0, I) / |df(z)/dz|) = log N(0, I) - \sum_j log |S_j|
        log_qs = base_log_probs - log_det_J
        # this is B

        return log_qs
    
    def compute_loss(self, feat_map, pred_map, traj_past, traj_futu, traj_futu_e, energy_map):
        # traj_past is B x T+1 x 2
        # traj_futu is B x T x 2
        # traj_futu_e is K x B x T x 2
        # energy_map is B x H x W x 1

        K, B, T_futu, D = list(traj_futu_e.shape)

        # ------------
        # forward loss
        # ------------
        # run the actual traj_futu through the net, to get its log prob
        CE_pqs = -1.0*self.log_prob(feat_map, pred_map, traj_past, traj_futu)
        # this is B
        CE_pq = torch.mean(CE_pqs)
        # summ_writer.summ_scalar('rpo/CE_pq', CE_pq.cpu().item())

        # ------------
        # reverse loss
        # ------------
        B, C, Z, X = list(energy_map.shape)
        xyz_cam = traj_futu_e.reshape([B*K, T_futu, 2])
        xyz_mem = utils_vox.Ref2Mem(self.add_fake_y(xyz_cam), Z, 10, X)
        # since we have multiple samples per image, we need to tile up energy_map
        energy_map = energy_map.unsqueeze(0).repeat(K, 1, 1, 1, 1).reshape(K*B, C, Z, X)
        CE_qphat_all = -1.0 * utils_misc.get_traj_loglike(xyz_mem, energy_map)
        # this is B*K x T_futu
        CE_qphat = torch.mean(CE_qphat_all)

        _forward_loss = CE_pq
        _reverse_loss = CE_qphat
        return _forward_loss, _reverse_loss

    def past_encoder(self, past_states):
        # here r2p2 has an rnn; 
        # for now i am just using a linear layer

        B, T_past, C = list(past_states.shape)
        assert(C==2 or C==3) # i expect xz or xyz
        past_states = past_states.reshape(B, T_past*C)
        past_encoding = self.past_encoder_layer(past_states)
        
        return past_encoding
    
    def prep_sta_feat(self, feat_map, past_encoding):
        B = list(feat_map.shape)[0]
        # as the "static" input,
        # concat the flattened feat_map with the flattened past_encoding
        # the feat_map is huge right now, so let's pool it first
        feat_map = F.max_pool2d(feat_map, kernel_size=8, stride=8)
        feat_ = feat_map.reshape(B, -1)
        past_ = past_encoding.reshape(B, -1)
        sta_feat = torch.cat([feat_, past_], dim=1)
        return sta_feat

    def prep_dyn_feat(self, pred_map, traj_past):
        B, C, Z, X = list(pred_map.shape)
        B, T, D = list(traj_past.shape)

        # as the "dynamic" input, we will sample from pred_map at each loc in traj_past

        traj_past_mem = utils_vox.Ref2Mem(self.add_fake_y(traj_past), Z, 10, X)
        x = traj_past_mem[:,:,0]
        z = traj_past_mem[:,:,2]

        feats = utils_samp.bilinear_sample2D(pred_map, x, z)
        # print('sampled these:', feats.shape)
        # this is B x T x C
        dyn_feat = feats.reshape(B, -1)
        # also, we will concat the actual traj_past
        dyn_feat = torch.cat([dyn_feat, traj_past.reshape(B, -1)], axis=1)
        # print('cat the traj itself, and got:', dyn_feat.shape)
        return dyn_feat

    def get_mu_sigma(self, feat):
        # feat is B x T+1 x 3
        # print('getting mu, sigma')
        # print('feat', feat.shape)
        B, C = list(feat.shape)
        pred = self.mlp(feat)
        # pred = pred*0.1 # do not move so fast

        # mu is good to go
        # mu = pred[:,:3]
        mu = pred[:,:2]
        # sig needs some cleaning
        # sig_raw = pred[:,3:].reshape(B, 3, 3)
        sig_raw = pred[:,2:].reshape(B, 2, 2)
        # sig_raw = sig_raw + 1e-6 * utils_geom.eye_3x3(B) # help it be diagonal
        sig_raw = sig_raw + 1e-8 * utils_geom.eye_2x2(B) # help it be diagonal
        sig_raw = self.expm_softclip(sig_raw, L=5)
        
        # print('mu_', sig_, sig_.shape)
        
        # make it positive semi-definite
        # (r2p2 used a matrix exponential to do this step)
        sig = utils_basic.matmul2(sig_raw, sig_raw.permute(0, 2, 1))
        # sig = sig + 1e-4 * utils_geom.eye_3x3(B) # help it be diagonal
        sig_inv = utils_basic.matmul2(-sig_raw, -sig_raw.permute(0, 2, 1))
        # sig_inv = sig.inverse()

        # print('mu', mu.shape)
        # print('sig', sig.shape)
        return mu, sig, sig_inv, sig_raw
    
    def compute_forward_mapping(self, feat_map, pred_map, noise, traj_past):
        # Compute the x (traj) corresponding to z (noise)
    
        def verlet_step(x_curr, x_prev, xdotdot, dt):
            return 2.*x_curr - x_prev + xdotdot * dt
        
        # feat_map is B x H_ x W_ x 8 (tiny, for flattening)
        # pred_map is B x H x W x 8 (bigger, for sampling from)
        # noise is B x T_futu x 3 (z in r2p2)
        # traj_past is B x T_past x 3
        B, T_futu, D = list(noise.shape)

        # print('noise', noise.shape)

        # reset the rnn hidden state
        hx = torch.autograd.Variable(torch.zeros(B, self.rnn_feat_dim)).cuda()
        
        # get the static feat
        past_encoding = self.past_encoder(traj_past)
        sta_feat = self.prep_sta_feat(feat_map, past_encoding)

        # print('past_encoding', past_encoding.shape)
        # print('sta_feat', sta_feat.shape)

        # we will use traj_past_ as a live history
        traj_past_ = traj_past.clone()
        traj_futu_ = []
        for t_idx in range(T_futu):
            # print('timestep %d' % t_idx)
            # on every timestep we generate mu and sig
            # mu is 2 x 1
            # sig is 2 x 2

            dyn_feat = self.prep_dyn_feat(pred_map, traj_past_)
            # this is B x T_past*3 (1 from pred_map, 2 from each xz)
            # print('dyn_feat', dyn_feat.shape)

            hx = self.rnn_cell(dyn_feat, hx)
            # print('hx', hx.shape)
            
            joint_feat = torch.cat([hx, sta_feat], dim=1)
            mu, sig, _, _ = self.get_mu_sigma(joint_feat)

            # using mu and sig, do a verlet step to estimate the next location

            noise_ = noise[:,t_idx]
            # noise_ is B x 3

            # print('noise_', noise_.shape)

            # xdotdot = mu + torch.sum(sig*noise_.reshape(B, 1, 3), dim=1)
            # xdotdot = xdotdot.reshape(B, 3) # this is acceleration
            xdotdot = mu + torch.sum(sig*noise_.reshape(B, 1, 2), dim=1)
            xdotdot = xdotdot.reshape(B, 2) # this is acceleration
            x_prev = traj_past_[:,-2] # prev pos
            x_curr = traj_past_[:,-1] # current pos
            x_next = verlet_step(x_curr=x_curr, x_prev=x_prev, xdotdot=xdotdot, dt=1.)
            # this is B x 3

            traj_futu_.append(x_next)

            # recenter traj_past, using the new estimate
            traj_past_ = torch.cat([traj_past_[:,1:], x_next.unsqueeze(1)], dim=1)

        traj_futu = torch.stack(traj_futu_, dim=1)
        # traj_futu is B x T x 3
        return traj_futu

    def forward(self, clist_cam, energy_map, occ_mems, summ_writer):
        
        total_loss = torch.tensor(0.0).cuda()

        B, S, C, Z, Y, X = list(occ_mems.shape)
        B2, S, D = list(clist_cam.shape)
        assert(B==B2)

        traj_past = clist_cam[:,:self.T_past]
        traj_futu = clist_cam[:,self.T_past:]
        
        # just xz
        traj_past = torch.stack([traj_past[:,:,0], traj_past[:,:,2]], dim=2) # xz
        traj_futu = torch.stack([traj_futu[:,:,0], traj_futu[:,:,2]], dim=2) # xz
        
        feat = occ_mems[:,0].permute(0, 1, 3, 2, 4).reshape(B, C*Y, Z, X)
        mask = 1.0 - (feat==0).all(dim=1, keepdim=True).float().cuda()
        halfgrid = utils_basic.meshgrid2D(B, int(Z/2), int(X/2), stack=True, norm=True).permute(0, 3, 1, 2)
        feat_map, _ = self.compressor(feat, mask, halfgrid)
        pred_map = self.conv2d(feat_map)
        # these are B x C x Z x X

        K = 12 # number of samples
        traj_past = traj_past.unsqueeze(0).repeat(K, 1, 1, 1)
        feat_map = feat_map.unsqueeze(0).repeat(K, 1, 1, 1, 1)
        pred_map = pred_map.unsqueeze(0).repeat(K, 1, 1, 1, 1)
        # to sample the K trajectories in parallel, we'll pack K onto the batch dim
        __p = lambda x: utils_basic.pack_seqdim(x, K)
        __u = lambda x: utils_basic.unpack_seqdim(x, K)
        traj_past_ = __p(traj_past)
        feat_map_ = __p(feat_map)
        pred_map_ = __p(pred_map)
        base_sample_ = torch.randn(K*B, self.T_futu, 2).cuda()
        traj_futu_e_ = self.compute_forward_mapping(feat_map_, pred_map_, base_sample_, traj_past_)
        traj_futu_e = __u(traj_futu_e_)
        # this is K x B x T x 2

        # print('traj_futu_e', traj_futu_e.shape, traj_futu_e[0,0])
        if summ_writer.save_this:
            o = []
            for k in list(range(K)):
                o.append(utils_improc.preprocess_color(
                    summ_writer.summ_traj_on_occ('',
                                                 utils_vox.Ref2Mem(self.add_fake_y(traj_futu_e[k]), Z, Y, X),
                                                 occ_mems[:,0], 
                                                 already_mem=True,
                                                 only_return=True)))
                summ_writer.summ_traj_on_occ('rponet/traj_futu_sample_%d' % k,
                                             utils_vox.Ref2Mem(self.add_fake_y(traj_futu_e[k]), Z, Y, X),
                                             occ_mems[:,0], 
                                             already_mem=True)
                
            mean_vis = torch.max(torch.stack(o, dim=0), dim=0)[0]
            summ_writer.summ_rgb('rponet/traj_futu_e_mean', mean_vis)
            
            summ_writer.summ_traj_on_occ('rponet/traj_futu_g',
                                         utils_vox.Ref2Mem(self.add_fake_y(traj_futu), Z, Y, X),
                                         occ_mems[:,0], 
                                         already_mem=True)
            
        # forward loss: neg logprob of GT samples under the model
        # reverse loss: neg logprob of estim samples under the (approx) GT (i.e., spatial prior)
        forward_loss, reverse_loss = self.compute_loss(feat_map[0], pred_map[0],
                                                       traj_past[0], traj_futu, traj_futu_e,
                                                       energy_map)
        total_loss = utils_misc.add_loss('rpo/forward_loss', total_loss, forward_loss, hyp.rpo2D_forward_coeff, summ_writer)
        total_loss = utils_misc.add_loss('rpo/reverse_loss', total_loss, reverse_loss, hyp.rpo2D_reverse_coeff, summ_writer)

        return total_loss

