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

class GRU(torch.nn.Module):
    # reset-able gru cell
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        
    def forward(self, inputs):  #and then in def forward:
        x, (hx, cx) = inputs
        x = x.view(x.size(0), -1)
        hx, cx = self.gru(x, (hx, cx))
        x = hx
        return x, (hx, cx)
                                                                        

class RpoNet(nn.Module):
    def __init__(self):
        super(RpoNet, self).__init__()

        print('RpoNet...')

        self.conv3d = nn.Conv3d(in_channels=(hyp.feat_dim*2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        print('conv3D, [in_channels={}, out_channels={}, ksize={}]'.format(hyp.feat_dim, 1, 1))

        out_dim = 32
        self.compressor = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
        ).cuda()
        # self.rnn_cell = nn.GRU(input_dim, 150, batch_first=True)
        self.rnn_cell = GRU(32, 150)

        # self.past_encoder_cell = nn.GRUCell(32, 150)
        # self.temporal_conv = torch.nn.Conv1d(32, 150, 3)

        self.past_encoder_layer = nn.Linear(2*3, 32).cuda()

    def past_encoder(self, past_states):
        # # encode the past states into a 150-dim feat, using an rnn
        # past_states = self.temporal_conv(past_states)
        # past_encoding = self.past_encoder_cell(past_states)
        # print('past_encoding', past_encoding.shape)
        # # # use encoding at last timestep. (B, T, R) -> (B, R) 
        # past_encoding = past_encoding[:, -1]
        # print('sliced past_encoding', past_encoding.shape)

        print('past encoder')
        print('past_states', past_states.shape)

        B, T_past, C = list(past_states.shape)
        past_states = past_states.reshape(B, T_past*C)
        past_encoding = self.past_encoder_layer(past_states)


        print('past_encoding', past_encoding.shape)
        
        return past_encoding
    
    def prep_sta_feat(self, feat_map, past_encoding):
        B = list(feat_map.shape)[0]
        # as the "static" input,
        # concat the flattened feat_map with the flattened past_encoding
        feat_ = feat_map.reshape(B, -1)
        past_ = past_encoding.reshape(B, -1)
        sta_feat = torch.cat([feat_, past_], dim=1)
        return sta_feat

    def prep_dyn_feat(self, pred_map, traj_past_):
        B, C, Z, Y, X = list(pred_map.shape)
        # as the "dynamic" input, sample from pred_map at each loc in traj_past_

        # xyzs_ = xyzs.permute(1, 0, 2, 3).reshape(B, K*T, 3)
        # scores_ = scores.permute(1, 0, 2).reshape(B, K*T)
        # x, y, z = torch.unbind(xyzs_, dim=2)
        # energy_per_timestep_ = utils_samp.bilinear_sample3D(energy_vol, x, y, z)
        # energy_per_timestep = energy_per_timestep_.reshape(B, K, T)
        

        
        x, y = tf.unstack(traj_past_, axis=2)
        x, y = utils_geom.Velo2Bird2(x, y)
        x *= 0.2 # this should be a hyp, specifying how much smaller the energy map is
        y *= 0.2 # this should be a hyp, specifying how much smaller the energy map is
        feats_along_traj_past = tf.map_fn(utils_geom.bilinear_sample_single, (pred_map, x, y), dtype=tf.float32)
        dyn_feat = tf.reshape(feats_along_traj_past, [B, -1])
        # also, we will concat the actual traj_past_
        traj_feat = tf.reshape(traj_past_, [B, -1])
        dyn_feat = tf.concat([dyn_feat, traj_feat], axis=1)
        return dyn_feat
    
    def sample_forward(self, feat_map, pred_map, noise, traj_past):
        # Compute the x corresponding to z

        # feat_map is B x H_ x W_ x 8 (tiny, for flattening)
        # pred_map is B x H x W x 8 (bigger, for sampling from)
        # noise is B x T x 3 (z in r2p2)
        # traj_past is B x T+1 x 3
        # T indicates how many timesteps to generate

        B, T, D = list(noise.shape)

        # # reset the rnn state
        # h_t = self.rnn_cell.zero_state(B, dtype=tf.float32)
        
        # cx = torch.autograd.Variable(torch.zeros(1, 150))
        # hx = torch.autograd.Variable(torch.zeros(1, 150))
        # # cx = Variable(cx.data)
        # # hx = Variable(hx.data)
    
        # get the static feat
        past_encoding = self.past_encoder(traj_past)
        sta_feat = self.prep_sta_feat(feat_map, past_encoding)

        print('past_encoding', past_encoding.shape)
        print('sta_feat', sta_feat.shape)
        return 0

        # # use traj_past_ as a live history
        # traj_past_ = tf.identity(traj_past)
        # traj_futu_ = []
        # for t_idx in range(T):
        #     print 'timestep %d' % t_idx
        #     # on every timestep we generate mu and sig
        #     # mu is 2 x 1
        #     # sig is 2 x 2

        #     dyn_feat = prep_dyn_feat(pred_map, traj_past_)
        #     rnn_output, h_t = rnn_cell(inputs=dyn_feat, state=h_t)
        #     joint_feat = tf.concat([rnn_output, sta_feat], axis=1)
        #     mu, sig, _, _ = get_mu_sigma(joint_feat, reuse=(reuse or t_idx > 0))

        #     # using mu and sig, do a verlet step to estimate the next location

        #     noise_ = noise[:,t_idx]
        #     # noise_ is B x 2

        #     xdotdot = mu + tf.einsum('bij,bj->bi', sig, noise_) # compute accel; B,2,2 x B,2 -> B,2
        #     x_prev = traj_past_[:,-2] # the second-last el here is the prev pos
        #     x_curr = traj_past_[:,-1] # the last el here is the current pos
        #     x_next = verlet_step(x_curr=x_curr, x_prev=x_prev, xdotdot=xdotdot, dt=1.)
        #     # this is B x 2

        #     traj_futu_.append(x_next)

        #     # recenter traj_past, using the new estimate
        #     traj_past_ = tf.concat([traj_past_[:,1:], tf.expand_dims(x_next, axis=1)], axis=1)
        # traj_futu = tf.stack(traj_futu_, axis=1)
        # # traj_futu is B x T x 2
        # return traj_futu
        

    def forward(self, feat, obj_lrtlist_cams, obj_scorelist_s, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)
        N, B2, S, D = list(obj_lrtlist_cams.shape)
        assert(B==B2)
        # obj_scorelist_s is N x B x S

        obj_lrtlist_cams_ = obj_lrtlist_cams.reshape(N*B, S, 19)
        obj_clist_cam_ = utils_geom.get_clist_from_lrtlist(obj_lrtlist_cams_)
        obj_clist_cam = obj_clist_cam_.reshape(N, B, S, 1, 3)
        # obj_clist_cam is N x B x S x 1 x 3
        obj_clist_cam = obj_clist_cam.squeeze(3)
        
        # # obj_clist_cam is N x B x S x 3
        # clist_cam = obj_clist_cam.reshape(N*B, S, 3)
        # clist_mem = utils_vox.Ref2Mem(clist_cam, Z, Y, X)
        # # this is N*B x S x 3
        # clist_mem = clist_mem.reshape(N, B, S, 3)

        # as with prinet, let's do this for a single object first
        traj_past = obj_clist_cam[0,:,:,:2]
        traj_futu = obj_clist_cam[0,:,:,2:]
        T_past = 2
        T_futu = S-2
        # traj_past is B x T_past x 3
        # traj_futu is B x T_futu x 3

        print('traj_past', traj_past.shape)
        print('traj_futu', traj_futu.shape)

        feat_map = self.compressor(feat)
        pred_map = self.conv3d(feat)
        # these are B x C x Z x Y x X

        # each component of the noise is IID Normal

        ## get K samples
        
        K = 5 # number of samples
        traj_past = traj_past.unsqueeze(0).repeat(K, 1, 1, 1)
        feat_map = feat_map.unsqueeze(0).repeat(K, 1, 1, 1, 1, 1)
        pred_map = pred_map.unsqueeze(0).repeat(K, 1, 1, 1, 1, 1)
        # to sample the K trajectories in parallel, we'll pack K onto the batch dim

        __p = lambda x: utils_basic.pack_seqdim(x, K)
        __u = lambda x: utils_basic.unpack_seqdim(x, K)
        traj_past_ = __p(traj_past)
        feat_map_ = __p(feat_map)
        pred_map_ = __p(pred_map)
        base_sample_ = torch.randn(K*B, T_futu, 3)
        traj_futu_e_ = self.sample_forward(feat_map_, pred_map_, base_sample_, traj_past_)
        # traj_futu_e = __u(traj_futu_e_)
        # # this is K x B x T x 3

        # print('traj_futu_e', traj_futu_e)
        # print(traj_futu_e.shape)
        
        # energy_vol = self.conv3d(feat)
        # # energy_vol is B x 1 x Z x Y x X
        # summ_writer.summ_oned('rpo/energy_vol', torch.mean(energy_vol, dim=3))
        # summ_writer.summ_histogram('rpo/energy_vol_hist', energy_vol)

        # summ_writer.summ_traj_on_occ('traj/obj%d_clist' % k,
        #                              obj_clist, occ_memXs[:,0], already_mem=False)
        

        return total_loss

