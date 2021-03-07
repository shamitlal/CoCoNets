import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import archs.bottle3D
import archs.sparse_invar_bottle3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_vox
import utils_geom
import utils_basic
import utils_samp

from scipy.special import fresnel
np.set_printoptions(precision=2)
np.random.seed(0)

class MotioncostNet(nn.Module):
    def __init__(self):
        super(MotioncostNet, self).__init__()

        print('MotioncostNet...')

        self.use_cost_vols = False


        if hyp.do_feat3D:
            in_dim = 66
        else:
            in_dim = 10

        self.T_past = 3
        self.T_futu = hyp.S - self.T_past

        self.dense_dim = 32
        self.densifier = archs.sparse_invar_encoder2D.CustomCNN(hyp.Y, self.dense_dim, 3).cuda()
        super_dim = self.dense_dim*self.T_past
        self.motioncoster = archs.encoder2D.Net2D(
            super_dim, mid_chans=64, out_chans=self.T_futu).cuda()

        # self.cost_motioncoster = archs.encoder3D.Net3D(
        #     in_channel=in_dim, pred_dim=out_dim).cuda()
        # self.cost_motioncoster = archs.encoder3D.ResNet3D(
        #     in_channel=in_dim, pred_dim=out_dim).cuda()

        # self.cost_motioncoster = nn.Sequential(
        #     nn.ReplicationPad3d(1),
        #     nn.Conv3d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=0),
        #     nn.BatchNorm3d(num_features=hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.ReplicationPad3d(1),
        #     nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=0),
        #     nn.BatchNorm3d(num_features=hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm3d(num_features=hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm3d(num_features=hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm3d(num_features=hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1),
        # ).cuda()

    def sample_traj(self, initial_vel):
        # accel = np.random.uniform(-0.2, 0.2)
        accel = np.random.uniform(-0.2, 0.2)
        theta = np.random.uniform(-0.1, 0.1)
        offset = np.random.uniform(0.0, 0.4)
        y_coeff = np.random.choice([-1, 1])
        alpha = np.random.uniform(6, 80)
        straight_line = np.random.choice([0, 1])
        # straight_line = False

        random_vel = np.random.choice([0, 1])
        if random_vel:
            initial_vel = np.random.uniform(0.1, 2.0)

        # we already cancelled yaw
        tangent = np.array([1, 0]).reshape(2, 1)
        normal = np.array([0, 1]).reshape(2, 1)

        vel_profile = np.zeros(self.T_futu)
        vel_profile = initial_vel + accel*np.linspace(0, self.T_futu-1, self.T_futu)
        while (initial_vel > 0.01) and (np.any(vel_profile < 0.00)):
            # resample the accel
            # print('resampling accel (was %.2f)' % accel)
            accel = np.random.uniform(-0.2, 0.2)
            vel_profile = initial_vel + accel*np.linspace(0, self.T_futu-1, self.T_futu)
            
        dist = np.zeros(self.T_futu)
        for t in list(range(1, self.T_futu)):
            dist[t] = dist[t-1] + initial_vel + t*accel
        
        # print('accel, theta, offset, y_coeff, alpha, straight_line',
        #       accel, theta, offset, y_coeff, alpha, straight_line)

        r0 = np.reshape(np.array([np.cos(theta), -np.sin(theta)]), (1, 2))
        r1 = np.reshape(np.array([np.sin(theta), np.cos(theta)]), (1, 2))
        rot = np.concatenate([r0,r1], axis=0)
        
        if straight_line:
            curve = np.zeros((self.T_futu, 2))
            curve[:,0] = dist
        else:
            ssa, csa = fresnel(offset+dist/alpha)
            curve = alpha * ((csa.T*tangent).T + (ssa.T*normal).T)
        curve = curve - curve[0]
        curve = (np.dot(rot, curve.T)).T
        curve[:,1] *= y_coeff
        return curve

    def sample_trajs(self, N, clist_cam):
        clist_cam = clist_cam.detach().cpu().numpy()
        clist_cam = np.stack([clist_cam[:,:,0], clist_cam[:,:,2]], axis=2)
        B, S, D = list(clist_cam.shape)
        vel = clist_cam[:,self.T_past-1] - clist_cam[:,self.T_past-2]
        vel = vel[:,0] # just the x part, since we already yaw-corrected

        trajs = np.zeros([B, N, S, 2])
        trajs[:,:,:self.T_past] = np.expand_dims(clist_cam[:,:self.T_past], 1)

        # print('vel', vel)
        for b in list(range(B)):
            for n in list(range(N)):
                traj_futu = self.sample_traj(vel[b])
                # connect it 
                traj_futu = traj_futu + clist_cam[b,self.T_past-1:self.T_past]
                trajs[b,n,self.T_past:] = traj_futu

        # make it 3d 
        trajs = np.stack([trajs[:,:,:,0], np.zeros([B, N, S]), trajs[:,:,:,1]], axis=3)
        trajs = torch.from_numpy(trajs).float().cuda()
        return trajs
                    
    def forward(self, clist_cam, occs, summ_writer, vox_util, suffix=''):
        total_loss = torch.tensor(0.0).cuda()
        B, S, C, Z, Y, X = list(occs.shape)
        B2, S2, D = list(clist_cam.shape)
        assert(B==B2, S==S2)
        assert(D==3)

        if summ_writer.save_this:
            summ_writer.summ_traj_on_occ('motioncost/actual_traj',
                                         clist_cam,
                                         occs[:,self.T_past],
                                         vox_util, 
                                         sigma=2)

        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)
        
        # occs_ = occs.reshape(B*S, C, Z, Y, X)
        occs_ = __p(occs)
        feats_ = occs_.permute(0, 1, 3, 2, 4).reshape(B*S, C*Y, Z, X)
        masks_ = 1.0 - (feats_==0).all(dim=1, keepdim=True).float().cuda()
        halfgrids_ = utils_basic.meshgrid2D(B*S, int(Z/2), int(X/2), stack=True, norm=True).permute(0, 3, 1, 2)
        # feats_ = torch.cat([feats_, grids_], dim=1)
        feats = __u(feats_)
        masks = __u(masks_)
        halfgrids = __u(halfgrids_)
        input_feats = feats[:,:self.T_past]
        input_masks = masks[:,:self.T_past]
        input_halfgrids = halfgrids[:,:self.T_past]
        dense_feats_, _ = self.densifier(__p(input_feats), __p(input_masks), __p(input_halfgrids))
        dense_feats = __u(dense_feats_)
        super_feat = dense_feats.reshape(B, self.T_past*self.dense_dim, int(Z/2), int(X/2))
        cost_maps = self.motioncoster(super_feat)
        cost_maps = F.interpolate(cost_maps, scale_factor=4, mode='bilinear')
        # this is B x T_futu x Z x X
        cost_maps = cost_maps.clamp(-1000, 1000) # raquel says this adds stability
        summ_writer.summ_histogram('motioncost/cost_maps_hist', cost_maps)
        summ_writer.summ_oneds('motioncost/cost_maps', torch.unbind(cost_maps.unsqueeze(2), dim=1))

        # next i need to sample some trajectories
        
        N = hyp.motioncost_num_negs
        sampled_trajs_cam = self.sample_trajs(N, clist_cam)
        # this is B x N x S x 3

        if summ_writer.save_this:
            # for n in list(range(np.min([N, 3]))):
            #     # this is 1 x S x 3
            #     summ_writer.summ_traj_on_occ('motioncost/sample%d_clist' % n,
            #                                  sampled_trajs_cam[0, n].unsqueeze(0),
            #                                  occs[:,self.T_past],
            #                                  # torch.max(occs, dim=1)[0], 
            #                                  # torch.zeros([1, 1, Z, Y, X]).float().cuda(),
            #                                  already_mem=False)
            o = []
            for n in list(range(N)):
                o.append(utils_improc.preprocess_color(
                    summ_writer.summ_traj_on_occ('',
                                                 sampled_trajs_cam[0,n].unsqueeze(0), 
                                                 occs[0:1,self.T_past],
                                                 vox_util,
                                                 only_return=True,
                                                 sigma=0.5)))
            summ_vis = torch.max(torch.stack(o, dim=0), dim=0)[0]
            summ_writer.summ_rgb('motioncost/all_sampled_trajs', summ_vis)

        # smooth loss
        cost_maps_ = cost_maps.reshape(B*self.T_futu, 1, Z, X)
        dz, dx = gradient2D(cost_maps_, absolute=True)
        dt = torch.abs(cost_maps[:,1:]-cost_maps[:,0:-1])
        smooth_spatial = torch.mean(dx+dz, dim=1, keepdims=True)
        smooth_time = torch.mean(dt, dim=1, keepdims=True)
        summ_writer.summ_oned('motioncost/smooth_loss_spatial', smooth_spatial)
        summ_writer.summ_oned('motioncost/smooth_loss_time', smooth_time)
        smooth_loss = torch.mean(smooth_spatial) + torch.mean(smooth_time)
        total_loss = utils_misc.add_loss('motioncost/smooth_loss', total_loss, smooth_loss, hyp.motioncost_smooth_coeff, summ_writer)

        # def clamp_xyz(xyz, X, Y, Z):
        #     x, y, z = torch.unbind(xyz, dim=-1)
        #     x = x.clamp(0, X)
        #     y = x.clamp(0, Y)
        #     z = x.clamp(0, Z)
        #     # if zero_y:
        #     #     y = torch.zeros_like(y)
        #     xyz = torch.stack([x,y,z], dim=-1)
        #     return xyz
        def clamp_xz(xz, X, Z):
            x, z = torch.unbind(xz, dim=-1)
            x = x.clamp(0, X)
            z = x.clamp(0, Z)
            xz = torch.stack([x,z], dim=-1)
            return xz

        clist_mem = utils_vox.Ref2Mem(clist_cam, Z, Y, X)
        # this is B x S x 3

        # sampled_trajs_cam is B x N x S x 3
        sampled_trajs_cam_ = sampled_trajs_cam.reshape(B, N*S, 3)
        sampled_trajs_mem_ = utils_vox.Ref2Mem(sampled_trajs_cam_, Z, Y, X)
        sampled_trajs_mem = sampled_trajs_mem_.reshape(B, N, S, 3)
        # this is B x N x S x 3
        
        xyz_pos_ = clist_mem[:,self.T_past:].reshape(B*self.T_futu, 1, 3)
        xyz_neg_ = sampled_trajs_mem[:,:,self.T_past:].permute(0, 2, 1, 3).reshape(B*self.T_futu, N, 3)
        # get rid of y
        xz_pos_ = torch.stack([xyz_pos_[:,:,0], xyz_pos_[:,:,2]], dim=2)
        xz_neg_ = torch.stack([xyz_neg_[:,:,0], xyz_neg_[:,:,2]], dim=2)
        xz_ = torch.cat([xz_pos_, xz_neg_], dim=1)
        xz_ = clamp_xz(xz_, X, Z)
        cost_maps_ = cost_maps.reshape(B*self.T_futu, 1, Z, X)
        cost_ = utils_samp.bilinear_sample2D(cost_maps_, xz_[:,:,0], xz_[:,:,1]).squeeze(1)
        # cost is B*T_futu x 1+N
        cost_pos = cost_[:,0:1] # B*T_futu x 1
        cost_neg = cost_[:,1:] # B*T_futu x N

        cost_pos = cost_pos.unsqueeze(2) # B*T_futu x 1 x 1
        cost_neg = cost_neg.unsqueeze(1) # B*T_futu x 1 x N

        utils_misc.add_loss('motioncost/mean_cost_pos', 0, torch.mean(cost_pos), 0, summ_writer)
        utils_misc.add_loss('motioncost/mean_cost_neg', 0, torch.mean(cost_neg), 0, summ_writer)
        utils_misc.add_loss('motioncost/mean_margin', 0, torch.mean(cost_neg-cost_pos), 0, summ_writer)

        xz_pos = xz_pos_.unsqueeze(2) # B*T_futu x 1 x 1 x 3
        xz_neg = xz_neg_.unsqueeze(1) # B*T_futu x 1 x N x 3
        dist = torch.norm(xz_pos-xz_neg, dim=3) # B*T_futu x 1 x N
        dist = dist / float(Z) * 5.0 # normalize for resolution, but upweight it a bit
        margin = F.relu(cost_pos - cost_neg + dist)
        margin = margin.reshape(B, self.T_futu, N)
        # mean over time (in the paper this is a sum)
        margin = torch.mean(margin, dim=1)
        # max over the negatives
        maxmargin = torch.max(margin, dim=1)[0] # B
        maxmargin_loss = torch.mean(maxmargin)
        total_loss = utils_misc.add_loss('motioncost/maxmargin_loss', total_loss,
                                         maxmargin_loss, hyp.motioncost_maxmargin_coeff, summ_writer)

        # now let's see some top k
        # we'll do this for the first el of the batch
        cost_neg = cost_neg.reshape(B, self.T_futu, N)[0].detach().cpu().numpy()
        futu_mem = sampled_trajs_mem[:,:,self.T_past:].reshape(B, N, self.T_futu, 3)[0:1]
        cost_neg = np.reshape(cost_neg, [self.T_futu, N])
        cost_neg = np.sum(cost_neg, axis=0)
        inds = np.argsort(cost_neg, axis=0)

        for n in list(range(2)):
            xyzlist_e_mem = futu_mem[0:1,inds[n]]
            xyzlist_e_cam = utils_vox.Mem2Ref(xyzlist_e_mem, Z, Y, X)
            # this is B x S x 3

            if summ_writer.save_this and n==0:
                print('xyzlist_e_cam', xyzlist_e_cam[0:1])
                print('xyzlist_g_cam', clist_cam[0:1,self.T_past:])

            dist = torch.norm(clist_cam[0:1,self.T_past:] - xyzlist_e_cam[0:1], dim=2)
            # this is B x T_futu
            meandist = torch.mean(dist)
            utils_misc.add_loss('motioncost/xyz_dist_%d' % n, 0, meandist, 0, summ_writer)
            
        if summ_writer.save_this:
            # plot the best and worst trajs
            # print('sorted costs:', cost_neg[inds])
            for n in list(range(2)):
                ind = inds[n]
                print('plotting good traj with cost %.2f' % (cost_neg[ind]))
                xyzlist_e_mem = sampled_trajs_mem[:,ind]
                # this is 1 x S x 3
                summ_writer.summ_traj_on_occ('motioncost/best_sampled_traj%d' % n,
                                             xyzlist_e_mem[0:1],
                                             occs[0:1,self.T_past],
                                             vox_util,
                                             already_mem=True,
                                             sigma=2)

            for n in list(range(2)):
                ind = inds[-(n+1)]
                print('plotting bad traj with cost %.2f' % (cost_neg[ind]))
                xyzlist_e_mem = sampled_trajs_mem[:,ind]
                # this is 1 x S x 3
                summ_writer.summ_traj_on_occ('motioncost/worst_sampled_traj%d' % n,
                                             xyzlist_e_mem[0:1],
                                             occs[0:1,self.T_past],
                                             vox_util,
                                             already_mem=True,
                                             sigma=2)

        # xyzlist_e_mem = utils_vox.Ref2Mem(xyzlist_e, Z, Y, X)
        # xyzlist_g_mem = utils_vox.Ref2Mem(xyzlist_g, Z, Y, X)
        # summ_writer.summ_traj_on_occ('motioncost/traj_e',
        #                              xyzlist_e_mem,
        #                              torch.max(occs, dim=1)[0], 
        #                              already_mem=True,
        #                              sigma=2)
        # summ_writer.summ_traj_on_occ('motioncost/traj_g',
        #                              xyzlist_g_mem,
        #                              torch.max(occs, dim=1)[0], 
        #                              already_mem=True,
        #                              sigma=2)
        
        # scorelist_here = scorelist[:,self.num_given:,0]
        # sql2 = torch.sum((vel_g-vel_e)**2, dim=2)

        # ## yes weightmask
        # weightmask = torch.arange(0, self.num_need, dtype=torch.float32, device=torch.device('cuda'))
        # weightmask = torch.exp(-weightmask**(1./4))
        # # 1.0000, 0.3679, 0.3045, 0.2682, 0.2431, 0.2242, 0.2091, 0.1966, 0.1860,
        # #         0.1769, 0.1689, 0.1618, 0.1555, 0.1497, 0.1445, 0.1397, 0.1353
        # weightmask = weightmask.reshape(1, self.num_need)
        # l2_loss = utils_basic.reduce_masked_mean(sql2, scorelist_here * weightmask)
        
        # utils_misc.add_loss('motioncost/l2_loss', 0, l2_loss, 0, summ_writer)
        
        # # # no weightmask:
        # # l2_loss = utils_basic.reduce_masked_mean(sql2, scorelist_here)
        
        # # total_loss = utils_misc.add_loss('motioncost/l2_loss', total_loss, l2_loss, hyp.motioncost_l2_coeff, summ_writer)

        # dist = torch.norm(xyzlist_e - xyzlist_g, dim=2)
        # meandist = utils_basic.reduce_masked_mean(dist, scorelist[:,:,0])
        # utils_misc.add_loss('motioncost/xyz_dist_0', 0, meandist, 0, summ_writer)

        # l2_loss_noexp = utils_basic.reduce_masked_mean(sql2, scorelist_here)
        # # utils_misc.add_loss('motioncost/vel_dist_noexp', 0, l2_loss, 0, summ_writer)
        # total_loss = utils_misc.add_loss('motioncost/l2_loss_noexp', total_loss, l2_loss_noexp, hyp.motioncost_l2_coeff, summ_writer)
        
        
        return total_loss

    
    
    
    
