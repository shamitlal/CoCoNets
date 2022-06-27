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

class ForecastNet(nn.Module):
    def __init__(self):
        super(ForecastNet, self).__init__()

        print('ForecastNet...')

        self.use_cost_vols = False
        if self.use_cost_vols:

            if hyp.do_feat:
                in_dim = 66
            else:
                in_dim = 10

            hidden_dim = 32
            out_dim = hyp.S

            self.cost_forecaster = archs.encoder3D.Net3D(
                in_channel=in_dim, pred_dim=out_dim).cuda()
            # self.cost_forecaster = archs.encoder3D.ResNet3D(
            #     in_channel=in_dim, pred_dim=out_dim).cuda()

            # self.cost_forecaster = nn.Sequential(
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

            
            library_mod = 'ab'
            traj_library_cam = np.load('../intphys_data/all_trajs_%s.npy' % library_mod)
            # traj_library_cam is L x 100, where L is huge

            # let's make it a bit more huge
            traj_library_cam = np.concatenate([traj_library_cam*0.8,
                                               traj_library_cam*1.0,
                                               traj_library_cam*1.2], axis=0)

            # print('traj_library_cam', traj_library_cam.shape)
            self.L = traj_library_cam.shape[0]

            self.frame_stride = 2
            traj_library_cam = traj_library_cam[:,::self.frame_stride]
            # traj_library_cam is L x M x 3
            self.M = traj_library_cam.shape[1]

            Z, Y, X = hyp.Z, hyp.Y, hyp.X
            Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
            traj_library_cam = torch.from_numpy(traj_library_cam).float().cuda()
            traj_library_cam_ = traj_library_cam.reshape(1, self.L*self.M, 3)
            traj_library_mem_ = utils_vox.Ref2Mem(traj_library_cam_, Z2, Y2, X2)
            traj_library_mem = traj_library_mem_.reshape(self.L, self.M, 3)
            # print('traj_library_mem', traj_library_mem.shape)
            # self.traj_library_mem = traj_library_mem
            self.traj_library_mem = traj_library_mem.detach().cpu().numpy()

            # self.traj_library_mem is L x M x 3
        else:
            self.num_given = 3
            self.num_need = hyp.S - self.num_given
            
            in_dim = 3
            out_dim = self.num_need * 3
            
            # self.regressor = archs.bottle3D.Bottle3D(
            #     in_channel=in_dim, pred_dim=out_dim).cuda()
            self.regressor = archs.sparse_invar_bottle3D.Bottle3D(
                in_channel=in_dim, pred_dim=out_dim).cuda()
            
            
    def sample_trajs_from_library(self, N, trajs_given):

        # trajs_given is B x S x 3
        B, S, D = list(trajs_given.shape)
        assert(D==3)

        trajs_given = trajs_given.detach().cpu().numpy()
        
        # first, trim the library to a manageable size; 
        # this also gives some diversity in training
        assert((self.L/4) > N)
        perm = np.random.permutation(self.L)
        perm = perm[:int(self.L/4)]
        trajs_lib = self.traj_library_mem[perm]
        # trajs_lib is N x M x 3
        trajs_lib = trajs_lib[:,:hyp.S]
        # trajs_lib is N x S x 3
        
        # # center on zero
        # trajs_given = trajs_given - trajs_given[:,0:1]
        # # trajs_given is B x S x 3
        # trajs = trajs - trajs[:,0:1]
        # # trajs is L x S x 3
        
        # # start each traj on the given startpoint
        # trajs_lib = trajs_lib - trajs_lib[:,0:1] + trajs_given[:,0:1]

        trajs_return = np.zeros((B, N, S, 3), np.float32)
        for b in list(range(B)):
            traj_given = trajs_given[b]
            # traj_given is S x 3

            # start the trajs_lib on the given startpoint
            trajs_lib_ = trajs_lib - trajs_lib[:,0:1] + np.expand_dims(traj_given[0:1], 0)

            # sort the trajs by the early match
            second_step_given = np.expand_dims(traj_given[1], 0)
            second_step_lib = trajs_lib_[:,1]
            # these are ? x 3
            dists = np.linalg.norm(second_step_given-second_step_lib, axis=1)
            # this is M
            inds = np.argsort(dists, axis=0)
            # default sort is ascending, which is good for us
            inds = inds[:N]
            trajs_return[b] = trajs_lib_[inds]

        trajs_return = torch.from_numpy(trajs_return).float().cuda()
        return trajs_return

    def forward(self, feats, xyzlist_cam, scorelist, vislist, occs, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()
        B, S, C, Z2, Y2, X2 = list(feats.shape)
        B, S, C, Z, Y, X = list(occs.shape)
        B2, S2, D = list(xyzlist_cam.shape)
        assert(B==B2, S==S2)
        assert(D==3)

        xyzlist_mem = utils_vox.Ref2Mem(xyzlist_cam, Z, Y, X)
        # these are B x S x 3
        scorelist = scorelist.unsqueeze(2)
        # this is B x S x 1
        vislist = vislist[:,0].reshape(B, 1, 1)
        # we only care that the object was visible in frame0
        scorelist = scorelist * vislist

        if self.use_cost_vols:
            if summ_writer.save_this:
                summ_writer.summ_traj_on_occ('forecast/actual_traj',
                                             xyzlist_mem*scorelist,
                                             torch.max(occs, dim=1)[0], 
                                             already_mem=True,
                                             sigma=2)
            

            Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
            Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
            occ_hint0 = utils_vox.voxelize_xyz(xyzlist_cam[:,0:1], Z4, Y4, X4)
            occ_hint1 = utils_vox.voxelize_xyz(xyzlist_cam[:,1:2], Z4, Y4, X4)
            occ_hint0 = occ_hint0*scorelist[:,0].reshape(B, 1, 1, 1, 1)
            occ_hint1 = occ_hint1*scorelist[:,1].reshape(B, 1, 1, 1, 1)
            occ_hint = torch.cat([occ_hint0, occ_hint1], dim=1)
            occ_hint = F.interpolate(occ_hint, scale_factor=4, mode='nearest')
            # this is B x 1 x Z x Y x X
            summ_writer.summ_occ('forecast/occ_hint', (occ_hint0 + occ_hint1).clamp(0,1))

            crops = []
            for s in list(range(S)):
                crop = utils_vox.center_mem_on_xyz(occs_highres[:,s], xyzlist_cam[:,s], Z2, Y2, X2)
                crops.append(crop)
            crops = torch.stack(crops, dim=0)
            summ_writer.summ_occs('forecast/crops', crops)

            # condition on the occ_hint
            feat = torch.cat([feat, occ_hint], dim=1)
            
            N = hyp.forecast_num_negs
            sampled_trajs_mem = self.sample_trajs_from_library(N, xyzlist_mem)

            if summ_writer.save_this:
                for n in list(range(np.min([N, 10]))):
                    xyzlist_mem = sampled_trajs_mem[0, n].unsqueeze(0)
                    # this is 1 x S x 3
                    summ_writer.summ_traj_on_occ('forecast/lib%d_xyzlist' % n,
                                                 xyzlist_mem,
                                                 torch.zeros([1, 1, Z, Y, X]).float().cuda(),
                                                 already_mem=True)
        
            cost_vols = self.cost_forecaster(feat)
            # cost_vols = F.sigmoid(cost_vols)
            cost_vols = F.interpolate(cost_vols, scale_factor=2, mode='trilinear')

            # cost_vols is B x S x Z x Y x X
            summ_writer.summ_histogram('forecast/cost_vols_hist', cost_vols)
            cost_vols = cost_vols.clamp(-1000, 1000) # raquel says this adds stability
            summ_writer.summ_histogram('forecast/cost_vols_clamped_hist', cost_vols)

            cost_vols_vis = torch.mean(cost_vols, dim=3).unsqueeze(2)
            # cost_vols_vis is B x S x 1 x Z x X
            summ_writer.summ_oneds('forecast/cost_vols_vis', torch.unbind(cost_vols_vis, dim=1))

            # smooth loss
            cost_vols_ = cost_vols.reshape(B*S, 1, Z, Y, X)
            dz, dy, dx = gradient3D(cost_vols_, absolute=True)
            dt = torch.abs(cost_vols[:,1:]-cost_vols[:,0:-1])
            smooth_vox_spatial = torch.mean(dx+dy+dz, dim=1, keepdims=True)
            smooth_vox_time = torch.mean(dt, dim=1, keepdims=True)
            summ_writer.summ_oned('forecast/smooth_loss_spatial', torch.mean(smooth_vox_spatial, dim=3))
            summ_writer.summ_oned('forecast/smooth_loss_time', torch.mean(smooth_vox_time, dim=3))
            smooth_loss = torch.mean(smooth_vox_spatial) + torch.mean(smooth_vox_time)
            total_loss = utils_misc.add_loss('forecast/smooth_loss', total_loss, smooth_loss, hyp.forecast_smooth_coeff, summ_writer)

            def clamp_xyz(xyz, X, Y, Z):
                x, y, z = torch.unbind(xyz, dim=-1)
                x = x.clamp(0, X)
                y = x.clamp(0, Y)
                z = x.clamp(0, Z)
                xyz = torch.stack([x,y,z], dim=-1)
                return xyz

            # obj_xyzlist_mem is K x B x S x 3
            # xyzlist_mem is B x S x 3
            # sampled_trajs_mem is B x N x S x 3
            xyz_pos_ = xyzlist_mem.reshape(B*S, 1, 3)
            xyz_neg_ = sampled_trajs_mem.permute(0, 2, 1, 3).reshape(B*S, N, 3)
            # xyz_pos_ = clamp_xyz(xyz_pos_, X, Y, Z)
            # xyz_neg_ = clamp_xyz(xyz_neg_, X, Y, Z)
            xyz_ = torch.cat([xyz_pos_, xyz_neg_], dim=1)
            xyz_ = clamp_xyz(xyz_, X, Y, Z)
            cost_vols_ = cost_vols.reshape(B*S, 1, Z, Y, X)
            x, y, z = torch.unbind(xyz_, dim=2)
            # x = x.clamp(0, X)
            # y = x.clamp(0, Y)
            # z = x.clamp(0, Z)
            cost_ = utils_samp.bilinear_sample3D(cost_vols_, x, y, z).squeeze(1)
            # cost is B*S x 1+N
            cost_pos = cost_[:,0:1] # B*S x 1
            cost_neg = cost_[:,1:] # B*S x N

            cost_pos = cost_pos.unsqueeze(2) # B*S x 1 x 1
            cost_neg = cost_neg.unsqueeze(1) # B*S x 1 x N

            utils_misc.add_loss('forecast/mean_cost_pos', 0, torch.mean(cost_pos), 0, summ_writer)
            utils_misc.add_loss('forecast/mean_cost_neg', 0, torch.mean(cost_neg), 0, summ_writer)
            utils_misc.add_loss('forecast/mean_margin', 0, torch.mean(cost_neg-cost_pos), 0, summ_writer)

            xyz_pos = xyz_pos_.unsqueeze(2) # B*S x 1 x 1 x 3
            xyz_neg = xyz_neg_.unsqueeze(1) # B*S x 1 x N x 3
            dist = torch.norm(xyz_pos-xyz_neg, dim=3) # B*S x 1 x N
            dist = dist / float(Z) * 5.0 # normalize for resolution, but upweight it a bit
            margin = F.relu(cost_pos - cost_neg + dist)
            margin = margin.reshape(B, S, N)
            # mean over time (in the paper this is a sum)
            margin = utils_basic.reduce_masked_mean(margin, scorelist.repeat(1, 1, N), dim=1)
            # max over the negatives
            maxmargin = torch.max(margin, dim=1)[0] # B
            maxmargin_loss = torch.mean(maxmargin)
            total_loss = utils_misc.add_loss('forecast/maxmargin_loss', total_loss,
                                             maxmargin_loss, hyp.forecast_maxmargin_coeff, summ_writer)

            cost_neg = cost_neg.reshape(B, S, N)[0].detach().cpu().numpy()
            sampled_trajs_mem = sampled_trajs_mem.reshape(B, N, S, 3)[0:1]
            cost_neg = np.reshape(cost_neg, [S, N])
            cost_neg = np.sum(cost_neg, axis=0)
            inds = np.argsort(cost_neg, axis=0)

            for n in list(range(2)):

                xyzlist_e_mem = sampled_trajs_mem[0:1,inds[n]]
                xyzlist_e_cam = utils_vox.Mem2Ref(xyzlist_e_mem, Z, Y, X)
                # this is B x S x 3

                # if summ_writer.save_this and n==0:
                #     print('xyzlist_e_cam', xyzlist_e_cam[0:1])
                #     print('xyzlist_g_cam', xyzlist_cam[0:1])
                #     print('scorelist', scorelist[0:1])

                dist = torch.norm(xyzlist_cam[0:1] - xyzlist_e_cam[0:1], dim=2)
                # this is B x S
                meandist = utils_basic.reduce_masked_mean(dist, scorelist[0:1].squeeze(2))
                utils_misc.add_loss('forecast/xyz_dist_%d' % n, 0, meandist, 0, summ_writer)
                # dist = torch.mean(torch.sum(torch.norm(xyzlist_cam[0:1] - xyzlist_e_cam[0:1], dim=2), dim=1))

                # mpe = torch.mean(torch.norm(xyzlist_cam[0:1,int(S/2)] - xyzlist_e_cam[0:1,int(S/2)], dim=1))
                # mpe = utils_basic.reduce_masked_mean(dist, scorelist[0:1])
                # utils_misc.add_loss('forecast/xyz_mpe_%d' % n, 0, dist, 0, summ_writer)

                # epe = torch.mean(torch.norm(xyzlist_cam[0:1,-1] - xyzlist_e_cam[0:1,-1], dim=1))
                # utils_misc.add_loss('forecast/xyz_epe_%d' % n, 0, dist, 0, summ_writer)

            if summ_writer.save_this:
                # plot the best and worst trajs
                # print('sorted costs:', cost_neg[inds])
                for n in list(range(2)):
                    ind = inds[n]
                    # print('plotting good traj with cost %.2f' % (cost_neg[ind]))
                    xyzlist_e_mem = sampled_trajs_mem[:,ind]
                    # this is 1 x S x 3
                    summ_writer.summ_traj_on_occ('forecast/best_sampled_traj%d' % n,
                                                 xyzlist_e_mem,
                                                 torch.max(occs[0:1], dim=1)[0], 
                                                 # torch.zeros([1, 1, Z, Y, X]).float().cuda(),
                                                 already_mem=True,
                                                 sigma=1)

                for n in list(range(2)):
                    ind = inds[-(n+1)]
                    # print('plotting bad traj with cost %.2f' % (cost_neg[ind]))
                    xyzlist_e_mem = sampled_trajs_mem[:,ind]
                    # this is 1 x S x 3
                    summ_writer.summ_traj_on_occ('forecast/worst_sampled_traj%d' % n,
                                                 xyzlist_e_mem,
                                                 torch.max(occs[0:1], dim=1)[0], 
                                                 # torch.zeros([1, 1, Z, Y, X]).float().cuda(),
                                                 already_mem=True,
                                                 sigma=1)
        else:

            # use some timesteps as input
            feat_input = feats[:,:self.num_given].squeeze(2)
            # feat_input is B x self.num_given x ZZ x ZY x ZX

            ## regular bottle3D
            # vel_e = self.regressor(feat_input)

            ## sparse-invar bottle3D
            comp_mask = 1.0 - (feat_input==0).all(dim=1, keepdim=True).float()
            summ_writer.summ_feat('forecast/feat_input', feat_input, pca=False)
            summ_writer.summ_feat('forecast/feat_comp_mask', comp_mask, pca=False)
            vel_e = self.regressor(feat_input, comp_mask)
            
            vel_e = vel_e.reshape(B, self.num_need, 3)
            vel_g = xyzlist_cam[:,self.num_given:] - xyzlist_cam[:,self.num_given-1:-1]

            xyzlist_e = torch.zeros_like(xyzlist_cam)
            xyzlist_g = torch.zeros_like(xyzlist_cam)
            for s in list(range(S)):
                # print('s = %d' % s)
                if s < self.num_given:
                    # print('grabbing from gt ind %s' % s)
                    xyzlist_e[:,s] = xyzlist_cam[:,s]
                    xyzlist_g[:,s] = xyzlist_cam[:,s]
                else:
                    # print('grabbing from s-self.num_given, which is ind %d' % (s-self.num_given))
                    xyzlist_e[:,s] = xyzlist_e[:,s-1] + vel_e[:,s-self.num_given]
                    xyzlist_g[:,s] = xyzlist_g[:,s-1] + vel_g[:,s-self.num_given]

        xyzlist_e_mem = utils_vox.Ref2Mem(xyzlist_e, Z, Y, X)
        xyzlist_g_mem = utils_vox.Ref2Mem(xyzlist_g, Z, Y, X)
        summ_writer.summ_traj_on_occ('forecast/traj_e',
                                     xyzlist_e_mem,
                                     torch.max(occs, dim=1)[0], 
                                     already_mem=True,
                                     sigma=2)
        summ_writer.summ_traj_on_occ('forecast/traj_g',
                                     xyzlist_g_mem,
                                     torch.max(occs, dim=1)[0], 
                                     already_mem=True,
                                     sigma=2)
        
        scorelist_here = scorelist[:,self.num_given:,0]
        sql2 = torch.sum((vel_g-vel_e)**2, dim=2)

        ## yes weightmask
        weightmask = torch.arange(0, self.num_need, dtype=torch.float32, device=torch.device('cuda'))
        weightmask = torch.exp(-weightmask**(1./4))
        # 1.0000, 0.3679, 0.3045, 0.2682, 0.2431, 0.2242, 0.2091, 0.1966, 0.1860,
        #         0.1769, 0.1689, 0.1618, 0.1555, 0.1497, 0.1445, 0.1397, 0.1353
        weightmask = weightmask.reshape(1, self.num_need)
        l2_loss = utils_basic.reduce_masked_mean(sql2, scorelist_here * weightmask)
        
        utils_misc.add_loss('forecast/l2_loss', 0, l2_loss, 0, summ_writer)
        
        # # no weightmask:
        # l2_loss = utils_basic.reduce_masked_mean(sql2, scorelist_here)
        
        # total_loss = utils_misc.add_loss('forecast/l2_loss', total_loss, l2_loss, hyp.forecast_l2_coeff, summ_writer)

        dist = torch.norm(xyzlist_e - xyzlist_g, dim=2)
        meandist = utils_basic.reduce_masked_mean(dist, scorelist[:,:,0])
        utils_misc.add_loss('forecast/xyz_dist_0', 0, meandist, 0, summ_writer)

        l2_loss_noexp = utils_basic.reduce_masked_mean(sql2, scorelist_here)
        # utils_misc.add_loss('forecast/vel_dist_noexp', 0, l2_loss, 0, summ_writer)
        total_loss = utils_misc.add_loss('forecast/l2_loss_noexp', total_loss, l2_loss_noexp, hyp.forecast_l2_coeff, summ_writer)
        
        
        return total_loss

    
