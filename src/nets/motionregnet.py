import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import archs.bottle3D
import archs.bottle2D
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

class Motionregnet(nn.Module):
    def __init__(self):
        super(Motionregnet, self).__init__()

        print('Motionregnet...')

        self.K = hyp.motionreg_num_slots
        self.T_past = hyp.motionreg_t_past
        self.T_futu = hyp.motionreg_t_futu
        self.S_cap = self.T_past+self.T_futu
        self.D = 3

        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.mse = torch.nn.MSELoss(reduction='none')

        self.bottle = archs.bottle3D.Bottle3D(
            hyp.feat3D_dim, pred_dim=32, chans=32).cuda()
        # self.bottle = archs.bottle2D.Bottle2D(
        #     hyp.feat3D_dim, pred_dim=32, chans=32).cuda()
        
        # self.hidden_dim = 8192
        # self.hidden_dim = 6912
        # self.hidden_dim = 2304
        # self.hidden_dim = 512
        
        # self.hidden_dim = 256
        self.hidden_dim = 128
        
        self.linear_layers = nn.Sequential(
            # nn.Linear(512 + self.T_past*(self.D+1), self.hidden_dim),
            nn.Linear(1024 + self.T_past*(self.D+1), self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.K*self.T_futu*3),
        ).cuda()

        print('bottle', self.bottle)
        print('linear', self.linear_layers)
        
    def forward(self, feat, clist_past, clist_mask, clist_futu=None, is_test=False, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat.shape)
        B2, S, D = list(clist_past.shape)
        assert(B==B2)
        assert(D==self.D)
        assert(S==self.T_past)

        # clist = clist[:,:self.S_cap] # clip to the length we are ready for
        # _, S, _ = list(clist.shape)
        # if not (S == self.S_cap):
        #     # we won't supervise
        #     have_gt = False
        # else:
        #     have_gt = True
            
        # print('feat', feat.shape)
        # feat = torch.mean(feat, dim=3) # mean along vertical dim
        # print('feat', feat.shape)
        feat_vec = self.bottle(feat)
        # print('feat_vec', feat_vec.shape)
        # clist_past = clist[:,:self.T_past]
        # this is B x T_past x self.D
        # print('clist_past', clist_past[0].detach().cpu().numpy())
        # print('clist_futu', clist_futu[0].detach().cpu().numpy())
        # input()
        bias = clist_past[:,-1].unsqueeze(1)
        # print('bias', bias.detach().cpu().numpy())
        clist_input = clist_past.clone()-clist_mask*bias

        # vel_past = clist_past[:,1:] - clist_past[:,:-1]
        # accel_past = vel_past[:,1:] - vel_past[:,:-1]
        
        # print('clist_input', clist_input[0].detach().cpu().numpy())
        # input()
        
        if hyp.motionreg_dropout and (not is_test):
            block_ind = np.random.randint(self.T_past)
            
            noise = np.random.uniform(low=-0.5, high=0.5, size=[B, self.T_past, 3])
            noise = torch.from_numpy(noise).float().cuda()
            clist_input = clist_input + noise
            
            # this is a value in 0,T_past-1
            # print('dropping up to %d' % block_ind)
            clist_mask[:,:block_ind] = 0.0
            clist_input[:,:block_ind] = 0.0
            # print('clist_input[0]', clist_input[0].detach().cpu().numpy())
            # print('clist_mask[0]', clist_mask[0].detach().cpu().numpy())
            # # clist_input = 
            # dropout_mask = torch.randint(0, self.T_past, clist_past.shape).cuda().float()
            # dropout_mask = torch.randint(0, self.T_past, clist_past.shape).cuda().float()
            # dropout_mask = torch.randint(0, 2, clist_past.shape).cuda().float()
        feat_vec_cat = torch.cat([feat_vec,
                                  clist_input.reshape(B, self.T_past*self.D),
                                  clist_mask.reshape(B, self.T_past*1),
        ], dim=1)
        # print('feat_vec_cat', feat_vec_cat.shape)
        pred = self.linear_layers(feat_vec_cat)

        clist_futus_e = pred.reshape(B, self.K, self.T_futu, self.D)*float(Z) + bias.unsqueeze(1)
        # print('clist_futus_e', clist_futus_e.shape)
        # print('clist_futu', clist_futu.shape)
        # input()

        if clist_futu is not None:
            # print('clist_futu', clist_futu.shape)

            # # clist_futu is B x S x 3
            # vel_futu = clist_futu[:,1:] - clist_futu[:,:-1]
            # accel_futu = vel_futu[:,1:] - vel_futu[:,:-1]
            # clist_futu_ = clist_futu.unsqueeze(1)
            # accel_futu_ = accel_futu.unsqueeze(1)
            # # print('accel_futu_[0,0]', accel_futu_[0,0].detach().cpu().numpy())
            # # print('accel_futus_e[0,0]', accel_futus_e[0,0].detach().cpu().numpy())

            # vel_futu = clist_futu[:,1:] - clist_futu[:,:-1]
            # clist_futu_ = clist_futu.unsqueeze(1)
            # vel_futu_ = vel_futu.unsqueeze(1)

            clist_futu_ = clist_futu.unsqueeze(1)
            
            # weightlist = torch.arange(0, self.T_futu, dtype=torch.float32, device=torch.device('cuda'))
            # weightlist = torch.exp(-weightlist**(1./4))
            # weightlist = weightlist.reshape(1, 1, self.T_futu, 1)
            # # print('weightlist', weightlist)
            # # weightlist = torch.ones_like(weightlist)
            weightlist = torch.linspace(1, 0.1, self.T_futu, dtype=torch.float32).reshape(1, 1, self.T_futu, 1).cuda()

            l1_diff = self.smoothl1(clist_futu_, clist_futus_e)*weightlist
            # this is B x K x T_futu x D
            l1_diff_per_traj = torch.mean(l1_diff, dim=(2, 3))
            # this is B x K
            min_l1_diff = torch.min(l1_diff_per_traj, dim=1)[0]
            max_l1_diff = torch.max(l1_diff_per_traj, dim=1)[0]
            # this is B 
            min_l1_loss = torch.mean(min_l1_diff)
            max_l1_loss = torch.mean(max_l1_diff)
            # this is []
            total_loss = utils_misc.add_loss('motionreg/l1_loss', total_loss, min_l1_loss, hyp.motionreg_l1_coeff, summ_writer)
            # total_loss = utils_misc.add_loss('motionreg/l1_loss_weak', total_loss, max_l1_loss, hyp.motionreg_l1_coeff*hyp.motionreg_weak_coeff, summ_writer)
            total_loss = utils_misc.add_loss('motionreg/l1_loss_weak', total_loss, torch.mean(l1_diff_per_traj), hyp.motionreg_l1_coeff*hyp.motionreg_weak_coeff, summ_writer)

            l2_diff = self.mse(clist_futu_, clist_futus_e)*weightlist
            # this is B x K x T_futu x D
            l2_diff_per_traj = torch.mean(l2_diff, dim=(2, 3))
            # this is B x K
            min_l2_diff = torch.min(l2_diff_per_traj, dim=1)[0]
            max_l2_diff = torch.max(l2_diff_per_traj, dim=1)[0]
            # this is B 
            min_l2_loss = torch.mean(min_l2_diff)
            max_l2_loss = torch.mean(max_l2_diff)
            # this is []
            total_loss = utils_misc.add_loss('motionreg/l2_loss', total_loss, min_l2_loss, hyp.motionreg_l2_coeff, summ_writer)
            # total_loss = utils_misc.add_loss('motionreg/l2_loss_weak', total_loss, max_l2_loss, hyp.motionreg_l2_coeff*hyp.motionreg_weak_coeff, summ_writer)
            total_loss = utils_misc.add_loss('motionreg/l2_loss_weak', total_loss, torch.mean(l2_diff_per_traj), hyp.motionreg_l2_coeff*hyp.motionreg_weak_coeff, summ_writer)

            # clist_past_ = clist_past.unsqueeze(1).repeat(1, self.K, 1, 1)
            # clists_e = torch.cat([clist_past_, clist_futus_e], dim=2)
            # # this is B x K x S x 3
            # vels_e = clists_e[:,:,1:] - clists_e[:,:,:-1]
            # smooth_loss = self.smoothl1(vels_e[:,:,1:], vels_e[:,:,:-1])


            # we can always at least ask that the velocity be similar to what we saw in traj_past
            # this is B x T_past x self.D
            past_vel = torch.mean(torch.norm(clist_past[:,:,1:] - clist_past[:,:,:-1], dim=2), dim=1)
            curr_vel = torch.mean(torch.norm(clist_futus_e[:,:,:,1:] - clist_futus_e[:,:,:,:-1], dim=3), dim=[1,2])
            vel_loss = self.smoothl1(past_vel, curr_vel)
            total_loss = utils_misc.add_loss('motionreg/vel_loss', total_loss, torch.mean(vel_loss), hyp.motionreg_vel_coeff, summ_writer)
            
            clist_past_ = clist_past.unsqueeze(1).repeat(1, self.K, 1, 1)
            clists_e = torch.cat([clist_past_, clist_futus_e], dim=2)
            # this is B x K x S x 3
            vels_e = clists_e[:,:,1:] - clists_e[:,:,:-1]
            smooth_loss = self.smoothl1(vels_e[:,:,1:], vels_e[:,:,:-1])
            total_loss = utils_misc.add_loss('motionreg/smooth_loss', total_loss, torch.mean(smooth_loss), hyp.motionreg_smooth_coeff, summ_writer)
            
            # total_loss = utils_misc.add_loss('motionreg/smooth_loss', total_loss, torch.mean(accel_loss), hyp.motionreg_smooth_coeff, summ_writer)
            # smooth_loss = self.smoothl1(accel_futus_e[:,:,1:], accel_futus_e[:,:,:-1])
            # smooth_loss = self.smoothl1(vel_futus_e[:,:,1:], vel_futus_e[:,:,:-1])
            
            if summ_writer is not None:
                l2_norm = torch.norm(clist_futu_ - clist_futus_e, dim=3)
                # this is B x K x S
                min_l2_norm = torch.min(l2_norm, dim=1)[0]
                # this is B x S
                for s in list(range(self.T_futu-2)):
                    summ_writer.summ_scalar('unscaled_motionreg/l2_on_step_%02d' % s, torch.mean(min_l2_norm[:,s]))
        # if have_gt:
        #     l2_norm = torch.norm(clist.unsqueeze(1) - clists_e, dim=3)
        #     # this is B x K x S
        #     min_l2_norm = torch.min(l2_norm, dim=1)[0]
        #     # this is B x S
        #     for s in list(range(self.S_cap)):
        #         summ_writer.summ_scalar('unscaled_motionreg/l2_on_step_%02d' % s, torch.mean(min_l2_norm[:,s]))
        #         # utils_misc.add_loss('motionreg_details/l2_on_step_%02d' % s, 0, torch.mean(min_l2_diff[:,s]), 0, summ_writer)

        # return total_loss, clists_e
        return total_loss, clist_futus_e
