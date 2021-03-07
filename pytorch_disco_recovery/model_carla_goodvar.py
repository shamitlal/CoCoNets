import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.centernet import CenterNet
from nets.matchnet import MatchNet
from nets.rendernet import RenderNet
from nets.occnet import OccNet
from nets.rgbnet import RgbNet
from nets.bkgnet import BkgNet
from backend import saverloader, inputs

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc
import utils.track

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

# the idea here is to optimize three vars:
# > object
# > bkg
# > object pose
# and minimize viewpred and occ losses in a video

class CARLA_GOODVAR(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaGoodvarModel()
        
        if hyp.do_freeze_feat3d:
            # print('freezing feat3d')
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
            
    # take over go() from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        set_nums = []
        set_names = []
        set_batch_sizes = []
        set_data_formats = []
        set_seqlens = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_data_formats.append(hyp.data_formats[set_name])
                set_seqlens.append(hyp.seqlens[set_name])
                set_names.append(set_name)
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))


        if hyp.train_trajvar or hyp.train_traj_via_scene:
            print('optimizing self.model.traj')
            self.optimizer = torch.optim.Adam([
                {'params': self.model.traj, 'lr': hyp.lr*10.0},
                {'params': self.model.obj, 'lr': hyp.lr*1.0},
            ])
        elif hyp.train_scenevar:
            self.optimizer = torch.optim.Adam([
                # {'params': self.model.parameters(), 'lr': hyp.lr},
                {'params': self.model.bkg, 'lr': hyp.lr*10.0},
                {'params': self.model.obj, 'lr': hyp.lr*10.0},
                # {'params': self.model.heats, 'lr': hyp.lr*10.0},
                # {'params': self.model.bkg_dict, 'lr': hyp.lr*10.0},
                # {'params': self.model.traj, 'lr': hyp.lr*10.0},
            ])
        else:
            print('optimizing model.parameters()')
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters(), 'lr': hyp.lr},
            ])
                
        model_state_dict = self.model.state_dict()
        for k in model_state_dict.keys():
            print('key', k)
        
        self.start_iter = saverloader.load_weights(self.model, None)
        if hyp.latents_init:
            bkg, obj, traj = saverloader.load_latents(hyp.latents_init)
            self.model.bkg.data = torch.from_numpy(bkg).float().cuda()
            self.model.obj.data = torch.from_numpy(obj).float().cuda()
            self.model.traj.data = torch.from_numpy(traj).float().cuda()
            print('initialized latents')
            
        print("------ Done loading weights ------")

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            # reset set_loader after each epoch
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0:
                    set_loaders[i] = iter(set_input)
            for (set_num,
                 set_data_format,
                 set_seqlen,
                 set_name,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 # set_latent_list,
                 # set_latent_optim_list,
                 # set_origin_T_cami_list,
                 set_loader
            ) in zip(
                set_nums,
                set_data_formats,
                set_seqlens,
                set_names,
                set_batch_sizes,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                # set_latent_lists,
                # set_latent_optim_lists,
                # set_origin_T_cami_lists,
                set_loaders
            ):   
                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                
                output_dict = dict()

                if log_this or set_do_backprop or hyp.do_test:
                    # print('%s: set_num %d; set_data_format %s; set_seqlen %s; log_this %d; set_do_backprop %d; ' % (
                    #     set_name, set_num, set_data_format, set_seqlen, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)

                    read_start_time = time.time()
                    if step==1:
                        feed, data_ind = next(set_loader)
                        data_ind = data_ind.detach().cpu().numpy()
                        # print('data_ind', data_ind)
                        
                        feed_cuda = {}
                        for k in feed:
                            try:
                                feed_cuda[k] = feed[k].cuda(non_blocking=True)
                            except:
                                # some things are not tensors (e.g., filename)
                                feed_cuda[k] = feed[k]

                    read_time = time.time() - read_start_time

                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_batch_size'] = set_batch_size

                    iter_start_time = time.time()
                    
                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_py = loss.cpu().item()

                    if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
                        self.optimizer.zero_grad()
                        
                        loss.backward()


                        print('traj grad', self.model.traj.grad)

                        
                        self.optimizer.step()
                        
                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (
                        hyp.name,
                        step,
                        hyp.max_iters,
                        total_time,
                        read_time,
                        iter_time,
                        loss_py,
                        set_name))
                    
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

                ind = set_names.index('train')
                saverloader.save_latents(self.model.bkg.data.detach().cpu().numpy(),
                                         self.model.obj.data.detach().cpu().numpy(),
                                         self.model.traj.data.detach().cpu().numpy(),
                                         self.checkpoint_dir,
                                         step)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

class CarlaGoodvarModel(nn.Module):
    def __init__(self):
        super(CarlaGoodvarModel, self).__init__()
        
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=1)

        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        
        self.bkg = torch.randn(
            [1, 4, int(hyp.Z/1), int(hyp.Y/1), int(hyp.X/1)],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))

        self.obj = torch.randn(
            [1, 4, int(hyp.Z/1), int(hyp.Y/1), int(hyp.X/2)],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))
        # self.obj.data[:,0] = 10.0

        self.heats = torch.randn(
            [1, hyp.S, 1, int(hyp.Z/1), int(hyp.Y/1), int(hyp.X/1)],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))

        self.traj = torch.randn(
            [1, hyp.S, 3],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))
        self.traj_init = torch.randn(
            [1, hyp.S, 3],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))
        
        self.diffs = None
        self.occ_memR0s = None
        self.free_memR0s = None
        self.vis_memR0s = None
        self.occ_memR0_median = None

        
        if hyp.do_render:
            self.rendernet = RenderNet()

        self.crop = (18,18,18)
        # self.crop = (4,4,4)
        # self.crop = (4,4,4)
        # self.crop = (6,6,6)
        # self.crop = (7,7,7)

    def crop_feat(self, feat_pad, crop):
        Z_pad, Y_pad, X_pad = crop
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    
    def pad_feat(self, feat, crop):
        Z_pad, Y_pad, X_pad = crop
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad
        
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        self.global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        # self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_cam0s = feed["origin_T_cam0s"]
        self.origin_T_cam1s = feed["origin_T_cam1s"]
        self.origin_T_cam2s = feed["origin_T_cam2s"]
        self.origin_T_cam3s = feed["origin_T_cam3s"]

        self.cam00s_T_cam0s = utils.geom.get_camM_T_camXs(self.origin_T_cam0s, ind=0)
        self.cam10s_T_cam1s = utils.geom.get_camM_T_camXs(self.origin_T_cam1s, ind=0)
        self.cam20s_T_cam2s = utils.geom.get_camM_T_camXs(self.origin_T_cam2s, ind=0)
        self.cam30s_T_cam3s = utils.geom.get_camM_T_camXs(self.origin_T_cam3s, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_cam0s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam0s)))
        self.camRs_T_cam3s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam3s)))
        self.cam0s_T_camRs = __u(__p(self.camRs_T_cam0s).inverse())
        self.cam3s_T_camRs = __u(__p(self.camRs_T_cam3s).inverse())
        self.cam0s_T_camR0s = __u(torch.matmul(__p(self.cam0s_T_camRs), __p(self.camRs_T_camR0s)))
        self.cam3s_T_camR0s = __u(torch.matmul(__p(self.cam3s_T_camRs), __p(self.camRs_T_camR0s)))
        self.cam0s_T_cam00s = __u(__p(self.cam00s_T_cam0s).inverse())

        self.camRs_T_cam0s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam0s)))
        self.camRs_T_cam1s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam1s)))
        self.camRs_T_cam2s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam2s)))
        self.camRs_T_cam3s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam3s)))

        self.camR0s_T_cam0s = __u(torch.matmul(__p(self.camR0s_T_camRs), __p(self.camRs_T_cam0s)))
        self.camR0s_T_cam1s = __u(torch.matmul(__p(self.camR0s_T_camRs), __p(self.camRs_T_cam1s)))
        self.camR0s_T_cam2s = __u(torch.matmul(__p(self.camR0s_T_camRs), __p(self.camRs_T_cam2s)))
        self.camR0s_T_cam3s = __u(torch.matmul(__p(self.camR0s_T_camRs), __p(self.camRs_T_cam3s)))

        # self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_camRs = feed['rgb_camRs']
        self.rgb_cam0s = feed['rgb_cam0s']
        self.rgb_cam1s = feed['rgb_cam1s']
        self.rgb_cam2s = feed['rgb_cam2s']
        self.rgb_cam3s = feed['rgb_cam3s']
        self.summ_writer.summ_rgbs('inputs/rgb_camRs', self.rgb_camRs.unbind(1))
        self.summ_writer.summ_rgbs('inputs/rgb_cam0s', self.rgb_cam0s.unbind(1))
        self.summ_writer.summ_rgbs('inputs/rgb_cam1s', self.rgb_cam1s.unbind(1))
        self.summ_writer.summ_rgbs('inputs/rgb_cam2s', self.rgb_cam2s.unbind(1))
        self.summ_writer.summ_rgbs('inputs/rgb_cam3s', self.rgb_cam3s.unbind(1))

        self.xyz_camRs = feed["xyz_camRs"]
        self.xyz_cam0s = feed["xyz_cam0s"]
        self.xyz_cam1s = feed["xyz_cam1s"]
        self.xyz_cam2s = feed["xyz_cam2s"]
        self.xyz_cam3s = feed["xyz_cam3s"]

        self.xyz_camRs0 = __u(utils.geom.apply_4x4(__p(self.camRs_T_cam0s), __p(self.xyz_cam0s)))
        self.xyz_camRs1 = __u(utils.geom.apply_4x4(__p(self.camRs_T_cam1s), __p(self.xyz_cam1s)))
        self.xyz_camRs2 = __u(utils.geom.apply_4x4(__p(self.camRs_T_cam2s), __p(self.xyz_cam2s)))
        self.xyz_camRs3 = __u(utils.geom.apply_4x4(__p(self.camRs_T_cam3s), __p(self.xyz_cam3s)))
        self.xyz_camRs_all = torch.cat([self.xyz_camRs,
                                        self.xyz_camRs0,
                                        self.xyz_camRs1,
                                        self.xyz_camRs2,
                                        self.xyz_camRs3,], dim=2)
        self.xyz_camR0s_all = __u(utils.geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs_all)))

        self.occ_memRs0 = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs0), self.Z1, self.Y1, self.X1))
        self.occ_memRs1 = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs1), self.Z1, self.Y1, self.X1))
        self.occ_memRs2 = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs2), self.Z1, self.Y1, self.X1))
        self.occ_memRs3 = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs3), self.Z1, self.Y1, self.X1))
        self.occ_memR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s_all), self.Z1, self.Y1, self.X1))
        # self.occ_halfmemR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s_all), self.Z2, self.Y2, self.X2))
        
        self.summ_writer.summ_occs('3d_feats/occ_memRs0', self.occ_memRs0.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs1', self.occ_memRs1.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs2', self.occ_memRs2.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs3', self.occ_memRs3.unbind(1))
        self.summ_writer.summ_occs('3d_feats/occ_memR0s', self.occ_memR0s.unbind(1))
        
        # i want the points visible from camRs, which are inbounds in camR0s
        self.depth_camRs_, self.valid_camRs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camRs), self.H, self.W)
        self.dense_xyz_camRs_ = utils.geom.depth2pointcloud(self.depth_camRs_, __p(self.pix_T_cams))
        self.dense_xyz_camR0s_ = utils.geom.apply_4x4(__p(self.camR0s_T_camRs), self.dense_xyz_camRs_)
        self.inbound_camRs_ = self.vox_util.get_inbounds(self.dense_xyz_camR0s_, self.Z, self.Y, self.X).float()
        self.inbound_camRs_ = torch.reshape(self.inbound_camRs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camRs = __u(self.depth_camRs_)
        self.valid_camRs = __u(self.valid_camRs_) * __u(self.inbound_camRs_)

        # also, i want the points visible from cam3s, which are inbounds in camR0s
        self.depth_cam3s_, self.valid_cam3s_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_cam3s), self.H, self.W)
        self.dense_xyz_cam3s_ = utils.geom.depth2pointcloud(self.depth_cam3s_, __p(self.pix_T_cams))
        self.dense_xyz_camR0s_ = utils.geom.apply_4x4(__p(self.camR0s_T_cam3s), self.dense_xyz_cam3s_)
        self.inbound_cam3s_ = self.vox_util.get_inbounds(self.dense_xyz_camR0s_, self.Z, self.Y, self.X).float()
        self.inbound_cam3s_ = torch.reshape(self.inbound_cam3s_, [self.B*self.S, 1, self.H, self.W])
        self.depth_cam3s = __u(self.depth_cam3s_)
        self.valid_cam3s = __u(self.valid_cam3s_) * __u(self.inbound_cam3s_)
        
        self.summ_writer.summ_oneds('inputs/depth_camRs', (self.depth_camRs*self.valid_camRs).unbind(1), maxval=32.0)
        self.summ_writer.summ_oneds('inputs/valid_camRs', (self.valid_camRs).unbind(1), norm=False)
        self.summ_writer.summ_oneds('inputs/depth_cam3s', (self.depth_cam3s*self.valid_cam3s).unbind(1), maxval=32.0)
        self.summ_writer.summ_oneds('inputs/valid_cam3s', (self.valid_cam3s).unbind(1), norm=False)
        
        return True # OK

    def train_feat3d(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()
        
        # self.summ_writer.summ_feats('latents/bkgs', self.bkgs.unbind(1), pca=True)
        self.summ_writer.summ_feat('latents/bkg', self.bkg, pca=True)
        self.summ_writer.summ_feat('latents/obj', self.obj, pca=True)
        self.summ_writer.summ_oneds('latents/heats', (self.heats).unbind(1), bev=True)

        s0 = 0
        s1 = int(self.S/2)
        s2 = self.S-1
        ss = [s0, s1, s2]

        if hyp.do_feat3d:
            feat_memR0s_input = self.occ_memR0s
            mask_memR0s = self.occ_memR0s
            
            feat_memR0s_input = __u(self.pad_feat(__p(feat_memR0s_input), [crop*2 for crop in self.crop]))
            mask_memR0s = __u(self.pad_feat(__p(mask_memR0s), [crop*2 for crop in self.crop]))
            
            feat_loss, feat_halfmemR0s_, feat_bunch = self.feat3dnet(
                __p(feat_memR0s_input), mask_input=__p(mask_memR0s), norm=False, summ_writer=None)
            total_loss += feat_loss
            feat_halfmemR0s = __u(feat_halfmemR0s_)
            self.summ_writer.summ_feats('3d_feats/feat_memR0s_inputs', feat_memR0s_input.unbind(1), pca=False)
            self.summ_writer.summ_feats('3d_feats/feat_halfmemR0s', feat_halfmemR0s.unbind(1), pca=True)
        
        if hyp.do_occ:
            occ_memR0s = []
            sup_memR0s = []
            occ_total = 0.0
            # for i, s in enumerate(ss):
            for s in list(range(self.S)):
                
                # scene_memR0 = self.bkgs[:,i]
                # scene_memR0 = F.interpolate(self.pad_feat(feat_halfmemR0s[:,s], self.crop), scale_factor=2, mode='trilinear')
                
                scene_memR0 = F.interpolate(feat_halfmemR0s[:,s], scale_factor=4, mode='trilinear')
                # scene_memR0 = feat_halfmemR0s[:,s]
                
                camR0_T_cam0 = self.camR0s_T_cam0s[:,s]
                camR0_T_cam1 = self.camR0s_T_cam1s[:,s]
                camR0_T_cam2 = self.camR0s_T_cam2s[:,s]
                camR0_T_cam3 = self.camR0s_T_cam3s[:,s]
                xyz_cam0 = self.xyz_cam0s[:,s]
                xyz_cam1 = self.xyz_cam1s[:,s]
                xyz_cam2 = self.xyz_cam2s[:,s]
                xyz_cam3 = self.xyz_cam3s[:,s]

                xyz_camXs = torch.stack([xyz_cam0,
                                         xyz_cam1,
                                         xyz_cam2,	
                                         xyz_cam3], dim=1)
                camR0s_T_camXs = torch.stack([camR0_T_cam0,
                                              camR0_T_cam1,
                                              camR0_T_cam2,
                                              camR0_T_cam3], dim=1)

                occ_memR0_sup, free_memR0_sup, _, _ = self.vox_util.prep_occs_supervision(
                    camR0s_T_camXs,
                    xyz_camXs,
                    self.Z1, self.Y1, self.X1,
                    agg=True)

                # be more conservative with "free"
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                free_memR0_sup = 1.0 - (F.conv3d(1.0 - free_memR0_sup, weights, padding=1)).clamp(0, 1)

                if s==int(self.S/2):
                    summ_writer = self.summ_writer
                else:
                    summ_writer = None

                occ_loss, occ_memR0_pred, loss_memR0 = self.occnet(
                    scene_memR0,
                    occ_memR0_sup,
                    free_memR0_sup,
                    torch.ones_like(free_memR0_sup),
                    summ_writer=summ_writer
                )
                occ_total += occ_loss / float(self.S)
                occ_memR0s.append(occ_memR0_pred)
                any_sup = (occ_memR0_sup + free_memR0_sup).clamp(0,1)
                sup_memR0s.append(any_sup)
            total_loss = utils.misc.add_loss('occ/occ_total', total_loss, occ_total, 1.0, self.summ_writer)
            occ_memR0s = torch.stack(occ_memR0s, dim=1)
            sup_memR0s = torch.stack(sup_memR0s, dim=1)
            
            no_sup_memR0s = 1.0 - sup_memR0s
            # no_sup_mask = no_sup_memR0s[:,1:] * no_sup_memR0s[:,:-1]
            temporal_smooth = self.smoothl1(occ_memR0s[:,1:], occ_memR0s[:,:-1].detach())
            temporal_smooth = utils.basic.reduce_masked_mean(temporal_smooth, no_sup_memR0s[:,1:])
            # temporal_smooth = utils.basic.reduce_masked_mean(temporal_smooth, no_sup_mask)
            total_loss = utils.misc.add_loss('occ/temporal_smooth_loss', total_loss, temporal_smooth, hyp.occ_temporal_coeff, self.summ_writer)

            self.summ_writer.summ_occs('occ/occ_memR0s', [F.sigmoid(occ) for occ in occ_memR0s.unbind(1)])
            # self.summ_writer.summ_oneds('occ/loss_memR0s', loss_memR0s.unbind(1), bev=True)
            
            _, _, _, Z_, Y_, X_ = list(occ_memR0s.shape)
            occ_memR0s_np = (occ_memR0s.detach().cpu().reshape(self.S, -1)).numpy()
            occ_memR0_median_np = np.median(occ_memR0s_np, axis=0)
            occ_memR0_median = torch.from_numpy(occ_memR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()
            self.summ_writer.summ_occ('latents/occ_memR0_median', F.sigmoid(occ_memR0_median))
            diffs = torch.norm(occ_memR0_median.unsqueeze(1) - occ_memR0s, dim=2, keepdim=True)
            self.summ_writer.summ_oneds('latents/occ_memR0_median_diffs', diffs.unbind(1), bev=True, norm=True)


        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
    
    def train_traj(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.summ_writer.summ_feat('latents/bkg', self.bkg, pca=True)
        self.summ_writer.summ_feat('latents/obj', self.obj, pca=True)
        self.summ_writer.summ_oneds('latents/heats', (self.heats).unbind(1), bev=True)

        # print('self.global_step', self.global_step)
        if self.global_step==1:

            if hyp.do_occ:
                occ_memR0s = []
                vis_memR0s = []
                for s in list(range(self.S)):

                    camR0_T_cam0 = self.camR0s_T_cam0s[:,s]
                    camR0_T_cam1 = self.camR0s_T_cam1s[:,s]
                    camR0_T_cam2 = self.camR0s_T_cam2s[:,s]
                    camR0_T_cam3 = self.camR0s_T_cam3s[:,s]
                    xyz_cam0 = self.xyz_cam0s[:,s]
                    xyz_cam1 = self.xyz_cam1s[:,s]
                    xyz_cam2 = self.xyz_cam2s[:,s]
                    xyz_cam3 = self.xyz_cam3s[:,s]

                    xyz_camXs = torch.stack([xyz_cam0,
                                             xyz_cam1,
                                             xyz_cam2,	
                                             xyz_cam3], dim=1)
                    camR0s_T_camXs = torch.stack([camR0_T_cam0,
                                                  camR0_T_cam1,
                                                  camR0_T_cam2,
                                                  camR0_T_cam3], dim=1)

                    occ_memR0_sup, free_memR0_sup, _, _ = self.vox_util.prep_occs_supervision(
                        camR0s_T_camXs,
                        xyz_camXs,
                        self.Z1, self.Y1, self.X1,
                        agg=True)

                    # be more conservative with "free"
                    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                    free_memR0_sup = 1.0 - (F.conv3d(1.0 - free_memR0_sup, weights, padding=1)).clamp(0, 1)

                    vis_memR0 = (occ_memR0_sup + free_memR0_sup).clamp(0, 1)
                    
                    occ_memR0s.append(occ_memR0_sup)
                    vis_memR0s.append(vis_memR0)
                occ_memR0s = torch.stack(occ_memR0s, dim=1)
                vis_memR0s = torch.stack(vis_memR0s, dim=1)

                print('computing median...')
                _, _, _, Z_, Y_, X_ = list(occ_memR0s.shape)
                occ_memR0s_np = (occ_memR0s.detach().cpu().reshape(self.S, -1)).numpy()
                vis_memR0s_np = (vis_memR0s.detach().cpu().reshape(self.S, -1)).numpy()
                occ_memR0_median_np_safe = np.median(occ_memR0s_np, axis=0)
                occ_memR0_median_np = utils.py.reduce_masked_median(
                    occ_memR0s_np.transpose(1, 0), vis_memR0s_np.transpose(1, 0), keep_batch=True)
                occ_memR0_median_np[np.isnan(occ_memR0_median_np)] = occ_memR0_median_np_safe[np.isnan(occ_memR0_median_np)]
                occ_memR0_median = torch.from_numpy(occ_memR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()

                diffs = vis_memR0s*torch.norm((occ_memR0_median.unsqueeze(1) - occ_memR0s), dim=2, keepdim=True)

                # blur out, to eliminate spurious peaks
                diffs = __u(F.interpolate(__p(diffs), scale_factor=0.25, mode='trilinear'))
                diffs = __u(F.interpolate(__p(diffs), scale_factor=4, mode='trilinear'))
                
            print('median done!')

            # next compute the argmaxes of the diffs
            box_vis = []
            xyz_memR0s = []
            xyz_camR0s = []
            lrt_camR0s = []
            for s in list(range(self.S)):
                xyz_offset = torch.zeros([self.B, 3], dtype=torch.float32).cuda()
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                occ_fat = F.conv3d(self.occ_memR0s[:,s], weights, padding=1).clamp(0, 1)
                xyz_mem = utils.track.convert_corr_to_xyz(diffs[:,s]*100.0, xyz_offset, hard=False)
                xyz_cam = self.vox_util.Mem2Ref(xyz_mem.unsqueeze(1), self.Z1, self.Y1, self.X1).squeeze(1)

                xyz_memR0s.append(xyz_mem)
                xyz_camR0s.append(xyz_cam)

                # box = torch.zeros((self.B, 9), dtype=torch.float32).cuda()
                rs = torch.zeros((self.B, 3), dtype=torch.float32).cuda()
                ls = 3.0*torch.ones((self.B, 3), dtype=torch.float32).cuda()
                ls[:,0] = 2.5
                ls[:,1] = 2.0
                ls[:,2] = 5.0
                box_cam = torch.cat([xyz_cam, ls, rs], dim=1)
                lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(box_cam.unsqueeze(1))
                lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam, self.Z1, self.Y1, self.X1)

                lrt_camR0s.append(lrtlist_cam.squeeze(1))

                box_vis.append(self.summ_writer.summ_lrtlist_bev(
                    '',
                    # utils.basic.normalize(occ_fat*self.heats[0:1,s]),
                    utils.basic.normalize(diffs[:,s]),
                    # utils.basic.normalize(loss_memR0s[0:1,s]),
                    lrtlist_mem[0:1],
                    torch.ones(1,1).float().cuda(), # scores
                    torch.ones(1,1).long().cuda(), # tids
                    self.vox_util,
                    already_mem=True,
                    only_return=True))

            self.summ_writer.summ_rgbs('latents/boxes_init', box_vis)

            xyz_memR0s = torch.stack(xyz_memR0s, dim=1)
            xyz_camR0s = torch.stack(xyz_camR0s, dim=1)
            lrt_camR0s = torch.stack(lrt_camR0s, dim=1)
            # these are B x S x 3

            self.traj.data = xyz_camR0s.clone()
            self.traj_init.data = xyz_camR0s.clone()

            self.diffs = diffs
            self.occ_memR0_median = occ_memR0_median
        # endif global_step==1

        traj_median = np.median(self.traj_init.data.detach().cpu().numpy(), axis=1)
        self.traj.data[:,int(self.S/2)] = torch.from_numpy(traj_median).float().cuda()

        self.summ_writer.summ_occ('latents/occ_memR0_median', F.sigmoid(self.occ_memR0_median))
        self.summ_writer.summ_oneds('latents/occ_memR0_median_diffs', self.diffs.unbind(1), bev=True, norm=True)

        # traj_prevprev = self.traj[:,:-2]
        # traj_prev = self.traj[:,1:-1]
        # traj_curr = self.traj[:,2:]
        # traj_elastic_loss = torch.mean(torch.sum(torch.abs(traj_curr - 2*traj_prev + traj_prevprev), dim=2))
        # total_loss = utils.misc.add_loss('latents/traj_elastic', total_loss, traj_elastic_loss, hyp.latent_traj_elastic_coeff, self.summ_writer)

        # xyz_mem = self.vox_util.Ref2Mem(self.traj, self.Z1, self.Y1, self.X1)
        # # this is B x S x 3
        # difflist_ = utils.samp.bilinear_sample3d(__p(self.diffs), __p(xyz_mem).unsqueeze(1))
        # # this is B*S x 1 x 1
        # difflist = __u(difflist_).squeeze(2)
        # # this is B x S x 1

        # # i want the traj to stay on top of diff signals
        # traj_diff_loss = torch.mean(torch.exp(-difflist))
        # total_loss = utils.misc.add_loss('latents/traj_diff', total_loss, traj_diff_loss, hyp.latent_traj_diff_coeff, self.summ_writer)

        box_vis = []
        xyz_memR0s = []
        xyz_camR0s = []
        lrt_camR0s = []
        for s in list(range(self.S)):
            # box = torch.zeros((self.B, 9), dtype=torch.float32).cuda()
            rs = torch.zeros((self.B, 3), dtype=torch.float32).cuda()
            ls = 3.0*torch.ones((self.B, 3), dtype=torch.float32).cuda()
            ls[:,0] = 2.5
            ls[:,1] = 1.5
            ls[:,2] = 5.0
            box_cam = torch.cat([self.traj[:,s], ls, rs], dim=1)
            lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(box_cam.unsqueeze(1))
            lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam, self.Z1, self.Y1, self.X1)

            lrt_camR0s.append(lrtlist_cam.squeeze(1))

            box_vis.append(self.summ_writer.summ_lrtlist_bev(
                '',
                # utils.basic.normalize(occ_fat*self.heats[0:1,s]),
                utils.basic.normalize(self.diffs[:,s]),
                # utils.basic.normalize(loss_memR0s[0:1,s]),
                # torch.zeros((1, 1, self.Z1, self.Y1, self.X1), dtype=torch.float32).cuda(),
                lrtlist_mem[0:1],
                torch.ones(1,1).float().cuda(), # scores
                torch.ones(1,1).long().cuda(), # tids
                self.vox_util,
                already_mem=True,
                only_return=True))
        self.summ_writer.summ_rgbs('latents/boxes_optim', box_vis)

        # print('self.traj', (self.traj.shape)

        self.summ_writer.summ_traj_on_occ(
            'latents/traj_optim',
            self.traj.data,
            utils.basic.normalize(self.diffs[:,0]),
            self.vox_util,
            traj_g=None,
            show_bkg=True,
            already_mem=False,
            sigma=2,
            only_return=False)

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def train_scene(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        obj_rev = torch.flip(self.obj, [4])
        self.obj_full = torch.cat([self.obj, obj_rev], dim=4)

        self.summ_writer.summ_feat('latents/bkg', self.bkg, pca=True)
        # self.summ_writer.summ_feat('latents/obj', self.obj, pca=True)
        self.summ_writer.summ_feat('latents/obj_full', self.obj_full, pca=True)
        self.summ_writer.summ_oneds('latents/heats', (self.heats).unbind(1), bev=True)

        # if np.random.uniform(0,1) > 0.5:
        #     self.obj = torch.flip(self.obj, [4])
            # # flip on X. (B, 1, Z, Y, X, so X=dim4)
            # obj_reverse = torch.flip(self.obj, [4])
            # obj_symm_loss = torch.mean(torch.abs(self.obj - obj_reverse))
            # total_loss = utils.misc.add_loss('latents/obj_symm', total_loss, obj_symm_loss, 1.0, self.summ_writer)

        traj_prevprev = self.traj[:,:-2]
        traj_prev = self.traj[:,1:-1]
        traj_curr = self.traj[:,2:]
        traj_elastic_loss = torch.mean(torch.sum(torch.abs(traj_curr - 2*traj_prev + traj_prevprev), dim=2))
        total_loss = utils.misc.add_loss('latents/traj_elastic', total_loss, traj_elastic_loss, hyp.latent_traj_elastic_coeff, self.summ_writer)
        
        box_vis = []
        xyz_memR0s = []
        xyz_camR0s = []
        lrt_camR0s = []
        for s in list(range(self.S)):
            # box = torch.zeros((self.B, 9), dtype=torch.float32).cuda()
            rs = torch.zeros((self.B, 3), dtype=torch.float32).cuda()
            ls = 3.0*torch.ones((self.B, 3), dtype=torch.float32).cuda()
            ls[:,0] = 2.5
            ls[:,1] = 1.5
            ls[:,2] = 5.0
            box_cam = torch.cat([self.traj[:,s], ls, rs], dim=1)
            lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(box_cam.unsqueeze(1))
            lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam, self.Z1, self.Y1, self.X1)

            lrt_camR0s.append(lrtlist_cam.squeeze(1))

            box_vis.append(self.summ_writer.summ_lrtlist_bev(
                '',
                self.occ_memR0s[0:1,0],
                # torch.zeros((1, 1, self.Z1, self.Y1, self.X1), dtype=torch.float32).cuda(),
                lrtlist_mem[0:1],
                torch.ones(1,1).float().cuda(), # scores
                torch.ones(1,1).long().cuda(), # tids
                self.vox_util,
                already_mem=True,
                only_return=True))
        self.summ_writer.summ_rgbs('latents/boxes_optim', box_vis)

        self.summ_writer.summ_traj_on_occ(
            'latents/traj_optim',
            self.traj.data,
            self.occ_memR0s[0:1,0],
            # torch.zeros((1, 1, self.Z1, self.Y1, self.X1), dtype=torch.float32).cuda(),
            self.vox_util,
            traj_g=None,
            show_bkg=True,
            already_mem=False,
            sigma=2,
            only_return=False)

        lrt_camR0s = torch.stack(lrt_camR0s, dim=1)

        # now that i have some boxes, i want to use them to render some new scenes
        camR0s_T_zoom = __u(self.vox_util.get_ref_T_zoom(__p(lrt_camR0s), self.Z1, self.Y1, self.X1))
        # we want a sample for each location in the mem grid
        # so we'll start with a gridcloud in mem, and find out which coords correspond in zoom
        xyz_mems_ = utils.basic.gridcloud3d(self.B*self.S, self.Z1, self.Y1, self.X1, norm=False)
        # this is B*S x Z*Y*X x 3
        xyz_camR0s_ = self.vox_util.Mem2Ref(xyz_mems_, self.Z1, self.Y1, self.X1)
        camR0s_T_zoom_ = __p(camR0s_T_zoom)
        zoom_T_camR0s_ = camR0s_T_zoom_.inverse() # note this is not a rigid transform
        xyz_zooms_ = utils.geom.apply_4x4(zoom_T_camR0s_, xyz_camR0s_)

        # we will do the whole traj at once (per obj)
        # note we just have one feat for the whole traj, so we tile up
        
        obj_feats = self.obj_full.unsqueeze(1).repeat(1, self.S, 1, 1, 1, 1)
        obj_feats_ = __p(obj_feats)
        # this is B*S x C x Z x Y x X
        self.summ_writer.summ_feats('latents/obj_feats', obj_feats.unbind(1), pca=True)

        obj_feat_memR0s_ = utils.samp.sample3d(obj_feats_, xyz_zooms_, self.Z1, self.Y1, self.X1)
        obj_feat_memR0s = __u(obj_feat_memR0s_)
        self.summ_writer.summ_feats('latents/obj_feat_memR0s', obj_feat_memR0s.unbind(1), pca=True)

        obj_mask_memR0s_ = utils.samp.sample3d(torch.ones_like(obj_feats_[:,0:1]), xyz_zooms_, self.Z1, self.Y1, self.X1)
        weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
        obj_mask_memR0s_ = F.conv3d(obj_mask_memR0s_, weights, padding=1).clamp(0, 1)
        obj_mask_memR0s = __u(obj_mask_memR0s_)
        self.summ_writer.summ_feats('latents/obj_mask_memR0s', obj_mask_memR0s.unbind(1), pca=False)

        scene_memR0s = self.bkg.unsqueeze(1) * (1.0 - obj_mask_memR0s) + obj_feat_memR0s * obj_mask_memR0s
        self.summ_writer.summ_feats('latents/scene_memR0s', scene_memR0s.unbind(1), pca=True)
        
        # print('self.global_step', self.global_step)
        if self.global_step==1:

            if hyp.do_occ:
                occ_memR0s = []
                free_memR0s = []
                vis_memR0s = []
                sup_memR0s = []
                for s in list(range(self.S)):
                    camR0_T_cam0 = self.camR0s_T_cam0s[:,s]
                    camR0_T_cam1 = self.camR0s_T_cam1s[:,s]
                    camR0_T_cam2 = self.camR0s_T_cam2s[:,s]
                    camR0_T_cam3 = self.camR0s_T_cam3s[:,s]
                    xyz_cam0 = self.xyz_cam0s[:,s]
                    xyz_cam1 = self.xyz_cam1s[:,s]
                    xyz_cam2 = self.xyz_cam2s[:,s]
                    xyz_cam3 = self.xyz_cam3s[:,s]

                    xyz_camXs = torch.stack([xyz_cam0,
                                             xyz_cam1,
                                             xyz_cam2,	
                                             xyz_cam3], dim=1)
                    camR0s_T_camXs = torch.stack([camR0_T_cam0,
                                                  camR0_T_cam1,
                                                  camR0_T_cam2,
                                                  camR0_T_cam3], dim=1)

                    occ_memR0_sup, free_memR0_sup, _, _ = self.vox_util.prep_occs_supervision(
                        camR0s_T_camXs,
                        xyz_camXs,
                        self.Z1, self.Y1, self.X1,
                        agg=True)
                    
                    # be more conservative with "free"
                    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                    free_memR0_sup = 1.0 - (F.conv3d(1.0 - free_memR0_sup, weights, padding=1)).clamp(0, 1)

                    vis_memR0 = (occ_memR0_sup + free_memR0_sup).clamp(0, 1)
                    vis_memR0s.append(vis_memR0)
                    occ_memR0s.append(occ_memR0_sup)
                    free_memR0s.append(free_memR0_sup)
                occ_memR0s = torch.stack(occ_memR0s, dim=1)
                free_memR0s = torch.stack(free_memR0s, dim=1)
                vis_memR0s = torch.stack(vis_memR0s, dim=1)

                _, _, _, Z_, Y_, X_ = list(occ_memR0s.shape)
                occ_memR0s_np = (occ_memR0s.detach().cpu().reshape(self.S, -1)).numpy()
                vis_memR0s_np = (vis_memR0s.detach().cpu().reshape(self.S, -1)).numpy()
                occ_memR0_median_np_safe = np.median(occ_memR0s_np, axis=0)
                occ_memR0_median_np = utils.py.reduce_masked_median(
                    occ_memR0s_np.transpose(1, 0), vis_memR0s_np.transpose(1, 0), keep_batch=True)
                occ_memR0_median_np[np.isnan(occ_memR0_median_np)] = occ_memR0_median_np_safe[np.isnan(occ_memR0_median_np)]
                occ_memR0_median = torch.from_numpy(occ_memR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()

                diffs = vis_memR0s*torch.norm((occ_memR0_median.unsqueeze(1) - occ_memR0s), dim=2, keepdim=True)

                # blur out, to eliminate spurious peaks
                diffs = __u(F.interpolate(__p(diffs), scale_factor=0.25, mode='trilinear'))
                diffs = __u(F.interpolate(__p(diffs), scale_factor=4, mode='trilinear'))

            self.occ_memR0s = occ_memR0s
            self.free_memR0s = free_memR0s
            self.vis_memR0s = vis_memR0s
            self.diffs = diffs
            self.occ_memR0_median = occ_memR0_median
        # endif global_step==1


        if hyp.do_occ:
            bkg_occ_total = 0.0
            obj_occ_total = 0.0
            for s in list(range(self.S)):
                scene_memR0 = scene_memR0s[:,s]

                # camR0_T_cam0 = self.camR0s_T_cam0s[:,s]
                # camR0_T_cam1 = self.camR0s_T_cam1s[:,s]
                # camR0_T_cam2 = self.camR0s_T_cam2s[:,s]
                # camR0_T_cam3 = self.camR0s_T_cam3s[:,s]
                # xyz_cam0 = self.xyz_cam0s[:,s]
                # xyz_cam1 = self.xyz_cam1s[:,s]
                # xyz_cam2 = self.xyz_cam2s[:,s]
                # xyz_cam3 = self.xyz_cam3s[:,s]

                # xyz_camXs = torch.stack([xyz_cam0,
                #                          xyz_cam1,
                #                          xyz_cam2,	
                #                          xyz_cam3], dim=1)
                # camR0s_T_camXs = torch.stack([camR0_T_cam0,
                #                               camR0_T_cam1,
                #                               camR0_T_cam2,
                #                               camR0_T_cam3], dim=1)
                # occ_memR0_sup, free_memR0_sup, _, _ = self.vox_util.prep_occs_supervision(
                #     camR0s_T_camXs,
                #     xyz_camXs,
                #     self.Z1, self.Y1, self.X1,
                #     agg=True)

                occ_memR0_sup = self.occ_memR0s[:,s]
                free_memR0_sup = self.free_memR0s[:,s]
                vis_memR0 = self.vis_memR0s[:,s]

                bkg_occ_loss, _, _ = self.occnet(
                    self.bkg,
                    (F.sigmoid(self.occ_memR0_median).round()==1).float(), # occ median
                    free_memR0_sup, # free on each frame
                    vis_memR0,
                )
                bkg_occ_total += bkg_occ_loss / float(self.S)
                
                obj_occ_loss, _, _ = self.occnet(
                    # obj_feat_memR0s[:,s],
                    scene_memR0,
                    occ_memR0_sup, # occ on each frame
                    free_memR0_sup, # free on each frame
                    torch.ones_like(free_memR0_sup),
                    # obj_mask_memR0s[:,s],
                )
                obj_occ_total += obj_occ_loss / float(self.S)
                
                # # pos = occ_memR0_sup
                # pos = (F.sigmoid(self.occ_memR0_median).round()==1).float()
                # neg = free_memR0_sup.clone()
                # pos = pos * (1.0 - free_memR0_sup)
                # # pred = self.bkg[:,0:1]
                # pred = scene_memR0[:,0:1]
                # label = pos*2.0 - 1.0
                # a = -label * pred
                # b = F.relu(a)
                # loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
                # pos_loss = utils.basic.reduce_masked_mean(loss, pos)
                # neg_loss = utils.basic.reduce_masked_mean(loss, neg)
                # balanced_loss = pos_loss + neg_loss
                # # neg_loss = utils.basic.reduce_masked_mean(loss, neg)
                # # occ_total += neg_loss / float(self.S)
                # occ_total += balanced_loss / float(self.S)
            total_loss = utils.misc.add_loss('latents/obj_occ_total', total_loss, obj_occ_total, hyp.latent_obj_occ_coeff, self.summ_writer)
            total_loss = utils.misc.add_loss('latents/bkg_occ_total', total_loss, bkg_occ_total, hyp.latent_bkg_occ_coeff, self.summ_writer)

        self.summ_writer.summ_occ('latents/occ_obj', F.sigmoid(self.obj_full[:,0:1]))
        self.summ_writer.summ_occ('latents/occ_bkg', F.sigmoid(self.bkg[:,0:1]))
        self.summ_writer.summ_occ('latents/occ_median', self.occ_memR0_median)
        self.summ_writer.summ_occs('latents/occ_scene', [F.sigmoid(scene) for scene in scene_memR0s[:,:,0:1].unbind(1)])
        occ_objs = F.sigmoid(obj_feat_memR0s[:,:,0:1])*obj_mask_memR0s
        # self.summ_writer.summ_occs('latents/occ_objs', [F.sigmoid(scene) for scene in obj_feat_memR0s[:,:,0:1].unbind(1)])
        self.summ_writer.summ_occs('latents/occ_objs', occ_objs.unbind(1))
        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(__p(scene_memR0s), absolute=True)
        smooth_vox = torch.mean(dz+dy+dx, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils.misc.add_loss('latents/scene_smooth_loss', total_loss, smooth_loss, hyp.latent_scene_smooth_coeff, self.summ_writer)

        dz, dy, dx = utils.basic.gradient3d(self.obj_full, absolute=True)
        smooth_vox = torch.mean(dz+dy+dx, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils.misc.add_loss('latents/obj_smooth_loss', total_loss, smooth_loss, hyp.latent_obj_smooth_coeff, self.summ_writer)
        
        if hyp.do_render:

            rgb_cam3s_e = []
            rgb_cam3s_g = []
            rgb_camRs_e = []
            rgb_camRs_g = []
            render_total = 0.0

            for s in list(range(self.S)):

                PH, PW = hyp.PH, hyp.PW
                sy = float(PH)/float(hyp.H)
                sx = float(PW)/float(hyp.W)
                assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
                assert(sy==0.5 or sy==1.0)
                projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

                scene_rgb_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,s], self.camRs_T_camR0s[:,s], scene_memR0s[:,s,1:4],
                    hyp.view_depth, PH, PW, noise_amount=2.0)
                scene_occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,s], self.camRs_T_camR0s[:,s], scene_memR0s[:,s,0:1],
                    hyp.view_depth, PH, PW, grid_z_vec=dists)

                rgb_g = self.rgb_camRs[:,s]
                valid_g = self.valid_camRs[:,s]
                rgb_g = F.interpolate(rgb_g, size=(PH, PW), mode='bilinear')
                valid_g = F.interpolate(valid_g, size=(PH, PW), mode='bilinear')

                if s==int(self.S/2):
                    summ_writer = self.summ_writer
                else:
                    summ_writer = None
                render_loss, rgb_e, _, _ = self.rendernet(
                    scene_rgb_proj,
                    scene_occ_proj,
                    dists,
                    rgb_g=rgb_g,
                    valid=valid_g,
                    summ_writer=summ_writer)
                render_total += render_loss / float(self.S)
                rgb_camRs_e.append(rgb_e)
                rgb_camRs_g.append(rgb_g*valid_g)
                
                # also render cam3
                scene_rgb_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,s], self.cam3s_T_camR0s[:,s], scene_memR0s[:,s,1:4],
                    hyp.view_depth, PH, PW, noise_amount=2.0)
                scene_occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,s], self.cam3s_T_camR0s[:,s], scene_memR0s[:,s,0:1],
                    hyp.view_depth, PH, PW, grid_z_vec=dists)
                rgb_g = self.rgb_cam3s[:,s]
                valid_g = self.valid_cam3s[:,s]
                rgb_g = F.interpolate(rgb_g, size=(PH, PW), mode='bilinear')
                valid_g = F.interpolate(valid_g, size=(PH, PW), mode='bilinear')
                render_loss, rgb_e, _, _ = self.rendernet(
                    scene_rgb_proj,
                    scene_occ_proj,
                    dists,
                    rgb_g=rgb_g,
                    valid=valid_g)
                render_total += render_loss
                rgb_cam3s_e.append(rgb_e)
                rgb_cam3s_g.append(rgb_g*valid_g)
                
            total_loss = utils.misc.add_loss('latents/render_total', total_loss, render_total, hyp.latent_render_coeff, self.summ_writer)
            self.summ_writer.summ_rgbs('render/rgb_camRs_e', rgb_camRs_e)
            self.summ_writer.summ_rgbs('render/rgb_camRs_g', rgb_camRs_g)
            self.summ_writer.summ_rgbs('render/rgb_cam3s_e', rgb_cam3s_e)
            self.summ_writer.summ_rgbs('render/rgb_cam3s_g', rgb_cam3s_g)

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)
        # data_ok = False
        
        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                # return self.run_train(feed)
                # return self.train_bkg_var(feed)
                if hyp.train_trajvar:
                    return self.train_traj(feed)
                elif hyp.train_scenevar:
                    return self.train_scene(feed)
                elif hyp.train_traj_via_scene:
                    return self.train_scene(feed)
                else:
                    return self.train_feat3d(feed)
            elif self.set_name=='val':
                return self.run_train(feed)
            elif self.set_name=='test':
                return self.run_sfm(feed)
                # return self.run_orb(feed)
                # return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
