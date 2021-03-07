import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.centernet import CenterNet
from nets.softargnet import SoftargNet
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

# the idea here is to optimize into a set of templates
# along with a detector net.
# together, they should output:
# what the background is, and where each of the objects are
# we use a fixed number of objects
# they do not all have to be present


class CARLA_PROTO(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaProtoModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        if hyp.do_freeze_center:
            self.model.centernet.eval()
            self.set_requires_grad(self.model.centernet, False)
        if hyp.do_freeze_softarg:
            self.model.softargnet.eval()
            self.set_requires_grad(self.model.softargnet, False)
        # if hyp.do_freeze_proto:
        #     self.model.protonet.eval()
        #     self.set_requires_grad(self.model.protonet, False)
            
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

        # set_latent_lists = []
        # set_latent_optim_lists = []
        # set_origin_T_cami_lists = []
        # for set_input, set_b in zip(set_inputs, set_batch_sizes):
        #     set_len = len(set_input.dataset)
        #     print('set_len:', set_len)
        #     set_latent_list = [None]*set_len
        #     set_latent_optim_list = []
        #     for si in list(range(set_len)):
        #         # set_latent_optim_list.append(torch.optim.SGD([self.model.zi], lr=hyp.lr*2.0))
        #         set_latent_optim_list.append(torch.optim.Adam([self.model.zi], lr=hyp.lr*100.0))
            
        #     set_origin_T_cami_list = [None]*set_len
        #     set_latent_lists.append(set_latent_list)
        #     set_latent_optim_lists.append(set_latent_optim_list)
        #     set_origin_T_cami_lists.append(set_origin_T_cami_list)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': hyp.lr},
            {'params': self.model.obj, 'lr': hyp.lr*10.0}
            # {'params': self.model.bkg_dict, 'lr': hyp.lr*10.0}
        ])
        
        model_state_dict = self.model.state_dict()
        for k in model_state_dict.keys():
            print('key', k)
        
        self.start_iter = saverloader.load_weights(self.model, None)
        # if hyp.latents_init:
        #     latent_list, latent_optim_list, origin_T_cami_list = saverloader.load_latents(hyp.latents_init)
        #     ind = set_names.index('train')
        #     print('putting these into ind', ind)
        #     set_latent_lists[ind] = latent_list
        #     set_latent_optim_lists[ind] = latent_optim_list
        #     set_origin_T_cami_lists[ind] = origin_T_cami_list
            
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

                    # # zi_np = np.random.randn(set_batch_size, 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)).astype(np.float32)
                    # zi_np = np.random.randn(set_batch_size, 4, int(hyp.Z), int(hyp.Y), int(hyp.X)).astype(np.float32)
                    # origin_T_cami_np = np.zeros((set_batch_size, 4, 4), np.float32)
                    # for b in list(range(set_batch_size)):
                    #     data_ind_b = data_ind[b]

                    #     zi_np_b = set_latent_list[data_ind_b]
                    #     origin_T_cami_np_b = set_origin_T_cami_list[data_ind_b]
                    #     if (zi_np_b is not None) and (origin_T_cami_np_b is not None):
                    #         # then this var must have been saved/optimized before
                    #         # use it
                    #         # print('using data from', data_ind_b)
                    #         zi_np[b] = zi_np_b
                    #         origin_T_cami_np[b] = origin_T_cami_np_b
                    #         # print('init with zi_np_b[0,0,:5,:5]', zi_np_b[0,0,:5,:5])
                    #     else:
                    #         print('this is the first time encountering index %d; initializing with random normal, and origin_T_camX0' % data_ind_b)
                    #         origin_T_camXs = feed["origin_T_camXs"]
                    #         origin_T_cami_np[b] = origin_T_camXs[b,0].cpu().numpy()
                    # # print('origin_T_cami_np', origin_T_cami_np.shape)
                            
                    # feed_cuda["origin_T_cami"] = torch.from_numpy(origin_T_cami_np).cuda()
                    # feed_cuda["zi"] = torch.from_numpy(zi_np).cuda()
                    
                    repeats = 1
                    iter_start_time = time.time()
                    for rep in list(range(repeats)):
                    
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
                            # for b in list(range(set_batch_size)):
                            #     zi_optim = set_latent_optim_list[data_ind[b]]
                            #     zi_optim.zero_grad()

                            loss.backward()

                            self.optimizer.step()
                            # for b in list(range(set_batch_size)):
                            #     zi_optim = set_latent_optim_list[data_ind[b]]
                            #     zi_optim.step()


                            # # ok, self.model.zi is holding new updated data, which we want to store
                            # zi_np = self.model.zi.data.detach().cpu().numpy()
                            # # zi_np = project_l2_ball_py(zi_np)
                            
                            # # print('zi_np', zi_np)
                            # for b in list(range(set_batch_size)):
                            #     # zi_cuda_b = zi_cuda[b]
                            #     # zi_np_b = zi_np[b]
                            #     # print('zi_cuda_b', zi_cuda_b.
                            #     # zi_np_b = set_latent_list[data_inds[b]]
                            #     # origin_T_cami[data_inds[b]] = zi_np_b
                            #     # origin_T_cami_np_b = set_origin_T_cami_list[data_inds[b]]
                            #     prev = set_latent_list[data_ind[b]]
                            #     new = zi_np[b]

                            #     set_latent_list[data_ind[b]] = zi_np[b]
                            #     set_origin_T_cami_list[data_ind[b]] = origin_T_cami_np[b]

                        # if hyp.do_emb3D:
                        #     def update_slow_network(slow_net, fast_net, beta=0.999):
                        #         param_k = slow_net.state_dict()
                        #         param_q = fast_net.named_parameters()
                        #         for n, q in param_q:
                        #             if n in param_k:
                        #                 param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                        #         slow_net.load_state_dict(param_k)
                        #     update_slow_network(self.model.featnet3D_slow, self.model.featnet3D)
                        
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

                # ind = set_names.index('train')
                # saverloader.save_latents(set_latent_lists[ind],
                #                          set_latent_optim_lists[ind],
                #                          set_origin_T_cami_lists[ind],
                #                          self.checkpoint_dir,
                #                          step)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

class CarlaProtoModel(nn.Module):
    def __init__(self):
        super(CarlaProtoModel, self).__init__()
        
        if hyp.do_bkg:
            self.bkg_k = 10
            self.bkg_dict = torch.randn(
                [1, self.bkg_k, 1, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)],
                requires_grad=True,
                dtype=torch.float32,
                device=torch.device('cuda'))
            self.bkgnet = BkgNet(self.bkg_k, resolution=int(hyp.Z/4))
            
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_center:
            self.centernet = CenterNet()
        if hyp.do_softarg:
            self.softargnet = SoftargNet()

        self.crop_low = (2,2,2)
        self.crop_mid = (8,8,8)
        self.crop = (18,18,18)

        self.obj = torch.randn(
            [1, 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)],
            requires_grad=True,
            dtype=torch.float32,
            device=torch.device('cuda'))
        self.obj.data[:,0] = 10.0

        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_render:
            self.rendernet = RenderNet()
            
    def zero_border(self, feat, crop):
        feat = self.crop_feat(feat, crop)
        feat = self.pad_feat(feat, crop)
        return feat
    
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
        global_step = feed['global_step']

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
        self.cam0s_T_camRs = __u(__p(self.camRs_T_cam0s).inverse())
        self.cam0s_T_cam00s = __u(__p(self.cam00s_T_cam0s).inverse())

        self.camRs_T_cam0s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam0s)))
        self.camRs_T_cam1s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam1s)))
        self.camRs_T_cam2s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam2s)))
        self.camRs_T_cam3s = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_cam3s)))

        self.camR0s_T_cam0s = __u(torch.matmul(__p(self.camR0s_T_camRs).inverse(), __p(self.camRs_T_cam0s)))
        self.camR0s_T_cam1s = __u(torch.matmul(__p(self.camR0s_T_camRs).inverse(), __p(self.camRs_T_cam1s)))
        self.camR0s_T_cam2s = __u(torch.matmul(__p(self.camR0s_T_camRs).inverse(), __p(self.camRs_T_cam2s)))
        self.camR0s_T_cam3s = __u(torch.matmul(__p(self.camR0s_T_camRs).inverse(), __p(self.camRs_T_cam3s)))

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
        # self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_cam0s), __p(self.xyz_cam0s)))
        # self.xyz_cam00s = __u(utils.geom.apply_4x4(__p(self.cam00s_T_cam0s), __p(self.xyz_cam0s)))
        # self.xyz_cam10s = __u(utils.geom.apply_4x4(__p(self.cam10s_T_cam1s), __p(self.xyz_cam1s)))
        # self.xyz_cam20s = __u(utils.geom.apply_4x4(__p(self.cam20s_T_cam2s), __p(self.xyz_cam2s)))
        # self.xyz_cam30s = __u(utils.geom.apply_4x4(__p(self.cam30s_T_cam3s), __p(self.xyz_cam3s)))

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
        self.occ_halfmemR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s_all), self.Z2, self.Y2, self.X2))
        
        self.summ_writer.summ_occs('3d_feats/occ_memRs0', self.occ_memRs0.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs1', self.occ_memRs1.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs2', self.occ_memRs2.unbind(1))
        # self.summ_writer.summ_occs('3d_feats/occ_memRs3', self.occ_memRs3.unbind(1))
        self.summ_writer.summ_occs('3d_feats/occ_memR0s', self.occ_memR0s.unbind(1))
        
        self.depth_camRs_, self.valid_camRs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camRs), self.H, self.W)
        self.dense_xyz_camRs_ = utils.geom.depth2pointcloud(self.depth_camRs_, __p(self.pix_T_cams))
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camR0s_ = utils.geom.apply_4x4(__p(self.camR0s_T_camRs), self.dense_xyz_camRs_)
        self.inbound_camRs_ = self.vox_util.get_inbounds(self.dense_xyz_camR0s_, self.Z, self.Y, self.X).float()
        self.inbound_camRs_ = torch.reshape(self.inbound_camRs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camRs = __u(self.depth_camRs_)
        self.valid_camRs = __u(self.valid_camRs_) * __u(self.inbound_camRs_)

        self.summ_writer.summ_oned('inputs/depth_camR0', self.depth_camRs[:,0]*self.valid_camRs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('inputs/valid_camR0', self.valid_camRs[:,0], norm=False)
        # self.summ_writer.summ_oned('inputs/valid_camR0_after', self.valid_camRs[:,0], norm=False)
        
        
        # # self.summ_writer.summ_oned('inputs/valid_camX0_before', self.valid_camXs[:,0], norm=False)

        # # weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
        # # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        # # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        
        # self.summ_writer.summ_oned('inputs/depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
        # self.summ_writer.summ_oned('inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        # # self.summ_writer.summ_oned('inputs/valid_camX0_after', self.valid_camXs[:,0], norm=False)
        
        # boxlists = feed["boxlists"]
        # self.scorelist_s = feed["scorelists"]
        # self.tidlist_s = feed["tidlists"]
        # boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
        # origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
        # lrtlist_camRs_, _ = utils.misc.parse_boxes(boxlists_, origin_T_camRs_)
        # self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        # self.lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        # self.lrtlist_camX0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        # # full_scorelist_s_ = utils_misc.rescore_lrtlist_with_inbound(
        # #     full_lrtlist_camX0s_, __p(full_tidlist_s), self.Z, self.Y, self.X, self.vox_util, pad=4.0)
        # # # full_scorelist_s = __u(full_scorelist_s_)
        # # self.full_scorelist_s = __u(full_scorelist_s_)
        # # self.full_tidlist_s = full_tidlist_s
        # # self.full_lrtlist_camX0s = __u(full_lrtlist_camX0s_)


        # print('rescoring lrtlist')
        # self.scorelist_s = __u(utils.misc.rescore_lrtlist_with_inbound(
        #     __p(self.lrtlist_camX0s), __p(self.tidlist_s), self.Z, self.Y, self.X, self.vox_util, pad=4.0))

        # self.summ_writer.summ_lrtlist(
        #     'inputs/boxlist_g',
        #     self.rgb_camXs[0:1,0],
        #     self.lrtlist_camXs[0:1,0],
        #     self.scorelist_s[0:1,0],
        #     self.tidlist_s[0:1,0],
        #     self.pix_T_cams[0:1,0])
        
        # # obj_masklist_memX0 = self.vox_util.assemble_padded_obj_masklist(
        # #     self.lrtlist_camX0s[:,0],
        # #     self.scorelist_s[:,0],
        # #     self.Z, self.Y, self.X)
        # # # this is B x N x 1 x Z x Y x X
        # # self.obj_mask_memX0 = torch.sum(obj_masklist_memX0, dim=1).clamp(0, 1)
        # # self.summ_writer.summ_oned('inputs/obj_mask_memX0', self.obj_mask_memX0, bev=True, max_along_y=True, norm=False)

        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.rgb_mem0s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam0s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs0 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam0s, self.rgb_mem0s)

        self.rgb_mem1s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam1s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs1 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam1s, self.rgb_mem1s)

        self.rgb_mem2s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam2s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs2 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam2s, self.rgb_mem2s)

        self.rgb_mem3s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam3s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs3 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam3s, self.rgb_mem3s)

        self.rgb_memRs__ = torch.stack([self.rgb_memRs0,
                                        self.rgb_memRs1,
                                        self.rgb_memRs2,
                                        self.rgb_memRs3], dim=0)
        self.occ_memRs__ = torch.stack([self.occ_memRs0,
                                        self.occ_memRs1,
                                        self.occ_memRs2,
                                        self.occ_memRs3], dim=0)
        self.rgb_memRs = utils.basic.reduce_masked_mean(self.rgb_memRs__, self.occ_memRs__.repeat(1, 1, 1, 3, 1, 1, 1), dim=0)
                                      
        self.rgb_memR0s = self.vox_util.apply_4x4s_to_voxs(self.camR0s_T_camRs, self.rgb_memRs)

        # utils.basic.reduce_masked_mean(
        #                                self.occ_memR0s[:,0]
        # occ_memR0_sup, free_memR0_sup, _, _ = self.vox_util.prep_occs_supervision(
        #     self.camR0s_T_cam0s,
        #     self.xyz_cam0s,
        #     self.Z2, self.Y2, self.X2,
        #     agg=False)
        # vis_memX0_sup = (occ_memX0_sup + free_memX0_sup).clamp(0,1)

        # self.obj.data[:,0] = 10.0
        self.summ_writer.summ_feat('3d_feats/obj', self.obj, pca=True)

        if hyp.do_feat3d:

            # feat_memR0s_input = torch.cat([
            #     self.occ_memRs0,
            #     self.occ_memRs0*self.rgb_memRs0], dim=2)
            feat_memR0s_input = torch.cat([
                self.occ_memR0s,
                self.occ_memR0s*self.rgb_memR0s], dim=2)
            feat_loss, feat_halfmemR0s_, feat_bunch = self.feat3dnet(
                __p(feat_memR0s_input), norm=False, summ_writer=None)
            total_loss += feat_loss
            feat_halfmemR0s = __u(feat_halfmemR0s_)
            self.summ_writer.summ_feats('3d_feats/feat_memR0s_inputs', feat_memR0s_input.unbind(1), pca=True)
            self.summ_writer.summ_feats('3d_feats/feat_halfmemR0', feat_halfmemR0s.unbind(1), pca=True)
            
            # feat_memX0_input = torch.cat([
            #     self.occ_memX0s[:,0],
            #     self.occ_memX0s[:,0]*self.rgb_memX0s[:,0]], dim=1)

            # self.lrtlist_camX0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))
            # obj_masklist_memX0 = self.vox_util.assemble_padded_obj_masklist(
            #     self.lrtlist_camX0s[:,0],
            #     self.scorelist_s[:,0],
            #     self.Z, self.Y, self.X, additive_coeff=1.0)
            # obj_mask_memX0 = torch.sum(obj_masklist_memX0, dim=1).clamp(0, 1)
            # self.summ_writer.summ_feat('3d_feats/feat_memX0_input_unmasked', feat_memX0_input, pca=True)
            # feat_memX0_input = feat_memX0_input * (1.0 - obj_mask_memX0)
            
            # self.summ_writer.summ_feat('3d_feats/feat_memX0_input', feat_memX0_input, pca=True)
            # feat_loss, feat_halfmemX0, feat_bunch = self.feat3dnet(
            #     feat_memX0_input, norm=False, summ_writer=None)
            # self.summ_writer.summ_feat('3d_feats/feat_halfmemX0', feat_halfmemX0, pca=True)

        if hyp.do_occ:

            occ_total = 0.0
            for s in list(range(self.S)):

                camR_T_cam0 = self.camRs_T_cam0s[:,s]
                camR_T_cam1 = self.camRs_T_cam1s[:,s]
                camR_T_cam2 = self.camRs_T_cam2s[:,s]
                camR_T_cam3 = self.camRs_T_cam3s[:,s]
                xyz_cam0 = self.xyz_cam0s[:,s]
                xyz_cam1 = self.xyz_cam1s[:,s]
                xyz_cam2 = self.xyz_cam2s[:,s]
                xyz_cam3 = self.xyz_cam3s[:,s]

                xyz_camXs = torch.stack([xyz_cam0,
                                         xyz_cam1,
                                         xyz_cam2,	
                                         xyz_cam3], dim=1)
                camRs_T_camXs = torch.stack([camR_T_cam0,
                                            camR_T_cam1,
                                            camR_T_cam2,
                                            camR_T_cam3], dim=1)
                                    
                occ_memR_sup, free_memR_sup, _, _ = self.vox_util.prep_occs_supervision(
                    camRs_T_camXs,
                    xyz_camXs,
                    self.Z2, self.Y2, self.X2,
                    agg=True)
                
                # be more conservative with "free"
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                free_memR_sup = 1.0 - (F.conv3d(1.0 - free_memR_sup, weights, padding=1)).clamp(0, 1)
                # we need to crop
                occ_memR_sup = self.crop_feat(occ_memR_sup, self.crop)
                free_memR_sup = self.crop_feat(free_memR_sup, self.crop)
                
                occ_loss, occ_memR_pred = self.occnet(
                    feat_halfmemR0s[:,s],
                    occ_memR_sup,
                    free_memR_sup,
                    torch.ones_like(free_memR_sup))
                occ_total += occ_loss
            self.summ_writer.summ_occ('occ/occ_memR_pred', F.sigmoid(occ_memR_pred))
            total_loss = utils.misc.add_loss('occ/occ_total', total_loss, occ_total, 1.0, self.summ_writer)

        if hyp.do_softarg:
            feat_lows_, feat_mids_, feat_highs_ = feat_bunch

            def assert_cropped_shape(feat, Z, Y, X, crop_guess):
                # make sure the shapes match what we expect
                _, _, Z_, Y_, X_ = list(feat.shape)
                Z_crop = int((Z - Z_)/2)
                Y_crop = int((Y - Y_)/2)
                X_crop = int((X - X_)/2)
                crop = (Z_crop, Y_crop, X_crop)
                if not (crop==crop_guess):
                    print('crop', crop)
                    assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
                return True
            assert_cropped_shape(feat_lows_, self.Z4, self.Y4, self.X4, self.crop_low)
            assert_cropped_shape(feat_mids_, self.Z4, self.Y4, self.X4, self.crop_mid)
            assert_cropped_shape(feat_highs_, self.Z2, self.Y2, self.X2, self.crop)
            feat_lows_ = self.pad_feat(feat_lows_, self.crop_low)
            feat_mids_ = self.pad_feat(feat_mids_, self.crop_mid)
            feat_highs_ = self.pad_feat(feat_highs_, self.crop)
            feat_lows_ = F.interpolate(feat_lows_, scale_factor=2, mode='trilinear')
            feat_mids_ = F.interpolate(feat_mids_, scale_factor=2, mode='trilinear')
            feat_lows_ = self.crop_feat(feat_lows_, self.crop)
            feat_mids_ = self.crop_feat(feat_mids_, self.crop)
            feat_highs_ = self.crop_feat(feat_highs_, self.crop)
            self.summ_writer.summ_feat('3d_feats/feat_lows_', feat_lows_, pca=True)
            self.summ_writer.summ_feat('3d_feats/feat_mids_', feat_mids_, pca=True)
            self.summ_writer.summ_feat('3d_feats/feat_highs_', feat_highs_, pca=True)

            feat_cats_ = torch.cat([feat_lows_,
                                    feat_mids_,
                                    feat_highs_], dim=1)
            feat_cats = __u(feat_cats_)
            # for s in list(range(self.S)):
            _, lrtlist_camR0s_, scorelist_s_ = self.centernet(
                feat_cats_, 
                self.crop,
                self.vox_util)
            # lrtlist_camR0s_ is B*S x K x 19
            lrt_camR0s = __u(lrtlist_camR0s_.squeeze(1))
            score_s = __u(scorelist_s_.squeeze(1))
            # lrt_camR0s is B x S x 19
            # # lrtlist_camR0s_ is B*S x 1 x 19
            # lrtlist_camR0s_ = lrtlist_camR0s_.reshape(self.B, self.S, 19)
            # # lrtlist_camR
            ls, rts = utils.geom.split_lrtlist(lrt_camR0s)
            # print('lrtlist_camR0s_', lrtlist_camR0s_.shape)
            # print('ls_', ls_.shape)
            # print('rts_', rts_.shape)
            # ls_ is B*S x 3
            # ls = __u(ls_)
            # # print('ls', ls.shape)
            ls = torch.mean(ls, dim=1, keepdim=True).repeat(1, self.S, 1)
            # ls_ = __p(ls)
            # lrtlist_camR0s_ = utils.geom.merge_lrtlist(ls_, rts_)
            lrt_camR0s = utils.geom.merge_lrtlist(ls, rts)
            
            # one more time but just zeroth frame and show some vis
            __ = self.centernet(
                feat_cats[:,0], 
                self.crop,
                self.vox_util,
                summ_writer=self.summ_writer)
            
            # lrt_camR0s is B x S x 19
            lrt_camRs = utils.geom.apply_4x4s_to_lrts(self.camRs_T_camR0s, lrt_camR0s)

            box_vis = []
            box_vis_bev = []
            for s in list(range(self.S)):
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.occ_memR0s[0:1,s],
                    # '', self.crop_feat(self.occ_halfmemR0s[0:1,s], self.crop),
                    lrt_camR0s[0:1,s:s+1],
                    score_s[0:1,s:s+1], # scores
                    torch.ones(1,50).long().cuda(), # tids
                    self.vox_util,
                    already_mem=False, only_return=True))

                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camRs[0:1,s],
                    lrt_camRs[0:1,s:s+1],
                    score_s[0:1,0:1],
                    torch.arange(50).reshape(1, 50).long().cuda(), # tids
                    self.pix_T_cams[0:1,0], only_return=True))
            self.summ_writer.summ_rgbs('center/boxes_e_pers', box_vis)
            self.summ_writer.summ_rgbs('center/boxes_e_bev', box_vis_bev)

            # now that i have some boxes, i want to use them to render some new scenes
            camRs_T_zoom = __u(self.vox_util.get_ref_T_zoom(__p(lrt_camRs), self.Z2, self.Y2, self.X2))
            # we want to sample for each location in the mem grid
            xyz_mems_ = utils.basic.gridcloud3d(self.B*self.S, self.Z1, self.Y1, self.X1, norm=False)
            # this is B*S x Z*Y*X x 3
            xyz_camRs_ = self.vox_util.Mem2Ref(xyz_mems_, self.Z1, self.Y1, self.X1)
            camRs_T_zoom_ = __p(camRs_T_zoom)
            zoom_T_camRs_ = camRs_T_zoom_.inverse() # note this is not a rigid transform
            xyz_zooms_ = utils.geom.apply_4x4(zoom_T_camRs_, xyz_camRs_)

            # we will do the whole traj at once (per obj)
            # note we just have one feat for the whole traj, so we tile up
            obj_feats = self.obj.unsqueeze(1).repeat(1, self.S, 1, 1, 1, 1)
            obj_feats_ = __p(obj_feats)
            # this is B*S x Z x Y x X x C

            # to sample, we need feats_ in ZYX order
            obj_featRs_ = utils.samp.sample3d(obj_feats_, xyz_zooms_, self.Z1, self.Y1, self.X1)
            obj_featRs = __u(obj_featRs_)
            self.summ_writer.summ_feats('3d_feats/obj_featRs', obj_featRs.unbind(1), pca=True)

            obj_maskRs_ = utils.samp.sample3d(torch.ones_like(obj_feats_[:,0:1]), xyz_zooms_, self.Z1, self.Y1, self.X1)
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            # obj_maskRs_ = 1.0 - (F.conv3d(1.0 - obj_maskRs_, weights, padding=1)).clamp(0, 1)
            obj_maskRs_ = F.conv3d(obj_maskRs_, weights, padding=1).clamp(0, 1)
            obj_maskRs = __u(obj_maskRs_)
            
            self.summ_writer.summ_feats('3d_feats/obj_maskRs', obj_maskRs.unbind(1), pca=False)

            bkg_memR0_input = feat_memR0s_input[:,0] * (1.0 - obj_maskRs[:,0])
            _, bkg_halfmemR0, _ = self.feat3dnet(bkg_memR0_input, norm=False)
            self.summ_writer.summ_feat('3d_feats/bkg_featR0', bkg_halfmemR0, pca=True)
            
        if hyp.do_render:

            rgbs_obj_e = []
            rgbs_full_e = []
            rgbs_g = []
            box_vis = []
            render_total = 0.0
            if hyp.do_center:
                bkg_memR0 = F.interpolate(self.pad_feat(bkg_halfmemR0, self.crop), scale_factor=2, mode='trilinear')

            for s in list(range(self.S)):
                # feat_halfmemR0 = feat_halfmemR0s[:,s]
                # feat_halfmemR0 = bkg_halfmemR0

                fullfeat_memR0 = F.interpolate(self.pad_feat(feat_halfmemR0s[:,s], self.crop), scale_factor=2, mode='trilinear')
                fullrgb_memR0 = fullfeat_memR0[:,1:]
                fullocc_memR0 = fullfeat_memR0[:,0:1]

                if hyp.do_center:
                    feat_memR0 = bkg_memR0.detach() * (1.0 - obj_maskRs[:,s]) + obj_featRs[:,s] * obj_maskRs[:,s]
                    rgb_memR0 = feat_memR0[:,1:]
                    occ_memR0 = feat_memR0[:,0:1]

                    obj_halfmask = self.crop_feat(F.interpolate(obj_maskRs[:,s], scale_factor=0.5, mode='trilinear'), self.crop)
                    obj_halffeat = self.crop_feat(F.interpolate(obj_featRs[:,s], scale_factor=0.5, mode='trilinear'), self.crop)
                    feat_halfmemR0 = bkg_halfmemR0.detach() * (1.0 - obj_halfmask) + obj_halffeat * obj_halfmask

                    if hyp.do_occ:

                        occ_total = 0.0

                        camR_T_cam0 = self.camRs_T_cam0s[:,s]
                        camR_T_cam1 = self.camRs_T_cam1s[:,s]
                        camR_T_cam2 = self.camRs_T_cam2s[:,s]
                        camR_T_cam3 = self.camRs_T_cam3s[:,s]
                        xyz_cam0 = self.xyz_cam0s[:,s]
                        xyz_cam1 = self.xyz_cam1s[:,s]
                        xyz_cam2 = self.xyz_cam2s[:,s]
                        xyz_cam3 = self.xyz_cam3s[:,s]

                        xyz_camXs = torch.stack([xyz_cam0,
                                                 xyz_cam1,
                                                 xyz_cam2,	
                                                 xyz_cam3], dim=1)
                        camRs_T_camXs = torch.stack([camR_T_cam0,
                                                     camR_T_cam1,
                                                     camR_T_cam2,
                                                     camR_T_cam3], dim=1)

                        occ_memR_sup, free_memR_sup, _, _ = self.vox_util.prep_occs_supervision(
                            camRs_T_camXs,
                            xyz_camXs,
                            self.Z1, self.Y1, self.X1,
                            agg=True)

                        # be more conservative with "free"
                        weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                        free_memR_sup = 1.0 - (F.conv3d(1.0 - free_memR_sup, weights, padding=1)).clamp(0, 1)
                        # occ_memR_sup = self.crop_feat(occ_memR_sup, self.crop)
                        # free_memR_sup = self.crop_feat(free_memR_sup, self.crop)

                        occ_loss, occ_memR_pred = self.occnet(
                            feat_memR0,
                            occ_memR_sup,
                            free_memR_sup,
                            obj_maskRs[:,s])
                            # torch.ones_like(free_memR_sup))
                        occ_total += occ_loss

                        

                        # occ_halfmemR_sup, free_halfmemR_sup, _, _ = self.vox_util.prep_occs_supervision(
                        #     camRs_T_camXs,
                        #     xyz_camXs,
                        #     self.Z2, self.Y2, self.X2,
                        #     agg=True)
                        # # be more conservative with "free"
                        # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                        # free_halfmemR_sup = 1.0 - (F.conv3d(1.0 - free_halfmemR_sup, weights, padding=1)).clamp(0, 1)
                        # occ_halfmemR_sup = self.crop_feat(occ_halfmemR_sup, self.crop)
                        # free_halfmemR_sup = self.crop_feat(free_halfmemR_sup, self.crop)
                        occ_loss, occ_halfmemR_pred = self.occnet(feat_halfmemR0)
                            # feat_halfmemR0,
                            # occ_halfmemR_sup,
                            # free_halfmemR_sup,
                            # torch.ones_like(free_memR_sup))
                        # occ_total += occ_loss
                        
                    # self.summ_writer.summ_occ('occ/obj_occ_memR_pred', F.sigmoid(occ_memR_pred))
                    self.summ_writer.summ_occ('occ/obj_occ_halfmemR_pred', F.sigmoid(occ_halfmemR_pred))
                    total_loss = utils.misc.add_loss('occ/obj_occ_total', total_loss, occ_total, 1.0, self.summ_writer)


                    
                # self.summ_writer.summ_occ('3d_feats/occ_halfmemR0', F.sigmoid(occ_memR0))

                PH, PW = hyp.PH, hyp.PW
                sy = float(PH)/float(hyp.H)
                sx = float(PW)/float(hyp.W)
                assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
                assert(sy==0.5 or sy==1.0)
                projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

                zoom_box = torch.from_numpy(np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)).cuda()
                zoom_box = zoom_box.reshape(1, 4).repeat(self.B, 1).cuda()
                zoompix_T_cam, _ = utils.geom.convert_box2d_to_intrinsics(
                    zoom_box, projpix_T_cams[:,s], PH, PW, use_image_aspect_ratio=True)

                fullfeat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                    zoompix_T_cam, self.camRs_T_camR0s[:,s], fullrgb_memR0,
                    hyp.view_depth, PH, PW, noise_amount=2.0)
                fullocc_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                    zoompix_T_cam, self.camRs_T_camR0s[:,s], fullocc_memR0,
                    hyp.view_depth, PH, PW, grid_z_vec=dists)

                if hyp.do_center:
                    feat_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                        zoompix_T_cam, self.camRs_T_camR0s[:,s], rgb_memR0,
                        hyp.view_depth, PH, PW, grid_z_vec=dists)
                    occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                        zoompix_T_cam, self.camRs_T_camR0s[:,s], occ_memR0,
                        hyp.view_depth, PH, PW, grid_z_vec=dists)

                rgb_R00 = self.rgb_camRs[:,s]
                depth_R00 = self.depth_camRs[:,s]
                valid_R00 = self.valid_camRs[:,s]
                # to do: replace this line with a crop_and_resize
                rgb_R00 = rgb_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                depth_R00 = depth_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                valid_R00 = valid_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                rgb_R00 = F.interpolate(rgb_R00, size=(PH, PW), mode='bilinear')
                depth_R00 = F.interpolate(depth_R00, size=(PH, PW), mode='bilinear')
                valid_R00 = F.interpolate(valid_R00, size=(PH, PW), mode='bilinear')

                if hyp.do_center:
                    render_loss, rgb_e, _, _ = self.rendernet(
                        feat_proj,
                        occ_proj,
                        dists,
                        rgb_g=rgb_R00,
                        depth_g=depth_R00,
                        valid=valid_R00)
                        # summ_writer=self.summ_writer)
                    render_total += render_loss
                    rgbs_obj_e.append(rgb_e)

                render_loss, rgb_e, _, _ = self.rendernet(
                    fullfeat_proj,
                    fullocc_proj,
                    dists,
                    rgb_g=rgb_R00,
                    depth_g=depth_R00,
                    valid=valid_R00)
                render_total += render_loss
                rgbs_full_e.append(rgb_e)
                
                rgbs_g.append(rgb_R00*valid_R00)

            # total_loss += render_loss
            total_loss = utils.misc.add_loss('render/render_total', total_loss, render_total, 1.0, self.summ_writer)
            self.summ_writer.summ_rgbs('3d_feats/rgbs_full_e', rgbs_full_e)
            if hyp.do_center:
                self.summ_writer.summ_rgbs('3d_feats/rgbs_obj_e', rgbs_obj_e)
            self.summ_writer.summ_rgbs('3d_feats/rgbs_g', rgbs_g)


        regular_centernet = False
        if hyp.do_center and regular_centernet:
            # this net achieves the following:
            # objectness: put 1 at each object center and 0 everywhere else
            # orientation: at the object centers, classify the orientation into a rough bin
            # size: at the object centers, regress to the object size

            # print('feat_halfmemX0', feat_halfmemX0.shape)
            
            lrtlist_camX0 = self.lrtlist_camX0s[:,0]
            lrtlist_halfmemX0 = self.vox_util.apply_mem_T_ref_to_lrtlist(
                lrtlist_camX0, self.Z2, self.Y2, self.X2)
            scorelist = self.scorelist_s[:,0]

            # lrtlist = self.lrtlist_camX0s[:,0]
            clist_camX0 = utils.geom.get_clist_from_lrtlist(lrtlist_camX0)
            # this is B x N x 3
            mask = self.vox_util.xyz2circles(clist_camX0, self.Z2, self.Y2, self.X2, radius=1.5, soft=True, already_mem=False)
            mask = self.crop_feat(mask, self.crop)
            mask = mask * self.scorelist_s[:,0].reshape(self.B, self.N, 1, 1, 1)
            self.center_mask = torch.max(mask, dim=1, keepdim=True)[0]
            self.summ_writer.summ_oned('center/center_mask', self.center_mask, bev=True)

            print('feat_halfmemX0', feat_halfmemX0.shape)

            feat_cat = torch.cat([feat_low,
                                  feat_mid,
                                  feat_high,
                                  feat_halfmemX0], dim=1)
            center_loss, lrtlist_camX0_e, scorelist_e = self.centernet(
                feat_cat, 
                self.crop,
                self.vox_util, 
                self.center_mask,
                lrtlist_camX0,
                lrtlist_halfmemX0,
                scorelist, 
                self.summ_writer)
            total_loss += center_loss
            print('cen total_loss', total_loss.detach().cpu().numpy())

            print('lrtlist_camX0_e', lrtlist_camX0_e.shape)
            print('lrtlist_camX0_g', lrtlist_camX0.shape)

            if lrtlist_camX0_e is not None:
                # lrtlist_camX_e = utils.geom.apply_4x4_to_lrtlist(self.camXs_T_camX0[:,0], lrtlist_camX0_e)
                # lrtlist_camR_e = utils.geom.apply_4x4_to_lrtlist(self.camRs_T_camXs[:,0], lrtlist_camXs_e)
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_e',
                    self.rgb_camXs[0:1,0],
                    lrtlist_camX0_e[0:1], 
                    scorelist_e[0:1],
                    torch.arange(50).reshape(1, 50).long().cuda(), # tids
                    self.pix_T_cams[0:1,0])
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_g',
                    self.rgb_camXs[0:1,0],
                    lrtlist_camX0[0:1],
                    scorelist[0:1],
                    self.tidlist_s[0:1,0],
                    self.pix_T_cams[0:1,0])

                lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
                    lrtlist_camX0_e[0:1], lrtlist_camX0[0:1], scorelist_e[0:1], scorelist[0:1])

                # print('lrtlist_e', lrtlist_e.shape)
                # print('lrtlist_g', lrtlist_g.shape)

                clist_e = utils.geom.get_clist_from_lrtlist(lrtlist_e)
                clist_g = utils.geom.get_clist_from_lrtlist(lrtlist_g)
                # print('clist_e', clist_e.detach().cpu().numpy())
                # print('clist_g', clist_g.detach().cpu().numpy())
                
                iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

                print('scorelist_e,g sum:',
                      torch.sum(scorelist_e).detach().cpu().numpy(),
                      torch.sum(scorelist_g).detach().cpu().numpy())
                
                if torch.sum(scorelist_g) > 0:
                    if torch.sum(scorelist_e) > 0:
                        maps_3d, maps_2d = utils.eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                        for ind, overlap in enumerate(iou_thresholds):
                            print('maps_3d_%d' % ind, maps_3d[ind])
                            print('maps_2d_%d' % ind, maps_2d[ind])
                            self.summ_writer.summ_scalar('ap_3d/%.2f_iou' % overlap, maps_3d[ind])
                            self.summ_writer.summ_scalar('ap_2d/%.2f_iou' % overlap, maps_2d[ind])
                    else:
                        print('scorelist_e is empty')
                        for ind, overlap in enumerate(iou_thresholds):
                            self.summ_writer.summ_scalar('ap_3d/%.2f_iou' % overlap, 0.0)
                            self.summ_writer.summ_scalar('ap_2d/%.2f_iou' % overlap, 0.0)
            else:
                print('output boxlist is none')

                    
                        
                
        
        if hyp.do_bkg:
            
            bkg_memX0_input = torch.cat([
                self.occ_memX0s[:,0],
                self.occ_memX0s[:,0]*self.rgb_memX0s[:,0]], dim=1)
            bkg_loss, bkg_memX0_pred = self.bkgnet(
                bkg_memX0_input,
                self.bkg_dict,
                target_occ=occ_memX0_sup,
                target_vis=vis_memX0_sup,
                is_train=(self.set_name=='train'),
                summ_writer=self.summ_writer)
            total_loss += bkg_loss
            
        # if hyp.do_occ:

        #     # # be more conservative with "free"
        #     # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
        #     # free_memi_sup = 1.0 - (F.conv3d(1.0 - free_memi_sup, weights, padding=1)).clamp(0, 1)

        #     occ_loss, occ_memX0_pred = self.occnet(
        #         feat_halfmemR0s[:,s],
        #         occ_memX0_sup,
        #         free_memX0_sup,
        #         torch.ones_like(free_memX0_sup),
        #         self.summ_writer)
        #     total_loss += occ_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False


    def run_softarg_train(self, feed):
        # in this block, i want to train softargnet on single frames, just to produce boxes with high center-surround
        
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.rgb_mem0s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam0s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs0 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam0s, self.rgb_mem0s)

        self.rgb_mem1s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam1s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs1 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam1s, self.rgb_mem1s)

        self.rgb_mem2s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam2s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs2 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam2s, self.rgb_mem2s)

        self.rgb_mem3s = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cam3s), self.Z1, self.Y1, self.X1, __p(self.pix_T_cams)))
        self.rgb_memRs3 = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_cam3s, self.rgb_mem3s)

        self.rgb_memRs__ = torch.stack([self.rgb_memRs0,
                                        self.rgb_memRs1,
                                        self.rgb_memRs2,
                                        self.rgb_memRs3], dim=0)
        self.occ_memRs__ = torch.stack([self.occ_memRs0,
                                        self.occ_memRs1,
                                        self.occ_memRs2,
                                        self.occ_memRs3], dim=0)
        self.rgb_memRs = utils.basic.reduce_masked_mean(self.rgb_memRs__, self.occ_memRs__.repeat(1, 1, 1, 3, 1, 1, 1), dim=0)
                                      
        self.rgb_memR0s = self.vox_util.apply_4x4s_to_voxs(self.camR0s_T_camRs, self.rgb_memRs)

        self.summ_writer.summ_feat('3d_feats/obj', self.obj, pca=True)

        if hyp.do_feat3d:

            feat_memR0s_input = torch.cat([
                self.occ_memR0s,
                self.occ_memR0s*self.rgb_memR0s], dim=2)
            feat_loss, feat_halfmemR0s_, feat_bunch = self.feat3dnet(
                __p(feat_memR0s_input), norm=False, summ_writer=None)
            total_loss += feat_loss
            feat_halfmemR0s = __u(feat_halfmemR0s_)
            self.summ_writer.summ_feats('3d_feats/feat_memR0s_inputs', feat_memR0s_input.unbind(1), pca=True)
            self.summ_writer.summ_feats('3d_feats/feat_halfmemR0', feat_halfmemR0s.unbind(1), pca=True)

            # _, _, _, Z_, Y_, X_ = list(feat_halfmemR0s.shape)
            # feat_halfmemR0s_np = (feat_halfmemR0s.detach().cpu().reshape(self.S, -1)).numpy()
            # feat_halfmemR0_median_np = np.median(feat_halfmemR0s_np, axis=0)
            # feat_halfmemR0_median = torch.from_numpy(feat_halfmemR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()
            # self.summ_writer.summ_feat('3d_feats/feat_halfmemR0_median', feat_halfmemR0_median, pca=True)

            # diffs = torch.norm(feat_halfmemR0_median.unsqueeze(1) - feat_halfmemR0s, dim=2, keepdim=True)
            # self.summ_writer.summ_oneds('3d_feats/feat_halfmemR0_median_diffs', diffs, bev=True, norm=True)
            
            
        if hyp.do_occ:

            occ_total = 0.0
            occ_halfmemR0s = []
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
                    self.Z2, self.Y2, self.X2,
                    agg=True)
                
                # be more conservative with "free"
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                free_memR0_sup = 1.0 - (F.conv3d(1.0 - free_memR0_sup, weights, padding=1)).clamp(0, 1)
                # we need to crop
                occ_memR0_sup = self.crop_feat(occ_memR0_sup, self.crop)
                free_memR0_sup = self.crop_feat(free_memR0_sup, self.crop)
                
                occ_loss, occ_halfmemR0_pred = self.occnet(
                    feat_halfmemR0s[:,s],
                    occ_memR0_sup,
                    free_memR0_sup,
                    torch.ones_like(free_memR0_sup))
                # occ_total += occ_loss
                
                occ_halfmemR0s.append(F.sigmoid(occ_halfmemR0_pred))
            # self.summ_writer.summ_occ('occ/occ_memR_pred', F.sigmoid(occ_memR_pred))
            self.summ_writer.summ_occs('occ/occ_halfmemR0s', occ_halfmemR0s)
            total_loss = utils.misc.add_loss('occ/occ_total', total_loss, occ_total, 1.0, self.summ_writer)
            occ_halfmemR0s = torch.stack(occ_halfmemR0s, dim=1)
            
            # _, _, _, Z_, Y_, X_ = list(feat_halfmemR0s.shape)
            # feat_halfmemR0s_np = (feat_halfmemR0s.detach().cpu().reshape(self.S, -1)).numpy()
            # feat_halfmemR0_median_np = np.median(feat_halfmemR0s_np, axis=0)
            # feat_halfmemR0_median = torch.from_numpy(feat_halfmemR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()
            # self.summ_writer.summ_feat('3d_feats/feat_halfmemR0_median', feat_halfmemR0_median, pca=True)

            # diffs = torch.norm(feat_halfmemR0_median.unsqueeze(1) - feat_halfmemR0s, dim=2, keepdim=True)
            # self.summ_writer.summ_oneds('3d_feats/feat_halfmemR0_median_diffs', diffs, bev=True, norm=True)

            _, _, _, Z_, Y_, X_ = list(occ_halfmemR0s.shape)
            occ_halfmemR0s_np = (occ_halfmemR0s.detach().cpu().reshape(self.S, -1)).numpy()
            occ_halfmemR0_median_np = np.median(occ_halfmemR0s_np, axis=0)
            occ_halfmemR0_median = torch.from_numpy(occ_halfmemR0_median_np).float().reshape(self.B, -1, Z_, Y_, X_).cuda()
            self.summ_writer.summ_occ('3d_feats/occ_halfmemR0_median', occ_halfmemR0_median)
            diffs = torch.norm(occ_halfmemR0_median.unsqueeze(1) - occ_halfmemR0s, dim=2, keepdim=True)
            self.summ_writer.summ_oneds('3d_feats/occ_halfmemR0_median_diffs', diffs.unbind(1), bev=True, norm=True)


        if hyp.do_softarg:
            feat_lows_, feat_mids_, feat_highs_ = feat_bunch

            def assert_cropped_shape(feat, Z, Y, X, crop_guess):
                # make sure the shapes match what we expect
                _, _, Z_, Y_, X_ = list(feat.shape)
                Z_crop = int((Z - Z_)/2)
                Y_crop = int((Y - Y_)/2)
                X_crop = int((X - X_)/2)
                crop = (Z_crop, Y_crop, X_crop)
                if not (crop==crop_guess):
                    print('crop', crop)
                    assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
                return True
            assert_cropped_shape(feat_lows_, self.Z4, self.Y4, self.X4, self.crop_low)
            assert_cropped_shape(feat_mids_, self.Z4, self.Y4, self.X4, self.crop_mid)
            assert_cropped_shape(feat_highs_, self.Z2, self.Y2, self.X2, self.crop)
            feat_lows_ = self.pad_feat(feat_lows_, self.crop_low)
            feat_mids_ = self.pad_feat(feat_mids_, self.crop_mid)
            feat_highs_ = self.pad_feat(feat_highs_, self.crop)
            feat_lows_ = F.interpolate(feat_lows_, scale_factor=2, mode='trilinear')
            feat_mids_ = F.interpolate(feat_mids_, scale_factor=2, mode='trilinear')
            feat_lows_ = self.crop_feat(feat_lows_, self.crop)
            feat_mids_ = self.crop_feat(feat_mids_, self.crop)
            feat_highs_ = self.crop_feat(feat_highs_, self.crop)
            self.summ_writer.summ_feat('3d_feats/feat_lows_', feat_lows_, pca=True)
            self.summ_writer.summ_feat('3d_feats/feat_mids_', feat_mids_, pca=True)
            self.summ_writer.summ_feat('3d_feats/feat_highs_', feat_highs_, pca=True)

            feat_cats_ = torch.cat([feat_lows_,
                                    feat_mids_,
                                    feat_highs_], dim=1)
            feat_cats = __u(feat_cats_)
            # for s in list(range(self.S)):
            # _, lrtlist_camR0s_, scorelist_s_ = self.softargnet(
            softarg_loss, lrtlist_camR0s_ = self.softargnet(
                feat_cats_, 
                self.crop,
                self.vox_util,
                summ_writer=self.summ_writer)
            total_loss += softarg_loss
            
            lrtlist_camR0s = __u(lrtlist_camR0s_)
            scorelist_s = torch.ones((self.B, self.S, 1), dtype=torch.float32).cuda()
            # lrtlist_camR0s is B x S x K x 19
            # scorelist_s is B x S x K
            lrtlist_camRs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camRs_T_camR0s), __p(lrtlist_camR0s)))

            box_vis = []
            box_vis_bev = []
            for s in list(range(self.S)):
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.occ_memR0s[0:1,s],
                    # '', self.crop_feat(self.occ_halfmemR0s[0:1,s], self.crop),
                    lrtlist_camR0s[0:1,s],
                    scorelist_s[0:1,s], # scores
                    torch.ones(1,50).long().cuda(), # tids
                    self.vox_util,
                    already_mem=False, only_return=True))

                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camRs[0:1,s],
                    lrtlist_camRs[0:1,s],
                    scorelist_s[0:1,s],
                    torch.arange(50).reshape(1, 50).long().cuda(), # tids
                    self.pix_T_cams[0:1,0], only_return=True))
            self.summ_writer.summ_rgbs('softarg/boxes_e_pers', box_vis)
            self.summ_writer.summ_rgbs('softarg/boxes_e_bev', box_vis_bev)

            if False: # this is not differentiable yet
                center_masks_ = self.vox_util.assemble_padded_obj_masklist(__p(lrtlist_camRs), __p(torch.ones_like(scorelist_s)), self.Z2, self.Y2, self.X2, coeff=0.9)
                surround_masks_ = self.vox_util.assemble_padded_obj_masklist(__p(lrtlist_camRs), __p(torch.ones_like(scorelist_s)), self.Z2, self.Y2, self.X2, coeff=1.0, additive_coeff=1.0)
                # these are B*S x K x 1 x Z x Y x X
                surround_masks_ = surround_masks_ - center_masks_
                print('center_masks_', center_masks_.shape)
                print('surround_masks_', surround_masks_.shape)

                center_masks = __u(self.crop_feat(center_masks_.squeeze(2), self.crop))
                surround_masks = __u(self.crop_feat(surround_masks_.squeeze(2), self.crop))
                # these are B x S x K x Z x Y x X

                self.summ_writer.summ_feats('softarg/center_masks', center_masks.unbind(1), pca=False)
                self.summ_writer.summ_feats('softarg/surround_masks', surround_masks.unbind(1), pca=False)

                print('center_masks', center_masks.shape)
                print('surround_masks', surround_masks.shape)

                K = 1
                occ_memRs_ = occ_memRs.repeat(1, 1, K, 1, 1, 1)
                print('occ_memRs_', occ_memRs_.shape)
                center_ = utils.basic.reduce_masked_mean(occ_memRs_, center_masks, dim=[3,4,5])
                surround_ = utils.basic.reduce_masked_mean(occ_memRs_, surround_masks, dim=[3,4,5])
                cs_score_ = center_ - surround_
                # this is B x S x K

                # ok. this score is high when things are good. so maybe this is the loss: 
                cs_loss = torch.exp(-cs_score_)
                print('cs_loss', cs_loss.detach().cpu().numpy(), cs_loss.shape)
                # cs_loss = utils.basic.reduce_masked_mean(cs_loss, utils.basic.normalize(scorelist_s))
                cs_loss = torch.mean(cs_loss)
                total_loss = utils.misc.add_loss('softarg/cs_loss', total_loss, cs_loss, hyp.softarg_coeff, self.summ_writer)
                # print('total_loss', total_loss.detach().cpu().numpy(), total_loss.shape)

            lrt_camRs = lrtlist_camRs.squeeze(2)
            score_s = scorelist_s.squeeze(2)
            
            # now that i have some boxes, i want to use them to render some new scenes
            camRs_T_zoom = __u(self.vox_util.get_ref_T_zoom(__p(lrt_camRs), self.Z2, self.Y2, self.X2))
            # we want to sample for each location in the mem grid
            xyz_mems_ = utils.basic.gridcloud3d(self.B*self.S, self.Z1, self.Y1, self.X1, norm=False)
            # this is B*S x Z*Y*X x 3
            xyz_camRs_ = self.vox_util.Mem2Ref(xyz_mems_, self.Z1, self.Y1, self.X1)
            camRs_T_zoom_ = __p(camRs_T_zoom)
            zoom_T_camRs_ = camRs_T_zoom_.inverse() # note this is not a rigid transform
            xyz_zooms_ = utils.geom.apply_4x4(zoom_T_camRs_, xyz_camRs_)

            # we will do the whole traj at once (per obj)
            # note we just have one feat for the whole traj, so we tile up
            obj_feats = self.obj.unsqueeze(1).repeat(1, self.S, 1, 1, 1, 1)
            obj_feats_ = __p(obj_feats)
            # this is B*S x Z x Y x X x C

            # to sample, we need feats_ in ZYX order
            obj_featRs_ = utils.samp.sample3d(obj_feats_, xyz_zooms_, self.Z1, self.Y1, self.X1)
            obj_featRs = __u(obj_featRs_)
            self.summ_writer.summ_feats('3d_feats/obj_featRs', obj_featRs.unbind(1), pca=True)

            obj_maskRs_ = utils.samp.sample3d(torch.ones_like(obj_feats_[:,0:1]), xyz_zooms_, self.Z1, self.Y1, self.X1)
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            # obj_maskRs_ = 1.0 - (F.conv3d(1.0 - obj_maskRs_, weights, padding=1)).clamp(0, 1)
            obj_maskRs_ = F.conv3d(obj_maskRs_, weights, padding=1).clamp(0, 1)
            obj_maskRs = __u(obj_maskRs_)
            
            self.summ_writer.summ_feats('3d_feats/obj_maskRs', obj_maskRs.unbind(1), pca=False)

            bkg_memR0_input = feat_memR0s_input[:,0] * (1.0 - obj_maskRs[:,0])
            _, bkg_halfmemR0, _ = self.feat3dnet(bkg_memR0_input, norm=False)
            self.summ_writer.summ_feat('3d_feats/bkg_featR0', bkg_halfmemR0, pca=True)
            
        if hyp.do_render:

            rgbs_obj_e = []
            rgbs_full_e = []
            rgbs_g = []
            box_vis = []
            render_total = 0.0
            occ_total = 0.0
            if hyp.do_softarg:
                bkg_memR0 = F.interpolate(self.pad_feat(bkg_halfmemR0, self.crop), scale_factor=2, mode='trilinear')

            for s in list(range(self.S)):
                # feat_halfmemR0 = feat_halfmemR0s[:,s]
                # feat_halfmemR0 = bkg_halfmemR0

                fullfeat_memR0 = F.interpolate(self.pad_feat(feat_halfmemR0s[:,s], self.crop), scale_factor=2, mode='trilinear')
                fullrgb_memR0 = fullfeat_memR0[:,1:]
                fullocc_memR0 = fullfeat_memR0[:,0:1]

                if hyp.do_softarg:
                    feat_memR0 = bkg_memR0.detach() * (1.0 - obj_maskRs[:,s]) + obj_featRs[:,s] * obj_maskRs[:,s]
                    rgb_memR0 = feat_memR0[:,1:]
                    occ_memR0 = feat_memR0[:,0:1]

                    obj_halfmask = self.crop_feat(F.interpolate(obj_maskRs[:,s], scale_factor=0.5, mode='trilinear'), self.crop)
                    obj_halffeat = self.crop_feat(F.interpolate(obj_featRs[:,s], scale_factor=0.5, mode='trilinear'), self.crop)
                    feat_halfmemR0 = bkg_halfmemR0.detach() * (1.0 - obj_halfmask) + obj_halffeat * obj_halfmask

                    if hyp.do_occ:


                        camR0_T_cam0 = self.camR0s_T_cam0s[:,s]
                        camR0_T_cam1 = self.camR0s_T_cam1s[:,s]
                        camR0_T_cam2 = self.camR0s_T_cam2s[:,s]
                        camR0_T_cam3 = self.camR0s_T_cam3s[:,s]
                        xyz_cam0 = self.xyz_cam0s[:,s]
                        xyz_cam1 = self.xyz_cam1s[:,s]
                        xyz_cam2 = self.xyz_cam2s[:,s]
                        xyz_cam3 = self.xyz_cam3s[:,s]

                        xyz_camXs = torch.stack([
                            xyz_cam0,
                            xyz_cam1,
                            xyz_cam2,	
                            xyz_cam3], dim=1)
                        camR0s_T_camXs = torch.stack([
                            camR0_T_cam0,
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
                        # occ_memR0_sup = self.crop_feat(occ_memR0_sup, self.crop)
                        # free_memR0_sup = self.crop_feat(free_memR0_sup, self.crop)

                        occ_loss, occ_memR0_pred = self.occnet(
                            feat_memR0,
                            occ_memR0_sup,
                            free_memR0_sup,
                            obj_maskRs[:,s])
                        occ_total += occ_loss

                        # for vis, let's also see occ_halfmemR
                        occ_loss, occ_halfmemR0_pred = self.occnet(feat_halfmemR0)
                        
                    # self.summ_writer.summ_occ('occ/obj_occ_memR_pred', F.sigmoid(occ_memR_pred))
                    self.summ_writer.summ_occ('occ/obj_occ_halfmemR0_pred', F.sigmoid(occ_halfmemR0_pred))
                total_loss = utils.misc.add_loss('occ/obj_occ_total', total_loss, occ_total, 1.0, self.summ_writer)
                    
                # self.summ_writer.summ_occ('3d_feats/occ_halfmemR0', F.sigmoid(occ_memR0))

                PH, PW = hyp.PH, hyp.PW
                sy = float(PH)/float(hyp.H)
                sx = float(PW)/float(hyp.W)
                assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
                assert(sy==0.5 or sy==1.0)
                projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

                zoom_box = torch.from_numpy(np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)).cuda()
                zoom_box = zoom_box.reshape(1, 4).repeat(self.B, 1).cuda()
                zoompix_T_cam, _ = utils.geom.convert_box2d_to_intrinsics(
                    zoom_box, projpix_T_cams[:,s], PH, PW, use_image_aspect_ratio=True)

                fullfeat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                    zoompix_T_cam, self.camRs_T_camR0s[:,s], fullrgb_memR0,
                    hyp.view_depth, PH, PW, noise_amount=2.0)
                fullocc_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                    zoompix_T_cam, self.camRs_T_camR0s[:,s], fullocc_memR0,
                    hyp.view_depth, PH, PW, grid_z_vec=dists)

                if hyp.do_softarg:
                    feat_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                        zoompix_T_cam, self.camRs_T_camR0s[:,s], rgb_memR0,
                        hyp.view_depth, PH, PW, grid_z_vec=dists)
                    occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                        zoompix_T_cam, self.camRs_T_camR0s[:,s], occ_memR0,
                        hyp.view_depth, PH, PW, grid_z_vec=dists)

                rgb_R00 = self.rgb_camRs[:,s]
                depth_R00 = self.depth_camRs[:,s]
                valid_R00 = self.valid_camRs[:,s]
                # to do: replace this line with a crop_and_resize
                rgb_R00 = rgb_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                depth_R00 = depth_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                valid_R00 = valid_R00[:,:,int(self.H*0.25):int(self.H*0.75), int(self.W*0.25):int(self.W*0.75)]
                rgb_R00 = F.interpolate(rgb_R00, size=(PH, PW), mode='bilinear')
                depth_R00 = F.interpolate(depth_R00, size=(PH, PW), mode='bilinear')
                valid_R00 = F.interpolate(valid_R00, size=(PH, PW), mode='bilinear')

                if hyp.do_softarg:
                    render_loss, rgb_e, _, _ = self.rendernet(
                        feat_proj,
                        occ_proj,
                        dists,
                        rgb_g=rgb_R00,
                        depth_g=depth_R00,
                        valid=valid_R00)
                        # summ_writer=self.summ_writer)
                    render_total += render_loss
                    rgbs_obj_e.append(rgb_e)

                render_loss, rgb_e, _, _ = self.rendernet(
                    fullfeat_proj,
                    fullocc_proj,
                    dists,
                    rgb_g=rgb_R00,
                    depth_g=depth_R00,
                    valid=valid_R00)
                render_total += render_loss
                rgbs_full_e.append(rgb_e)
                
                rgbs_g.append(rgb_R00*valid_R00)

            # total_loss += render_loss
            total_loss = utils.misc.add_loss('render/render_total', total_loss, render_total, 1.0, self.summ_writer)
            self.summ_writer.summ_rgbs('3d_feats/rgbs_full_e', rgbs_full_e)
            if hyp.do_softarg:
                self.summ_writer.summ_rgbs('3d_feats/rgbs_obj_e', rgbs_obj_e)
            self.summ_writer.summ_rgbs('3d_feats/rgbs_g', rgbs_g)

            
            
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
                # return self.run_cs_train(feed)
                return self.run_softarg_train(feed)
            elif self.set_name=='val':
                return self.run_train(feed)
            elif self.set_name=='test':
                return self.run_sfm(feed)
                # return self.run_orb(feed)
                # return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
