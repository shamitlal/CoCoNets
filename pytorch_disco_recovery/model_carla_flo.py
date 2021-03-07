import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from tensorboardX import SummaryWriter
from backend import saverloader, inputs

from model_base import Model
from nets.featnet3D import FeatNet3D
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

from utils_basic import *
import vox_util
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net

np.set_printoptions(precision=2)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_FLO(Model):
    # take over __init__() from base
    def __init__(self, checkpoint_dir, log_dir):

        print('------ CREATING NEW MODEL ------')
        print(hyp.name)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.all_inputs = inputs.get_inputs()
        print("------ Done getting inputs ------")
        
        self.device = torch.device("cuda")
        
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaFloModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

    # take over go() from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        # print('there seem to be %d examples'
        # self.Z = np.empty((len(train_loader.dataset), code_dim))
        

        # self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        # self.z = torch.zeros([B, hyp.feat3D_dim, self.Z, self.Y, self.X], torch.float32).cuda()
        # self.z = torch.autograd.Variable(self.z, requires_grad=True)

        
        set_nums = []
        set_names = []
        set_batch_sizes = []
        set_data_formats = []
        set_data_names = []
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
                set_data_names.append(hyp.data_names[set_name])
                set_seqlens.append(hyp.seqlens[set_name])
                set_names.append(set_name)
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))
            
        if hyp.do_test:
            all_ious = np.zeros([hyp.max_iters, hyp.S_test], np.float32)
            all_diff_2ds = np.zeros([hyp.max_iters, hyp.S_test], np.float32)
            all_diff_3ds = np.zeros([hyp.max_iters, hyp.S_test], np.float32)

            all_best_ious = np.zeros([hyp.max_iters], np.float32)
            all_selected_ious = np.zeros([hyp.max_iters], np.float32)
            all_worst_ious = np.zeros([hyp.max_iters], np.float32)
            all_avg_ious = np.zeros([hyp.max_iters], np.float32)
            
            test_count = 0
            
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': hyp.lr},
        ])
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
            
        print("------ Done loading weights ------")

                

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            # reset set_loader after each epoch
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0:
                    set_loaders[i] = iter(set_input)
            for (set_num,
                 set_data_format,
                 set_data_name,
                 set_seqlen,
                 set_name,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_data_formats,
                set_data_names,
                set_seqlens,
                set_names,
                set_batch_sizes,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders
            ):   
                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                
                output_dict = dict()


                if log_this or set_do_backprop or hyp.do_test:
                    # print('%s: set_num %d; set_data_format %s; set_batch_size %d; set_seqlen %s; log_this %d; set_do_backprop %d; ' % (
                    #     set_name, set_num, set_data_format, set_batch_size, set_seqlen, log_this, set_do_backprop))
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
                    feed_cuda['data_ind'] = data_ind
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_data_name'] = set_data_name
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_batch_size'] = set_batch_size


                    
                    
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
                            loss.backward()
                            self.optimizer.step()

                        if hyp.do_test and (not returned_early) and False:
                            ious = results['ious']
                            ious = ious[0].cpu().numpy()
                            all_ious[test_count] = ious

                            best_ious = results['best_iou']
                            best_ious = best_ious.squeeze().cpu().numpy()
                            all_best_ious[test_count] = best_ious

                            selected_ious = results['selected_iou']
                            selected_ious = selected_ious.squeeze().cpu().numpy()
                            all_selected_ious[test_count] = selected_ious

                            avg_ious = results['avg_iou']
                            avg_ious = avg_ious.squeeze().cpu().numpy()
                            print('avg_ious', avg_ious)
                            all_avg_ious[test_count] = avg_ious
                            
                            worst_ious = results['worst_iou']
                            worst_ious = worst_ious.squeeze().cpu().numpy()
                            print('worst_ious', worst_ious)
                            all_worst_ious[test_count] = worst_ious

                            diff_2ds = results['diff_2ds']
                            diff_2ds = diff_2ds[0].cpu().numpy()
                            all_diff_2ds[test_count] = diff_2ds

                            diff_3ds = results['diff_3ds']
                            diff_3ds = diff_3ds[0].cpu().numpy()
                            all_diff_3ds[test_count] = diff_3ds
                            
                            test_count += 1
                            # print('all_ious', all_ious[:test_count])
                            mean_ious = np.mean(all_ious[:test_count], axis=0)
                            mean_best_ious = np.mean(all_best_ious[:test_count], axis=0)
                            mean_avg_ious = np.mean(all_avg_ious[:test_count], axis=0)
                            mean_worst_ious = np.mean(all_worst_ious[:test_count], axis=0)
                            mean_selected_ious = np.mean(all_selected_ious[:test_count], axis=0)
                            # mean_confs = np.mean(all_confs[:test_count], axis=0)
                            print('-'*10)
                            print('mean_ious', mean_ious)
                            print('mean_best_ious', mean_best_ious)
                            print('mean_selected_ious', mean_selected_ious)
                            print('mean_avg_ious', mean_avg_ious)
                            print('mean_worst_ious', mean_worst_ious)
                            # print('mean_confs', mean_confs)

                            corr_2d = np.corrcoef(np.reshape(all_diff_2ds[:test_count], [-1]),
                                                  np.reshape(all_ious[:test_count], [-1]))[0,1]
                            corr_3d = np.corrcoef(np.reshape(all_diff_3ds[:test_count], [-1]),
                                                  np.reshape(all_ious[:test_count], [-1]))[0,1]
                            print('corr_2d', corr_2d)
                            print('corr_3d', corr_3d)
                            

                        if hyp.do_emb3D:
                            def update_slow_network(slow_net, fast_net, beta=0.999):
                                param_k = slow_net.state_dict()
                                param_q = fast_net.named_parameters()
                                for n, q in param_q:
                                    if n in param_k:
                                        param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                                slow_net.load_state_dict(param_k)
                            update_slow_network(self.model.featnet3D_slow, self.model.featnet3D)
                        
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
                
        for writer in set_writers: # close writers to flush cache into file
            writer.close()

class CarlaFloModel(nn.Module):
    def __init__(self):
        super(CarlaFloModel, self).__init__()
        self.crop_guess = (18,18,18)
        self.featnet3D = FeatNet3D(in_dim=4)
        self.occnet = OccNet()
        self.flownet = FlowNet()
        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

    def crop_feat(self, feat_pad):
        Z_pad, Y_pad, X_pad = self.crop_guess
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    def pad_feat(self, feat):
        Z_pad, Y_pad, X_pad = self.crop_guess
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad
    
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=16,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.set_data_name = feed['set_data_name']
        self.S = feed["set_seqlen"]
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils_geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        # if self.set_name=='test':
        
        self.anchor = int(self.S/2)

        self.camXAs_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=self.anchor)
        self.camXs_T_camXAs = __u(utils_geom.safe_inverse(__p(self.camXAs_T_camXs)))
        self.xyz_camXAs = __u(utils_geom.apply_4x4(__p(self.camXAs_T_camXs), __p(self.xyz_camXs)))
        # _, self.scene_centroid = utils_geom.split_rt(self.origin_T_camRs[:,self.anchor])

        all_ok = False
        num_tries = 0
        while not all_ok:
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            num_tries += 1
            all_ok = True
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

            occ_memXAA = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,self.anchor], self.Z2, self.Y2, self.X2)
            occ_memXAA = self.crop_feat(occ_memXAA)

            occ_memXAA = occ_memXAA.reshape(self.B, -1)
            num_inb = torch.sum(occ_memXAA, dim=1)
            # this is B
            if torch.mean(num_inb) < 300:
                # print('num_inb', num_inb.detach().cpu().numpy())
                all_ok = False
                
            if num_tries > 20:
                print('cannot find a good centroid; returning early')
                return False

        # print('scene_centroid', scene_centroid)
        
        self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
        # scene_centroid_x = 0.0
        # scene_centroid_y = 1.5 # 1.0 is a bit too high up
        # scene_centroid_z = 18.0
        # scene_centroid = np.array([scene_centroid_x,
        #                            scene_centroid_y,
        #                            scene_centroid_z]).reshape([1, 3])
        # self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        # self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

        self.rgb_camXs = feed['rgb_camXs']
        # visX_e = []
        # for s in list(range(0, self.S, 2)):
        #     visX_e.append(self.summ_writer.summ_lrtlist(
        #         '', self.rgb_camXs[:,s],
        #         self.lrtlist_camXs[:,s],
        #         self.scorelist_s[:,s],
        #         self.tidlist_s[:,s],
        #         self.pix_T_cams[:,s], only_return=True))
        # self.summ_writer.summ_rgbs('obj/box_camXs_g', visX_e)

        # print('set_name', self.set_name)
        # print('vox_size_X', self.vox_size_X)
        # print('vox_size_Y', self.vox_size_Y)
        # print('vox_size_Z', self.vox_size_Z)


        # ## projected depth, and inbound mask
        # self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # # we need to go to X0 to see what will be inbounds
        # self.dense_xyz_camX0s_ = utils_geom.apply_4x4(__p(self.camX0s_T_camXs), self.dense_xyz_camXs_)
        # self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camX0s_, self.Z, self.Y, self.X).float()
        # self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        # self.depth_camXs = __u(self.depth_camXs_)
        # self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        # self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
        # self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        
        return True # OK
            
    def run_synth_flow(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0).cuda()
        S = feed["set_seqlen"]
        K = 8
        __p = lambda x: pack_seqdim(x, self.B)
        __u = lambda x: unpack_seqdim(x, self.B)

        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))
        
        # # extract the data unique to this mode
        # boxlist_camRs = feed["boxlists"]
        # tidlist_s = feed["tidlists"] # coordinate-less and plural
        # scorelist_s = feed["scorelists"] # coordinate-less and plural

        # lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))).reshape(self.B, S, self.N, 19)
        # lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(lrtlist_camRs)))
        # # stabilize boxes for ego/cam motion
        # lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camX0_T_camXs), __p(lrtlist_camXs)))
        # # these are is B x S x N x 19
        
        self.occ_memX0s, self.rgb_memX0s, self.flow_memX0, _ = utils_misc.get_synth_flow_v2(
            self.xyz_camX0s[:,0], 
            self.occ_memX0s[:,0], 
            self.rgb_memX0s[:,0],
            self.vox_util, 
            summ_writer=self.summ_writer,
            sometimes_zero=False)
        # self.occ_memX0s, self.unp_memX0s, self.flow_memX0, _ = utils_misc.get_synth_flow(
        #     self.occ_memX0s,
        #     self.unp_memX0s,
        #     summ_writer=self.summ_writer,
        #     sometimes_zero=False,
        #     do_vis=False)

        self.summ_writer.summ_occs('flow_g/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        self.summ_writer.summ_3D_flow('flow_g/synth_flow_memX0', self.flow_memX0, clip=0.0)
        occ_memX1_backwarped = utils_samp.backwarp_using_3D_flow(self.occ_memX0s[:,1], self.flow_memX0, binary_feat=True)
        self.summ_writer.summ_occs('flow_g/occ_memX0s_aligned_g', [self.occ_memX0s[:,0], occ_memX1_backwarped])
        
        self.flow_memX0_half = utils_basic.downsample3Dflow(self.flow_memX0, 2)

        # ego-stab
        if hyp.do_feat3D:
            # occ_memX0s is B x S x 1 x H x W x D
            # rgb_memX0s is B x S x 3 x H x W x D
            feat_memX0s_input = torch.cat([self.occ_memX0s, self.occ_memX0s*self.rgb_memX0s], dim=2)
            feat_memX0s_input_ = __p(feat_memX0s_input)
            feat_loss, feat_memX0s_, _ = self.featnet3D(
                feat_memX0s_input_,
                self.summ_writer)
            total_loss += feat_loss
            feat_memX0s = __u(feat_memX0s_)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s_input', torch.unbind(feat_memX0s_input, dim=1), pca=True)
            self.summ_writer.summ_feats('3D_feats/feat_memX0s_output', torch.unbind(feat_memX0s, dim=1), pca=True)

        if hyp.do_flow:
            flow_loss, flowX0_pred = self.flownet(
                feat_memX0s[:,0],
                feat_memX0s[:,1],
                self.crop_feat(self.flow_memX0_half),
                self.crop_feat(self.occ_memX0s_half[:,0]),
                self.summ_writer)
            total_loss += flow_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_real_flow(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0).cuda()
        # S = feed["set_seqlen"]
        # K = 8
        # __p = lambda x: pack_seqdim(x, self.B)
        # __u = lambda x: unpack_seqdim(x, self.B)
        
        # # extract the data unique to this mode
        # boxlist_camRs = feed["boxlists"]
        # tidlist_s = feed["tidlists"] # coordinate-less and plural
        # scorelist_s = feed["scorelists"] # coordinate-less and plural

        # lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))).reshape(self.B, S, self.N, 19)
        # lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(lrtlist_camRs)))
        # # stabilize boxes for ego/cam motion
        # lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camX0_T_camXs), __p(lrtlist_camXs)))
        # # these are is B x S x N x 19

        # (obj_lrtlist_camX0s,
        #  obj_scorelist_s,
        # ) = utils_misc.collect_object_info(lrtlist_camX0s,
        #                                    tidlist_s,
        #                                    scorelist_s,
        #                                    K, mod='X0',
        #                                    do_vis=False)
        # (obj_lrtlist_camXs,
        #  obj_scorelist_s,
        # ) = utils_misc.collect_object_info(lrtlist_camXs,
        #                                    tidlist_s,
        #                                    scorelist_s,
        #                                    K, mod='X',
        #                                    do_vis=False)

        # flowX0 = utils_misc.get_gt_flow(
        #     obj_lrtlist_camX0s,
        #     obj_scorelist_s,
        #     utils_geom.eye_4x4s(self.B, S),
        #     self.Z, self.Y, self.X,
        #     K=K, 
        #     mod='X0',
        #     vis=False,
        #     summ_writer=self.summ_writer)

        # flowX0_half = utils_misc.get_gt_flow(
        #     obj_lrtlist_camX0s,
        #     obj_scorelist_s,
        #     utils_geom.eye_4x4s(self.B, S),
        #     self.Z2, self.Y2, self.X2,
        #     K=K, 
        #     mod='X0',
        #     vis=False,
        #     summ_writer=self.summ_writer)

        # for b in list(range(self.B)):
        #     # ensure there is at least one voxel we will supervise with
        #     if torch.sum(torch.abs(flowX0_half)) < 1.0:
        #         return total_loss, results, True

        # # include_egomotion = False
        # # if include_egomotion:
        # (obj_lrtlist_camX0s,
        #  obj_scorelist_s,
        # ) = utils_misc.collect_object_info(lrtlist_camX0s,
        #                                    tidlist_s,
        #                                    scorelist_s,
        #                                    K, mod='X0',
        #                                    do_vis=False)
        # (obj_lrtlist_camXs,
        #  obj_scorelist_s,
        # ) = utils_misc.collect_object_info(lrtlist_camXs,
        #                                    tidlist_s,
        #                                    scorelist_s,
        #                                    K, mod='X',
        #                                    do_vis=False)

        # flowX = utils_misc.get_gt_flow(
        #     obj_lrtlist_camX0s,
        #     # obj_lrtlist_camXs,
        #     obj_scorelist_s,
        #     self.camX0_T_camXs, 
        #     # utils_geom.eye_4x4s(self.B, S),
        #     self.Z, self.Y, self.X,
        #     K=K, 
        #     mod='X',
        #     vis=False,
        #     summ_writer=self.summ_writer)

        # flowX_half = utils_misc.get_gt_flow(
        #     obj_lrtlist_camX0s,
        #     # obj_lrtlist_camXs,
        #     obj_scorelist_s,
        #     self.camX0_T_camXs,
        #     # utils_geom.eye_4x4s(self.B, S),
        #     self.Z2, self.Y2, self.X2,
        #     K=K, 
        #     mod='X',
        #     vis=False,
        #     summ_writer=self.summ_writer)

        # self.summ_writer.summ_lrtlist('obj/lrtlist_camX0', self.rgb_camXs[:,0], lrtlist_camXs[:,0],
        #                               scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        # self.summ_writer.summ_lrtlist('obj/lrtlist_camX1', self.rgb_camXs[:,1], lrtlist_camXs[:,1],
        #                               scorelist_s[:,1], tidlist_s[:,1], self.pix_T_cams[:,1])
        # self.summ_writer.summ_3D_flow('flow_g/flowX', flowX, clip=0.0)
        # self.summ_writer.summ_3D_flow('flow_g/flowX0', flowX0, clip=0.0)
        # occX1_backwarped = utils_samp.backwarp_using_3D_flow(self.occ_memXs[:,1], flowX, binary_feat=True)
        # self.summ_writer.summ_occs('flow/occXs_aligned_g', [self.occ_memXs[:,0], occX1_backwarped])
        # occX1_backwarped = utils_samp.backwarp_using_3D_flow(self.occ_memX0s[:,1], flowX0, binary_feat=True)
        # self.summ_writer.summ_occs('flow/occX0s_aligned_g', [self.occ_memXs[:,0], occX1_backwarped])
        

        # #     for b in list(range(self.B)):
        # #         # ensure there is at least one voxel we will supervise with
        # #         if torch.sum(torch.abs(flowX_half)) < 1.0:
        # #             return total_loss, results, True

        # #     self.summ_writer.summ_lrtlist('obj/lrtlist_camX0', self.rgb_camXs[:,0], lrtlist_camXs[:,0],
        # #                                   scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        # #     self.summ_writer.summ_lrtlist('obj/lrtlist_camX1', self.rgb_camXs[:,1], lrtlist_camXs[:,1],
        # #                                   scorelist_s[:,1], tidlist_s[:,1], self.pix_T_cams[:,1])
        # #     self.summ_writer.summ_3D_flow('flow_g/flowX', flowX, clip=0.0)
        # #     occX1_backwarped = utils_samp.backwarp_using_3D_flow(self.occ_memXs[:,1], flowX, binary_feat=True)
        # #     self.summ_writer.summ_occs('flow/occs_aligned_g', [self.occ_memXs[:,0], occX1_backwarped])


        # flip = torch.rand(1)
        # if flip > 0.5:
        # # if flip > 1:
        #     # include egomotion
        #     if hyp.do_feat3D:
        #         # occ_memXs is B x S x 1 x H x W x D
        #         # unp_memXs is B x S x 3 x H x W x D
        #         featXs_input = torch.cat([self.occ_memXs, self.occ_memXs*self.unp_memXs], dim=2)
        #         featXs_input_ = __p(featXs_input)
        #         feat_loss, featXs_ = self.featnet3D(
        #             featXs_input_,
        #             self.summ_writer)
        #         total_loss += feat_loss
        #         featXs = __u(featXs_)
        #         self.summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
        #         self.summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)

        #     if hyp.do_flow:
        #         flow_loss, flowX_pred = self.flownet(
        #             featXs[:,0],
        #             featXs[:,1],
        #             flowX_half,
        #             self.occ_memXs_half[:,0],
        #             False,
        #             self.summ_writer)
        #         total_loss += flow_loss
        # else:
        #     # ego-stab
        #     if hyp.do_feat3D:
        #         # occ_memX0s is B x S x 1 x H x W x D
        #         # unp_memX0s is B x S x 3 x H x W x D
        #         featX0s_input = torch.cat([self.occ_memX0s, self.occ_memX0s*self.unp_memX0s], dim=2)
        #         featX0s_input_ = __p(featX0s_input)
        #         feat_loss, featX0s_ = self.featnet3D(
        #             featX0s_input_,
        #             self.summ_writer)
        #         total_loss += feat_loss
        #         featX0s = __u(featX0s_)
        #         self.summ_writer.summ_feats('3D_feats/featX0s_input', torch.unbind(featX0s_input, dim=1), pca=True)
        #         self.summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), pca=True)

        #     if hyp.do_flow:
        #         flow_loss, flowX0_pred = self.flownet(
        #             featX0s[:,0],
        #             featX0s[:,1],
        #             flowX0_half,
        #             self.occ_memX0s_half[:,0],
        #             False,
        #             self.summ_writer)
        #         total_loss += flow_loss
            

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):

        ok = self.prepare_common_tensors(feed)
        if not ok:
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            set_name = feed['set_name']
            set_data_format = feed['set_data_format']

            if set_name=='train':
                return self.run_synth_flow(feed)
            elif set_name=='val':
                return self.run_real_flow(feed)
        
        # if set_data_format=='seq':
        #     return self.run_flow(feed)
        # elif set_data_format=='traj':
        #     return self.run_tracker(feed)
            
        print('weird set_name:', set_name)
        assert(False)
        
