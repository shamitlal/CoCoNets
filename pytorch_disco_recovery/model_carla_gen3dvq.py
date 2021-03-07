import time
import torch
from tensorboardX import SummaryWriter
from backend import saverloader, inputs
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans

from model_base import Model
from nets.featnet3D import FeatNet3D
from nets.upnet3D import UpNet3D
from nets.occnet import OccNet
from nets.segnet import SegNet
from nets.centernet import CenterNet
from nets.vq3dnet import Vq3dNet
from nets.viewnet import ViewNet
from nets.gen3dnet import Gen3dNet
from nets.sigen3dnet import Sigen3dNet

import torch.nn.functional as F

# from utils_basic import *
# import utils_vox
import vox_util
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes


# the purpose of this mode is to train gen3d/sigen3d, after feat/vq/view have been trained and frozen
# i am separating it out because here i want to aggregate lots of views, and do everything in memR (except rendering)

class CARLA_GEN3DVQ(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaGen3dvqModel()
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_up3D:
            self.model.upnet3D.eval()
            self.set_requires_grad(self.model.upnet3D, False)
        if hyp.do_freeze_vq3d:
            self.model.vq3dnet.eval()
            self.set_requires_grad(self.model.vq3dnet, False)
        if hyp.do_freeze_sigen3d:
            self.model.sigen3dnet.eval()
            self.set_requires_grad(self.model.sigen3dnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_center:
            self.model.centernet.eval()
            self.set_requires_grad(self.model.centernet, False)
        if hyp.do_freeze_seg:
            self.model.segnet.eval()
            self.set_requires_grad(self.model.segnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)

    # take over go() from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")
        
        if hyp.lr > 0:
            params_to_optimize = self.model.parameters()
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=hyp.lr)
        else:
            self.optimizer = None
            
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        print("------ Done loading weights ------")

        dataset = np.load(hyp.ind_dataset, allow_pickle=True)
        dataset = dataset.item()
        dataset = dataset['ind_list']
        np.random.shuffle(dataset)

        dataset_np = np.stack(dataset)
        dataset_cuda = torch.from_numpy(dataset_np).long().cuda()
        
        data_len = len(dataset)
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
                set_loaders
            ):   
                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                
                output_dict = dict()

                if log_this or set_do_backprop or hyp.do_test:
                    print('%s: set_num %d; set_data_format %s; set_seqlen %s; log_this %d; set_do_backprop %d; ' % (
                        set_name, set_num, set_data_format, set_seqlen, log_this, set_do_backprop))
                    print('log_this = %s' % log_this)
                    print('set_do_backprop = %s' % set_do_backprop)
                    
                    read_start_time = time.time()
                    perm = np.random.permutation(data_len).astype(np.int32)
                    batch_inds = perm[:set_batch_size]
                    batch_inds = np.reshape(batch_inds, (-1))
                    print('batch_inds', batch_inds)
                    # batch = []
                    # for batch_ind in batch_inds:
                    #     batch.append(dataset[batch_ind])
                    # batch = np.stack(batch, axis=0)
                    # print('batch', batch.shape)
                    feed_cuda = {}
                    # feed_cuda['ind_memR'] = torch.from_numpy(batch).long().cuda()
                    feed_cuda['ind_memR'] = dataset_cuda[batch_inds]
                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_batch_size'] = set_batch_size
                    read_time = time.time() - read_start_time
                    
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

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

class CarlaGen3dvqModel(nn.Module):
    def __init__(self):
        super(CarlaGen3dvqModel, self).__init__()

        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=4)

        if hyp.do_up3D:
            self.upnet3D = UpNet3D()
            
        if hyp.do_occ:
            self.occnet = OccNet()

        if hyp.do_center:
            self.centernet = CenterNet()
            
        if hyp.do_seg:
            self.num_seg_labels = 13 # note label0 is "none"
            # we will predict all 12 valid of these, plus one "air" class
            self.segnet = SegNet(self.num_seg_labels)
        
        if hyp.do_vq3d:
            self.vq3dnet = Vq3dNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
            
        if hyp.do_view:
            self.viewnet = ViewNet()
            
        if hyp.do_gen3d:
            self.gen3dnet = Gen3dNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()
        if hyp.do_sigen3d:
            self.sigen3dnet = Sigen3dNet(
                # input_dim=2,
                # num_layers=2,
            ).cuda()

    def prepare_common_tensors(self, feed):
        results = dict()
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True,
        )

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        # self.crop_guess = (52,52,52)
        self.crop_guess = (50,50,50)
        # self.comp_guess = (8,8,8)
        self.comp_guess = (16,16,16)


        # if self.set_name=='test':
        #     self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        # elif self.set_name=='val':
        #     self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        # else:
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        # self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        
        # self.vox_util = vox_util.Vox_util(feed['set_name'], delta_coeff=0.25)
        # self.vox_util = vox_util.Vox_util(feed['set_name'], delta=(0.25, 0.0, 0.25), assert_cube=False)
        scene_centroid_x = np.random.uniform(-8.0, 8.0)
        scene_centroid_y = np.random.uniform(-1.5, 3.0)
        scene_centroid_z = np.random.uniform(10.0, 26.0)
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        # self.vox_util = vox_util.Vox_util(feed['set_name'], delta=(0.1, 0.0, 0.1), assert_cube=False)


        if False:
            self.rgb_camRs = feed["rgb_camRs"]
            self.rgb_camXs = feed["rgb_camXs"]
            self.pix_T_cams = feed["pix_T_cams"]

            self.origin_T_camRs = feed["origin_T_camRs"]
            self.origin_T_camXs = feed["origin_T_camXs"]

            self.camX0_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
            # self.camX1_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=1)
            self.camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
            self.camXs_T_camRs = __u(utils_geom.safe_inverse(__p(self.camRs_T_camXs)))
            self.camX0_T_camR = utils_basic.matmul2(self.camX0_T_camXs[:,0], self.camXs_T_camRs[:,0])

            self.xyz_camXs = feed["xyz_camXs"]
            self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))

            # self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))

            # # some of the raw boxes may be out of bounds
            # _boxlist_camRs = feed["boxlists"]
            # _tidlist_s = feed["tidlists"] # coordinate-less and plural
            # _scorelist_s = feed["scorelists"] # coordinate-less and plural
            # _scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(
            #     utils_geom.eye_4x4(self.B*self.S),
            #     __p(_boxlist_camRs),
            #     __p(_tidlist_s),
            #     self.Z, self.Y, self.X,
            #     self.vox_util,
            #     only_cars=False, pad=2.0))
            # boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            #     __p(_boxlist_camRs), __p(_tidlist_s), __p(_scorelist_s))
            # self.boxlist_camRs = __u(boxlist_camRs_)
            # self.tidlist_s = __u(tidlist_s_)
            # self.scorelist_s = __u(scorelist_s_)
            # self.lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))

            # self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
            self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
            self.occ_memRs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
            self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
            # self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

            self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
                __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
            self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
                __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
            self.unp_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)

            # ## projected depth, and inbound mask
            # self.depth_camXs_, self.valid_camXs_ = utils_geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
            # self.dense_xyz_camXs_ = utils_geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
            # self.dense_xyz_camRs_ = utils_geom.apply_4x4(__p(self.camRs_T_camXs), self.dense_xyz_camXs_)
            # self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camRs_, self.Z, self.Y, self.X).float()
            # self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
            # self.depth_camXs = __u(self.depth_camXs_)
            # self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)

            free_memXs = __u(self.vox_util.get_freespace(__p(self.xyz_camXs), __p(self.occ_memXs_half))).cuda()
            free_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, free_memXs)
            self.vis_memRs = (free_memRs + self.occ_memRs_half).clamp(0, 1)

            #####################
            ## visualize what we got
            #####################
            # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
            # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
            # self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            # self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            # self.summ_writer.summ_unps('3D_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))

            self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
            # self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
            # self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        set_name = feed['set_name']
        if 'train' in set_name:
            is_train = True
        else:
            is_train = False

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_gen3d:
            ind_memR = feed['ind_memR']
            gen3d_loss, ind_vox = self.gen3dnet(
                ind_memR,
                self.summ_writer,
            )
            total_loss += gen3d_loss

            quant_vox = self.vq3dnet.convert_inds_to_embeds(ind_vox)
            self.summ_writer.summ_feat('gen3d/quant', quant_vox, pca=True)

        if hyp.do_sigen3d:
            # vis_memR0 = self.vis_memRs[:,0].round()
            # vis_memR = torch.sum(self.vis_memRs, dim=1).clamp(0.0, 1.0).round()
            # self.summ_writer.summ_oned('sigen3d/vis', vis_memR, bev=True, norm=False)
            
            ind_memR = feed['ind_memR']
            sigen3d_loss, ind_vox = self.sigen3dnet(
                ind_memR, # input
                self.vox_util, 
                None, # torch.ones_like(vis_memR),  # total vis
                None, # vis_memR0, # view vis
                self.summ_writer,
                is_train=is_train,
            )
            total_loss += sigen3d_loss
            quant_vox = self.vq3dnet.convert_inds_to_embeds(ind_vox)
            self.summ_writer.summ_feat('sigen3d/emb_output', quant_vox, pca=True)
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)
        
        # ind_vox = self.gen3dnet.generate_sample(1, self.Z8, self.Y8, self.X8)
        # self.summ_writer.summ_oned('sample/ind_vox', ind_vox.unsqueeze(1)/512.0, bev=True, norm=False)
        # quant_vox, quant_vox_up = self.vq3dnet.convert_inds_to_embeds(ind_vox)
        # self.summ_writer.summ_feat('sample/quant', quant_vox, pca=True)
        # # quant_vox = self.vq3dnet._post_vq_conv(quant_vox)
        # # quant_vox = self.vq3dnet._post_vq_unpack(quant_vox)
        # self.summ_writer.summ_feat('sample/quant_up', quant_vox_up, pca=True)
        # feat_memR = quant_vox_up.clone()


        # ind_memR being 64 x 8 x 64 (i think):
        # Z4 x Y2 x X2 takes 97s (1.5 mins)
        # Z2 x Y2 x X2 takes 357s (6 mins)
        # Z2 x Y2 x X2 stopped early takes 174s (3 mins) < ok makes sense

        # Z2 x Y2 x X2 stopped after 5k two times takes 519s (8.66 mins)

        # stopping after 10k one time: 534.23; this may suffice
        
        # 52 masks
        

        # 64 x 4 x 64 stopped after 10k: 52 masks; 267s (4.45 mins)

        # 64 x 4 x 64 stopped after 10k, double time: 27 masks; 135s (2.25 mins)
        # 64 x 4 x 64 stopped after 10k, speedup=4: 18 masks; 103s (1.72 mins)



        # right now:
        # 64 x 4 x 64: 349.37s 5.81 mins

        
        print('generating ind_vox sized %d, %d, %d' % (self.Z2, self.Y2, self.X2))
        if hyp.do_gen3d:
            # ind_memR = self.gen3dnet.generate_sample(1, self.Z2, self.Y2, self.X2, stop_early=True)
            # ind_memR = self.gen3dnet.generate_sample(1, self.Z2, self.Y2, self.X2, stop_early=False)
            ind_memR = self.gen3dnet.generate_sample(
                1, self.comp_guess[0], self.comp_guess[1], self.comp_guess[2], stop_early=False)
            self.summ_writer.summ_oned('sample/ind_memR', ind_memR.unsqueeze(1)/512.0, bev=True, norm=False)
            quant_memR = self.vq3dnet.convert_inds_to_embeds(ind_memR)
            self.summ_writer.summ_feat('sample/quant_memR', quant_memR, pca=True)
            feat_memR = quant_memR.clone()
            
        elif hyp.do_sigen3d:
            feat_memR = torch.cat([
                self.occ_memRs[0:1,0],
                self.unp_memRs[0:1,0]*self.occ_memRs[0:1,0],
            ], dim=1)
            _, feat_memR, _ = self.featnet3D(
                feat_memR,
                self.summ_writer,
            )
            print('featnet passed...')
            _, feat_memR, _, _, ind_memR = self.vq3dnet._vq_vae(feat_memR)
            ind_memR = ind_memR.reshape(1, self.Z2, self.Y2, self.X2)
            print('vq3d passed...')

            # print('visibility is wrong here; it needs to be computed in X')
            # # we need to find out what was visible
            # free_memR = self.vox_util.get_freespace(self.xyz_camRs[0:1,0], self.occ_memRs_half[0:1,0])
            # vis_memR = (self.occ_memRs_half[0:1,0]+free_memR).clamp(0.0, 1.0)

            vis_memR = self.vis_memRs[0:1,0].round().float()
            # manually set the bottom to zero
            # vis_memR[:,:,self.Z2//2:,:,:] = 0.0
            vis_memR[:,:,:self.Z2//2,:,:] = 0.0
            
            self.summ_writer.summ_oned('sample/vis_memR', vis_memR, bev=True, norm=False)

            # print('vis_memR', vis_memR.shape)
            # print('ind_memR', ind_memR.shape)
            ind_memR = (ind_memR * vis_memR[:,0]).long()
            self.summ_writer.summ_feat('sample/feat_memR_orig', feat_memR, pca=True)
            feat_memR = feat_memR * vis_memR
            self.summ_writer.summ_feat('sample/feat_memR', feat_memR, valid=vis_memR, pca=True)
            # print('prep passed...')

            ind_memR, vis_memR = self.sigen3dnet.generate_cond_sample(
                ind_memR,
                vis_memR,
                summ_writer=self.summ_writer,
                speed=8)
            print('sigen3d passed...')
            
            self.summ_writer.summ_oned('sample/ind_memR', ind_memR.unsqueeze(1)/512.0, bev=True, norm=False)
            quant_memR = self.vq3dnet.convert_inds_to_embeds(ind_memR)
            self.summ_writer.summ_feat('sample/quant_memR', quant_memR, valid=vis_memR, pca=True)
            feat_memR = quant_memR.clone()

        if hyp.do_up3D:
            up3D_loss, feat_memR = self.upnet3D(feat_memR, self.summ_writer)
            total_loss += up3D_loss
            print('up total_loss', total_loss.detach().cpu().numpy())

            valid_R = torch.ones_like(feat_memR[:,0:1])

            _, _, Z2, Y2, X2 = list(feat_memR.shape)
            Z_crop = int((self.Z - Z2)/2)
            Y_crop = int((self.Y - Y2)/2)
            X_crop = int((self.X - X2)/2)
            crop = (Z_crop, Y_crop, X_crop)
            if not (crop==self.crop_guess):
                print('crop', crop)
            assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
            self.summ_writer.summ_feat('sample/feat_memR_up', feat_memR, pca=True)
            
        if hyp.do_occ:
            _, occ_memR_pred = self.occnet(
                feat_memR, 
                torch.ones_like(feat_memR[:,0:1]),
                torch.ones_like(feat_memR[:,0:1]),
                torch.ones_like(feat_memR[:,0:1]),
                self.summ_writer)
            self.summ_writer.summ_occ('sample/occ_memR_pred', occ_memR_pred)

        if hyp.do_center:
            # this net achieves the following:
            # objectness: put 1 at each object center and 0 everywhere else
            # orientation: at the object centers, classify the orientation into a rough bin
            # size: at the object centers, regress to the object size

            # lrtlist_camX0 = self.lrtlist_camX0s[:,0]
            # lrtlist_memX0 = self.vox_util.apply_mem_T_ref_to_lrtlist(
            #     lrtlist_camX0, self.Z, self.Y, self.X)
            # scorelist = self.scorelist_s[:,0]
            
            center_loss, lrtlist_camR_e, scorelist_e = self.centernet(
                feat_memR, 
                self.crop_guess,
                self.vox_util,
                None, # center mask
                None, # lrt cam
                None, # lrt mem
                None, # scores
                self.summ_writer)
            total_loss += center_loss
            print('cen total_loss', total_loss.detach().cpu().numpy())

            # if lrtlist_camR_e is not None:
            #     # lrtlist_camX_e = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camR[:,0], lrtlist_camR_e)
            #     # lrtlist_camR_e = utils_geom.apply_4x4_to_lrtlist(self.camRs_T_camXs[:,0], lrtlist_camXs_e)
            #     self.summ_writer.summ_lrtlist(
            #         'center/boxlist_e',
            #         self.rgb_camXs[0:1,0],
            #         lrtlist_camR_e[0:1], 
            #         scorelist_e[0:1],
            #         torch.arange(50).reshape(1, 50).long().cuda(), # tids
            #         self.pix_T_cams[0:1,0])
            #     # self.summ_writer.summ_lrtlist(
            #     #     'center/boxlist_g',
            #     #     self.rgb_camXs[0:1,0],
            #     #     self.lrtlist_camXs[0:1,0],
            #     #     self.scorelist_s[0:1,0],
            #     #     self.tidlist_s[0:1,0],
            #     #     self.pix_T_cams[0:1,0])
            
            
        if hyp.do_seg:
            # seg_camXs = feed['seg_camXs']
            # self.summ_writer.summ_seg('seg/seg_camR', seg_camXs[:,0])
            # seg_memR = utils_misc.parse_seg_into_mem(
            #     seg_camXs, self.num_seg_labels, self.occ_memRs,
            #     self.pix_T_cams, self.camRs_T_camXs, self.vox_util)
            # seg_memR = seg_memR[:,
            #                       crop[0]:-crop[0],
            #                       crop[1]:-crop[2],
            #                       crop[2]:-crop[2]]
            # seg_memR_vis = torch.max(seg_memR, dim=2)[0]
            # # self.summ_writer.summ_seg('seg/seg_memR_vis', seg_memR_vis)

            # # occ_memR = torch.max(self.occ_memXs, dim=1)[0]
            # # occ_memR = occ_memR[:,:,
            # #                       crop[0]:-crop[0],
            # #                       crop[1]:-crop[2],
            # #                       crop[2]:-crop[2]]
            
            seg_loss, seg_memR_pred = self.segnet(
                feat_memR, 
                None, # seg_memR,
                None, # occ_memR_sup,
                None, # free_memR_sup,
                None) # summ writer
                # self.summ_writer)
            # total_loss += seg_loss

            seg_e_vis = torch.max(torch.max(seg_memR_pred, dim=1)[1], dim=2)[0]
            self.summ_writer.summ_seg('sample/seg_e', seg_e_vis)
            
            print('seg total_loss', total_loss.detach().cpu().numpy())
            
            
        # if hyp.do_view:
        #     PH, PW = hyp.PH, hyp.PW
        #     sy = float(PH)/float(hyp.H)
        #     sx = float(PW)/float(hyp.W)
        #     assert(sx==0.5) # else we need a fancier downsampler
        #     assert(sy==0.5)
        #     rgbs = []
        #     for k in list(range(self.S)):
        #         print('generating view %d' % k)
        #         _, rgb_e, _ = self.viewnet(
        #             self.pix_T_cams[0:1,0],
        #             self.camXs_T_camRs[0:1,k],
        #             feat_memR,
        #             self.rgb_camXs[0:1,0], 
        #             torch.ones_like(self.rgb_camXs[0:1,0,0:1]),
        #             self.summ_writer,
        #             suffix='_%d' % k)
        #         rgbs.append(rgb_e.clamp(-0.5, 0.5))
        #     self.summ_writer.summ_rgbs('sample/rgbs', rgbs)
        
        return total_loss, None, False
            
    def forward(self, feed):
        self.prepare_common_tensors(feed)
        
        set_name = feed['set_name']
        if set_name=='train':
            return self.run_train(feed)
        elif set_name=='val':
            return self.run_train(feed)
        elif set_name=='test':
            return self.run_test(feed)
        else:
            print('weird set_name:', set_name)
            assert(False)
