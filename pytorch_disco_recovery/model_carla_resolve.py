import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.matchnet import MatchNet
from nets.rendernet import RenderNet
from nets.occnet import OccNet
from nets.resolvenet import ResolveNet
from nets.rgbnet import RgbNet
from nets.sigen3dnet import Sigen3dNet
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

# the idea here is to resolve to higher resolutions in a smart way,
# following the advice of PointRend and even NERF

class CARLA_RESOLVE(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaResolveModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        # if hyp.do_freeze_resolve:
        #     self.model.resolvenet.eval()
        #     self.set_requires_grad(self.model.resolvenet, False)
        if hyp.do_freeze_sigen3d:
            self.model.sigen3dnet.eval()
            self.set_requires_grad(self.model.sigen3dnet, False)
            
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

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': hyp.lr},
        ])
        
        model_state_dict = self.model.state_dict()
        for k in model_state_dict.keys():
            print('key', k)
        
        self.start_iter = saverloader.load_weights(self.model, None)
            
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

class CarlaResolveModel(nn.Module):
    def __init__(self):
        super(CarlaResolveModel, self).__init__()
        
        # self.crop_2x = (18*2,18*2,18*2)
        # self.crop = (18,18,18)
        # self.crop_low = (2,2,2)
        # self.crop_mid = (8,8,8)

        # self.crop = (0,0,0)
        # self.crop = (4,4,4)
        # self.crop = (5,5,5)
        # self.crop = (6,6,6)
        # self.crop = (7,7,7)
        # self.crop = (8,8,8)
        # self.crop = (9,9,9)
        self.crop = (10,10,10)
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)

        if hyp.do_render:
            self.rendernet = RenderNet()

        if hyp.do_occ:
            self.occnet = OccNet()

        if hyp.do_resolve:
            self.resolvenet = ResolveNet()

        if hyp.do_sigen3d:
            self.sigen3dnet1 = Sigen3dNet().cuda()
            self.sigen3dnet2 = Sigen3dNet().cuda()
            
        if hyp.do_rgb:
            self.rgbnet = RgbNet()
            
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

        self.H, self.W, self.V = hyp.H, hyp.W, hyp.V
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

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        # scene_centroid_x = np.random.uniform(-8.0, 8.0)
        # scene_centroid_y = np.random.uniform(0.0, 2.0)
        # scene_centroid_z = np.random.uniform(8.0, 26.0)
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_camXs = feed['rgb_camXs']
        self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))


        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camX0s_ = utils.geom.apply_4x4(__p(self.camX0s_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camX0s_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        # self.summ_writer.summ_oned('inputs/valid_camX0_before', self.valid_camXs[:,0], norm=False)

        # weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
        # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        
        self.summ_writer.summ_oned('inputs/depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        # self.summ_writer.summ_oned('inputs/valid_camX0_after', self.valid_camXs[:,0], norm=False)
        

        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        
        # self.summ_writer.summ_rgb('inputs/rgb_camX0', self.rgb_camXs[:,0])

        self.summ_writer.summ_rgb('inputs/rgb_camX0', self.rgb_camXs[:,0])


        # my plan here is:
        # i can have a feat3dnet pre-trained, and even trained concurrently
        # then, at uncertain points,
        # i extract some multiscale features, at two stages of the feature pyramid

        if hyp.do_feat3d:
            
            self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
            self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

            # input from index 1, but in X0 coords
            feat_memX1_input = torch.cat([
                self.occ_memX0s[:,1],
                self.occ_memX0s[:,1]*self.rgb_memX0s[:,1]], dim=1)
            feat_loss, feat_halfmemX0, feat_bunch = self.feat3dnet(
                feat_memX1_input, norm=False, summ_writer=self.summ_writer)
            total_loss += feat_loss
            valid_halfmemX0 = torch.ones_like(feat_halfmemX0[:,0:1])

            rgb_halfmemX0 = feat_halfmemX0[:,1:]
            occ_halfmemX0 = feat_halfmemX0[:,0:1]
                
        if hyp.do_occ:
            occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2,
                agg=True)

            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_memX0_sup = 1.0 - (F.conv3d(1.0 - free_memX0_sup, weights, padding=1)).clamp(0, 1)
            
            # we need to crop
            occ_memX0_sup = self.crop_feat(occ_memX0_sup, self.crop)
            free_memX0_sup = self.crop_feat(free_memX0_sup, self.crop)
            
            occ_loss, occ_memX0_logit = self.occnet(
                feat_halfmemX0[:,0:1],
                occ_memX0_sup,
                free_memX0_sup,
                valid_halfmemX0,
                self.summ_writer)
            total_loss += occ_loss
        
        if hyp.do_resolve:
            
            occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z, self.Y, self.X,
                agg=True)
            have_label = (occ_memX0_sup + free_memX0_sup).clamp(0, 1)
            
            low_feat, mid_feat, high_feat = feat_bunch
            full_feat = feat_halfmemX0.clone()
            # print('low_feat', low_feat.shape)
            # print('mid_feat', mid_feat.shape)
            # print('high_feat', high_feat.shape)
            # print('full_feat', full_feat.shape)

            low_feat = self.pad_feat(low_feat, self.crop_low)
            mid_feat = self.pad_feat(mid_feat, self.crop_mid)
            high_feat = self.pad_feat(high_feat, self.crop)
            full_feat = self.pad_feat(full_feat, self.crop)
            occ_logit = self.pad_feat(occ_memX0_logit, self.crop)

            low_feat = F.interpolate(low_feat, scale_factor=4, mode='trilinear')
            mid_feat = F.interpolate(mid_feat, scale_factor=4, mode='trilinear')
            high_feat = F.interpolate(high_feat, scale_factor=2, mode='trilinear')
            full_feat = F.interpolate(full_feat, scale_factor=2, mode='trilinear')
            occ_logit = F.interpolate(occ_logit, scale_factor=2, mode='trilinear')

            cat_feat = torch.cat([low_feat, mid_feat, high_feat, full_feat], dim=1)
            # print('cat_feat', cat_feat.shape)
                
            self.summ_writer.summ_feat('feat3d/low_feat', low_feat, pca=True)
            self.summ_writer.summ_feat('feat3d/mid_feat', mid_feat, pca=True)
            self.summ_writer.summ_feat('feat3d/high_feat', high_feat, pca=True)
            self.summ_writer.summ_feat('feat3d/full_feat', full_feat, pca=True)
            self.summ_writer.summ_feat('feat3d/cat_feat', cat_feat, pca=True)

            # next i want:
            # an uncertainty metric
            # the code says: use L1 between 0.0 and the logit
            # let's have a look at this

            # i need to re-up and THEN pad, so that we do not use padded values as topk
            occ_logit_ = F.interpolate(occ_memX0_logit, scale_factor=2, mode='trilinear')
            uncertainty = -torch.abs(occ_logit_)
            uncertainty = uncertainty - torch.min(uncertainty)
            uncertainty = self.pad_feat(uncertainty, self.crop_2x)
            # print('uncertainty', uncertainty.shape)
            # utils.basic.print_stats('uncertainty', uncertainty)
            self.summ_writer.summ_oned('feat3d/uncertainty', uncertainty, bev=True, norm=True)

            def top_k(uncertainty, K=10):
                B, C, Z, Y, X = list(uncertainty.shape)
                assert(C==1)

                scorelist, indlist = torch.topk(uncertainty.view(B, C, -1), K)
                # print('indlist', indlist.shape)
                
                indlist_z = indlist // (Y*X)
                indlist_y = (indlist % (Y*X)) // X
                indlist_x = (indlist % (Y*X)) % X

                scorelist = scorelist.reshape(B, K)
                indlist_z = indlist_z.reshape(B, K)
                indlist_y = indlist_y.reshape(B, K)
                indlist_x = indlist_x.reshape(B, K)

                xyzlist = torch.stack([indlist_x, indlist_y, indlist_z], dim=2).float()
                # this is B x K x 3
                return scorelist, xyzlist, indlist
            
            # def rand_k(uncertainty, K=10):
            #     B, C, Z, Y, X = list(uncertainty.shape)
            #     assert(C==1)

            #     indlist = torch.randint(low=0, high=Z*Y*X, size=(B, 1, K)).cuda()
            #     # print('indlist2', indlist.shape)

            #     indlist_z = indlist // (Y*X)
            #     indlist_y = (indlist % (Y*X)) // X
            #     indlist_x = (indlist % (Y*X)) % X

            #     indlist_z = indlist_z.reshape(B, K)
            #     indlist_y = indlist_y.reshape(B, K)
            #     indlist_x = indlist_x.reshape(B, K)

            #     xyzlist = torch.stack([indlist_x, indlist_y, indlist_z], dim=2).float()
            #     # this is B x K x 3
            #     return xyzlist, indlist

            
            # this strategy is maybe different from what kaiming is suggesting
            # he says select K*N uniformly first,
            # then take the top M<N that are uncertain
            # and use N-M more 
            
            self.K = 2048
            _, samp_xyz, samp_ind = top_k(uncertainty*have_label, K=self.K)
            _, samp_xyz2, samp_ind2 = top_k(torch.randn(have_label.shape).float().cuda()*have_label, K=int(self.K/2))
            # samp_xyz2, samp_ind2 = rand_k(uncertainty, K=int(self.K/2))
            samp_xyz = torch.cat([samp_xyz, samp_xyz2], dim=1)
            samp_ind = torch.cat([samp_ind, samp_ind2], dim=2)
            # samp_xyz is B x K x 3
            # samp_ind is B x 1 x K

            scat = torch.zeros_like(uncertainty).reshape(self.B, 1, -1).scatter_(2, samp_ind, 1).view(
                self.B, 1, self.Z, self.Y, self.X)
            self.summ_writer.summ_oned('resolve/scat', self.crop_feat(scat, self.crop_2x), max_along_y=True, bev=True, norm=True)

            # topk_vis = self.vox_util.xyz2circles(samp_xyz, self.Z, self.Y, self.X, radius=1.0, soft=False, already_mem=True)
            # self.summ_writer.summ_oned('feat3d/topk', topk_vis, bev=True, max_along_y=True, norm=True)
            

            # next, i need to sample these locs
            # then compute new answers
            # then replace the original answers with the new ones

            samp_feat = utils.samp.bilinear_sample3d(cat_feat, samp_xyz)
            # this is B x C x K

            samp_occ = utils.samp.bilinear_sample3d(occ_memX0_sup, samp_xyz)
            samp_free = utils.samp.bilinear_sample3d(free_memX0_sup, samp_xyz)

            resolve_loss, samp_logit = self.resolvenet(
                samp_feat,
                samp_occ,
                samp_free,
                self.summ_writer)
            total_loss += resolve_loss
            
            # now i need to scatter these values into occ_memX0_logit
            
            self.summ_writer.summ_occ('resolve/occ_before', self.crop_feat(F.sigmoid(occ_logit), self.crop_2x))
            self.summ_writer.summ_occ('resolve/occ_sup', self.crop_feat(occ_memX0_sup, self.crop_2x))

            # print('samp_xyz', samp_xyz.shape)
            # print('samp_feat', samp_feat.shape)
            # print('samp_logit', samp_logit.shape)
            
            # occ_logit = occ_logit.reshape(self.B, 1, self.Z*self.Y*self.X).scatter_(2, samp_xyz.long(), samp_logit).view(
            #     self.B, 1, self.Z, self.Y, self.X)
            # samp_xyz = samp_xyz.reshape(self.B, 
            # occ_logit = occ_logit.reshape(self.B, 1, self.Z*self.Y*self.X).scatter_(2, samp_ind, samp_logit).view(


            occ_g = occ_memX0_sup.clone()
            free_g = free_memX0_sup.clone()
            occ_e = F.sigmoid(occ_logit)
            occ_e_binary = occ_e.round()
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g)
            acc_free = utils.basic.reduce_masked_mean(free_match, free_g)
            acc_total = utils.basic.reduce_masked_mean(either_match, either_have)
            acc_bal = (acc_occ + acc_free)*0.5
            self.summ_writer.summ_scalar('super_resolve/acc_occ_before', acc_occ.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_free_before', acc_free.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_total_before', acc_total.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_bal_before', acc_bal.cpu().item())
            
            
            occ_logit = occ_logit.reshape(self.B, 1, self.Z*self.Y*self.X).scatter_(2, samp_ind, samp_logit).view(
                self.B, 1, self.Z, self.Y, self.X)
            self.summ_writer.summ_occ('resolve/occ_after', self.crop_feat(F.sigmoid(occ_logit), self.crop_2x))
            new_logit = torch.zeros_like(occ_logit).reshape(self.B, 1, self.Z*self.Y*self.X).scatter_(2, samp_ind, samp_logit).view(
                self.B, 1, self.Z, self.Y, self.X)
            # self.summ_writer.summ_oned('resolve/new_logit', self.crop_feat(new_logit, self.crop_2x), max_along_y=True, bev=True, norm=True)
            self.summ_writer.summ_oned('resolve/new_logit', self.crop_feat(new_logit, self.crop_2x), max_along_y=True, bev=True, norm=True)
            # self.summ_writer.summ_occ('resolve/occ_after', self.crop_feat(F.sigmoid(occ_logit), self.crop_2x))

            occ_g = occ_memX0_sup.clone()
            free_g = free_memX0_sup.clone()
            occ_e = F.sigmoid(occ_logit)
            occ_e_binary = occ_e.round()
            occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g)
            acc_free = utils.basic.reduce_masked_mean(free_match, free_g)
            acc_total = utils.basic.reduce_masked_mean(either_match, either_have)
            acc_bal = (acc_occ + acc_free)*0.5
            self.summ_writer.summ_scalar('super_resolve/acc_occ_after', acc_occ.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_free_after', acc_free.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_total_after', acc_total.cpu().item())
            self.summ_writer.summ_scalar('super_resolve/acc_bal_after', acc_bal.cpu().item())
            
        if hyp.do_sigen3d:

            rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
                self.rgb_camXs[:,0], self.Z1, self.Y1, self.X1, self.pix_T_cams[:,0])
            occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z1, self.Y1, self.X1)
            vis_memX0 = self.vox_util.get_freespace(self.xyz_camX0s[:,0], occ_memX0)
            vis_memX0 = (vis_memX0 + occ_memX0).clamp(0,1)
            input_memX0 = torch.cat([occ_memX0, occ_memX0*rgb_memX0], dim=1)

            occ_memX0_sup, free_memX0_sup, occs, frees = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z1, self.Y1, self.X1,
                agg=True)
            # occ_memX0_sup = F.interpolate(occ_memX0_sup, scale_factor=0.5, mode='trilinear')
            # free_memX0_sup = F.interpolate(free_memX0_sup, scale_factor=0.5, mode='trilinear')
            # occ_memX0_sup = (occ_memX0_sup > 0.6).float()
            # free_memX0_sup = (free_memX0_sup > 0.6).float()

            occ_memX0_sup = self.zero_border(occ_memX0_sup, self.crop)
            free_memX0_sup = self.zero_border(free_memX0_sup, self.crop)
            
            sigen3d_loss, occ_logit2, occ_e2 = self.sigen3dnet1(
                input_memX0,
                torch.ones_like(vis_memX0),
                occ_memX0_sup,
                free_memX0_sup,
                summ_writer=self.summ_writer,
                # summ_writer=None,
            )
            total_loss += sigen3d_loss


            if False:
                # let's add a second scale right here

                occ_logit4 = occ_logit4.detach()
                occ_e4 = occ_e4.detach()

                self.summ_writer.summ_occ('resolve/occ_e4', occ_e4)
                self.summ_writer.summ_occ('resolve/occ_g4', occ_memX0_sup)

                occ_logit2 = F.interpolate(occ_logit4, scale_factor=2, mode='trilinear')
                occ_e2 = F.interpolate(occ_e4, scale_factor=2, mode='trilinear')
                # uncertainty = -torch.abs(occ_logit2)
                # uncertainty = torch.exp(-torch.abs(occ_e2 - 0.5))
                uncertainty = 1 - torch.abs(occ_e2 - 0.5) # values at 0.5 will have 1.0; others decline
                uncertainty = uncertainty - torch.min(uncertainty)
                self.summ_writer.summ_oned('resolve/uncertainty', uncertainty, bev=True, norm=True)

                def indlist_to_xyzlist(indlist, Z, Y, X):
                    B, K = list(indlist.shape)
                    indlist_z = indlist // (Y*X)
                    indlist_y = (indlist % (Y*X)) // X
                    indlist_x = (indlist % (Y*X)) % X

                    indlist_z = indlist_z.reshape(B, K)
                    indlist_y = indlist_y.reshape(B, K)
                    indlist_x = indlist_x.reshape(B, K)

                    xyzlist = torch.stack([indlist_x, indlist_y, indlist_z], dim=2).float()
                    # this is B x K x 3
                    return xyzlist

                def rand_k(B, Z, Y, X, K=10):
                    indlist = torch.randint(low=0, high=Z*Y*X, size=(B, K)).cuda()
                    # this is B x K
                    xyzlist = indlist_to_xyzlist(indlist, Z, Y, X)               
                    # this is B x K x 3
                    return xyzlist, indlist

                # pointrend suggests to select K*N uniformly first,
                # then take the top N1 that are uncertain
                # and use N2 more, where N=N1+N2
                K = 3
                N = 2048
                beta = 0.75
                N1 = int(beta*N)
                N2 = N - N1
                samp_xyz, samp_ind = rand_k(self.B, self.Z2, self.Y2, self.X2, K=K*N)
                # samp_xyz is B x K*N x 3
                # samp_ind is B x K*N

                # samp_uncertainty = utils.samp.bilinear_sample3d(uncertainty, samp_xyz)
                # _, indlist1 = torch.topk(samp_uncertainty, N1, dim=2)
                # indlist1 = indlist1.squeeze(1)
                # indlist2 = samp_ind[:,:N2]
                # indlist = torch.cat([indlist1, indlist2], dim=1)
                # # this is B x N

                # indlist = samp_ind[:,:N]



                rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_camXs[:,0], self.Z2, self.Y2, self.X2, self.pix_T_cams[:,0])
                occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z2, self.Y2, self.X2)
                vis_memX0 = self.vox_util.get_freespace(self.xyz_camX0s[:,0], occ_memX0)
                vis_memX0 = (vis_memX0 + occ_memX0).clamp(0,1)
                input_memX0 = torch.cat([occ_memX0, occ_memX0*rgb_memX0], dim=1)
                occ_memX0_sup, free_memX0_sup, occs, frees = self.vox_util.prep_occs_supervision(
                    self.camX0s_T_camXs,
                    self.xyz_camXs,
                    self.Z2, self.Y2, self.X2,
                    agg=True)
                # occ_memX0_sup = self.zero_border(occ_memX0_sup, self.crop)
                # free_memX0_sup = self.zero_border(free_memX0_sup, self.crop)

                scat = torch.zeros((self.B, 1, self.Z2*self.Y2*self.X2), dtype=torch.float32, device=torch.device('cuda')).scatter_(2, samp_ind.unsqueeze(1), 1).view(
                    self.B, 1, self.Z2, self.Y2, self.X2)
                self.summ_writer.summ_oned('resolve/scat_rand', scat, max_along_y=False, bev=True, norm=True)

                # samp_occ = utils.samp.bilinear_sample3d(occ_memX0_sup, samp_xyz)
                # samp_occ = 
                # samp_occ = torch.gather(occ_memX0_sup.reshape(self.B, 1, -1), 1, samp_ind.unsqueeze(1))

                assert(self.B==1)
                samp_occ_ = occ_memX0_sup.reshape(-1)
                samp_occ_ = samp_occ_[samp_ind.reshape(-1)]
                samp_occ = samp_occ_.reshape(self.B, 1, K*N)
                samp_free_ = free_memX0_sup.reshape(-1)
                samp_free_ = samp_free_[samp_ind.reshape(-1)]
                samp_free = samp_free_.reshape(self.B, 1, K*N)

                # print('samp_occ', samp_occ)
                _, indlist = torch.topk((samp_occ+samp_free).clamp(0,1), N, dim=2)
                indlist = indlist.squeeze(1)
                # indlist = samp_ind.reshape(-1)[indlist.reshape(-1)].reshape(self.B, -1)
                indlist = torch.gather(samp_ind, 1, indlist)

                scat = torch.zeros_like(occ_memX0).reshape(self.B, 1, -1).scatter_(2, indlist.unsqueeze(1), 1).view(
                    self.B, 1, self.Z2, self.Y2, self.X2)

                # scat = torch.zeros_like(occ_memX0).reshape(-1)
                # scat[indlist.reshape(-1)] = 1.0
                # scat = scat.reshape(self.B, 1, self.Z2, self.Y2, self.X2)

                # self.summ_writer.summ_oned('resolve/scat', self.crop_feat(scat, self.crop_2x), max_along_y=True, bev=True, norm=True)
                # self.summ_writer.summ_oned('resolve/scat', scat, max_along_y=True, bev=True, norm=True)
                self.summ_writer.summ_oned('resolve/scat_sort', scat, max_along_y=False, bev=True, norm=True)

                # # now, i want to re-compute at these locations
                # xyzlist = indlist_to_xyzlist(indlist, self.Z2, self.Y2, self.X2)

                self.summ_writer.summ_occ('resolve/occ_e2A', occ_e2)
                sigen3d_loss, occ_logit2_new, occ_e2_new = self.sigen3dnet2(
                    input_memX0,
                    scat,
                    # vis_memX0,
                    occ_memX0_sup,
                    free_memX0_sup,
                    # summ_writer=None,
                    summ_writer=self.summ_writer,
                )
                total_loss += sigen3d_loss

                self.summ_writer.summ_occ('resolve/occ_e2B', occ_e2_new)
                self.summ_writer.summ_occ('resolve/occ_g2', occ_memX0_sup)

                # # now, at the scat locations, i want to use the new answers
                # occ_e2 = occ_e2.reshape(self.B, 1, -1).scatter_(2, indlist.unsqueeze(1), occ_e2_new).view(
                #     self.B, 1, self.Z2, self.Y2, self.X2)

                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                scat2 = (F.conv3d(scat, weights, padding=1)).clamp(0, 1)

                occ_e2[scat2 > 0] = occ_e2_new[scat2 > 0]
                self.summ_writer.summ_occ('resolve/occ_e2C', occ_e2)

        if hyp.do_render:

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
            assert(sy==0.5 or sy==1.0)
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            # assert(S==2) # else we should warp each feat in 1:

            rgb_halfmem = self.pad_feat(feat_halfmemX0[:,1:], self.crop)
            occ_halfmem = self.pad_feat(feat_halfmemX0[:,0:1], self.crop)

            feat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camX0s[:,0], rgb_halfmem,
                hyp.view_depth, PH, PW, noise_amount=2.0)
            occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camX0s[:,0], occ_halfmem,
                hyp.view_depth, PH, PW, grid_z_vec=dists)

            if sx==0.5:
                rgb_X00 = utils.basic.downsample(self.rgb_camXs[:,0], 2)
                valid_X00 = utils.basic.downsample(self.valid_camXs[:,0], 2)
            else:
                rgb_X00 = self.rgb_camXs[:,0]
                depth_X00 = self.depth_camXs[:,0]
                valid_X00 = self.valid_camXs[:,0]
            
            print('dists', dists.detach().cpu().numpy())
            print('rgb_X00', rgb_X00.shape)
            print('feat_proj', feat_proj.shape)
            print('occ_proj', occ_proj.shape)
            print('dists', dists.shape)

            # decode the perspective volume into an image
            render_loss, rgb_e, _, _ = self.rendernet(
                feat_proj,
                occ_proj,
                dists,
                rgb_g=rgb_X00,
                depth_g=depth_X00,
                valid=valid_X00,
                summ_writer=self.summ_writer)
            total_loss += render_loss


        # # assert(self.S==2)

        # origin_T_cam0 = self.origin_T_cams[:, 0]
        # origin_T_cam1 = self.origin_T_cams[:, 1]
        # cam0_T_cam1 = utils.basic.matmul2(utils.geom.safe_inverse(origin_T_cam0), origin_T_cam1)

        # # let's immediately discard the true motion and make some fake motion

        # xyz0_cam0 = self.xyz_cams[:,0]
        # xyz1_cam0 = utils.geom.apply_4x4(cam0_T_cam1, self.xyz_cams[:,1])

        # # camX_T_cam0 = utils.geom.get_random_rt(
        # xyz_cam_g, rx, ry, rz = utils.geom.get_random_rt(
        #     self.B,
        #     r_amount=4.0,
        #     t_amount=2.0,
        #     sometimes_zero=False,
        #     return_pieces=True)
        # rot = utils.geom.eul2rotm(rx*0.1, ry, rz*0.1)
        # camX_T_cam0 = utils.geom.merge_rt(rot, xyz_cam_g)
        
        # cam0_T_camX = utils.geom.safe_inverse(camX_T_cam0)
        # xyz1_camX = utils.geom.apply_4x4(camX_T_cam0, xyz1_cam0)

        # occ0_mem0 = self.vox_util.voxelize_xyz(xyz0_cam0, self.Z, self.Y, self.X)
        # occ1_memX = self.vox_util.voxelize_xyz(xyz1_camX, self.Z, self.Y, self.X)

        # rgb0_mem0 = self.vox_util.unproject_rgb_to_mem(
        #     self.rgb_cams[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
        # rgb1_mem1 = self.vox_util.unproject_rgb_to_mem(
        #     self.rgb_cams[:,1], self.Z, self.Y, self.X, self.pix_T_cams[:,1])
        
        # rgb1_memX = self.vox_util.apply_4x4_to_vox(
        #     utils.basic.matmul2(camX_T_cam0, cam0_T_cam1), rgb1_mem1)
        
        # self.summ_writer.summ_occs('inputs/occ_mems', [occ0_mem0, occ1_memX])
        # self.summ_writer.summ_unps('inputs/rgb_mems', [rgb0_mem0, rgb1_memX], [occ0_mem0, occ1_memX])

        # if hyp.do_feat3d:
        #     feat_mem0_input = torch.cat([occ0_mem0, occ0_mem0*rgb0_mem0], dim=1)
        #     feat_memX_input = torch.cat([occ1_memX, occ1_memX*rgb1_memX], dim=1)
        #     feat_loss0, feat_halfmem0 = self.feat3dnet(feat_mem0_input, self.summ_writer)
        #     feat_loss1, feat_halfmemX = self.feat3dnet(feat_memX_input, self.summ_writer)
        #     total_loss += feat_loss0 + feat_loss1

        # # if hyp.do_resolve:
        # #     assert(hyp.do_feat3d)
        # #     resolve_loss, cam0_T_cam1_e, _ = self.resolvenet(
        # #         feat_halfmem0,
        # #         feat_halfmemX,
        # #         cam0_T_camX,
        # #         self.vox_util,
        # #         self.summ_writer)
        # #     total_loss += resolve_loss

        # if hyp.do_match:
        #     assert(hyp.do_feat3d)

        #     occ_rs = []
        #     rgb_rs = []
        #     feat_rs = []
        #     feat_rs_trimmed = []
        #     for ind, rad in enumerate(self.radlist):
        #         rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
        #         occ_r, rgb_r = self.place_scene_at_dr(
        #             rgb0_mem0, self.xyz_cams[:,0], rad_,
        #             self.Z, self.Y, self.X, self.vox_util)
        #         occ_rs.append(occ_r)
        #         rgb_rs.append(rgb_r)

        #         inp_r = torch.cat([occ_r, occ_r*rgb_r], dim=1)
        #         _, feat_r = self.feat3dnet(inp_r)
        #         feat_rs.append(feat_r)
        #         feat_r_trimmed = feat_r[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
        #         # print('feat_r_trimmed', feat_r_trimmed.shape)
        #         feat_rs_trimmed.append(feat_r_trimmed)
                
        #     self.summ_writer.summ_occs('resolve/occ_rs', occ_rs)
        #     self.summ_writer.summ_unps('resolve/rgb_rs', rgb_rs, occ_rs)
        #     self.summ_writer.summ_feats('resolve/feat_rs', feat_rs, pca=True)
        #     self.summ_writer.summ_feats('resolve/feat_rs_trimmed', feat_rs_trimmed, pca=True)

        #     match_loss, camX_T_cam0_e, cam0_T_camX_e = self.matchnet(
        #         torch.stack(feat_rs_trimmed, dim=1), # templates
        #         feat_halfmemX, # search region
        #         self.vox_util,
        #         xyz_cam_g=xyz_cam_g,
        #         rad_g=ry,
        #         summ_writer=self.summ_writer)
        #     total_loss += match_loss

        #     occ1_mem0_e = self.vox_util.apply_4x4_to_vox(cam0_T_camX_e, occ1_memX)
        #     occ1_mem0_g = self.vox_util.apply_4x4_to_vox(cam0_T_camX, occ1_memX)

        #     self.summ_writer.summ_occs('resolve/occ_mems_0', [occ0_mem0, occ1_memX])
        #     self.summ_writer.summ_occs('resolve/occ_mems_e', [occ0_mem0, occ1_mem0_e.round()])
        #     self.summ_writer.summ_occs('resolve/occ_mems_g', [occ0_mem0, occ1_mem0_g.round()])
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        
        if hyp.do_sigen3d:
            rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
                self.rgb_camXs[:,0], self.Z4, self.Y4, self.X4, self.pix_T_cams[:,0])
            occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z4, self.Y4, self.X4)
            vis_memX0 = self.vox_util.get_freespace(self.xyz_camX0s[:,0], occ_memX0)
            vis_memX0 = (vis_memX0 + occ_memX0).clamp(0,1)
            input_memX0 = torch.cat([occ_memX0, occ_memX0*rgb_memX0], dim=1)

            occ_memX0_sup, free_memX0_sup, occs, frees = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z4, self.Y4, self.X4,
                agg=True)
            # occ_memX0_sup = self.crop_feat(occ_memX0_sup, self.crop)
            # occ_memX0_sup = F.interpolate(occ_memX0_sup, scale_factor=0.5, mode='trilinear')
            # free_memX0_sup = F.interpolate(free_memX0_sup, scale_factor=0.5, mode='trilinear')
            # occ_memX0_sup = (occ_memX0_sup > 0.6).float()
            # free_memX0_sup = (free_memX0_sup > 0.6).float()

            # occ_memX0_sup = self.crop_feat(occ_memX0_sup, self.crop)
            # free_memX0_sup = self.crop_feat(free_memX0_sup, self.crop)

            sigen3d_loss, occ_memX0 = self.sigen3dnet(
                input_memX0,
                vis_memX0,
                occ_memX0_sup,
                free_memX0_sup,
                summ_writer=self.summ_writer)
            total_loss += sigen3d_loss
            print('sigen3d passed...')

            # self.summ_writer.summ_occ('sample/occ_g', occ_memX0_sup)
            # self.summ_writer.summ_occ('sample/output_occ', occ_memX0)
            # self.summ_writer.summ_occ('sample/output_vis', vis_memX0)

            # occ_g = occ_memX0_sup.clone()
            # free_g = free_memX0_sup.clone()
            # occ_e_binary = occ_memX0.round()
            # occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
            # free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
            # either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
            # either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
            # acc_occ = utils.basic.reduce_masked_mean(occ_match, occ_g)
            # acc_free = utils.basic.reduce_masked_mean(free_match, free_g)
            # acc_total = utils.basic.reduce_masked_mean(either_match, either_have)
            # acc_bal = (acc_occ + acc_free)*0.5
            # self.summ_writer.summ_scalar('unscaled_sigen3d/acc_occ', acc_occ.cpu().item())
            # self.summ_writer.summ_scalar('unscaled_sigen3d/acc_free', acc_free.cpu().item())
            # self.summ_writer.summ_scalar('unscaled_sigen3d/acc_total', acc_total.cpu().item())
            # self.summ_writer.summ_scalar('unscaled_sigen3d/acc_bal', acc_bal.cpu().item())
        
        return total_loss, None, False

    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)
        # data_ok = False
        
        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            elif self.set_name=='val':
                return self.run_test(feed)
            elif self.set_name=='test':
                return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
