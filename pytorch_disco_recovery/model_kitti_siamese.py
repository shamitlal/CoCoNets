import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs

from model_base import Model
# from nets.featnet2D import FeatNet2D
from nets.featnet3D import FeatNet3D
from nets.matchnet import MatchNet
# from nets.embnet3D import EmbNet3D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
# from nets.viewnet import ViewNet
# from nets.mocnet3D import MocNet3D

from tensorboardX import SummaryWriter
import torch.nn.functional as F

# from utils_moc import MocTrainer
# from utils_basic import *
# import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc
import utils_track
import vox_util

np.set_printoptions(precision=2)
np.random.seed(0)

class KITTI_SIAMESE(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = KittiSiameseModel()
        if hyp.do_feat3D and hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
            
    # override go from base
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
        # self.start_iter = 0

        set_nums = []
        set_names = []
        set_seqlens = []
        set_batch_sizes = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_names.append(set_name)
                set_seqlens.append(hyp.seqlens[set_name])
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=1000000, flush_secs=1000000))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        if hyp.do_test:
            all_ious = np.zeros([hyp.max_iters, hyp.S_test], np.float32)
            test_count = 0
            
        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    set_loaders[i] = iter(set_input)
            for (set_num,
                 set_name,
                 set_seqlen,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_names,
                set_seqlens,
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
                    # print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)
                          
                    read_start_time = time.time()

                    feed = next(set_loader)
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
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_seqlen'] = set_seqlen
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

                    if hyp.do_test and (not returned_early):
                        ious = results['ious']
                        ious = ious[0].cpu().numpy()
                        all_ious[test_count] = ious
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        print('mean_ious', mean_ious)

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0)):
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

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

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_py,
                                                                                        set_name))
                    if log_this:
                    # if log_this and (not returned_early):
                        set_writer.flush()
                    
            if hyp.do_save_outputs:
                out_fn = '%s_output_dict.npy' % (hyp.name)
                np.save(out_fn, output_dict)
                print('saved %s' % out_fn)
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

        if hyp.do_test:
            mean_ious = np.mean(all_ious[:test_count], axis=0)
            print('mean_ious', mean_ious)

            
class KittiSiameseModel(nn.Module):
    def __init__(self):
        super(KittiSiameseModel, self).__init__()
        self.featnet3D = FeatNet3D()
        self.matchnet = MatchNet()
            
    def prepare_common_tensors(self, feed, prep_summ=True):
        results = dict()
        
        if prep_summ:
            self.summ_writer = utils_improc.Summ_writer(
                writer=feed['writer'],
                global_step=feed['global_step'],
                log_freq=feed['set_log_freq'],
                fps=8,
                just_gif=feed['just_gif'],
            )
        else:
            self.summ_writer = None

        self.include_vis = hyp.do_include_vis

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K

        # print('set_name', self.set_name)
        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        # print('Z, Y, X = %d, %d, %d' % (self.Z, self.Y, self.X))
        
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()                                                                 
        self.vox_util = vox_util.Vox_util(
            self.Z, self.Y, self.X, 
            self.set_name, scene_centroid=self.scene_centroid,
            assert_cube=True)
        
        self.vox_size_X = self.vox_util.vox_size_X
        self.vox_size_Y = self.vox_util.vox_size_Y
        self.vox_size_Z = self.vox_util.vox_size_Z
        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX

        # # zoom stuff
        # self.ZZ = int(self.Z/2)
        # self.ZY = int(self.Y/2)
        # self.ZX = int(self.X/2)
        
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camXs = feed["origin_T_camXs"]

        self.cams_T_velos = feed["cams_T_velos"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.xyz_veloXs = feed["xyz_veloXs"]
        self.xyz_camXs = __u(utils_geom.apply_4x4(__p(self.cams_T_velos), __p(self.xyz_veloXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        self.boxlist_camXs = feed["boxlists"]
        self.scorelist_s = feed["scorelists"]
        self.tidlist_s = feed["tidlists"]

        boxlist_camXs_ = __p(self.boxlist_camXs)
        scorelist_s_ = __p(self.scorelist_s)
        tidlist_s_ = __p(self.tidlist_s)
        boxlist_camXs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            boxlist_camXs_, tidlist_s_, scorelist_s_)
        self.boxlist_camXs = __u(boxlist_camXs_)
        self.scorelist_s = __u(scorelist_s_)
        self.tidlist_s = __u(tidlist_s_)

        self.lrtlist_camXs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camXs)))

        (self.lrt_camXs,
         self.box_camXs,
         self.score_s,
        ) = utils_misc.collect_object_info(self.lrtlist_camXs,
                                           self.boxlist_camXs,
                                           self.tidlist_s,
                                           self.scorelist_s,
                                           1, mod='X',
                                           do_vis=False,
                                           summ_writer=None)
        # these have a leading K dim, which is 1
        self.lrt_camXs = self.lrt_camXs.squeeze(0)
        self.box_camXs = self.box_camXs.squeeze(0)
        self.score_s = self.score_s.squeeze(0)
        self.tid_s = torch.ones_like(self.score_s).long()
        
        self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)

        if prep_summ and self.include_vis:
            visX_g = []
            for s in list(range(self.S)):
                visX_g.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,s], self.lrtlist_camXs[:,s],
                    self.scorelist_s[:,s], self.tidlist_s[:,s], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('2D_inputs/box_camXs', visX_g)
            # visX_g = []
            # for s in list(range(self.S)):
            #     visX_g.append(self.summ_writer.summ_lrtlist(
            #         'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
            #         self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            # self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)
        
        # if self.set_name=='test':
        #     # center on an object, so that it does not fall out of bounds
        #     self.scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
        #     self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
        #         self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        # else:
        #     # center randomly
        #     scene_centroid_x = np.random.uniform(-8.0, 8.0)
        #     scene_centroid_y = np.random.uniform(-1.5, 3.0)
        #     scene_centroid_z = np.random.uniform(10.0, 26.0)
        #     scene_centroid = np.array([scene_centroid_x,
        #                                scene_centroid_y,
        #                                scene_centroid_z]).reshape([1, 3])
        #     self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        #     # center on a random non-outlier point

        #     all_ok = False
        #     num_tries = 0
        #     while not all_ok:
        #         scene_centroid_x = np.random.uniform(-8.0, 8.0)
        #         scene_centroid_y = np.random.uniform(-1.5, 3.0)
        #         scene_centroid_z = np.random.uniform(10.0, 26.0)
        #         scene_centroid = np.array([scene_centroid_x,
        #                                    scene_centroid_y,
        #                                    scene_centroid_z]).reshape([1, 3])
        #         self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        #         num_tries += 1

        #         # try to vox
        #         self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X,
        #                                           self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        #         all_ok = True

        #         # we want to ensure this gives us a few points inbound for each batch el
        #         inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X, already_mem=False))
        #         num_inb = torch.sum(inb.float(), axis=2)
        #         if torch.min(num_inb) < 100:
        #             all_ok = False

        #         if num_tries > 100:
        #             return False
        #     self.summ_writer.summ_scalar('zoom_sampling/num_tries', num_tries)
        #     self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())

        
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))

        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)

        if prep_summ and self.include_vis:
            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
            # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
            # self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
            # self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)

            
        self.obj_clist_camXs = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)
        self.obj_scorelist = self.score_s

        self.summ_writer.summ_traj_on_occ('track/true_traj',
                                          self.obj_clist_camXs*self.obj_scorelist.unsqueeze(2),
                                          # torch.max(self.occ_memXs, dim=1)[0],
                                          self.occ_memXs[:,0],
                                          self.vox_util, 
                                          already_mem=False,
                                          sigma=2)
        
        # we want to generate a "crop" at the same resolution as the regular voxels
        self.obj_occXs_, _ = self.vox_util.voxelize_near_xyz(
            __p(self.xyz_camXs),
            __p(self.obj_clist_camXs),
            self.ZZ,
            self.ZY,
            self.ZX,
            sz=(self.ZZ*self.vox_size_Z),
            sy=(self.ZY*self.vox_size_Y),
            sx=(self.ZX*self.vox_size_Y))
        self.obj_occXs = __u(self.obj_occXs_)
        self.summ_writer.summ_occs('3D_inputs/obj_occXs', torch.unbind(self.obj_occXs, dim=1))
            
        return True

    def train_over_pair(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        assert(self.S==2)

        search_everywhere = False
        # search_everywhere = True
        
        if hyp.do_test:
            # return early
            return total_loss, results, True
        
        if hyp.do_feat3D:
            
            # featurize the object in frame0
            _, obj_featX0, _ = self.featnet3D(
                self.obj_occXs[:,0],
                None)
            self.summ_writer.summ_feat('3D_feats/obj_featX0_output', obj_featX0, pca=True)

            if search_everywhere:
                # featurize everything in frame1
                feat_loss, featX1, validX1 = self.featnet3D(
                    self.occ_memXs[:,1],
                    self.summ_writer)
            else:
                # we'll make a search region containing the object
                # we'll search midway between the objects time0 and time1 locs, plus some noise
                search_loc_ref = (self.obj_clist_camXs[:,0] + self.obj_clist_camXs[:,1])*0.5
                search_loc_ref += np.random.normal()
                SZ, SY, SX = self.ZZ*2, self.ZY*2, self.ZX*2
                search_occX1, search_lrt = self.vox_util.voxelize_near_xyz(
                    self.xyz_camXs[:,1],
                    search_loc_ref,
                    SZ, SY, SX,
                    sz=(SZ*self.vox_size_Z),
                    sy=(SY*self.vox_size_Y),
                    sx=(SX*self.vox_size_Y))
                feat_loss, featX1, validX1 = self.featnet3D(
                    search_occX1,
                    self.summ_writer)

            total_loss += feat_loss
            self.summ_writer.summ_feat('3D_feats/featX1_output', featX1, pca=True)

        if hyp.do_match:
            assert(hyp.do_feat3D)
            
            if search_everywhere:
                obj_loc_halfmem = self.vox_util.Ref2Mem(
                    self.obj_clist_camXs[:,1].unsqueeze(1),
                    self.Z2, self.Y2, self.X2).squeeze(1)
                # check this visually (yes looks good)
                self.summ_writer.summ_traj_on_occ('match/obj_loc',
                                                  obj_loc_halfmem.unsqueeze(1)*2.0,
                                                  self.occ_memXs[:,1],
                                                  self.vox_util,
                                                  already_mem=True,
                                                  sigma=2)
            else:
                obj_loc_halfmem = self.vox_util.Ref2Zoom(
                    self.obj_clist_camXs[:,1].unsqueeze(1), search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
                # check this visually (yes looks good)
                self.summ_writer.summ_traj_on_occ('match/obj_loc',
                                                  obj_loc_halfmem.unsqueeze(1)*2.0,
                                                  search_occX1,
                                                  self.vox_util,
                                                  already_mem=True,
                                                  sigma=2)
                
            corr, match_loss = self.matchnet(
                obj_featX0, # template
                featX1, # search region
                obj_loc_halfmem, # gt position in search coords
                self.summ_writer)
            total_loss += match_loss

            # utils_basic.print_stats_py('train corr', corr.detach().cpu().numpy())
            self.summ_writer.summ_histogram('corr_train', corr)

            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
    
    def track_over_seq(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        for b in list(range(self.B)):
            if self.score_s[b,0] < 1.0:
                # we need the template to exist
                print('returning early, since score_s[%d,0] = %.1f' % (b, self.score_s[b,0].cpu().numpy()))
                return total_loss, results, True
            # if torch.sum(self.score_s[b]) < (self.S/2):
            if not (torch.sum(self.score_s[b]) == self.S):
                # the full traj should be valid
                print('returning early, since sum(score_s) = %d, while S = %d' % (torch.sum(self.score_s).cpu().numpy(), self.S))
                return total_loss, results, True
        

        one_shot = True
        one_shot = False
        if one_shot:
            if hyp.do_feat3D:
                # featurize the object in frame0
                _, obj_featX0, _ = self.featnet3D(
                    self.obj_occXs[:,0],
                    None)
                self.summ_writer.summ_feat('3D_feats/obj_featX0_output', obj_featX0, pca=True)

                # featurize everything in frames 0 to N
                # (really only 1 to N is necessary, but 0 is good for debug)
                _, featXs_, validXs_ = self.featnet3D(
                    __p(self.occ_memXs),
                    self.summ_writer)
                featXs = __u(featXs_)
                self.summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), pca=True)

            if hyp.do_match:
                # tile out the crop to get all the templates
                templates = obj_featX0.reshape(
                    self.B, 1, hyp.feat_dim, int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2)).repeat(1, self.S, 1, 1, 1, 1)
                search_regions = featXs.clone()
                corrs_, xyz_offset = utils_track.cross_corr_with_template(__p(search_regions), __p(templates))
                corrs = __u(corrs_)
                # corrs is B x S x 1 x Z x Y x X
                # xyz_offset is 1 x 3

                # normalize each corr map, mostly for vis purposes
                corrs = __u(utils_basic.normalize(__p(corrs)))
                self.summ_writer.summ_histogram('corr_test', corrs)
                self.summ_writer.summ_oneds('track/corrs', torch.unbind(torch.mean(corrs, dim=4), dim=1))

                peak_xyzs_halfmem = __u(utils_track.convert_corr_to_xyz(__p(corrs), xyz_offset))
                # this is B x S x 3, and in halfmem coords (i.e., search coords)
                peak_xyzs_mem = peak_xyzs_halfmem * 2.0
                # this is B x S x 3, and in mem coords

                self.summ_writer.summ_traj_on_occ('track/estim_traj',
                                                  peak_xyzs_mem,
                                                  # torch.max(self.occ_memXs, dim=1)[0],
                                                  self.occ_memXs[:,0],
                                                  self.vox_util,
                                                  already_mem=True,
                                                  sigma=2)

                peak_xyzs_cam = self.vox_util.Mem2Ref(peak_xyzs_mem, self.Z, self.Y, self.X)
                # this is B x S x 3
                dists = torch.norm(peak_xyzs_cam - self.obj_clist_camXs, dim=2)
                # this is B x S
                dist = utils_basic.reduce_masked_mean(dists, self.obj_scorelist)
                # this is []
                self.summ_writer.summ_scalar('track/centroid_dist', dist.cpu().item())
        else:
            assert(hyp.do_feat3D and hyp.do_match)

            # featurize the object in frame0
            _, obj_featX0, _ = self.featnet3D(
                self.obj_occXs[:,0],
                None)
            self.summ_writer.summ_feat('3D_feats/obj_featX0_output', obj_featX0, pca=True)

            # init the obj location with gt of frame0
            obj_loc_ref = self.obj_clist_camXs[:,0]

            # track in frames 0 to N
            # (really only 1 to N is necessary, but 0 is good for debug)
            search_occXs = []
            search_featXs = []
            peak_xyzs_mem = []
            corrs = []

            # make the search size 2x the zoom size
            SZ, SY, SX = self.ZZ*2, self.ZY*2, self.ZX*2
            
            for s in list(range(0, self.S)):
                search_occXi, _ = self.vox_util.voxelize_near_xyz(
                    self.xyz_camXs[:,s],
                    obj_loc_ref,
                    SZ, SY, SX,
                    sz=(SZ*self.vox_size_Z),
                    sy=(SY*self.vox_size_Y),
                    sx=(SX*self.vox_size_Y))
                search_occXs.append(search_occXi)

                _, search_featXi, _ = self.featnet3D(
                    search_occXi,
                    None)
                search_featXs.append(search_featXi)
                
                corr, xyz_offset = utils_track.cross_corr_with_template(search_featXi, obj_featX0)
                # corr is B x 1 x Z x Y x X
                # xyz_offset is 1 x 3
                
                self.summ_writer.summ_oned('match/corr_%d' % s, torch.mean(corr, dim=3)) # reduce the vertical dim
                

                use_window = False
                # use_window = True
                if use_window:
                    z_window = np.reshape(np.hanning(corr.shape[2]), [1, 1, corr.shape[2], 1, 1])
                    y_window = np.reshape(np.hanning(corr.shape[3]), [1, 1, 1, corr.shape[3], 1])
                    x_window = np.reshape(np.hanning(corr.shape[4]), [1, 1, 1, 1, corr.shape[4]])
                    z_window = torch.from_numpy(z_window).float().cuda()
                    y_window = torch.from_numpy(y_window).float().cuda()
                    x_window = torch.from_numpy(x_window).float().cuda()
                    window_weight = 0.25
                    corr = corr*(1.0-window_weight) + corr*z_window*window_weight
                    corr = corr*(1.0-window_weight) + corr*y_window*window_weight
                    corr = corr*(1.0-window_weight) + corr*x_window*window_weight

                # normalize each corr map, mostly for vis purposes
                corr = utils_basic.normalize(corr)
                corrs.append(torch.mean(corr, dim=3))
                
                peak_xyz_search = utils_track.convert_corr_to_xyz(corr, xyz_offset)
                # this is B x 3, and in search coords
                # print('peak_xyz_search', peak_xyz_search.detach().cpu().numpy())

                # the middle of the halfsearch coords is obj_loc_ref from the previous iter
                search_mid = np.array([SX/4.0, SY/4.0, SZ/4.0], np.float32).reshape([1, 3])
                search_mid = torch.from_numpy(search_mid).float().to('cuda')
                obj_loc_halfmem = self.vox_util.Ref2Mem(obj_loc_ref.unsqueeze(1), self.Z2, self.Y2, self.X2).squeeze(1)
                peak_xyz_halfmem = (peak_xyz_search - search_mid) + obj_loc_halfmem
                peak_xyz_mem = peak_xyz_halfmem * 2.0
                # this is B x 3, and in mem coords

                # print('search_mid', (search_mid.detach().cpu().numpy()))
                # print('obj_loc_halfmem', (obj_loc_halfmem.detach().cpu().numpy()))
                # print('peak_xyz_halfmem', peak_xyz_halfmem.detach().cpu().numpy())

                peak_xyzs_mem.append(peak_xyz_mem)
                
                obj_loc_ref = self.vox_util.Mem2Ref(peak_xyz_mem.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
                
            peak_xyzs_mem = torch.stack(peak_xyzs_mem, dim=1)
            self.summ_writer.summ_traj_on_occ('track/estim_traj',
                                              peak_xyzs_mem,
                                              # torch.max(self.occ_memXs, dim=1)[0],
                                              self.occ_memXs[:,0],
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
            
            obj_clist_camXs_e = self.vox_util.Mem2Ref(peak_xyzs_mem, self.Z, self.Y, self.X)
            # this is B x S x 3
            dists = torch.norm(obj_clist_camXs_e - self.obj_clist_camXs, dim=2)
            # this is B x S
            dist = utils_basic.reduce_masked_mean(dists, self.obj_scorelist)
            # this is []
            self.summ_writer.summ_scalar('track/centroid_dist', dist.cpu().item())

            # obj_clist_camRs_e = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(obj_clist_camXs_e).unsqueeze(1))).squeeze(2)
            # obj_clist_camXs_e = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(obj_clist_camXs_e).unsqueeze(1))).squeeze(2)
            # this is B x S x 3

            box_camXs_e = torch.cat([obj_clist_camXs_e, self.box_camXs[:,:,3:]], dim=2)
            # this is B x S x 9
            
            lrt_camXs_e = utils_geom.convert_boxlist_to_lrtlist(box_camXs_e)
            # this is B x S x 19

            ious = torch.zeros([self.B, self.S]).float().cuda()
            for s in list(range(self.S)):
                ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camXs[:,s:s+1], lrt_camXs_e[:,s:s+1]).squeeze(1)
            # print('ious', ious.detach().cpu().numpy())
            results['ious'] = ious
            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            
        # score_s = feed["score_traj"]
        # tid_s = torch.ones_like(score_s).long()
        # # box_camRs is B x S x 9
        # lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(box_camRs)
            
            
        #     self.summ_writer.summ_occs('3D_inputs/search_occXs', search_occXs)
        #     self.summ_writer.summ_feats('3D_feats/search_featXs', search_featXs, pca=True)
        #     self.summ_writer.summ_oneds('track/corrs', corrs)


        return total_loss, results, False
    
    def forward(self, feed):
        
        set_name = feed['set_name']
        if set_name=='test':
            just_gif = True
        else:
            just_gif = False
        feed['just_gif'] = just_gif

        if set_name=='train' or set_name=='val':
            ok = self.prepare_common_tensors(feed)
            if not ok:
                total_loss = torch.tensor(0.0).cuda()
                return total_loss, None, True
            return self.train_over_pair(feed)
        elif set_name=='test':
            ok = self.prepare_common_tensors(feed)
            if not ok:
                total_loss = torch.tensor(0.0).cuda()
                return total_loss, None, True
            return self.track_over_seq(feed)
        
        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)
