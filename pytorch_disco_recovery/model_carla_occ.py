import itertools
import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs

from model_base import Model
from nets.linclassnet import LinClassNet
from nets.featnet2D import FeatNet2D
from nets.featnet3D import FeatNet3D
from nets.upnet3D import UpNet3D
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet

from nets.vq3dnet import Vq3dNet
from nets.occnet import OccNet
from nets.preoccnet import PreoccNet
from nets.centernet import CenterNet
from nets.segnet import SegNet


# from nets.mocnet2D import MocNet2D
# from nets.mocnet3D import MocNet3D

from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D


from tensorboardX import SummaryWriter
import torch.nn.functional as F

from utils_moc import MocTrainer
# from utils_basic import *
# import utils_vox
import utils_track
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import utils_eval
import utils_py
import utils_misc
import vox_util

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10


# the idea of this mode is to generate object proposals with just a preoccnet, to see if this outperforms featnet+occnet



class CARLA_OCC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaOccModel()
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_preocc:
            self.model.preoccnet.eval()
            self.set_requires_grad(self.model.preoccnet, False)
        if hyp.do_freeze_center:
            self.model.centernet.eval()
            self.set_requires_grad(self.model.centernet, False)
        if hyp.do_freeze_seg:
            self.model.segnet.eval()
            self.set_requires_grad(self.model.segnet, False)
        if hyp.do_freeze_vq3d:
            self.model.vq3dnet.eval()
            self.set_requires_grad(self.model.vq3dnet, False)

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
            
        if hyp.do_test:

            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            num_ious = len(iou_thresholds)
            all_proposal_maps_3d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
            all_proposal_maps_2d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
            all_proposal_maps_pers = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
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
                    feed_cuda['data_ind'] = data_ind
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_seqlen'] = set_seqlen
                    # feed_cuda['set_data_name'] = set_data_name
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

                        if hyp.do_test and (not returned_early):
                            proposal_maps_3d = results['all_proposal_maps_3d']
                            all_proposal_maps_3d[test_count] = proposal_maps_3d
                            proposal_maps_2d = results['all_proposal_maps_2d']
                            all_proposal_maps_2d[test_count] = proposal_maps_2d
                            proposal_maps_pers = results['all_proposal_maps_pers']
                            all_proposal_maps_pers[test_count] = proposal_maps_pers
                            
                            test_count += 1

                            print('-'*10)
                            
                            mean_proposal_maps_3d = np.mean(all_proposal_maps_3d[:test_count], axis=0)
                            print('mean_proposal_maps_3d', np.mean(mean_proposal_maps_3d, axis=0))
                            
                            mean_proposal_maps_2d = np.mean(all_proposal_maps_2d[:test_count], axis=0)
                            print('mean_proposal_maps_2d', np.mean(mean_proposal_maps_2d, axis=0))
                            
                            mean_proposal_maps_pers = np.mean(all_proposal_maps_pers[:test_count], axis=0)
                            print('mean_proposal_maps_pers', np.mean(mean_proposal_maps_pers, axis=0))
                            
                            # mean_zoom_proposal_maps_3d = np.mean(all_zoom_proposal_maps_3d[:test_count], axis=0)
                            # print('mean_zoom_proposal_maps_3d', np.mean(mean_zoom_proposal_maps_3d, axis=0))
                            
                            # mean_zoom_proposal_maps_2d = np.mean(all_zoom_proposal_maps_2d[:test_count], axis=0)
                            # print('mean_zoom_proposal_maps_2d', np.mean(mean_zoom_proposal_maps_2d, axis=0))
                            
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
            
class CarlaOccModel(nn.Module):
    def __init__(self):
        super(CarlaOccModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            
        self.crop_guess = (18,18,18)
        if hyp.do_feat3D:
            # self.crop_guess = (19,19,19)
            self.featnet3D = FeatNet3D(in_dim=4)#, crop=self.crop_guess)
            
        if hyp.do_up3D:
            self.upnet3D = UpNet3D()
        
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=4)#, crop=self.crop_guess)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
            
        if hyp.do_view:
            self.viewnet = ViewNet()
            
        if hyp.do_render:
            self.rendernet = RenderNet()

        if hyp.do_vq3d:
            self.vq3dnet = Vq3dNet()
            self.labelpools = [utils_misc.SimplePool(100) for i in list(range(hyp.vq3d_num_embeddings))]
            print('declared labelpools')

        if hyp.do_linclass:
            self.linclassnet = LinClassNet(hyp.feat3D_dim)

        if hyp.do_occ:
            self.occnet = OccNet()
            
        if hyp.do_preocc:
            self.preoccnet = PreoccNet()
            
        if hyp.do_center:
            self.centernet = CenterNet()
            
        if hyp.do_seg:
            self.num_seg_labels = 13 # note label0 is "none"
            # we will predict all 12 valid of these, plus one "air" class
            self.segnet = SegNet(self.num_seg_labels)

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

        self.anchor = int(self.S/2)
        self.camXAs_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=self.anchor)
        self.camXs_T_camXAs = __u(utils_geom.safe_inverse(__p(self.camXAs_T_camXs)))
        self.xyz_camXAs = __u(utils_geom.apply_4x4(__p(self.camXAs_T_camXs), __p(self.xyz_camXs)))

        if self.set_name=='test':
            self.box_camRs = feed["box_traj_camR"]
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            self.lrt_camRs = utils_misc.parse_boxes(self.box_camRs, self.origin_T_camRs)
            self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            self.lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)

        else:
            # we don't really need boxes then, but...
            
            origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
            boxlists = feed["boxlists"]
            self.scorelist_s = feed["scorelists"]
            self.tidlist_s = feed["tidlists"]
            # print('boxlists', boxlists.shape)
            boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
            lrtlist_camRs_ = utils_misc.parse_boxes(boxlists_, origin_T_camRs_)
            self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        
            # origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4)
            # self.lrtlist_camRs = utils_misc.parse_boxes(box_camRs, origin_T_camRs)
            # self.lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))
            self.lrtlist_camR0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), __p(self.lrtlist_camRs)))
            self.lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
            self.lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        
        if self.set_name=='test':
            # center on an object, so that it does not fall out of bounds
            self.scene_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        else:
            # center randomly
            all_ok = False
            num_tries = 0
            while not all_ok:
                scene_centroid_x = np.random.uniform(-8.0, 8.0)
                scene_centroid_y = np.random.uniform(-1.5, 3.0)
                scene_centroid_z = np.random.uniform(10.0, 26.0)
                # scene_centroid_x = 0.0
                # scene_centroid_y = 1.0
                # scene_centroid_z = 18.0
                scene_centroid = np.array([scene_centroid_x,
                                           scene_centroid_y,
                                           scene_centroid_z]).reshape([1, 3])
                self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
                num_tries += 1
                all_ok = True
                self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                # we want to ensure this gives us a few points inbound for each batch el
                inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z, self.Y, self.X, already_mem=False, padding=28.0))
                # this is B x S x N
                num_inb = torch.sum(inb.float(), axis=2)
                # this is B x S
                if torch.min(num_inb) < 300:
                    all_ok = False
                if num_tries > 100:
                    return False
            self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
            self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())

        # scene_centroid_x = 0.0
        # scene_centroid_y = 1.0
        # scene_centroid_z = 18.0
        # scene_centroid = np.array([scene_centroid_x,
        #                            scene_centroid_y,
        #                            scene_centroid_z]).reshape([1, 3])
        # self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        # self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
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

        # for b in list(range(self.B)):
        #     # if torch.sum(scorelist_s[b,0]) == 0:
        #     if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
        #         return 0.0, None, True

        # lrtlist_camRs_, obj_lens_ = utils_misc.parse_boxes(__p(feed["boxlists"]), __p(self.origin_T_camRs))
        
        
        
        # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)


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
    
    def run_train(self, feed):
        results = dict()
        

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)


        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        
        self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        self.summ_writer.summ_unps('3D_inputs/rgb_memX0s', torch.unbind(self.rgb_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        

        if hyp.do_preocc:
            # pre-occ is a pre-estimate of occupancy
            # as another mnemonic, it marks the voxels we will preoccupy ourselves with

            crop = self.crop_guess
            Z_, Y_, X_ = self.Z2 - crop[0], self.Y2 - crop[1], self.X2 - crop[2]
            
            occ_memX0_sup, free_memX0_sup, occ_memXs, free_memXs = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2,
                agg=True)

            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_memX0_sup = 1.0 - (F.conv3d(1.0 - free_memX0_sup, weights, padding=1)).clamp(0, 1)
            
            vis_memX0 = (occ_memXs[:,0] + free_memXs[:,0]).clamp(0, 1)
            
            occ_memX0_sup = self.crop_feat(occ_memX0_sup)
            free_memX0_sup = self.crop_feat(free_memX0_sup)
            vis_memX0 = self.crop_feat(vis_memX0)
            
            preocc_input_memX0 = torch.cat([
                self.occ_memX0s[:,0],
                self.rgb_memX0s[:,0]*self.occ_memX0s[:,0],
            ], dim=1)

            # dropout_mask = torch.randint(0, 2, (self.B, 1, Z_, Y_, X_)).cuda().float()
            # print('dropout_mask', dropout_mask.shape)
            # print('preocc_input_memX0', preocc_input_memX0.shape)
            # preocc_input_memX0 = preocc_input_memX0 * dropout_mask

            density_coeff = np.random.uniform(0.01, 0.99)
            # print('coeff = %.2f' % coeff)
            input_mask = (torch.rand((self.B, 1, self.Z, self.Y, self.X)).cuda() < density_coeff).float()
            print('coeff %.3f; density %.3f' % (density_coeff, torch.mean(input_mask).detach().cpu().numpy()))
            # dropout_mask = torch.randint(0, 2, (self.B, 1, self.Z, self.Y, self.X)).cuda().float()
            preocc_input_memX0 = preocc_input_memX0 * input_mask
            
            preocc_loss, occ_memX0_pred = self.preoccnet(
                preocc_input_memX0,
                occ_g=occ_memX0_sup,
                free_g=free_memX0_sup,
                valid=torch.ones_like(occ_memX0_sup),
                summ_writer=self.summ_writer)
            total_loss += preocc_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_explain(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.box_camRs = feed["box_traj_camR"]
        # box_camRs is B x S x 9
        self.score_s = feed["score_traj"]
        self.tid_s = torch.ones_like(self.score_s).long()
        self.lrt_camRs = utils_misc.parse_boxes(self.box_camRs, self.origin_T_camRs)
        self.lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
        self.lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
        self.lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)

        # self.lrt_camRAs = utils_geom.apply_4x4s_to_lrts(self.camRAs_T_camRs, self.lrt_camRs)
        
        full_boxlist_camRs = feed["full_boxlist_camR"]
        full_scorelist_s = feed["full_scorelist"]
        full_tidlist_s = feed["full_tidlist"]


        # full_boxlist_camRs is B x S x N x 9
        N = full_scorelist_s.shape[2]

        # print('full_boxlist_camRs', full_boxlist_
        for b in list(range(self.B)):
            for s in list(range(self.S)):
                for n in list(range(N)):
                    box = full_boxlist_camRs[b,s,n]
                    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box, axis=0)
                    if lx < 1.0:
                        y = y - ly/2.0
                        ly = ly * 2.0
                    box = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=0)
                    full_boxlist_camRs[b,s,n] = box

        
        full_origin_T_camRs = self.origin_T_camRs.unsqueeze(2).repeat(1, 1, N, 1, 1)
        full_lrtlist_camRs_ = utils_misc.parse_boxes(__p(full_boxlist_camRs), __p(full_origin_T_camRs))
        full_lrtlist_camR0s_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), full_lrtlist_camRs_)
        full_lrtlist_camXs_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), full_lrtlist_camRs_)
        full_lrtlist_camX0s_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), full_lrtlist_camXs_)

        self.full_scorelist_s = full_scorelist_s
        self.full_tidlist_s = full_tidlist_s
        self.full_lrtlist_camRs = __u(full_lrtlist_camRs_)
        self.full_lrtlist_camR0s = __u(full_lrtlist_camR0s_)
        self.full_lrtlist_camXs = __u(full_lrtlist_camXs_)
        self.full_lrtlist_camX0s = __u(full_lrtlist_camX0s_)

        self.full_lrtlist_camXAs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXAs_T_camXs), __p(self.full_lrtlist_camXs)))
        # self.moving_lrtlist_camXA0 = utils_geom.apply_4x4_to_lrtlist(self.camXAs_T_camXs[:,0], self.moving_lrtlist_camX00)
        # note the default vox size is in fullres; we want the halfmem
        pad = (self.vox_util.default_vox_size_X*2.0) * self.crop_guess[0]
        # print('pad: %.2f meters' % pad)
        
        full_scorelist_s_ = utils_misc.rescore_lrtlist_with_inbound(
            __p(self.full_lrtlist_camXAs), __p(self.full_tidlist_s), self.Z, self.Y, self.X, self.vox_util, pad=pad)
        self.full_scorelist_s = __u(full_scorelist_s_)

        # rescore based on motion
        new_scorelist_s = torch.zeros_like(self.full_scorelist_s)
        for n in list(range(self.N)):
            for s0 in list(range(self.S)):
                if s0==0:
                    s1 = s0+1
                else:
                    s1 = s0-1
                target = self.full_lrtlist_camX0s[:,s0,n] # B x 19
                score = self.full_scorelist_s[:,s0,n] # B
                for b in list(range(self.B)):
                    if score[b] > 0.5 and target[b,0] > 0.01:
                        ious = np.zeros((self.N), dtype=np.float32)
                        for i in list(range(self.N)):
                            if self.full_scorelist_s[b,s1,i] > 0.5  and self.full_lrtlist_camX0s[b,s1,i,0] > 0.01:
                                iou_3d, _ = utils_geom.get_iou_from_corresponded_lrtlists(
                                    target[b:b+1].unsqueeze(1), self.full_lrtlist_camX0s[b:b+1,s1,i:i+1])
                                ious[i] = np.squeeze(iou_3d[0,0])
                        if float(np.max(ious)) < 0.97:
                            # the object must have moved
                            new_scorelist_s[b,s0,n] = 1.0
        self.full_scorelist_s = new_scorelist_s * self.full_scorelist_s

        print('objects detectable across the entire seq:', torch.sum(self.full_scorelist_s).detach().cpu().numpy())
        if torch.sum(self.full_scorelist_s) == 0:
            # return early, since no objects are inbound AND moving
            return total_loss, results, True

        # return total_loss, results, True
                
        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        vis = []
        # for s in list(range(0, self.S, 2)):
        for s in list(range(0, self.S)):
            vis.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,s],
                self.full_lrtlist_camXs[:,s],
                self.full_scorelist_s[:,s],
                self.full_tidlist_s[:,s],
                self.pix_T_cams[:,s],
                only_return=True))
        self.summ_writer.summ_rgbs('2D_inputs/lrtlist_camXs', vis)

        # return total_loss, results, True
        

        # # self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        # #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # # self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        
        # # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3D_inputs/rgb_memX0s', torch.unbind(self.rgb_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))

        have_feats = False
        have_medians = False
        # have_boxes = False

        use_feat_cache = False
        data_ind = feed['data_ind']
                
        with torch.no_grad():
            vis_memXAI_all = []
            self.occ_memXAI_all = []
            occrel_memXAI_all = []

            for I in list(range(self.S)):
                print('computing feats for I', I)

                occ_memXAIs, free_memXAIs, _, _ = self.vox_util.prep_occs_supervision(
                    self.camXAs_T_camXs[:,I:I+1],
                    self.xyz_camXs[:,I:I+1],
                    self.Z2, self.Y2, self.X2,
                    agg=False)

                occ_memXAI_g = self.crop_feat(occ_memXAIs.squeeze(1))
                free_memXAI_g = self.crop_feat(free_memXAIs.squeeze(1))

                vis_memXAI = (occ_memXAI_g + free_memXAI_g).clamp(0, 1)

                self.rgb_memXII = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_camXs[:,I], self.Z, self.Y, self.X, self.pix_T_cams[:,I])
                # self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                self.occ_memXAI = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,I], self.Z, self.Y, self.X)

                preocc_memXAI_input = torch.cat([
                    self.occ_memXAI,
                    self.rgb_memXAI*self.occ_memXAI,
                ], dim=1)
                _, occ_memXAI = self.preoccnet(preocc_memXAI_input)

                # _, occrel_memXAI = self.occrelnet(feat_memXAI)
                occrel_memXAI = torch.ones_like(occ_memXAI)

                use_occrel = False
                if not use_occrel:
                    occrel_memXAI = torch.ones_like(occrel_memXAI)

                _, _, Z2, Y2, X2 = list(occ_memXAI.shape)
                Z_crop = int((self.Z2 - Z2)/2)
                Y_crop = int((self.Y2 - Y2)/2)
                X_crop = int((self.X2 - X2)/2)
                crop = (Z_crop, Y_crop, X_crop)
                if not (crop==self.crop_guess):
                    print('crop', crop)
                assert(crop==self.crop_guess) # otw we need to rewrite self.crop above

                vis_memXAI_all.append(vis_memXAI)
                self.occ_memXAI_all.append(occ_memXAI)
                occrel_memXAI_all.append(occrel_memXAI)

        self.summ_writer.summ_occs('3D_feats/occ_memXAI', self.occ_memXAI_all)

        occ_memXAI_all_np = (torch.stack(self.occ_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
        vis_memXAI_all_np = (torch.stack(vis_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
        occ_memXAI_median_np_safe = np.median(occ_memXAI_all_np, axis=0)
        occ_memXAI_median_np = utils_py.reduce_masked_median(
            occ_memXAI_all_np.transpose(1, 0), vis_memXAI_all_np.transpose(1, 0), keep_batch=True)
        occ_memXAI_median_np[np.isnan(occ_memXAI_median_np)] = occ_memXAI_median_np_safe[np.isnan(occ_memXAI_median_np)]
        self.occ_memXAI_median = torch.from_numpy(occ_memXAI_median_np).float().reshape(1, -1, Z2, Y2, X2).cuda()

        # occ_memXAI_diff_np = np.mean(np.abs(occ_memXAI_all_np[1:] - occ_memXAI_all_np[:-1]), axis=0)
        # occ_memXAI_diff = torch.from_numpy(occ_memXAI_diff_np).float().reshape(1, 1, Z2, Y2, X2).cuda()
                
        self.summ_writer.summ_occ('3D_feats/occ_memXAI_median', self.occ_memXAI_median)
        
        # now, i should be able to walk through a second time, and collect great diff signals
        # if use_feat_cache:

        self.diff_memXAI_all = []
        for I in list(range(self.S)):
            vis_memXAI = vis_memXAI_all[I]

            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
            vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)

            occ_memXAI = self.occ_memXAI_all[I]
            occrel_memXAI = occrel_memXAI_all[I]

            use_occrel = False
            if not use_occrel:
                occrel_memXAI = torch.ones_like(occrel_memXAI)

            # diff_memXAI_all.append(torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(occ_memXAI * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(occ_memXAI.round() * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))

            # diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True)
            diff = torch.norm(occ_memXAI - self.occ_memXAI_median, dim=1, keepdim=True) * occrel_memXAI
            # diff = torch.nn.functional.relu(diff - occ_memXAI_diff)

            # diff = occ_memXAI.round() * vis_memXAI * diff
            diff = vis_memXAI * diff
            self.diff_memXAI_all.append(diff)

        
        self.K = 32

        super_lrtlist = []
        super_scorelist = []
        super_tidlist = []

        for super_iter in list(range(4)):
            print('-'*100)
            print('super_iter %d' % super_iter)

            diff_memXAI_vis = []
            for I in list(range(self.S)):
                diff_memXAI_vis.append(self.summ_writer.summ_oned('', self.diff_memXAI_all[I], bev=True, max_along_y=True, norm=False, only_return=True))
            self.summ_writer.summ_rgbs('3D_feats/diff_memXAI_all_%d' % super_iter, diff_memXAI_vis)
            
            lrtlist_memXAI_all, connlist_memXAI_all, scorelist_all, blue_vis_all = utils_misc.propose_boxes_by_differencing(
                self.K, self.S, self.occ_memXAI_all, self.diff_memXAI_all, self.crop_guess,
                None, data_ind, super_iter, use_box_cache=False)
                # self.set_data_name, data_ind, super_iter, use_box_cache=False)
            self.summ_writer.summ_rgbs('proposals/blue_boxes_%d' % super_iter, blue_vis_all)

            camXs_T_camXAs_all = list(self.camXs_T_camXAs.unbind(1))
            lrtlist_camXAI_all = [self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_memXAI, self.Z2, self.Y2, self.X2)
                                  for lrtlist_memXAI in lrtlist_memXAI_all]
            lrtlist_camXI_all = [utils_geom.apply_4x4_to_lrtlist(camXI_T_camXA, lrtlist_camXAI)
                                 for (camXI_T_camXA, lrtlist_camXAI) in zip(camXs_T_camXAs_all, lrtlist_camXAI_all)]

            if super_iter == 0:
                # quick eval:
                # note that since B=1, if i pack then i'll have tensors shaped S x N x 19
                super_lrtlist_ = __p(torch.stack(lrtlist_camXI_all, dim=1))
                super_scorelist_ = __p(torch.stack(scorelist_all, dim=1))
                full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
                full_scorelist_s_ = __p(self.full_scorelist_s)
                iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                all_maps_3d = np.zeros([self.S, len(iou_thresholds)])
                all_maps_2d = np.zeros([self.S, len(iou_thresholds)])
                all_maps_pers = np.zeros([self.S, len(iou_thresholds)])
                all_maps_valid = np.zeros([self.S])
                for s in list(range(self.S)):
                    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                        super_lrtlist_[s:s+1], full_lrtlist_camXs_[s:s+1], super_scorelist_[s:s+1], full_scorelist_s_[s:s+1])

                    if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:
                        all_maps_valid[s] = 1.0
                        maps_3d, maps_2d = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                        all_maps_3d[s] = maps_3d
                        all_maps_2d[s] = maps_2d
                        boxlist_e = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_e)
                        boxlist_g = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_g)
                        maps_pers = utils_eval.get_mAP_from_2d_boxlists(boxlist_e, scorelist_e, boxlist_g, iou_thresholds)
                        all_maps_pers[s] = maps_pers
                    elif torch.sum(scorelist_g) > 0:
                        all_maps_valid[s] = 1.0
                        all_maps_3d[s] = 0.0
                        all_maps_2d[s] = 0.0
                        all_maps_pers[s] = 0.0
                    
                for ind, overlap in enumerate(iou_thresholds):
                    maps_3d = all_maps_3d[:,ind]
                    maps_2d = all_maps_2d[:,ind]
                    maps_pers = all_maps_pers[:,ind]
                    
                    map_3d_val = utils_py.reduce_masked_mean(maps_3d, all_maps_valid)
                    map_2d_val = utils_py.reduce_masked_mean(maps_2d, all_maps_valid)
                    map_pers_val = utils_py.reduce_masked_mean(maps_pers, all_maps_valid)
                    
                    # if len(maps_3d):
                        # map_3d_val = np.mean(maps_3d)
                        # map_2d_val = np.mean(maps_2d)
                        # map_pers_val = np.mean(maps_pers)
                    # else:
                    #     map_3d_val = 0.0
                    #     map_2d_val = 0.0
                    #     map_pers_val = 0.0
                    self.summ_writer.summ_scalar('proposal_ap_3d/%.2f_iou' % overlap, map_3d_val)
                    self.summ_writer.summ_scalar('proposal_ap_2d/%.2f_iou' % overlap, map_2d_val)
                    self.summ_writer.summ_scalar('proposal_ap_pers/%.2f_iou' % overlap, map_pers_val)
                results['all_proposal_maps_3d'] = all_maps_3d
                results['all_proposal_maps_2d'] = all_maps_2d
                results['all_proposal_maps_pers'] = all_maps_pers
                
            box_vis_bev = []
            box_vis = []
            for I in list(range(self.S)):
                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,I],
                    torch.cat([self.full_lrtlist_camXs[:,I], lrtlist_camXI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.pix_T_cams[:,0], frame_id=I, only_return=True))
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.pad_feat(self.occ_memXAI_all[I]),
                    torch.cat([self.full_lrtlist_camXAs[:,I], lrtlist_camXAI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.vox_util, frame_id=I, only_return=True))
            self.summ_writer.summ_rgbs('proposals/all_boxes_bev_%d' % super_iter, box_vis_bev)
            self.summ_writer.summ_rgbs('proposals/all_boxes_%d' % super_iter, box_vis)

            # return here if you just want proposal eval
            return total_loss, results, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        
        # if set_name=='moc2D_init':
        #     self.prepare_common_tensors(feed, prep_summ=False)
        #     return self.prep_neg_emb2D(feed)
        
        # if set_name=='moc3D_init':
        #     self.prepare_common_tensors(feed, prep_summ=False)
        #     return self.prep_neg_emb3D(feed)

        ok = self.prepare_common_tensors(feed)
        if not ok:
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if set_name=='train':
                return self.run_train(feed)
            elif set_name=='test':
                return self.run_explain(feed)

