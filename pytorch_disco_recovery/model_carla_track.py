import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.flownet import FlowNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_TRACK(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaTrackModel()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_preocc:
            self.model.preoccnet.eval()
            self.set_requires_grad(self.model.preoccnet, False)
        if hyp.do_freeze_emb2D:
            self.model.embnet2D.eval()
            self.set_requires_grad(self.model.embnet2D, False)

    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")
        if hyp.lr > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
            self.start_iter = saverloader.load_weights(self.model, self.optimizer)
            print("------ Done loading weights ------")
        else:
            self.start_iter = 0

        if hyp.do_save_outputs:
            out_dir = 'outs/%s' % (hyp.name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        set_nums = []
        set_names = []
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
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        if hyp.do_eval_map:
            all_ious = {}
            for s in list(range(1,hyp.S)):
                all_ious['%d' % s] = []
                
        actual_step = 0
        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    set_loaders[i] = iter(set_input)

            for (set_num,
                 set_name,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_names,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders
            ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                if log_this or set_do_backprop or hyp.do_save_outputs:
                          
                    read_start_time = time.time()

                    feed = next(set_loader)
                    feed_cuda = {}
                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True)
                        except:
                            # some things are not tensors (e.g., filename)
                            feed_cuda[k] = feed[k]

                    # feed_cuda = next(iter(set_input))
                    read_time = time.time() - read_start_time
                    
                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_name'] = set_name


                    filename = feed_cuda['filename'][0]
                    # print('filename = %s' % filename)
                    tokens = filename.split('/')
                    filename = tokens[-1]
                    # print('new filename = %s' % filename)
                    
                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_vis = loss.cpu().item()

                    print(results)
                    if hyp.do_eval_map and (not returned_early):
                        for s in list(range(1,hyp.S)):
                            iou = results['iou_%d' % s]
                            all_ious['%d' % s].append(iou)
                            all_ious_ = np.stack(all_ious['%d' % s], axis=0)
                            mean_ious_ = np.mean(all_ious_)
                            std_ious_ = np.std(all_ious_)
                            print('iou_%d = %.2f +- %.2f' % (s, mean_ious_, std_ious_))
                            
                    if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_vis,
                                                                                        set_name))
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()
            
class CarlaTrackModel(nn.Module):
    def __init__(self):
        super(CarlaTrackModel, self).__init__()
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()

        self.feat_net = frozen_flow_net.FrozenFeatNet(
            '../discovery/%s_feat_model.pb' % hyp.pb_model)
        self.flow_net = frozen_flow_net.FrozenFlowNet(
            '../discovery/%s_flow_model.pb' % hyp.pb_model)
            
        self.device = torch.device("cuda")
        
        self.include_image_summs = hyp.do_include_summs

        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=True)
        global_step = feed['global_step']
        npz_filename = feed['filename']
        # print('npz_filename', npz_filename)

        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9

        rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        cam_T_velos = feed["cam_T_velos"]

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camXs = feed["origin_T_camXs"]

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camR0_T_camRs = utils_geom.get_camM_T_camXs(origin_T_camRs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        camR0_T_camXs = __u(torch.matmul(__p(camR0_T_camRs), __p(camRs_T_camXs)))
        
        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))
        xyz_camR0s = __u(utils_geom.apply_4x4(__p(camR0_T_camXs), __p(xyz_camXs)))
                            
        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        # occX0s = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z, Y, X))
        # occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        occR0s = __u(utils_vox.voxelize_xyz(__p(xyz_camR0s), Z, Y, X))
        occR0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camR0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        # unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
        #     __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))
        # unpX0s = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs)
        # unpRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, unpXs)
        unpR0s = utils_vox.apply_4x4s_to_voxs(camR0_T_camXs, unpXs)
                                              
        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        ## boxes
        boxlist_camRs = feed["boxes3D"]
        tidlist_s = feed["tids"] # coordinate-less and plural
        scorelist_s = feed["scores"] # coordinate-less and plural
        scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(camR0_T_camRs), __p(boxlist_camRs), __p(tidlist_s), Z, Y, X))

        for b in list(range(B)):
            if torch.sum(scorelist_s[b,0]) == 0:
                return total_loss, results, True
        
        # we have ensured there is at least one car; but let's not penalize for detecting bikes
        # so let's recompute the scores
        scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(camR0_T_camRs), __p(boxlist_camRs), __p(tidlist_s), Z, Y, X, only_cars=False, pad=0.0))
        boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            __p(boxlist_camRs), __p(tidlist_s), __p(scorelist_s))
        boxlist_camRs = __u(boxlist_camRs_)
        tidlist_s = __u(tidlist_s_)
        scorelist_s = __u(scorelist_s_)
        lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))).reshape(B, S, N, 19)
        lrtlist_camR0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camR0_T_camRs), __p(lrtlist_camRs)))
        lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
        
        (obj_lrtlist_camR0s,
         obj_scorelist_s,
        ) = utils_misc.collect_object_info(lrtlist_camR0s,
                                           tidlist_s,
                                           scorelist_s,
                                           K, mod='R',
                                           do_vis=False,
                                           summ_writer=summ_writer)
        (obj_lrtlist_camXs,
         obj_scorelist_s,
        ) = utils_misc.collect_object_info(lrtlist_camXs,
                                           tidlist_s,
                                           scorelist_s,
                                           K, mod='X',
                                           do_vis=False,
                                           summ_writer=summ_writer)
        # these are N x B x S x 19

        for b in list(range(B)):
            print('sum of obj0 scorelist', torch.sum(obj_scorelist_s[0,b,:]))
            if (not torch.sum(obj_scorelist_s[0,b,:]) == S):
                return total_loss, results, True
        
        #####################
        ## visualize what we got
        #####################

        mask_memR0s = utils_vox.assemble_padded_obj_masklist(
            obj_lrtlist_camR0s[0], obj_scorelist_s[0], Z2, Y2, X2, coeff=0.9)
        # this is B x S x 1 x Z x Y x X
        mask_mems_g = mask_memR0s

        if self.include_image_summs:
            summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
            summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
            summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
            summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
            # summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
            # summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
            summ_writer.summ_occs('3D_inputs/occR0s', torch.unbind(occR0s, dim=1))
            summ_writer.summ_unps('3D_inputs/unpR0s', torch.unbind(unpR0s, dim=1), torch.unbind(occR0s, dim=1))
            summ_writer.summ_occs('obj/mask_mems_g', torch.unbind(mask_memR0s, dim=1))

        lrt_camR0s_e, all_ious = utils_track.track_via_chained_flows(
            obj_lrtlist_camR0s[0], 
            mask_memR0s[:,0],
            self,
            occR0s,
            occR0s_half,
            unpR0s,
            summ_writer,
            include_image_summs=self.include_image_summs,
        )

        print('all_ious', all_ious)
        for s in range(1, hyp.S):
            results['iou_%d' % s] = torch.mean(all_ious[s-1]).cpu().item()

        mask_memR0s_e = utils_vox.assemble_padded_obj_masklist(
            lrt_camR0s_e, scorelist_s[:,:,0], Z2, Y2, X2, coeff=0.9)

        if self.include_image_summs:
            summ_writer.summ_occs('obj/mask_mems_e', torch.unbind(mask_memR0s_e, dim=1))
            summ_writer.summ_lrtlist('obj/box_cam%02d' % 0, rgb_camRs[:,0], obj_lrtlist_camR0s[0],
                                     obj_scorelist_s[0], tidlist_s[:,0], pix_T_cams[:,0])

        # take to X coords (in two steps)
        lrt_camR0s_e_ = lrt_camR0s_e.unsqueeze(2)
        lrt_camRs_e_ = __u(utils_geom.apply_4x4_to_lrtlist(utils_geom.safe_inverse(__p(camR0_T_camRs)), __p(lrt_camR0s_e_)))
        lrt_camXs_e_ = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrt_camRs_e_)))

        if self.include_image_summs:
            visX_e = []
            visX_g = []
            for s in list(range(S)):
                visX_e.append(summ_writer.summ_lrtlist('obj/box_camX%d_e' % s, rgb_camXs[:,s], lrt_camXs_e_[:,s],
                                                       obj_scorelist_s[0,:,s:s+1], tidlist_s[:,0], pix_T_cams[:,0], only_return=True))
                visX_g.append(summ_writer.summ_lrtlist('obj/box_camX%d_g' % s, rgb_camXs[:,s], obj_lrtlist_camXs[0,:,s].unsqueeze(1),
                                                       obj_scorelist_s[0,:,s:s+1], tidlist_s[:,0], pix_T_cams[:,0], only_return=True))
            summ_writer.summ_rgbs('obj/box_camXs_e', visX_e)
            summ_writer.summ_rgbs('obj/box_camXs_g', visX_g)

        if hyp.do_save_outputs:
            flow_memRs = []
            for s in list(range(S-1)):
                print('computing basic flow from %d to %d' % (s, s+1))
                input_mem0 = input_mems[:,s]
                input_mem1 = input_mems[:,s+1]
                featnet_inputs = torch.stack([input_mem0, input_mem1], dim=1)
                featnet_outputs = self.feat_net.infer_pt(featnet_inputs)
                featnet_output_mem0 = featnet_outputs[:,0]
                featnet_output_mem1 = featnet_outputs[:,1]
                flow_mem0 = self.flow_net.infer_pt([featnet_output_mem0,
                                                    featnet_output_mem1])
                flow_memRs.append(flow_mem0)
            flow_memRs = torch.stack(flow_memRs, dim=1)
        
        if hyp.do_save_outputs:
            results['lrtlist_camR0s_e'] = lrtlist_camR0s_e
            results['lrtlist_camRs_e'] = lrtlist_camRs_e
            results['flow_memRs'] = flow_memRs

        
        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False


