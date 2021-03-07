import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
# from nets.featnet import FeatNet
# from nets.occnet import OccNet
# from nets.flownet import FlowNet
# from nets.viewnet import ViewNet
# from nets.embnet2D import EmbNet2D
# from nets.embnet3D import EmbNet3D
from nets.pwcnet import PWCNet

import torch.nn.functional as F

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_eval

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_PWC(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaPWCNet().to(self.device)

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
            all_maps_v0 = []
            all_maps_v1 = []
            all_maps_v2 = []
            all_maps_v3 = []
            all_maps_v4 = []
            all_maps_v5 = []
            
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

                    # if hyp.do_save_outputs:
                    #     out_fn = '%s/%s_flow_memRs.npy' % (out_dir, filename)
                    #     flow_memRs = results['flow_memRs'][0]
                    #     flow_memRs = flow_memRs.detach().cpu()
                    #     np.save(out_fn, flow_memRs)
                    #     print('saved %s' % out_fn)
                    #     print(flow_memRs.shape)

                    if hyp.do_eval_map and (not returned_early):
                        maps_v0 = results['maps_v0']
                        all_maps_v0.append(maps_v0)
                        all_maps_v0_ = np.stack(all_maps_v0, axis=0)
                        all_maps_v0_ = np.mean(all_maps_v0_, axis=0)
                        print('all_maps_v0_:', all_maps_v0_)
                        
                        maps_v1 = results['maps_v1']
                        all_maps_v1.append(maps_v1)
                        all_maps_v1_ = np.stack(all_maps_v1, axis=0)
                        all_maps_v1_ = np.mean(all_maps_v1_, axis=0)
                        print('all_maps_v1_:', all_maps_v1_)
                        
                        maps_v2 = results['maps_v2']
                        all_maps_v2.append(maps_v2)
                        all_maps_v2_ = np.stack(all_maps_v2, axis=0)
                        all_maps_v2_ = np.mean(all_maps_v2_, axis=0)
                        print('all_maps_v2_:', all_maps_v2_)
                        
                        maps_v3 = results['maps_v3']
                        all_maps_v3.append(maps_v3)
                        all_maps_v3_ = np.stack(all_maps_v3, axis=0)
                        all_maps_v3_ = np.mean(all_maps_v3_, axis=0)
                        print('all_maps_v3_:', all_maps_v3_)
                        
                        maps_v4 = results['maps_v4']
                        all_maps_v4.append(maps_v4)
                        all_maps_v4_ = np.stack(all_maps_v4, axis=0)
                        all_maps_v4_ = np.mean(all_maps_v4_, axis=0)
                        print('all_maps_v4_:', all_maps_v4_)
                        
                        maps_v5 = results['maps_v5']
                        all_maps_v5.append(maps_v5)
                        all_maps_v5_ = np.stack(all_maps_v5, axis=0)
                        all_maps_v5_ = np.mean(all_maps_v5_, axis=0)
                        print('all_maps_v5_:', all_maps_v5_)
                        
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
        

class CarlaPWCNet(nn.Module):
    def __init__(self):
        super(CarlaPWCNet, self).__init__()
        self.pwcnet = PWCNet()

        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.autograd.set_detect_anomaly(True)
        
        self.include_image_summs = hyp.do_include_summs

    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)
        
        writer = feed['writer']
        global_step = feed['global_step']

        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        D = 9

        rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs"]
        pix_T_cams = feed["pix_T_cams"]
        cam_T_velos = feed["cam_T_velos"]
        
        if (not hyp.flow_do_synth_rt) or feed['set_name']=='val':
            boxlist_camRs = feed["boxes3D"]
            tidlist_s = feed["tids"] # coordinate-less and plural
            scorelist_s = feed["scores"] # coordinate-less and plural
            # # postproc the boxes:
            # scorelist_s = __u(utils_misc.rescore_boxlist_with_inbound(__p(boxlist_camRs), __p(tidlist_s), Z, Y, X))

            boxlist_camRs_, tidlist_s_, scorelist_s_ = __p(boxlist_camRs), __p(tidlist_s), __p(scorelist_s)
            boxlist_camRs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
                boxlist_camRs_, tidlist_s_, scorelist_s_)
            boxlist_camRs = __u(boxlist_camRs_)
            tidlist_s = __u(tidlist_s_)
            scorelist_s = __u(scorelist_s_)

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed["origin_T_camXs"]
        origin_T_camXs_ = __p(origin_T_camXs)

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camX0_T_camXs_ = __p(camX0_T_camXs)
        camRs_T_camXs_ = torch.matmul(origin_T_camRs_.inverse(), origin_T_camXs_)
        camXs_T_camRs_ = camRs_T_camXs_.inverse()
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        # occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        occX0s = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z, Y, X))
        # occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpX0s = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs)
        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))
        unpX0s_half = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs_half)

        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        inbound_camXs = __u(inbound_camXs_)
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)



        depth_camX0s_, valid_camX0s_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camX0s), H, W)
        depth_camX0s = __u(depth_camX0s_)
        

        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        # summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occX0s', torch.unbind(occX0s, dim=1))
        # summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpX0s', torch.unbind(unpX0s, dim=1), torch.unbind(occX0s, dim=1))


        if (not hyp.flow_do_synth_rt) or feed['set_name']=='val':
            lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(boxlist_camRs_)).reshape(B, S, N, 19)
            lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
            # stabilize boxes for ego/cam motion
            lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camX0_T_camXs), __p(lrtlist_camXs)))
            # these are is B x S x N x 19

            summ_writer.summ_lrtlist('lrtlist_camR0', rgb_camRs[:,0], lrtlist_camRs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('lrtlist_camR1', rgb_camRs[:,1], lrtlist_camRs[:,1],
                                     scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
            summ_writer.summ_lrtlist('lrtlist_camX0', rgb_camXs[:,0], lrtlist_camXs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('lrtlist_camX1', rgb_camXs[:,1], lrtlist_camXs[:,1],
                                     scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
            (obj_lrtlist_camXs,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camXs,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='X',
                                               do_vis=True,
                                               summ_writer=summ_writer)
            (obj_lrtlist_camRs,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camRs,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='R',
                                               do_vis=True,
                                               summ_writer=summ_writer)
            (obj_lrtlist_camX0s,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camX0s,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='X0',
                                               do_vis=False)

            masklist_memR = utils_vox.assemble_padded_obj_masklist(
                lrtlist_camRs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
            masklist_memX = utils_vox.assemble_padded_obj_masklist(
                lrtlist_camXs[:,0], scorelist_s[:,0], Z, Y, X, coeff=1.0)
            # obj_mask_memR is B x N x 1 x Z x Y x X
            summ_writer.summ_occ('obj/masklist_memR', torch.sum(masklist_memR, dim=1))
            summ_writer.summ_occ('obj/masklist_memX', torch.sum(masklist_memX, dim=1))

        # flow_pwc = self.pwcnet(rgb_camRs[:, 0] + 0.5, rgb_camRs[:, 1] + 0.5) # only 1/255 normalization in pwcnet 
        # summ_writer.summ_flow('flow/flow_pwc', flow_pwc)

        # compute flow from X0 to X1
        # (to be fair with our own models, we use inputs in X coordinates)
        pwcflow_camX0 = self.pwcnet(rgb_camXs[:, 0] + 0.5, rgb_camXs[:, 1] + 0.5) # only 1/255 normalization in pwcnet 
        summ_writer.summ_flow('flow/pwcflow_camX0', pwcflow_camX0)

        
        camX0_T_camX1 = camX0_T_camXs[:,1]
        camX1_T_camX0 = utils_geom.safe_inverse(camX0_T_camX1)
        depth_camX0 = depth_camXs[:,0]
        egoflow_camX0 = utils_geom.depthrt2flow(depth_camX0, camX1_T_camX0, pix_T_cams[:,0])
        summ_writer.summ_flow('flow/egoflow_camX0', egoflow_camX0)
        rgb_camX1_egostab = utils_samp.backwarp_using_2D_flow(rgb_camXs[:,1], egoflow_camX0)
        valid_camX1_egostab = utils_samp.backwarp_using_2D_flow(torch.ones_like(rgb_camXs[:,1,0:1]), egoflow_camX0)
        valid_camX1_flowstab = utils_samp.backwarp_using_2D_flow(torch.ones_like(rgb_camXs[:,1,0:1]), pwcflow_camX0)
        egostab_rgbX0 = rgb_camXs[:,0]*valid_camX1_egostab
        egostab_rgbX1 = rgb_camX1_egostab*valid_camX1_egostab
        summ_writer.summ_rgbs('flow/rgb_stab', [egostab_rgbX0, egostab_rgbX1])

        # compute stabflow from X0 to X1
        stabpwcflow_camX0 = self.pwcnet(egostab_rgbX0 + 0.5, egostab_rgbX1 + 0.5) # only 1/255 normalization in pwcnet 
        summ_writer.summ_flow('flow/stabpwcflow_camX0', stabpwcflow_camX0*inbound_camXs[:,0])

        # prep occR0, which we will use to mask the 3D flows
        occR0 = utils_vox.voxelize_xyz(xyz_camRs[:,0], Z2, Y2, X2)
        
        # v0: 2D flow estimated from real frames, minus egoflow, unprojected
        v0flow_camX0 = pwcflow_camX0 - egoflow_camX0
        summ_writer.summ_flow('flow/v0flow_camX0', v0flow_camX0*inbound_camXs[:,0])
        v0flow_memX0 = utils_vox.unproject_rgb_to_mem(
            v0flow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v0flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v0flow_memX0)
        v0flow_memR0 = torch.cat([v0flow_memR0, torch.ones_like(v0flow_memR0[:,0:1])], dim=1)
        v0flow_memR0 = v0flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v0flow_memR0', v0flow_memR0, clip=0.0)

        # v1: 2D flow estimated from ego-stabilized frames, unprojected
        v1flow_memX0 = utils_vox.unproject_rgb_to_mem(
            stabpwcflow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v1flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v1flow_memX0)
        v1flow_memR0 = torch.cat([v1flow_memR0, torch.ones_like(v1flow_memR0[:,0:1])], dim=1)
        v1flow_memR0 = v1flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v1flow_memR0', v1flow_memR0, clip=0.0)

        # v2: 3D flow estimated by backwarping depth1, unprojected
        depth_camX1_flowstab = utils_samp.backwarp_using_2D_flow(depth_camXs[:,1], pwcflow_camX0)
        xyz0 = utils_geom.depth2pointcloud(depth_camXs[:,0], pix_T_cams[:,0])
        xyz1 = utils_geom.depth2pointcloud(depth_camX1_flowstab, pix_T_cams[:,0])
        v2flow_camX0 = xyz1-xyz0
        v2flow_camX0 = v2flow_camX0.reshape([B, H, W, 3]).permute(0, 3, 1, 2)*valid_camX1_flowstab
        summ_writer.summ_flow('flow/v2flow_camX0', v2flow_camX0[:,:2]*inbound_camXs[:,0])
        v2flow_memX0 = utils_vox.unproject_rgb_to_mem(
            v2flow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v2flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v2flow_memX0)
        v2flow_memR0 = v2flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v2flow_memR0', v2flow_memR0, clip=0.0)

        # v3: 3D flow estimated by backwarping the ego-stabilized pointcloud
        depth_camX1_egostab = utils_samp.backwarp_using_2D_flow(depth_camXs[:,1], egoflow_camX0)
        depth_camX1_egostab_and_flowstab = utils_samp.backwarp_using_2D_flow(depth_camX1_egostab, stabpwcflow_camX0)
        xyz0 = utils_geom.depth2pointcloud(depth_camXs[:,0], pix_T_cams[:,0])
        xyz1 = utils_geom.depth2pointcloud(depth_camX1_egostab_and_flowstab, pix_T_cams[:,0])
        v3flow_camX0 = xyz1-xyz0
        v3flow_camX0 = v3flow_camX0.reshape([B, H, W, 3]).permute(0, 3, 1, 2)*valid_camX1_egostab
        summ_writer.summ_flow('flow/v3flow_camX0_2chan', v3flow_camX0[:,:2]*inbound_camXs[:,0])
        v3flow_memX0 = utils_vox.unproject_rgb_to_mem(
            v3flow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v3flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v3flow_memX0)
        v3flow_memR0 = v3flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v3flow_memR0', v3flow_memR0, clip=0.0)


        # v4: 2D flow estimated from real frames, unprojected
        v4flow_camX0 = pwcflow_camX0
        summ_writer.summ_flow('flow/v4flow_camX0', v4flow_camX0*inbound_camXs[:,0])
        v4flow_memX0 = utils_vox.unproject_rgb_to_mem(
            v4flow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v4flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v4flow_memX0)
        v4flow_memR0 = torch.cat([v4flow_memR0, torch.ones_like(v4flow_memR0[:,0:1])], dim=1)
        v4flow_memR0 = v4flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v4flow_memR0', v4flow_memR0, clip=0.0)


        # v5: 3D flow estimated by backwarping the geometrically-stabilized pointcloud
        depth_camX1_egostab_and_flowstab = utils_samp.backwarp_using_2D_flow(depth_camX0s[:,1], stabpwcflow_camX0)
        xyz0 = utils_geom.depth2pointcloud(depth_camXs[:,0], pix_T_cams[:,0])
        xyz1 = utils_geom.depth2pointcloud(depth_camX1_egostab_and_flowstab, pix_T_cams[:,0])
        v5flow_camX0 = xyz1-xyz0
        v5flow_camX0 = v5flow_camX0.reshape([B, H, W, 3]).permute(0, 3, 1, 2)*valid_camX1_egostab
        summ_writer.summ_flow('flow/v5flow_camX0_2chan', v5flow_camX0[:,:2]*inbound_camXs[:,0])
        v5flow_memX0 = utils_vox.unproject_rgb_to_mem(
            v5flow_camX0, Z2, Y2, X2, pix_T_cams[:,0])
        v5flow_memR0 = utils_vox.apply_4x4_to_vox(camRs_T_camXs[:,0], v5flow_memX0)
        v5flow_memR0 = v5flow_memR0 * occR0
        summ_writer.summ_3D_flow('flow/v5flow_memR0', v5flow_memR0, clip=0.0)
        

        if hyp.do_eval_map:
            maps_v0 = self.discover(v0flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            maps_v1 = self.discover(v1flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            maps_v2 = self.discover(v2flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            maps_v3 = self.discover(v3flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            maps_v4 = self.discover(v4flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            maps_v5 = self.discover(v5flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            results['maps_v0'] = maps_v0
            results['maps_v1'] = maps_v1
            results['maps_v2'] = maps_v2
            results['maps_v3'] = maps_v3
            results['maps_v4'] = maps_v4
            results['maps_v5'] = maps_v5
        else:
            maps_v0 = self.discover(v0flow_memR0, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer)
            
        return total_loss, results, False

    
    def discover(self, flow, K, occR0, camXs_T_camRs, rgb_camRs, rgb_camXs, boxlist_camRs, scorelist_s, pix_T_cams, B, Z2, Y2, X2, summ_writer):
        flow_mag = torch.norm(flow, dim=1)
        # this is B x Z2 x Y2 x X2
        occ_flow_mag = flow_mag * occR0[:,0]

        # get K boxes
        det_image, boxlist_memR, scorelist, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(occ_flow_mag, K)
        # boxlist_memR is B x K x 9
        boxlist_camR = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR, Z2, Y2, X2)
        lrtlist_camR = utils_geom.convert_boxlist_to_lrtlist(boxlist_camR)

        masklist_1 = utils_vox.assemble_padded_obj_masklist(
            lrtlist_camR, scorelist, Z2, Y2, X2, coeff=0.8)
        masklist_2 = utils_vox.assemble_padded_obj_masklist(
            lrtlist_camR, scorelist, Z2, Y2, X2, coeff=1.2)
        masklist_3 = utils_vox.assemble_padded_obj_masklist(
            lrtlist_camR, scorelist, Z2, Y2, X2, coeff=1.8)
        # these are B x K x 1 x Z2 x Y2 x X2

        # use_center_surround = False
        use_center_surround = True

        if use_center_surround:
            # the idea of a center-surround feature is:
            # there should be stuff in the center but not in the surround
            # so, i need the density of the center
            # and the density of the surround
            # then, the score is center minus surround
            center_mask = (masklist_1).squeeze(2)
            surround_mask = (masklist_3-masklist_2).squeeze(2)
            # these are B x K x Z x Y x X

            # it could be that this scoring would work better with estimated occs,
            # since they are thicker

            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            occ = F.conv3d(occR0, weights, padding=1)
            occ = torch.clamp(occ, 0, 1)
            occ = occ.squeeze(1)

            if self.include_image_summs:
                summ_writer.summ_3D_flow('flow/occ_flow', occ.unsqueeze(1)*flow)
                summ_writer.summ_rgb('obj/det_image', det_image)
                summ_writer.summ_lrtlist('obj/det_boxlist', rgb_camRs[:,0], lrtlist_camR,
                                         scorelist, tidlist, pix_T_cams[:,0])
                
            occ_flow_mag = flow_mag * occ
            occ_flow_mag_ = occ_flow_mag.unsqueeze(1).repeat(1, K, 1, 1, 1)
            center_ = utils_basic.reduce_masked_mean(occ_flow_mag_, center_mask, dim=[2,3,4])
            surround_ = utils_basic.reduce_masked_mean(occ_flow_mag_, surround_mask, dim=[2,3,4])

            scorelist = center_ - surround_
            # scorelist is B x K, with arbitrary range
            scorelist = torch.clamp(torch.sigmoid(scorelist), min=1e-4)
            # scorelist is B x K, in the range [0,1]

        if self.include_image_summs:
            summ_writer.summ_lrtlist('obj/scored_boxlist', rgb_camRs[:,0], lrtlist_camR,
                                     scorelist, tidlist, pix_T_cams[:,0])
            lrtlist_camX = utils_geom.apply_4x4_to_lrtlist(camXs_T_camRs[:,0], lrtlist_camR)
            summ_writer.summ_lrtlist('obj/scored_boxlistX', rgb_camXs[:,0], lrtlist_camX,
                                     scorelist, tidlist, pix_T_cams[:,0])

        boxlist_e = boxlist_camR.detach().cpu().numpy()
        boxlist_g = boxlist_camRs[:,0].detach().cpu().numpy()
        scorelist_e = scorelist.detach().cpu().numpy()
        scorelist_g = scorelist_s[:,0].detach().cpu().numpy()

        assert(B==1)
        boxlist_e, boxlist_g, scorelist_e, _ = utils_eval.drop_invalid_boxes(
            boxlist_e, boxlist_g, scorelist_e, scorelist_g)

        ious = np.linspace(0.1, 0.9, 9)
        maps = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
        return maps

