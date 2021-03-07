import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import os

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

class CARLA_PRECOMPUTE(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaPrecomputeNet()
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
                    print('filename = %s' % filename)
                    tokens = filename.split('/')
                    filename = tokens[-1]
                    print('new filename = %s' % filename)
                    
                    
                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results = self.model(feed_cuda)
                    loss_vis = loss.cpu().item()

                    if hyp.do_save_outputs:
                        out_fn = '%s/%s_flow_memRs.npy' % (out_dir, filename)
                        flow_memRs = results['flow_memRs'][0]
                        flow_memRs = flow_memRs.detach().cpu()
                        np.save(out_fn, flow_memRs)
                        print('saved %s' % out_fn)
                        print(flow_memRs.shape)
                    
                    if set_do_backprop and hyp.lr > 0:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
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
            

class CarlaPrecomputeNet(nn.Module):
    def __init__(self):
        super(CarlaPrecomputeNet, self).__init__()

        self.feat_net = frozen_flow_net.FrozenFeatNet(
            '/projects/katefgroup/cvpr2020_share/frozen_flow_net/feats_model.pb')
        self.flow_net = frozen_flow_net.FrozenFlowNet(
            '/projects/katefgroup/cvpr2020_share/frozen_flow_net/flow_model_no_dep.pb')
            
        self.device = torch.device("cuda")
        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=False)
        global_step = feed['global_step']
        npz_filename = feed['filename']
        print('npz_filename', npz_filename)

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
                            
        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, unpXs)
                                              
        summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))

        input_mems = torch.cat([occRs, unpRs], dim=2)
        
        flow_memRs = []
        for s in list(range(S-1)):
            # on each step, i need to warp frame1 to frame0
    
            print('computing basic flow from %d to %d' % (s, s+1))
            input_mem0 = input_mems[:,s]
            input_mem1 = input_mems[:,s+1]

            featnet_inputs = torch.stack([input_mem0, input_mem1], dim=1)
            featnet_outputs = self.feat_net.infer_pt(featnet_inputs)
            featnet_output_mem0 = featnet_outputs[:,0]
            featnet_output_mem1 = featnet_outputs[:,1]

            origin_T_cam0 = origin_T_camRs[:,s]
            origin_T_cam1 = origin_T_camRs[:,s+1]
            cam0_T_cam1 = torch.matmul(utils_geom.safe_inverse(origin_T_cam0), origin_T_cam1)
            featnet_output_mem1 = utils_vox.apply_4x4_to_vox(cam0_T_cam1, featnet_output_mem1)

            flow_mem0 = self.flow_net.infer_pt([featnet_output_mem0,
                                                featnet_output_mem1])
            flow_memRs.append(flow_mem0)
            summ_writer.summ_3D_flow('flow/flow_mem0_%02d' % s, flow_mem0, clip=0.0)

        flow_memRs = torch.stack(flow_memRs, dim=1)
        
        if hyp.do_save_outputs:
            results['flow_memRs'] = flow_memRs

        summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results


