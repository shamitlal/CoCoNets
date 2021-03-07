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

from utils_basic import *
import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_OBJ(Model):
    def go(self):
        self.start_time = time.time()
        self.declare_model()

        if hyp.lr > 0:
            self.optimizer = torch.optim.Adam(self.model.var_params, lr=hyp.lr)
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)

        print("------ Done loading weights ------")

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

        print('initializing obj/bkg vars')
        set_ind = hyp.set_names.index('train')
        set_name = set_names[set_ind]
        set_num = set_nums[set_ind]
        set_input = set_inputs[set_ind]
        set_writer = set_writers[set_ind]
        set_loader = set_loaders[set_ind]
        feed = next(set_loader)
        feed_cuda = {}
        for k in feed:
            try:
                feed_cuda[k] = feed[k].cuda(non_blocking=True)
            except:
                # some things are not tensors (e.g., filename)
                feed_cuda[k] = feed[k]
        feed_cuda['writer'] = set_writer
        feed_cuda['global_step'] = 0
        feed_cuda['set_num'] = set_num
        feed_cuda['set_name'] = set_name
        
        # flows_e = []
        # unps_e = []
        # for s in list(range(hyp.S)):
        #     flow_fn = '../discovery/flow_e_%02d_v3.npy' % (34+s*3+1)
        #     unp_fn = '../discovery/vis_a_%02d_v3.npy' % (34+s*3+1)
        #     print('loading %s' % flow_fn)
        #     flow_e = np.load(flow_fn)
        #     flows_e.append(flow_e)
        #     unp_e = np.load(unp_fn)
        #     unps_e.append(unp_e)
        # flows_e = np.stack(flows_e)
        # unps_e = np.stack(unps_e)
        # print(flows_e.shape)
        # print(unps_e.shape)
        # # flows_e is S x H x W x D x 3
        # flows_e = np.transpose(flows_e, axes=[0, 4, 3, 1, 2])
        # # unps_e = np.transpose(unps_e, axes=[0, 3, 1, 2])
        # unps_e = np.transpose(unps_e, axes=[0, 3, 2, 1])
        # # flows_e is S x 3 x D x H x W
        # # flows_e = np.unsqueeze(flows_e, axis=0)
        # flows_e = np.expand_dims(flows_e, axis=0)
        # unps_e = np.expand_dims(unps_e, axis=0)
        # feed_cuda['flows_e'] = torch.FloatTensor(flows_e).to(self.device)
        # feed_cuda['unps_e'] = torch.FloatTensor(unps_e).to(self.device)

        self.model.eval()
        with torch.no_grad():
            loss, results = self.model(feed_cuda)
        bkg = results['bkg'].detach().cpu().numpy()
        obj = results['obj'].detach().cpu().numpy()
        obj_alist = results['obj_alist'].detach().cpu().numpy()
        obj_tlist = results['obj_tlist'].detach().cpu().numpy()
        obj_l = results['obj_l'].detach().cpu().numpy()

        if not hyp.optim_init:
            self.model.obj.data = torch.FloatTensor(obj).to(self.device)
            self.model.obj_alist.data = torch.FloatTensor(obj_alist).to(self.device)
            self.model.obj_tlist.data = torch.FloatTensor(obj_tlist/self.model.t_speed).to(self.device)
            self.model.obj_l.data = torch.FloatTensor(obj_l).to(self.device)
            print('assigned obj data into vars!')
            self.model.bkg.data = torch.FloatTensor(bkg).to(self.device)
            print('assigned bkg data into vars!')
        else:
            print('using data from optim_init!')
        print('obj_tlist', self.model.obj_tlist.data.cpu().numpy())
        
        # for step in list(range(self.start_iter+1, hyp.max_iters+1)):
        for step in list(range(1, hyp.max_iters+1)):
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
                self.model.obj_tlist.data[:,0] = torch.FloatTensor(obj_tlist[:,0]/self.model.t_speed).to(self.device)
                self.model.obj_alist.data[:,0] = torch.FloatTensor(obj_alist[:,0]).to(self.device)
                self.model.obj_l.data = torch.FloatTensor(obj_l).to(self.device)
                # print(self.model.obj_tlist.data.cpu().numpy())
                
                # print('optim_dict tlist:', self.model.optim_dict['obj_tlist']) 
                
                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                if log_this or set_do_backprop:
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
                    feed_cuda['set_name'] = set_name

                    flows_e = []
                    unps_e = []
                    for s in list(range(hyp.S)):
                        flow_fn = '../discovery/flow_e_%02d_v3.npy' % (34+s*3+1)
                        unp_fn = '../discovery/vis_a_%02d_v3.npy' % (34+s*3+1)
                        flow_e = np.load(flow_fn)
                        flows_e.append(flow_e)
                        unp_e = np.load(unp_fn)
                        unps_e.append(unp_e)
                    flows_e = np.stack(flows_e)
                    unps_e = np.stack(unps_e)
                    # flows_e is S x H x W x D x 3
                    flows_e = np.transpose(flows_e, axes=[0, 4, 3, 1, 2])
                    unps_e = np.transpose(unps_e, axes=[0, 3, 2, 1])
                    # flows_e is S x 3 x D x H x W
                    # flows_e = np.unsqueeze(flows_e, axis=0)
                    flows_e = np.expand_dims(flows_e, axis=0)
                    unps_e = np.expand_dims(unps_e, axis=0)
                    feed_cuda['flows_e'] = torch.FloatTensor(flows_e).to(self.device)
                    feed_cuda['unps_e'] = torch.FloatTensor(unps_e).to(self.device)

                    iter_start_time = time.time()
                    if set_do_backprop:
                        self.model.train()
                        loss, results = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results = self.model(feed_cuda)
                    loss_vis = loss.cpu().item()

                    if set_do_backprop and (hyp.lr > 0):
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
                        loss_vis,
                        set_name,
                    ))
            if np.mod(step, hyp.snap_freq) == 0 and (hyp.lr > 0):
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)
                saverloader.save_optim(self.model, self.checkpoint_dir, step)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

    def declare_model(self):
        print("------ DECLARING SELF.MODEL ------")
        self.model = CarlaObjNet().to(self.device)
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_emb2D:
            self.model.embnet2D.eval()
            self.set_requires_grad(self.model.embnet2D, False)

class CarlaObjNet(nn.Module):
    def __init__(self):
        super(CarlaObjNet, self).__init__()
        # self.featnet = FeatNet()
        # self.occnet = OccNet()
        # self.flownet = FlowNet()
        # self.viewnet = ViewNet()

        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        # if hyp.do_render:
        #     self.rendernet = RenderNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
        
        self.device = torch.device("cuda")
        self.t_speed = 1.0
        
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)

        self.optim_dict = {}
        self.optim_dict['bkg'] = torch.zeros((1, hyp.feat_dim, Z2, Y2, X2), device=torch.device('cuda'))
        self.optim_dict['obj'] = torch.zeros((1, hyp.feat_dim, Z4, Y4, X4), device=torch.device('cuda'))
        self.optim_dict['obj_alist'] = torch.zeros((1, hyp.S, 3), device=torch.device('cuda')) # rotation
        self.optim_dict['obj_tlist'] = torch.zeros((1, hyp.S, 3), device=torch.device('cuda')) # translation
        self.optim_dict['obj_l'] = torch.zeros((1, 3), device=torch.device('cuda')) # length

        
        # self.bkg = torch.zeros((1, hyp.feat_dim, Z2, Y2, X2), device=torch.device('cuda'))
        # self.obj = torch.zeros((1, hyp.feat_dim, Z4, Y4, X4), device=torch.device('cuda'))
        # self.obj_alist = torch.zeros((1, hyp.S, 3), device=torch.device('cuda')) # rotation
        # self.obj_tlist = torch.zeros((1, hyp.S, 3), device=torch.device('cuda')) # translation
        # self.obj_l = torch.zeros((1, 3), device=torch.device('cuda')) # length
        
        self.optim_dict['obj'] = torch.autograd.Variable(self.optim_dict['obj'], requires_grad=True)
        self.optim_dict['bkg'] = torch.autograd.Variable(self.optim_dict['bkg'], requires_grad=True)
        self.optim_dict['obj_alist'] = torch.autograd.Variable(self.optim_dict['obj_alist'], requires_grad=True)
        self.optim_dict['obj_tlist'] = torch.autograd.Variable(self.optim_dict['obj_tlist'], requires_grad=True)
        self.optim_dict['obj_l'] = torch.autograd.Variable(self.optim_dict['obj_l'], requires_grad=True)

        # self.obj = torch.autograd.Variable(self.obj, requires_grad=True)
        # self.bkg = torch.autograd.Variable(self.bkg, requires_grad=True)
        # self.obj_alist = torch.autograd.Variable(self.obj_alist, requires_grad=True)
        # self.obj_tlist = torch.autograd.Variable(self.obj_tlist, requires_grad=True)
        # self.obj_l = torch.autograd.Variable(self.obj_l, requires_grad=True)

        
        self.obj = self.optim_dict['obj']
        self.bkg = self.optim_dict['bkg']
        self.obj_alist = self.optim_dict['obj_alist']
        self.obj_tlist = self.optim_dict['obj_tlist']
        self.obj_l = self.optim_dict['obj_l']

        # self.optimizer.add_param_group({'optim': [self.bkg})
        # self.optimizer.add_param_group({'optim': self.bkg})
        # self.optimizer.add_param_group({'optim': self.bkg})
        # self.optimizer.add_param_group({'optim': self.bkg})
        
        self.var_params = [
            self.obj,
            self.bkg,
            self.obj_alist,
            self.obj_tlist,
            self.obj_l,
        ]

        # self.state_dict().update(self.obj)
        # self.state_dict().update(self.bkg)
        # self.state_dict().update(self.obj_alist)
        # self.state_dict().update(self.obj_tlist)
        # self.state_dict().update(self.obj_l)
                        

    def forward(self, feed):
        results = dict()

        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8)
        # writer = feed['writer']
        global_step = feed['global_step']
        npz_filename = feed['filename'][0]

        track_outputs_fn = '01_m128x32x128_picked_ns_snap57_output_dict.npy'
        track_outputs = np.load(track_outputs_fn, allow_pickle=True).item()
        # print('loaded %s' % track_outputs_fn)
        track_outputs_here = track_outputs[npz_filename]
        
        
        total_loss = torch.tensor(0.0).cuda()
        

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z0, Y0, X0 = int(Z*2), int(Y*2), int(X*2)
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        Z8, Y8, X8 = int(Z/8), int(Y/8), int(X/8)
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

            # print('boxlist_camRs[:,0,0]:', end=' ')
            # print(boxlist_camRs[:,0,0])

        origin_T_camRs = feed["origin_T_camRs"]
        origin_T_camRs_ = __p(origin_T_camRs)
        origin_T_camXs = feed["origin_T_camXs"]
        origin_T_camXs_ = __p(origin_T_camXs)

        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camX0_T_camXs_ = __p(camX0_T_camXs)
        camRs_T_camXs_ = torch.matmul(utils_geom.safe_inverse(origin_T_camRs_), origin_T_camXs_)
        camXs_T_camRs_ = utils_geom.safe_inverse(camRs_T_camXs_)
        camRs_T_camXs = __u(camRs_T_camXs_)
        camXs_T_camRs = __u(camXs_T_camRs_)

        xyz_veloXs = feed["xyz_veloXs"]
        xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        occX0s = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z, Y, X))
        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))
        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))
        unpX0s = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs)
        unpX0s_half = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs, unpXs_half)

        ## projected depth, and inbound mask
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
        dense_xyz_camX0s_ = utils_geom.apply_4x4(__p(camX0_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camX0s_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)
        
        #####################
        ## visualize what we got
        #####################
        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1))
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        # summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
        summ_writer.summ_occs('3D_inputs/occX0s', torch.unbind(occX0s, dim=1))
        # summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpX0s', torch.unbind(unpX0s, dim=1), torch.unbind(occX0s, dim=1))

        if (not hyp.flow_do_synth_rt) or feed['set_name']=='val':
            lrtlist_camRs = __u(utils_geom.convert_boxlist_to_lrtlist(boxlist_camRs_)).reshape(B, S, N, 19)
            lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(lrtlist_camRs)))
            # stabilize boxes for ego/cam motion
            lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camX0_T_camXs), __p(lrtlist_camXs)))
            # these are is B x S x N x 19

            summ_writer.summ_lrtlist('2D_inputs/lrtlist_camR0', rgb_camRs[:,0], lrtlist_camRs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('2D_inputs/lrtlist_camR1', rgb_camRs[:,1], lrtlist_camRs[:,1],
                                     scorelist_s[:,1], tidlist_s[:,1], pix_T_cams[:,1])
            summ_writer.summ_lrtlist('2D_inputs/lrtlist_camX0', rgb_camXs[:,0], lrtlist_camXs[:,0],
                                     scorelist_s[:,0], tidlist_s[:,0], pix_T_cams[:,0])
            summ_writer.summ_lrtlist('2D_inputs/lrtlist_camX1', rgb_camXs[:,1], lrtlist_camXs[:,1],
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
                lrtlist_camRs[:,0], scorelist_s[:,0], Z2, Y2, X2, coeff=1.2)
            masklist_memX = utils_vox.assemble_padded_obj_masklist(
                lrtlist_camXs[:,0], scorelist_s[:,0], Z2, Y2, X2, coeff=1.2)
            # masklist_memX is B x N x 1 x Z x Y x X
            summ_writer.summ_occ('obj/masklist_memR', torch.sum(masklist_memR, dim=1))
            summ_writer.summ_occ('obj/masklist_memX', torch.sum(masklist_memX, dim=1))

        # lrtlist_camXs is B x S x N x 19
        lenlist, rtlist = utils_geom.split_lrtlist(lrtlist_camXs[:,:,0])
        # these are B x S x 3 and B x S x 4 x 4
        rlist_, tlist_ = utils_geom.split_rt(__p(rtlist))
        rlist = __u(rlist_)
        tlist = __u(tlist_)
        # tlist is B x S x 3
        rx_, ry_, rz_ = utils_geom.rotm2eul(rlist_)
        rx, ry, rz = __u(rx_), __u(ry_), __u(rz_)
        angles = torch.stack([rx[:,0], ry[:,0], rz[:,0]], dim=1)
        # angles is B x 3

        tlist_g = tlist

        # self.obj_tlist[:,0] = tlist[:,0]

        if global_step==0 and hyp.do_feat:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            featX0s_input = torch.cat([occX0s, occX0s*unpX0s], dim=2)
            featX0s_input_ = __p(featX0s_input)
            # featX0s_, validX0s_, feat_loss = self.featnet(featX0s_input_, summ_writer, comp_mask=__p(occX0s))
            featX0s_, validX0s_, feat_loss = self.featnet(featX0s_input_, summ_writer, comp_mask=None)
            total_loss += feat_loss
            featX0s = __u(featX0s_)
            validX0s = __u(validX0s_)
            summ_writer.summ_feats('3D_feats/featX0s_input', torch.unbind(featX0s_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), pca=True)
            cropX0_obj0 = utils_vox.crop_zoom_from_mem(featX0s[:,0], lrtlist_camXs[:,0,0], Z4, Y4, X4)
            # featX0_bkg0 = featX0s[:,0]*(1.0-masklist_memX[:,0]) + masklist_memX[:,0]*torch.mean(featX0s[:,0], dim=1, keepdim=True)
            featX0_bkg0 = featX0s[:,0]*(1.0-masklist_memX[:,0]) + masklist_memX[:,0]*featX0s[:,-1]

            summ_writer.summ_feat('crops/cropX0_obj0', cropX0_obj0, pca=True)
            summ_writer.summ_feat('3D_feats/featX0_bkg0', featX0_bkg0, pca=True)

            results['obj'] = cropX0_obj0
            results['bkg'] = featX0_bkg0
            # results['obj_tlist'] = tlist
            # # # lazy init:
            # results['obj_tlist'] = (tlist[:,0].unsqueeze(1).repeat(1, hyp.S, 1) + tlist)*0.5
            results['obj_tlist'] = tlist[:,0].unsqueeze(1).repeat(1, hyp.S, 1)
            # rough init:
            # results['obj_tlist'] = tlist + 0.5
            results['obj_alist'] = angles.unsqueeze(1).repeat(1, hyp.S, 1)
            results['obj_l'] = lenlist[:,0] # first len

            use_flow_init = False
            use_track_init = True
            if use_flow_init:
                # note the fed flow is in R coords
                flows_e = feed["flows_e"]
                flows_e_ = __p(flows_e)
                flows_e_ = F.interpolate(flows_e_*2.0, scale_factor=2)
                flows_e = __u(flows_e_)

                unps_e = feed["unps_e"]
                unps_e_ = __p(unps_e)
                unps_e_ = F.interpolate(unps_e_, scale_factor=2)
                unps_e = __u(unps_e_)
                
                obj_lrtR = lrtlist_camRs[:,0,0]
                obj_lrtlist_camRs = torch.zeros([B, S, 1, 19], dtype=torch.float32, device=torch.device('cuda'))
                for s in list(range(hyp.S)):
                    obj_lrtlist_camRs[:,s,0] = obj_lrtR
                    summ_writer.summ_3D_flow('flow/flow%d_e' % s, flows_e[:,s], clip=3.0)
                    summ_writer.summ_3D_flow('flow/occ_flow%d_e' % s, flows_e[:,s]*occRs[:,s])
                    # summ_writer.summ_oned('flow/vis_%d' % s, torch.mean(unps_e[:,s], dim=4))
                    summ_writer.summ_oned('flow/vis_%d' % s, torch.mean(unps_e[:,s], dim=1, keepdim=True))
                    
                    print('s = %d; cropping...' % s)
                    flow_crop = utils_vox.crop_zoom_from_mem(flows_e[:,s], obj_lrtR, Z4, Y4, X4)
                    occR__ = occRs[:,s].repeat(1, 3, 1, 1, 1)
                    occ_crop = utils_vox.crop_zoom_from_mem(occR__, obj_lrtR, Z4, Y4, X4)
                    mean_t = torch.mean(flow_crop, dim=[2, 3, 4])
                    # mean_t = reduce_masked_mean(flow_crop, occ_crop, dim=[2, 3, 4])
                    print('mean_t:', end=' ')
                    print(mean_t)

                    masks = utils_vox.assemble_padded_obj_masklist(
                        obj_lrtR.unsqueeze(1), torch.ones(B, 1, device=torch.device('cuda')),
                        Z2, Y2, X2, coeff=1.0)
                    summ_writer.summ_oned('flow/mask_%d' % s, torch.mean(masks, dim=(1,4)))

                    # obj_lrt is B x 19
                    l, rt = utils_geom.split_lrt(obj_lrtR)
                    # these are B x 3 and B x 4 x 4
                    r, t = utils_geom.split_rt(rt)
                    rt = utils_geom.merge_rt(r, t+mean_t)
                    obj_lrtR = utils_geom.merge_lrt(l, rt)
                    print(obj_lrtR)

                print(obj_lrtlist_camRs.shape)
                obj_lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(obj_lrtlist_camRs)))
                print(obj_lrtlist_camXs.shape)
                obj_lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(camX0_T_camXs), __p(obj_lrtlist_camXs)))
                print(obj_lrtlist_camX0s.shape)
                maskX0s = utils_vox.assemble_padded_obj_masklist(
                    obj_lrtlist_camX0s[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                    Z2, Y2, X2, coeff=1.0)
                maskXs = utils_vox.assemble_padded_obj_masklist(
                    obj_lrtlist_camXs[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                    Z2, Y2, X2, coeff=1.0)
                maskRs = utils_vox.assemble_padded_obj_masklist(
                    obj_lrtlist_camRs[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                    Z2, Y2, X2, coeff=1.0)
                summ_writer.summ_oned('flow/maskX0s', torch.mean(maskX0s, dim=(1,4)))
                summ_writer.summ_oned('flow/maskXs', torch.mean(maskXs, dim=(1,4)))
                summ_writer.summ_oned('flow/maskRs', torch.mean(maskRs, dim=(1,4)))

                lenlist, rtlist = utils_geom.split_lrtlist(obj_lrtlist_camXs[:,:,0])
                # these are B x S x 3 and B x S x 4 x 4
                rlist_, tlist_ = utils_geom.split_rt(__p(rtlist))
                rlist = __u(rlist_)
                tlist = __u(tlist_)
                # for flow-based init, we do this:
                # results['obj_tlist'] = tlist
            elif use_track_init:

                obj_lrtlist_camRs = track_outputs_here['lrtlist_camRs_e']
                # this is B x S x 1 x 19
                flow_memRs = track_outputs_here['flow_memRs']
                # this is B x S-1 x 3 x Z2 x Y2 x X2

                flow_memRs_ = __p(flow_memRs)
                flow_memRs_ = F.interpolate(flow_memRs_*2.0, scale_factor=2)
                flow_memRs = __u(flow_memRs_)
                
                for s in list(range(S-1)):
                    summ_writer.summ_3D_flow('flow/flow%d_e' % s, flow_memRs[:,s], clip=3.0)
                    summ_writer.summ_3D_flow('flow/occ_flow%d_e' % s, flow_memRs[:,s]*occRs[:,s])
                
                

                obj_lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camXs_T_camRs), __p(obj_lrtlist_camRs)))
                print(obj_lrtlist_camXs.shape)

                maskXs = utils_vox.assemble_padded_obj_masklist(
                    obj_lrtlist_camXs[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                    Z2, Y2, X2, coeff=1.0)
                maskRs = utils_vox.assemble_padded_obj_masklist(
                    obj_lrtlist_camRs[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                    Z2, Y2, X2, coeff=1.0)
                summ_writer.summ_oned('track/maskXs', torch.mean(maskXs, dim=(1,4)))
                summ_writer.summ_oned('track/maskRs', torch.mean(maskRs, dim=(1,4)))
                
                
                # print('got lrtlist_camR0s_e:', lrtlist_camR0s_e)
                # print(lrtlist_camR0s_e.shape)

                lenlist, rtlist = utils_geom.split_lrtlist(obj_lrtlist_camXs[:,:,0])
                # these are B x S x 3 and B x S x 4 x 4
                rlist_, tlist_ = utils_geom.split_rt(__p(rtlist))
                rlist = __u(rlist_)
                tlist = __u(tlist_)

                # overwrite the basic init
                results['obj_tlist'] = tlist
                

        if global_step > 0:
            # tlist_diff = (tlist - self.obj_tlist*self.t_speed)
            # # this is B x S x 3
            # tlist_l2 = torch.mean(utils_basic.l2_on_axis(tlist_diff, 2))
            # utils_misc.add_loss('optim/tlist_l2', 0, tlist_l2, 0, summ_writer)
            
            occXs_double = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z0, Y0, X0))
            occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
            occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))

            
            
            summ_writer.summ_feat('optim/obj', self.obj, pca=True)
            summ_writer.summ_feat('optim/bkg', self.bkg, pca=True)
            # self.obj_alist is B x S x 3
            angles_ = __p(self.obj_alist)
            r_ = utils_geom.eul2rotm(angles_[:,0], angles_[:,1], angles_[:,2])
            # r_ is B*S x 3 x 3
            t_ = __p(self.obj_tlist*self.t_speed)
            # t_ is B*S x 3
            rtlist = __u(utils_geom.merge_rt(r_, t_))
            lenlist = self.obj_l.unsqueeze(1).repeat(1, hyp.S, 1)
            optim_lrtlist = utils_geom.merge_lrtlist(lenlist, rtlist)
            # this is B x S x 19
            optim_masklist = utils_vox.assemble_padded_obj_masklist(
                optim_lrtlist, torch.ones(B, S, device=torch.device('cuda')),
                Z2, Y2, X2, coeff=1.0)
            # optim_masklist is B x S x 1 x Z x Y x X
            summ_writer.summ_oned('optim/masks_ax4', torch.mean(optim_masklist, dim=(1,4)))
            summ_writer.summ_oned('optim/masks_ax3', torch.mean(optim_masklist, dim=(1,3)))

            # summ_writer.summ_occs('occs/occXs_double', torch.unbind(occXs_double, dim=1))
            # occXs_crop = __u(utils_vox.crop_zoom_from_mem(__p(occXs_double), __p(optim_lrtlist), Z2, Y2, X2, additive_pad=5.0))
            # summ_writer.summ_occs('occs/occXs_crop', torch.unbind(occXs_crop, dim=1))

            vis_e = []
            for s in list(range(hyp.S)):
                vis_e.append(summ_writer.summ_lrtlist('optim/lrtlist_camX%d' % s, rgb_camXs[:,s], optim_lrtlist.detach()[:,s:s+1],
                                                      torch.ones(B, 1, device=torch.device('cuda')).float(),
                                                      torch.ones(B, 1, device=torch.device('cuda')).long(),
                                                      pix_T_cams[:,s], only_return=True))
            summ_writer.summ_rgbs('optim/lrtlist_camXs', vis_e)
                
            # bkg_fullres = F.interpolate(self.bkg, scale_factor=2.0)

            # camXs_T_zoom_ = utils_vox.get_ref_T_zoom(__p(optim_lrtlist), Z4, Y8, X8)
            # camXs_T_zoom = __u(camXs_T_zoom_)
            
            obj_halfres = F.interpolate(self.obj, scale_factor=0.5)
            camXs_T_zoom_ = utils_vox.get_ref_T_zoom(__p(optim_lrtlist), Z8, Y8, X8)
            camXs_T_zoom = __u(camXs_T_zoom_)
            assembled_featXs, assembled_validXs, bkg_featXs, obj_featXs = utils_vox.assemble(
                self.bkg,
                obj_halfres,
                origin_T_camXs,
                camXs_T_zoom)
            # assembled_featXs is B x C x S x Z2 x Y2 x X2
            summ_writer.summ_feats('optim/assembled_featXs', torch.unbind(assembled_featXs, dim=1))

            # bkg_doubleres = F.interpolate(self.bkg, scale_factor=2.0)
            # camXs_T_zoom_ = utils_vox.get_ref_T_zoom(__p(optim_lrtlist), Z4, Y4, X4)
            # camXs_T_zoom = __u(camXs_T_zoom_)
            # assembled_featXs_double, assembled_validXs_double, _, _ = utils_vox.assemble(
            #     bkg_doubleres,
            #     self.obj,
            #     origin_T_camXs,
            #     camXs_T_zoom)
            # # assembled_featXs is B x C x S x Z0 x Y0 x X0
            # summ_writer.summ_feats('optim/assembled_featXs_double', torch.unbind(assembled_featXs_double, dim=1))
            # assembled_featXs_crop = __u(utils_vox.crop_zoom_from_mem(__p(assembled_featXs_double), __p(optim_lrtlist), Z2, Y2, X2, additive_pad=5.0))
            # assembled_validXs_crop = __u(utils_vox.crop_zoom_from_mem(__p(assembled_validXs_double), __p(optim_lrtlist), Z2, Y2, X2, additive_pad=5.0))
            # summ_writer.summ_feats('optim/assembled_featXs_crop', torch.unbind(assembled_featXs_crop, dim=1))
            
            if hyp.do_view:
                assert(hyp.do_feat)
                # we warped the features into the canonical view
                # now we resample to the target view and decode

                PH, PW = hyp.PH, hyp.PW
                sy = float(PH)/float(hyp.H)
                sx = float(PW)/float(hyp.W)
                assert(sx==0.5) # else we need a fancier downsampler
                assert(sy==0.5)
                projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

                feat_projXs = __u(utils_vox.apply_pixX_T_memR_to_voxR(
                    __p(projpix_T_cams), utils_geom.eye_4x4(B*S), __p(assembled_featXs),
                    hyp.view_depth, PH, PW))
                summ_writer.summ_feats('optim/feat_projXs', torch.unbind(torch.mean(feat_projXs, dim=3), dim=1))
                rgb_Xs_ = downsample(__p(rgb_camXs), 2)
                valid_Xs_ = downsample(__p(valid_camXs), 2)

                # decode the perspective volume into an image
                view_loss, rgb_Xs_e_, emb2D_Xs_e_ = self.viewnet(
                    __p(feat_projXs),
                    rgb_Xs_,
                    valid_Xs_,
                    # torch.ones_like(valid_Xs_),
                    summ_writer)
                total_loss += view_loss
                summ_writer.summ_rgbs('optim/rgb_Xs_e', torch.unbind(__u(rgb_Xs_e_), dim=1))
                summ_writer.summ_feats('optim/emb2D_Xs_e', torch.unbind(__u(emb2D_Xs_e_), dim=1))

            if hyp.do_occ:
                occXs_sup, freeXs_sup, _, _ = utils_vox.prep_occs_supervision(
                    utils_geom.eye_4x4s(B, S),
                    xyz_camXs,
                    Z2, Y2, X2, 
                    agg=False)
                
                summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_sup, dim=1))
                summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs_sup, dim=1))

                occ_loss, occXs_pred_ = self.occnet(__p(assembled_featXs),
                                                    __p(occXs_sup),
                                                    __p(freeXs_sup),
                                                    __p(assembled_validXs),
                                                    summ_writer)
                occXs_pred = __u(occXs_pred_)
                summ_writer.summ_occs('optim/occXs', torch.unbind(occXs_pred, dim=1))
                total_loss += occ_loss





                # occXs_sup_double, freeXs_sup_double, _, _ = utils_vox.prep_occs_supervision(
                #     utils_geom.eye_4x4s(B, S),
                #     xyz_camXs,
                #     Z0, Y0, X0, 
                #     agg=False)
                # occXs_sup_crop = __u(utils_vox.crop_zoom_from_mem(__p(occXs_sup_double), __p(optim_lrtlist), Z2, Y2, X2, additive_pad=5.0))
                # freeXs_sup_crop = __u(utils_vox.crop_zoom_from_mem(__p(freeXs_sup_double), __p(optim_lrtlist), Z2, Y2, X2, additive_pad=5.0))
                # summ_writer.summ_occs('occ_sup/occXs_sup_crop', torch.unbind(occXs_sup_crop, dim=1))
                # summ_writer.summ_occs('occ_sup/freeXs_sup_crop', torch.unbind(freeXs_sup_crop, dim=1))

                # occ_loss, occXs_pred_crop_ = self.occnet(__p(assembled_featXs_crop),
                #                                          __p(occXs_sup_crop),
                #                                          __p(freeXs_sup_crop),
                #                                          __p(assembled_validXs_crop),
                #                                          summ_writer,
                #                                          suffix='_crop')
                # occXs_pred_crop = __u(occXs_pred_crop_)
                # summ_writer.summ_occs('optim/occXs_crop', torch.unbind(occXs_pred_crop, dim=1))
                # total_loss += occ_loss

                


            if hyp.do_emb2D:
                assert(hyp.do_view)
                # create an embedding image, representing the bottom-up 2D feature tensor

                emb_loss_2D, emb2D_Xs_g_ = self.embnet2D(
                    __p(rgb_camXs),
                    emb2D_Xs_e_,
                    __p(valid_camXs),
                    summ_writer)
                total_loss += emb_loss_2D
                summ_writer.summ_feats('emb2D/emb2D_Xs_g', torch.unbind(__u(emb2D_Xs_g_), dim=1))

            if hyp.do_feat:
                featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
                featXs_input_ = __p(featXs_input)
                featXs_, validXs_, feat_loss = self.featnet(featXs_input_, summ_writer, comp_mask=None)
                total_loss += feat_loss
                featXs = __u(featXs_)
                validXs = __u(validXs_)

                summ_writer.summ_feats('3D_feats/featXs', torch.unbind(featXs, dim=1), pca=True)
                
                freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
                freeXs = __u(freeXs_)
                visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)
                visXs = visXs * validXs

            if hyp.do_emb3D:
                emb_loss_3D = self.embnet3D(
                    __p(assembled_featXs),
                    __p(featXs),
                    __p(assembled_validXs),
                    __p(visXs),
                    summ_writer)
                total_loss += emb_loss_3D
            
            

            ioulist = utils_geom.get_iou_from_corresponded_lrtlists(optim_lrtlist, lrtlist_camXs[:,:,0])
            # print('got ioulist:', ioulist)
            # ioulist is B x S
            ioulist_ = torch.mean(ioulist, dim=0)
            for s in list(range(hyp.S)):
                summ_writer.summ_scalar('optim/iou_%d' % s, ioulist_[s].cpu().item())
            summ_writer.summ_scalar('optim/mean_iou', torch.mean(ioulist).cpu().item())
            
            
            
            # optim_masklist is B x S x 1 x Z x Y x X

            # optim_lrtlist is B x S x 19
            optim_lrtlist_Xs = optim_lrtlist.unsqueeze(2)
            # optim_lrtlist_Xs is B x S x 1 x 19
            optim_lrtlist_Rs = __u(utils_geom.apply_4x4_to_lrtlist(__p(camRs_T_camXs), __p(optim_lrtlist_Xs)))
            # optim_lrtlist_Rs is B x S x 1 x 19

            
            lenlist, rtlist = utils_geom.split_lrtlist(optim_lrtlist_Rs[:,:,0])
            # these are B x S x 3 and B x S x 4 x 4
            rlist_, tlist_ = utils_geom.split_rt(__p(rtlist))
            rlist = __u(rlist_)
            tlist = __u(tlist_)
            # tlist is B x S x 3


            (obj_lrtlist_camRs,
             obj_scorelist_s,
            ) = utils_misc.collect_object_info(lrtlist_camRs,
                                               tidlist_s,
                                               scorelist_s,
                                               pix_T_cams, 
                                               K, mod='R',
                                               do_vis=False,
                                               summ_writer=summ_writer)
            t_l2_total = 0.0

            maskRs = utils_vox.assemble_padded_obj_masklist(
                optim_lrtlist_Rs[:,:,0], torch.ones(B, S, device=torch.device('cuda')),
                Z, Y, X, coeff=1.0)

            # # note the fed flow is in R coords
            # flowRs = feed["flows_e"]
            # flowRs_ = __p(flowRs)
            # flowRs_ = F.interpolate(flowRs_*2.0, scale_factor=2)
            # flowRs = __u(flowRs_)

            # flow_memRs = feed["flows_e"]
            flow_memRs = track_outputs_here['flow_memRs']
            # this is B x S-1 x 3 x Z2 x Y2 x X2
            flow_memRs_ = __p(flow_memRs)
            flow_memRs_ = F.interpolate(flow_memRs_*2.0, scale_factor=2)
            flow_memRs = __u(flow_memRs_)
            
            
            for s in list(range(hyp.S-1)):

                # flowR = utils_misc.get_gt_flow(
                #     obj_lrtlist_camRs[:,:,s:s+2],
                #     obj_scorelist_s[:,:,s:s+2],
                #     utils_geom.eye_4x4s(B, 2),
                #     Z2, Y2, X2,
                #     K=K, 
                #     mod='R%d' % s,
                #     vis=True,
                #     summ_writer=summ_writer)
                # flow_crop = utils_vox.crop_zoom_from_mem(flowR, optim_lrtlist_Rs[:,s,0], Z4, Y4, X4)

                # maskR = maskRs[:,s].repeat(1, 3, 1, 1, 1)
                # occR = occRs[:,s]
                # flowR = flow_memRs[:,s]
                # flow_t_vec = reduce_masked_mean(flowR, maskR*occR, dim=[2, 3, 4])
                
                flow_crop = utils_vox.crop_zoom_from_mem(flow_memRs[:,s], optim_lrtlist_Rs[:,s,0], Z4, Y4, X4)
                flow_t_vec = torch.mean(flow_crop, dim=[2, 3, 4])
                
                optim_t_vec = tlist[:,s+1]-tlist[:,s]

                # print('optim_t_vec:', end=' ')
                # print(optim_t_vec)

                # print('flow_t_vec:', end=' ')
                # print(flow_t_vec)

                t_diff = optim_t_vec - flow_t_vec
                # print(optim_t_vec.shape)
                # print(flow_t_vec.shape)
                # print(t_diff.shape)
                t_l2 = torch.mean(utils_basic.l2_on_axis(t_diff, 1))
                t_l2_total = t_l2_total + t_l2
            total_loss = utils_misc.add_loss('optim/flow_l2', total_loss, t_l2_total, 0.01, summ_writer)
            

            summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results


