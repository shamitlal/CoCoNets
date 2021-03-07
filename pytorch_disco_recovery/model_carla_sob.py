import itertools
import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import imageio,scipy
from sklearn.cluster import KMeans
from backend import saverloader, inputs
from spatial_correlation_sampler import SpatialCorrelationSampler
import os

from model_base import Model
from nets.linclassnet import LinClassNet
from nets.featnet2D import FeatNet2D
from nets.featnet3D import FeatNet3D
from nets.upnet3D import UpNet3D
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet

from nets.vq3dnet import Vq3dNet
from nets.occnet import OccNet
from nets.occrelnet import OccrelNet
from nets.subnet import SubNet
from nets.centernet import CenterNet
from nets.segnet import SegNet
from nets.motnet import MotNet
from nets.flownet import FlowNet


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

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 5, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data.fill_(0)
        self.conv.weight.data[0,0,0,1] = 1
        self.conv.weight.data[1,0,1,0] = 1
        self.conv.weight.data[2,0,1,1] = 1
        self.conv.weight.data[3,0,1,2] = 1
        self.conv.weight.data[4,0,2,1] = 1
    def forward(self, input_mask):
        return self.conv(input_mask).min(dim=1)
    
class Dilate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 5, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data.fill_(0)
        self.conv.weight.data[0,0,0,1] = 1
        self.conv.weight.data[1,0,1,0] = 1
        self.conv.weight.data[2,0,1,1] = 1
        self.conv.weight.data[3,0,1,2] = 1
        self.conv.weight.data[4,0,2,1] = 1
    def forward(self, input_mask):
        return self.conv(input_mask).max(dim=1)

def project_l2_ball_py(z):
    # project the vectors in z onto the l2 unit norm ball
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1, keepdims=True)), 1)

# def project_l2_ball_pt(z):
#     # project the vectors in z onto the l2 unit norm ball
#     return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)
        

# the idea of this mode is to overfit to a few examples and prove to myself that i can generate sob outputs

class CARLA_SOB(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaSobModel()
        if hyp.do_freeze_feat2D:
            self.model.featnet2D.eval()
            self.set_requires_grad(self.model.featnet2D, False)
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
        if hyp.do_freeze_up3D:
            self.model.upnet3D.eval()
            self.set_requires_grad(self.model.upnet3D, False)
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
        if hyp.do_freeze_occrel:
            self.model.occrelnet.eval()
            self.set_requires_grad(self.model.occrelnet, False)
        if hyp.do_freeze_sub:
            self.model.subnet.eval()
            self.set_requires_grad(self.model.subnet, False)
        if hyp.do_freeze_center:
            self.model.centernet.eval()
            self.set_requires_grad(self.model.centernet, False)
        if hyp.do_freeze_seg:
            self.model.segnet.eval()
            self.set_requires_grad(self.model.segnet, False)
        if hyp.do_freeze_mot:
            self.model.motnet.eval()
            self.set_requires_grad(self.model.motnet, False)
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
            
class CarlaSobModel(nn.Module):
    def __init__(self):
        super(CarlaSobModel, self).__init__()

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
            self.featnet3D_slow = FeatNet3D(in_dim=4, crop=self.crop_guess)
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
            
        if hyp.do_occrel:
            self.occrelnet = OccrelNet()
            
        if hyp.do_sub:
            self.subnet = SubNet()
            
        if hyp.do_center:
            self.centernet = CenterNet()
            
        if hyp.do_seg:
            self.num_seg_labels = 13 # note label0 is "none"
            # we will predict all 12 valid of these, plus one "air" class
            self.segnet = SegNet(self.num_seg_labels)
            
        self.diff_pool = utils_misc.SimplePool(100000, version='np')

        if hyp.do_flow:
            self.flownet = FlowNet()
        
        # self.heatmap_size = hyp.flow_heatmap_size
        # dilation = 3
        # grid_z, grid_y, grid_x = utils_basic.meshgrid3D(
        #     1, self.heatmap_size,
        #     self.heatmap_size,
        #     self.heatmap_size)
        # self.max_disp = int(dilation * (self.heatmap_size - 1) / 2)
        # self.offset_grid = torch.stack([grid_z, grid_y, grid_x], dim=1) - int(self.heatmap_size/2)
        # # this is 1 x 3 x H x H x H, with 0 in the middle
        # self.offset_grid = self.offset_grid.reshape(1, 3, -1, 1, 1, 1) * dilation
        # self.correlation_sampler = SpatialCorrelationSampler(
        #     kernel_size=1,
        #     patch_size=self.heatmap_size,
        #     stride=1,
        #     padding=0,
        #     dilation_patch=dilation,
        # ).cuda()

        
        # if hyp.do_mot:
        #     # # self.num_mot_labels = 3 # air, bkg, obj
        #     # # self.motnet = MotNet(self.num_mot_labels)
        #     # self.motnet = MotNet(1)

        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.randn([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.randn([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim*4, int(hyp.Z/4), int(hyp.Y/4), int(hyp.X/4)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim*4, int(hyp.Z/4), int(hyp.Y/4), int(hyp.X/4)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 128, int(hyp.Z/4), int(hyp.Y/4), int(hyp.X/4)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 128, int(hyp.Z/8), int(hyp.Y/8), int(hyp.X/8)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z/4), int(hyp.Y/4), int(hyp.X/4)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z/4), int(hyp.Y/4), int(hyp.X/4)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 128, int(hyp.Z/8), int(hyp.Y/8), int(hyp.X/8)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 128, int(hyp.Z/16), int(hyp.Y/16), int(hyp.X/16)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z), int(hyp.Y), int(hyp.X)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], 1, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.randn([hyp.batch_sizes['train'], 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim*4, int(hyp.Z/8), int(hyp.Y/8), int(hyp.X/8)]).float().cuda()
        # self.zi = torch.zeros([hyp.batch_sizes['train'], hyp.feat3D_dim, int(hyp.Z), int(hyp.Y), int(hyp.X)]).float().cuda()
        # self.zi = torch.autograd.Variable(self.zi, requires_grad=True)
        
        # self.zi_origin_T_camX0 = utils_geom.eye_4x4(hyp.batch_sizes['train'])
        # self.zi_origin_T_camX0 = torch.autograd.Variable(self.zi_origin_T_camX0, requires_grad=False)
        
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

        # all_ok = False
        # num_tries = 0
        # while not all_ok:
        #     scene_centroid_x = np.random.uniform(-8.0, 8.0)
        #     scene_centroid_y = np.random.uniform(-1.5, 3.0)
        #     scene_centroid_z = np.random.uniform(10.0, 26.0)
        #     scene_centroid = np.array([scene_centroid_x,
        #                                scene_centroid_y,
        #                                scene_centroid_z]).reshape([1, 3])
        #     self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        #     num_tries += 1
        #     all_ok = True
        #     self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

        #     occ_memXAA = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,self.anchor], self.Z2, self.Y2, self.X2)
        #     occ_memXAA = self.crop_feat(occ_memXAA)

        #     occ_memXAA = occ_memXAA.reshape(self.B, -1)
        #     num_inb = torch.sum(occ_memXAA, dim=1)
        #     # this is B
        #     if torch.mean(num_inb) < 300:
        #         # print('num_inb', num_inb.detach().cpu().numpy())
        #         all_ok = False
                
        #     if num_tries > 20:
        #         print('cannot find a good centroid; returning early')
        #         return False
        # # print('scene_centroid', scene_centroid)
        # self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        # self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
        
        scene_centroid_x = 0.0
        scene_centroid_y = 1.5 # 1.0 is a bit too high up
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

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

        return True # OK

    def run_train(self, feed):
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

        full_boxlist_camRs = feed["full_boxlist_camR"]
        full_scorelist_s = feed["full_scorelist"]
        full_tidlist_s = feed["full_tidlist"]

        # full_boxlist_camRs is B x S x N x 9
        N = full_scorelist_s.shape[2]
        full_origin_T_camRs = self.origin_T_camRs.unsqueeze(2).repeat(1, 1, N, 1, 1)
        full_lrtlist_camRs_ = utils_misc.parse_boxes(__p(full_boxlist_camRs), __p(full_origin_T_camRs))
        full_lrtlist_camXs_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), full_lrtlist_camRs_)
        full_lrtlist_camX0s_ = utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), full_lrtlist_camXs_)

        self.full_scorelist_s = full_scorelist_s
        self.full_tidlist_s = full_tidlist_s
        self.full_lrtlist_camRs = __u(full_lrtlist_camRs_)
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
                                ious[i] = np.squeeze(utils_geom.get_iou_from_corresponded_lrtlists(
                                    target[b:b+1].unsqueeze(1), self.full_lrtlist_camX0s[b:b+1,s1,i:i+1])[0,0])
                        if float(np.max(ious)) < 0.97:
                            # the object must have moved
                            new_scorelist_s[b,s0,n] = 1.0
        self.full_scorelist_s = new_scorelist_s * self.full_scorelist_s

        print('objects detectable across the entire seq:', torch.sum(self.full_scorelist_s).detach().cpu().numpy())
        if torch.sum(self.full_scorelist_s) == 0:
            # return early, since no objects are inbound AND moving
            return total_loss, results, True

        # self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        
        # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/rgb_memX0s', torch.unbind(self.rgb_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))

        have_feats = False
        have_medians = False

        use_feat_cache = False
        if hyp.do_use_cache and use_feat_cache:
            data_ind = feed['data_ind']
            feat_cache_fn = 'cache/%s_%06d_s%d_feat.npz' % (self.set_data_name, data_ind, self.S)
            # check if the thing exists
            if os.path.isfile(feat_cache_fn):
                print('found feat cache at %s; we will use this' % feat_cache_fn)
                cache = np.load(feat_cache_fn, allow_pickle=True)['save_dict'].item()
                # cache = cache['save_dict']
                have_feats = True

                feat_memXAI_all = torch.from_numpy(cache['feat_memXAI_all']).cuda().unbind(1)
                occ_memXAI_all = torch.from_numpy(cache['occ_memXAI_all']).cuda().unbind(1)
                occrel_memXAI_all = torch.from_numpy(cache['occrel_memXAI_all']).cuda().unbind(1)
                vis_memXAI_all = torch.from_numpy(cache['vis_memXAI_all']).cuda().unbind(1)

                # feat_memXAI_input_vis = torch.from_numpy(cache['feat_memXAI_input_vis']).unbind(1)
                # feat_memXAI_vis = torch.from_numpy(cache['feat_memXAI_vis']).unbind(1)
                self.scene_centroid = torch.from_numpy(cache['scene_centroid']).cuda()
                self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

                # make sure the shapes match what we expect
                _, _, Z2, Y2, X2 = list(feat_memXAI_all[0].shape)
                Z_crop = int((self.Z2 - Z2)/2)
                Y_crop = int((self.Y2 - Y2)/2)
                X_crop = int((self.X2 - X2)/2)
                crop = (Z_crop, Y_crop, X_crop)
                if not (crop==self.crop_guess):
                    print('crop', crop)
                    assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
            else:
                print('could not find feat cache at %s; we will write this' % feat_cache_fn)
                
        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # vis = []
        # # for s in list(range(0, self.S, 2)):
        # for s in list(range(0, self.S)):
        #     vis.append(self.summ_writer.summ_lrtlist(
        #         '', self.rgb_camXs[:,s],
        #         self.full_lrtlist_camXs[:,s],
        #         self.full_scorelist_s[:,s],
        #         self.full_tidlist_s[:,s],
        #         self.pix_T_cams[:,s],
        #         only_return=True))
        # self.summ_writer.summ_rgbs('2D_inputs/lrtlist_camXs', vis)

        # global_step = feed['global_step']
        # if global_step < 5:
        #     return total_loss, results, False
    
        if not have_feats and use_feat_cache:
            with torch.no_grad():
                vis_memXAI_all = []
                feat_memXAI_all = []
                occ_memXAI_all = []
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
                    self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                    self.occ_memXAI = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,I], self.Z, self.Y, self.X)

                    feat_memXAI_input = torch.cat([
                        self.occ_memXAI,
                        self.rgb_memXAI*self.occ_memXAI,
                    ], dim=1)
                    _, feat_memXAI, _ = self.featnet3D(feat_memXAI_input)
                    _, occ_memXAI_pred, _, _ = self.occnet(feat_memXAI)
                    occ_memXAI = F.sigmoid(occ_memXAI_pred)

                    # these boosts seem to hurt:
                    # occ_memXAI = (occ_memXAI + occ_memXAI_g).clamp(0, 1) * (1.0 - free_memXAI_g)
                    # occ_memXAI = (occ_memXAI + occ_memXAI_g).clamp(0, 1)

                    # use_occrel = False
                    # if use_occrel:
                    _, occrel_memXAI = self.occrelnet(feat_memXAI)

                    _, _, Z2, Y2, X2 = list(feat_memXAI.shape)
                    Z_crop = int((self.Z2 - Z2)/2)
                    Y_crop = int((self.Y2 - Y2)/2)
                    X_crop = int((self.X2 - X2)/2)
                    crop = (Z_crop, Y_crop, X_crop)
                    if not (crop==self.crop_guess):
                        print('crop', crop)
                    assert(crop==self.crop_guess) # otw we need to rewrite self.crop above

                    vis_memXAI_all.append(vis_memXAI)
                    feat_memXAI_all.append(feat_memXAI)
                    occ_memXAI_all.append(occ_memXAI)
                    occrel_memXAI_all.append(occrel_memXAI)

                if hyp.do_use_cache:
                    # save this, so that we have it all next time
                    save_dict = {}
                    save_dict['scene_centroid'] = self.scene_centroid.detach().cpu().numpy()
                    save_dict['vis_memXAI_all'] = torch.stack(vis_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['feat_memXAI_all'] = torch.stack(feat_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['occ_memXAI_all'] = torch.stack(occ_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['occrel_memXAI_all'] = torch.stack(occrel_memXAI_all, dim=1).detach().cpu().numpy()
                    np.savez(feat_cache_fn, save_dict=save_dict)
                    print('saved feats to %s cache, for next time' % feat_cache_fn)
                    # return early, to not apply grads
                    return total_loss, results, True

        if hyp.do_use_cache:
            data_ind = feed['data_ind']
            med_cache_fn = 'cache/%s_%06d_s%d_med.npz' % (self.set_data_name, data_ind, self.S)
            # check if the thing exists
            if os.path.isfile(med_cache_fn):
                print('found median cache at %s; we will use this' % med_cache_fn)
                cache = np.load(med_cache_fn, allow_pickle=True)['save_dict'].item()
                have_medians = True
                occ_memXAI_median = torch.from_numpy(cache['occ_memXAI_median']).cuda()
                feat_memXAI_median = torch.from_numpy(cache['feat_memXAI_median']).cuda()
            else:
                print('could not find median cache at %s; we will write this' % med_cache_fn)
        
        if not have_medians:
            feat_memXAI_all_np = (torch.stack(feat_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            feat_memXAI_median_np = np.median(feat_memXAI_all_np, axis=0)
            feat_memXAI_median = torch.from_numpy(feat_memXAI_median_np).float().reshape(1, -1, Z2, Y2, X2).cuda()

            occ_memXAI_all_np = (torch.stack(occ_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            vis_memXAI_all_np = (torch.stack(vis_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            occ_memXAI_median_np_safe = np.median(occ_memXAI_all_np, axis=0)
            occ_memXAI_median_np = utils_py.reduce_masked_median(
                occ_memXAI_all_np.transpose(1, 0), vis_memXAI_all_np.transpose(1, 0), keep_batch=True)
            occ_memXAI_median_np[np.isnan(occ_memXAI_median_np)] = occ_memXAI_median_np_safe[np.isnan(occ_memXAI_median_np)]
            occ_memXAI_median = torch.from_numpy(occ_memXAI_median_np).float().reshape(1, -1, Z2, Y2, X2).cuda()

            # occ_memXAI_diff_np = np.mean(np.abs(occ_memXAI_all_np[1:] - occ_memXAI_all_np[:-1]), axis=0)
            # occ_memXAI_diff = torch.from_numpy(occ_memXAI_diff_np).float().reshape(1, 1, Z2, Y2, X2).cuda()

            if hyp.do_use_cache:
                # save this, so that we have it all next time
                save_dict = {}
                save_dict['occ_memXAI_median'] = occ_memXAI_median.detach().cpu().numpy()
                save_dict['feat_memXAI_median'] = feat_memXAI_median.detach().cpu().numpy()
                np.savez(med_cache_fn, save_dict=save_dict)
                print('saved medians to %s cache, for next time' % med_cache_fn)
                
        self.summ_writer.summ_feat('3D_feats/feat_memXAI_median', feat_memXAI_median, pca=True)
        self.summ_writer.summ_occ('3D_feats/occ_memXAI_median', occ_memXAI_median)
        
        # now, i should be able to walk through a second time, and collect great diff signals
        if use_feat_cache:
            diff_memXAI_all = []
            for I in list(range(self.S)):
                vis_memXAI = vis_memXAI_all[I]

                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
                # vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)

                occ_memXAI = occ_memXAI_all[I]
                occrel_memXAI = occrel_memXAI_all[I]

                use_occrel = True
                if not use_occrel:
                    occrel_memXAI = torch.ones_like(occrel_memXAI)

                # diff_memXAI_all.append(torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(occ_memXAI * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(occ_memXAI.round() * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))

                # diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True)
                diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True) * occrel_memXAI
                # diff = torch.nn.functional.relu(diff - occ_memXAI_diff)

                diff = occ_memXAI.round() * vis_memXAI * diff
                diff_memXAI_all.append(diff)
            
        # if self.summ_writer.save_this:
        #     diff_memXAI_vis = []
        #     cumdiff_memXAI_vis = []
        #     for I in list(range(self.S-5)):
        #         diff_memXAI_vis.append(self.summ_writer.summ_oned('', diff_memXAI_all[I], bev=True, max_along_y=True, norm=False, only_return=True))
        #     self.summ_writer.summ_rgbs('3D_feats/diff_memXAI', diff_memXAI_vis)

        if hyp.do_sub:
            input_batch = []
            obj_batch = []
            bkg_batch = []
            label_batch = []
            valid_batch = []
            # for I in list(range(0, self.S, 4)):


            # diff_all = torch.stack(diff_memXAI_all).detach().cpu().numpy()
            # diff_mean = np.mean(diff_all)
            # diff_std = np.std(diff_all)
            
            # mean_diff = torch.mean(torch.stack(diff_memXAI_all))
            inds = []
            for I in list(range(0, self.S, 8)):

                if use_feat_cache:
                    feat_memXAI = feat_memXAI_all[I]
                    diff_memXAI = diff_memXAI_all[I]
                else:
                    self.rgb_memXII = self.vox_util.unproject_rgb_to_mem(
                        self.rgb_camXs[:,I], self.Z, self.Y, self.X, self.pix_T_cams[:,I])
                    self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                    self.occ_memXAI = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,I], self.Z, self.Y, self.X)

                    occ_memXAIs, free_memXAIs, _, _ = self.vox_util.prep_occs_supervision(
                        self.camXAs_T_camXs[:,I:I+1],
                        self.xyz_camXs[:,I:I+1],
                        self.Z2, self.Y2, self.X2,
                        agg=False)

                    occ_memXAI_g = self.crop_feat(occ_memXAIs.squeeze(1))
                    free_memXAI_g = self.crop_feat(free_memXAIs.squeeze(1))

                    vis_memXAI = (occ_memXAI_g + free_memXAI_g).clamp(0, 1)
                    
                    feat_memXAI_input = torch.cat([
                        self.occ_memXAI,
                        self.rgb_memXAI*self.occ_memXAI,
                    ], dim=1)
                    _, feat_memXAI, _ = self.featnet3D(feat_memXAI_input)

                    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                    vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
                    _, occ_memXAI_pred, _, _ = self.occnet(feat_memXAI)
                    occ_memXAI = F.sigmoid(occ_memXAI_pred)
                    _, occrel_memXAI = self.occrelnet(feat_memXAI)

                    diff_memXAI = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True) * occrel_memXAI
                    diff_memXAI = occ_memXAI.round() * vis_memXAI * diff_memXAI
                    inds.append(I)

                cat_memXAI = torch.cat([feat_memXAI, feat_memXAI_median], dim=1)

                # obj_memXAI = (diff_memXAI_all[I] > (diff_mean + diff_std)).float()
                # bkg_memXAI = (diff_memXAI_all[I] < np.max([diff_mean - diff_std, 0])).float()
                # obj_memXAI = (diff_memXAI_all[I] > 0.95).float()
                # bkg_memXAI = (diff_memXAI_all[I] < 0.05).float()
                obj_memXAI = (diff_memXAI > 0.95).float()
                # bkg_memXAI = (diff_memXAI < 0.05).float()
                bkg_memXAI = (diff_memXAI < 0.5).float()

                val_memXAI = (obj_memXAI + bkg_memXAI).clamp(0, 1)
                # take the top 50%
                
                # lab_memXAI = (diff_memXAI_all[I] > mean_diff).float()
                # val_memXAI = (lab_memXAI > 0.0)
                input_batch.append(cat_memXAI)
                obj_batch.append(obj_memXAI)
                bkg_batch.append(bkg_memXAI)
                valid_batch.append(val_memXAI)
                # label_batch.append(lab_memXAI)
                # valid_batch.append(val_memXAI)

            # self.summ_writer.summ_occs('sub/obj_label_g', obj_batch)
            # self.summ_writer.summ_occs('sub/bkg_label_g', bkg_batch)
            self.summ_writer.summ_oneds('sub/obj_label_g', obj_batch, bev=True, norm=True)
            self.summ_writer.summ_oneds('sub/bkg_label_g', bkg_batch, bev=True, norm=True)
                
            input_batch = torch.cat(input_batch, dim=0)
            obj_batch = torch.cat(obj_batch, dim=0)
            bkg_batch = torch.cat(bkg_batch, dim=0)
            valid_batch = torch.cat(valid_batch, dim=0)
            # label_batch = torch.cat(label_batch, dim=0)
            
            sub_loss, sub_memXAI = self.subnet(
                input_batch,
                obj_batch,
                bkg_batch,
                valid_batch,
                # torch.ones_like(label_batch), # valid
                self.summ_writer)
            total_loss += sub_loss

            sub_memXAIs = __u(sub_memXAI).unbind(1)
            self.summ_writer.summ_occs('sub/sub', sub_memXAIs)
            self.summ_writer.summ_oneds('sub/sub_oned', sub_memXAIs, bev=True, max_along_y=True, norm=False)

            all_acc_obj = []
            all_acc_bkg = []
            all_acc_bal = []
            mask_memXAIs = []
            for I in list(range(len(sub_memXAIs))):
                I2 = inds[I]
                mask_memXAI = self.vox_util.assemble_padded_obj_masklist(
                    self.full_lrtlist_camXAs[:,I2], self.full_scorelist_s[:,I2], self.Z2, self.Y2, self.X2, coeff=1.0)
                mask_memXAI = torch.sum(mask_memXAI, dim=1).clamp(0, 1)
                mask_memXAI = self.crop_feat(mask_memXAI)
                mask_memXAIs.append(mask_memXAI)

                obj_g = mask_memXAI.clone()
                bkg_g = 1.0 - obj_g
                obj_e = sub_memXAIs[I].round()
                bkg_e = 1.0 - obj_e
                
                # obj_match = obj_g*torch.eq(obj_e, obj_g).float()
                # bkg_match = bkg_g*torch.eq(1.0-obj_e, bkg_g).float()
                # acc_obj = utils_basic.reduce_masked_mean(obj_match, obj_g)
                # acc_bkg = utils_basic.reduce_masked_mean(bkg_match, bkg_g)
                acc_obj = utils_basic.reduce_masked_mean(obj_g*obj_e, (obj_g+obj_e).clamp(0,1))
                acc_bkg = utils_basic.reduce_masked_mean(bkg_g*bkg_e, (bkg_g+bkg_e).clamp(0,1))
                acc_bal = (acc_obj + acc_bkg)*0.5

                all_acc_obj.append(acc_obj)
                all_acc_bkg.append(acc_bkg)
                all_acc_bal.append(acc_bal)
            self.summ_writer.summ_scalar('unscaled_sub/full_acc_obj', torch.mean(torch.stack(all_acc_obj)))
            self.summ_writer.summ_scalar('unscaled_sub/full_acc_bkg', torch.mean(torch.stack(all_acc_bkg)))
            self.summ_writer.summ_scalar('unscaled_sub/full_acc_bal', torch.mean(torch.stack(all_acc_bal)))
            self.summ_writer.summ_oneds('sub/mask_oned', mask_memXAIs, bev=True, max_along_y=True, norm=False)

        # if hyp.do_feat3D:
        #     feat_memX0s_input = torch.cat([
        #         self.occ_memX0s,
        #         self.rgb_memX0s*self.occ_memX0s,
        #     ], dim=2)
        #     feat_memX0_input = utils_basic.reduce_masked_mean(
        #         feat_memX0s_input[:,1:],
        #         self.occ_memX0s[:,1:].repeat(1, 1, 4, 1, 1, 1),
        #         dim=1)
        #     feat3D_loss, feat_memX0, valid_memX0 = self.featnet3D(
        #         feat_memX0_input,
        #         self.summ_writer,
        #     )
        #     total_loss += feat3D_loss
        #     self.summ_writer.summ_feat('3D_feats/feat_memX0_input', feat_memX0_input, pca=True)
        #     self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, pca=True)
        #     # print('feat total_loss', total_loss.detach().cpu().numpy())

        # if hyp.do_occ:
        #     occ_memX0_sup, free_memX0_sup, occ_memXs, free_memXs = self.vox_util.prep_occs_supervision(
        #         self.camX0s_T_camXs,
        #         self.xyz_camXs,
        #         self.Z2, self.Y2, self.X2,
        #         agg=True)
        #     vis_memX0 = (occ_memXs[:,0] + free_memXs[:,0]).clamp(0, 1)
        #     occ_memX0_sup = self.crop_feat(occ_memX0_sup)
        #     free_memX0_sup = self.crop_feat(free_memX0_sup)
        #     vis_memX0 = self.crop_feat(vis_memX0)

        #     occ_loss, occ_memX0_pred, occ_memX0_loss, occ_memX0_match = self.occnet(
        #         feat_memX0, 
        #         occ_memX0_sup,
        #         free_memX0_sup,
        #         valid_memX0, 
        #         self.summ_writer)
        #     total_loss += occ_loss

        # if hyp.do_occrel:
        #     occrel_loss, occrel_memX0_pred = self.occrelnet(
        #         feat_memX0, 
        #         occ_memX0_match,
        #         occ_memX0_sup,
        #         free_memX0_sup,
        #         vis_memX0*valid_memX0, 
        #         self.summ_writer)
        #     total_loss += occrel_loss
        #     self.summ_writer.summ_occ('occrel/occrel_occ', F.sigmoid(occ_memX0_pred) * occrel_memX0_pred)

        
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
        
    
    def run_sub(self, feed):
        # assert(hyp.do_feat3D)
        # assert(hyp.do_occ)
        assert(self.B==1)

        results = dict()
        start_time = time.time()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # this effort failed:
        if False:
            (obj_lrtlist_camXAs, obj_tidlist_s, obj_scorelist_s,
            ) = utils_misc.collect_object_info(self.full_lrtlist_camXAs,
                                               full_tidlist_s,
                                               full_scorelist_s,
                                               hyp.N, mod='R',
                                               do_vis=True,
                                               summ_writer=self.summ_writer)
            # obj_lrtlist_camXAs is N x B x S x 19
            obj_lrtlist_camXAs_ = obj_lrtlist_camXAs.reshape(hyp.N*self.B, self.S, 19)
            obj_scorelist_s_ = obj_scorelist_s.reshape(hyp.N*self.B, self.S)
            obj_clist_camXAs_ = utils_geom.get_clist_from_lrtlist(obj_lrtlist_camXAs_)
            # this is N*B x S x 3
            print('obj_clist_camXAs_', obj_clist_camXAs_.shape)
            print('obj_scorelist_s_', obj_scorelist_s_.shape)
            obj_motion_s_ = torch.norm(obj_clist_camXAs_[:,1:]-obj_clist_camXAs_[:,:-1], dim=2)
            print('obj_motion_s_', obj_motion_s_.detach().cpu().numpy(), obj_motion_s_.shape)
            obj_motion_s_ = obj_motion_s_ * (obj_scorelist_s_[:,1:]*obj_scorelist_s_[:,:-1])
            # this is N*B x S-1
            obj_motion_s_ = torch.cat([obj_motion_s_, obj_motion_s_[:,0:1]], dim=1)
            # obj_motion_ = torch.norm(obj_clist_camXAs_[:,0]-obj_clist_camXAs_[:,-1], dim=1) * (obj_scorelist_s_[:,0]*obj_scorelist_s_[:,-1])
            # this is N*B
            # obj_motion_s = obj_motion_s_.reshape(hyp.N, self.B, self.S)
            # print('obj_motion_s', obj_motion_s.detach().cpu().numpy())
            # obj_scorelist_s = obj_scorelist_s * (obj_motion_s > 1.0).float()
            # print('obj_scorelist_s', obj_scorelist_s.detach().cpu().numpy())
            # obj_motion_s = 
            obj_scorelist_s = (obj_motion_s_ > 1.0).float().reshape(hyp.N, self.B, self.S)
            print('obj_scorelist_s', obj_scorelist_s.detach().cpu().numpy(), obj_scorelist_s.shape)

            # self.moving_lrtlist_camXA0 = obj_lrtlist_camXAs[:,:,0].permute(1, 0, 2)
            # self.moving_scorelist = (obj_scorelist_s[:,:,0] * (obj_motion > 1.0).float()).permute(1, 0)
            # print('moving_lrtlist_camXA0', self.moving_lrtlist_camXA0.shape)
            # print('moving_scorelist', self.moving_scorelist.shape)

            self.moving_lrtlist_camXAs = obj_lrtlist_camXAs.permute(1, 2, 0, 3)
            self.moving_scorelist_s = obj_scorelist_s.permute(1, 2, 0)
            self.moving_tidlist_s = obj_tidlist_s.permute(1, 2, 0)

            self.moving_lrtlist_camXs = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camXAs), __p(self.moving_lrtlist_camXAs)))
        elif False:
            # how about this as an alternative:
            # for each object, look at the pointcloud in its box in this frame and a neighbor frame;
            # if the pointcloud changed, the object probably moved (or the camera moved a lot)
            # we can count #inbounds for example

            # that failed too

            # full_lrtlist_camXs is B x S x N x 19
            # full_scorelist_s is B x S x N
            for n in list(range(self.N)):
                lrt_camXs = self.full_lrtlist_camXs[:,:,n]
                scores = self.full_scorelist_s[:,:,n]
                for s0 in list(range(self.S)):
                    if s0==0:
                        s1 = s0+1
                    else:
                        s1 = s0-1
                    # get the zoomed pointcloud at two timesteps, with the s0 box
                    xyz1_zoom = self.vox_util.Ref2Zoom(self.xyz_camXs[:,s0], lrt_camXs[:,s0], self.Z, self.Y, self.X)
                    xyz2_zoom = self.vox_util.Ref2Zoom(self.xyz_camXs[:,s1], lrt_camXs[:,s0], self.Z, self.Y, self.X)
                    inb1 = self.vox_util.get_inbounds(xyz1_zoom, self.Z, self.Y, self.X, already_mem=True)
                    inb2 = self.vox_util.get_inbounds(xyz2_zoom, self.Z, self.Y, self.X, already_mem=True)
                    # these are B x P
                    absdiff = torch.abs(torch.sum(inb2.float(), dim=1)-torch.sum(inb1.float(), dim=1))
                    for b in list(range(self.B)):
                        self.full_scorelist_s[b,s0,n] = absdiff
                        # if absdiff[b] < 10:
                        #     # print('b %d, n %d, s %d; absdiff' % (b, n, s0), absdiff[b].detach().cpu().numpy())
                        #     self.full_scorelist_s[b,s0,n] = 0.0
                        # else:
                        #     print('b %d, n %d, s %d; absdiff' % (b, n, s0), absdiff[b].detach().cpu().numpy())

        # what i want to do here is:
        # run through the seq in order,
        # get features from each frame, in the coords of the anchor frame

        if hyp.do_flow:
            # occ_memXA_sup, free_memXA_sup, occ_memXAs, free_memXAs = self.vox_util.prep_occs_supervision(
            #     self.camXAs_T_camXs,
            #     self.xyz_camXs,
            #     self.Z2, self.Y2, self.X2,
            #     agg=True)
            # vis_memXAs = (occ_memXAs + free_memXAs).clamp(0, 1)
            # vis_memXAs_ = __p(vis_memXAs)
            # vis_memXAs_ = self.crop_feat(vis_memXAs_)
            # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            # vis_memXAs_ = (F.conv3d(vis_memXAs_, weights, padding=1)).clamp(0, 1)
            # vis_memXAs_ = (F.conv3d(vis_memXAs_, weights, padding=1)).clamp(0, 1)
            # vis_memXAs = __u(vis_memXAs_)
            
            # print('computing flow...')
            flow01_all = []
            flow12_all = []
            flow23_all = []
            flow34_all = []
            flow45_all = []
            flow02_all = []
            flow03_all = []
            flow04_all = []
            flow05_all = []
            flow10_all = []
            flow01_vis = []
            flow04_vis = []
            flow05_vis = []
            flow10_vis = []
            consistent_flow05_all = []
            consistent_flow05_vis = []
            consistent_mask05_vis = []
            feat_vis = []
            occ0_vis = []
            featmag_vis = []
            flowmag_vis = []
            clip = 1.0
            for s in list(range(self.S-5)):
                # vis0 = vis_memXAs[:,s]
                # print('working on s=%d' % s)
                occ0 = occ_memXAI_all[s]
                rel0 = occrel_memXAI_all[s]
                occ1 = occ_memXAI_all[s+1]
                occ2 = occ_memXAI_all[s+2]
                occ3 = occ_memXAI_all[s+3]
                occ4 = occ_memXAI_all[s+4]
                occ5 = occ_memXAI_all[s+5]
                feat0 = feat_memXAI_all[s]
                feat1 = feat_memXAI_all[s+1]
                feat2 = feat_memXAI_all[s+2]
                feat3 = feat_memXAI_all[s+3]
                feat4 = feat_memXAI_all[s+4]
                feat5 = feat_memXAI_all[s+5]

                # print('occ0', occ0.shape)
                # print('feat0', feat0.shape)
                _, flow01 = self.flownet(feat0, feat1)
                _, flow12 = self.flownet(feat1, feat2)
                _, flow23 = self.flownet(feat2, feat3)
                _, flow34 = self.flownet(feat3, feat4)
                _, flow45 = self.flownet(feat4, feat5)

                # _, flow01 = self.flownet(feat0, feat1)
                # _, flow12 = self.flownet(feat1, feat2)
                
                
                # # _, flow02 = self.flownet(feat0, feat2)
                _, flow54 = self.flownet(feat5, feat4)
                _, flow43 = self.flownet(feat4, feat3)
                _, flow32 = self.flownet(feat3, feat2)
                _, flow21 = self.flownet(feat2, feat1)
                _, flow10 = self.flownet(feat1, feat0)
                # # _, flow20 = self.flownet(feat2, feat0)

                # flow = flow * occ_memXAI_median
                
                # print('flow', flow.shape)
                # flowmag_vis.append(self.summ_writer.summ_feat('', flow*occ0, pca=False, only_return=True))
                # featmag_vis.append(self.summ_writer.summ_feat('', feat0*occ0, pca=False, only_return=True))
                # featmag_vis.append(self.summ_writer.summ_feat('', feat0, pca=False, only_return=True))
                # flowmag_vis.append(self.summ_writer.summ_feat('', flow01*occ0*(1.0 - occ_memXAI_median), pca=False, only_return=True))
                # flow_vis.append(self.summ_writer.summ_3D_flow('', flow, occ=occ0, clip=clip, only_return=True))
                flow01_vis.append(self.summ_writer.summ_3D_flow('', flow01, occ=rel0*occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                # flow10_vis.append(self.summ_writer.summ_3D_flow('', flow10, occ=occ1*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                # feat_vis.append(self.summ_writer.summ_feat('', feat0, pca=True, only_return=True))
                occ0_vis.append(self.summ_writer.summ_occ('', occ0, only_return=True))

                # print('apending flow')
                # flow10_all.append(flow10)

                # # diff1_aligned = utils_samp.backwarp_using_3D_flow(diff1, flow01)
                # flow10_aligned = utils_samp.backwarp_using_3D_flow(flow10, flow01)
                # flow20_aligned = utils_samp.backwarp_using_3D_flow(flow20, flow02)
                # mask1 = torch.exp(-torch.norm(flow10_aligned+flow01, dim=1, keepdim=True))
                # mask2 = torch.exp(-torch.norm(flow20_aligned+flow02, dim=1, keepdim=True))
                # mask = mask1*mask2
                # consistent_flow01_vis.append(self.summ_writer.summ_3D_flow(
                #     '', flow01, occ=mask*occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                # # consistent_flow01
                
                # flow01 = utils_samp.backwarp_using_3D_flow(flow10, flow01)
                # flow01
                
                # i want to chain together the flow
                # flow01 is good
                # flow12 is cool but not aligned with flow01; since it is in 1 coords, i can warp it with that

                # flow01 is good
                flow02 = flow01 + utils_samp.apply_flowAB_to_voxB(flow01, flow12)
                flow03 = flow02 + utils_samp.apply_flowAB_to_voxB(flow02, flow23)
                flow04 = flow03 + utils_samp.apply_flowAB_to_voxB(flow03, flow34)
                flow05 = flow04 + utils_samp.apply_flowAB_to_voxB(flow04, flow45)

                flow01_all.append(flow01)
                flow02_all.append(flow02)
                flow03_all.append(flow03)
                flow04_all.append(flow04)
                flow05_all.append(flow05)

                flow12_all.append(flow12)
                flow23_all.append(flow23)
                flow34_all.append(flow34)
                flow45_all.append(flow45)
                
                # these chained flows look surprisingly good

                # now i want flow40, and to backwarp it with flow04
                
                # flow42 = flow43 + utils_samp.backwarp_using_3D_flow(flow34, flow03)

                flow53 = flow54 + utils_samp.apply_flowAB_to_voxB(flow54, flow43)
                flow52 = flow53 + utils_samp.apply_flowAB_to_voxB(flow53, flow32)
                flow51 = flow52 + utils_samp.apply_flowAB_to_voxB(flow52, flow21)
                flow50 = flow51 + utils_samp.apply_flowAB_to_voxB(flow51, flow10)

                # utils_samp.apply_flow01_to_vox1(flow01, vox1, binary_feat=False)                

                # flow42 = flow43 + utils_samp.backwarp_using_3D_flow(flow32, flow43)
                
                # flow03 = flow02 + utils_samp.backwarp_using_3D_flow(flow23, flow02)
                # flow04 = flow03 + utils_samp.backwarp_using_3D_flow(flow34, flow03)
                
                flow05_vis.append(self.summ_writer.summ_3D_flow('', flow05*rel0, occ=occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))

                # # flow43 is good
                # flow43 = flow01 + utils_samp.backwarp_using_3D_flow(flow12, flow01)
                # flow03 = flow02 + utils_samp.backwarp_using_3D_flow(flow23, flow02)
                # flow04 = flow03 + utils_samp.backwarp_using_3D_flow(flow34, flow03)
                flow50_aligned = utils_samp.apply_flowAB_to_voxB(flow05, flow50)
                flow_cycle = flow05 + flow50_aligned
                mask = torch.exp(-torch.norm(flow_cycle, dim=1, keepdim=True))
                # mask = mask * vis0


                # flow10_aligned = utils_samp.backwarp_using_3D_flow(flow10, flow01)
                # flow20_aligned = utils_samp.backwarp_using_3D_flow(flow20, flow02)
                # mask1 = torch.exp(-torch.norm(flow10_aligned+flow01, dim=1, keepdim=True))
                # mask2 = torch.exp(-torch.norm(flow20_aligned+flow02, dim=1, keepdim=True))
                # mask = mask1*mask2
                consistent_flow05_vis.append(self.summ_writer.summ_3D_flow(
                    '', flow05*rel0, occ=mask*occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                consistent_mask05_vis.append(self.summ_writer.summ_oned('', mask, bev=True, norm=False, only_return=True))
                # # consistent_flow01

                consistent_flow05_all.append(flow05 * mask * rel0 * occ0 * (1.0 - occ_memXAI_median))
                            

                # feat1_aligned = utils_samp.backwarp_using_3D_flow(feat1, flow_total)
                
            # print('summing..')

            # self.summ_writer.summ_rgbs('3D_feats/flowmag', flowmag_vis)
            # self.summ_writer.summ_rgbs('3D_feats/featmag', featmag_vis)
            self.summ_writer.summ_rgbs('3D_feats/flow01', flow01_vis)
            self.summ_writer.summ_rgbs('3D_feats/flow05', flow05_vis)
            self.summ_writer.summ_rgbs('3D_feats/consistent_flow05', consistent_flow05_vis)
            self.summ_writer.summ_rgbs('3D_feats/consistent_mask05', consistent_mask05_vis)
            # self.summ_writer.summ_rgbs('3D_feats/flow10', flow10_vis)
            # self.summ_writer.summ_rgbs('3D_feats/feat', feat_vis)
            self.summ_writer.summ_rgbs('3D_feats/occ0', occ0_vis)

        # self.summ_writer.summ_rgbs('3D_feats/feat_memXAI_input', feat_memXAI_input_vis)
        # self.summ_writer.summ_rgbs('3D_feats/feat_memXAI', feat_memXAI_vis)
        # self.summ_writer.summ_rgbs('3D_feats/occ_memXAI', occ_memXAI_vis)
        # self.summ_writer.summ_rgbs('3D_feats/occrel_memXAI', occrel_memXAI_vis)

        # return total_loss, results, False
        
        diff_memXAI_all = []
        for I in list(range(self.S)):
            # feat_memXAI = feat_memXAI_all[I]
            # diff_memXAI_all.append(torch.norm(feat_memXAI - feat_memXAI_median, dim=1, keepdim=True))

            vis_memXAI = vis_memXAI_all[I]

            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
            # vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
            
            occ_memXAI = occ_memXAI_all[I]
            occrel_memXAI = occrel_memXAI_all[I]

            use_occrel = True
            if not use_occrel:
                occrel_memXAI = torch.ones_like(occrel_memXAI)

            # diff_memXAI_all.append(torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(occ_memXAI * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
            # diff_memXAI_all.append(occ_memXAI.round() * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))

            # diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True)
            diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True) * occrel_memXAI
            # diff = torch.nn.functional.relu(diff - occ_memXAI_diff)

            diff = occ_memXAI.round() * vis_memXAI * diff
            diff_memXAI_all.append(diff)
            
            # diff_memXAI = torch.norm(feat_memXAI - feat_memXAI_median, dim=1, keepdim=True)
            # diff_memXAI_vis.append(self.summ_writer.summ_oned('', diff_memXAI, bev=True, only_return=True))

        cumdiff_memXAI_all = []
        for I in list(range(self.S-5)):
            diff0 = diff_memXAI_all[I]
            diff1 = diff_memXAI_all[I+1]
            diff2 = diff_memXAI_all[I+2]
            diff3 = diff_memXAI_all[I+3]
            diff4 = diff_memXAI_all[I+4]
            diff5 = diff_memXAI_all[I+5]
            flow01 = flow01_all[I]
            flow02 = flow02_all[I]
            flow03 = flow03_all[I]
            flow04 = flow04_all[I]
            flow05 = flow05_all[I]
            diff1_aligned = utils_samp.backwarp_using_3D_flow(diff1, flow01)
            diff2_aligned = utils_samp.backwarp_using_3D_flow(diff2, flow02)
            diff3_aligned = utils_samp.backwarp_using_3D_flow(diff3, flow03)
            diff4_aligned = utils_samp.backwarp_using_3D_flow(diff4, flow04)
            diff5_aligned = utils_samp.backwarp_using_3D_flow(diff5, flow05)

            # flow12 = flow12_all[I]
            # flow23 = flow23_all[I]
            # flow34 = flow34_all[I]
            # flow45 = flow45_all[I]
            
            diff4 = diff4 + utils_samp.apply_flowAB_to_voxB(flow45, diff5)
            diff3 = diff3 + utils_samp.apply_flowAB_to_voxB(flow34, diff4)
            diff2 = diff2 + utils_samp.apply_flowAB_to_voxB(flow23, diff3)
            diff1 = diff1 + utils_samp.apply_flowAB_to_voxB(flow12, diff2)
            diff0 = diff0 + utils_samp.apply_flowAB_to_voxB(flow01, diff1)
            # flow05 = flow04 + utils_samp.apply_flowAB_to_voxB(flow04, flow45)

            # diff = flow01 + utils_samp.apply_flowAB_to_voxB(flow01, flow12)
            # flow03 = flow02 + utils_samp.apply_flowAB_to_voxB(flow02, flow23)
            # flow04 = flow03 + utils_samp.apply_flowAB_to_voxB(flow03, flow34)
            # flow05 = flow04 + utils_samp.apply_flowAB_to_voxB(flow04, flow45)

            
            # diff1_aligned = utils_samp.backwarp_using_3D_flow(diff1, flow01*0.0)
            # cumdiff = diff0 * diff1_aligned
            cumdiff_memXAI_all.append(diff0/5.0)
            
            # cumdiff = diff0 + diff1_aligned + diff2_aligned + diff3_aligned + diff4_aligned + diff5_aligned
            # cumdiff_memXAI_all.append(cumdiff/5.0)
            # print('cumdiff_memXAI_all

        # diff_memXAI_all = torch.stack(diff_memXAI_all, dim=1)
        # utils_basic.print_stats('diff_memXAI_all', diff_memXAI_all)
        # diff_memXAI_all = 1.0-torch.exp(-diff_memXAI_all)
        # utils_basic.print_stats('1.0-exp(-diff_memXAI_all)', diff_memXAI_all)
        # diff_memXAI_all = utils_basic.normalize(diff_memXAI_all)
        # diff_memXAI_all = diff_memXAI_all.unbind(dim=1)
        
        if self.summ_writer.save_this:
            super_vis = []
            
            feat_memXAI_vis = []
            feat_memXAI_input_vis = []
            occ_memXAI_vis = []

            diff_memXAI_vis = []
            cumdiff_memXAI_vis = []
            for I in list(range(self.S-5)):
                # diff_memXAI_vis.append(self.summ_writer.summ_oned('', diff_memXAI_all[I], bev=True, norm=True, only_return=True))
                # diff_memXAI_vis.append(self.summ_writer.summ_oned('', diff_memXAI_all[I], bev=True, norm=False, only_return=True))
                diff_memXAI_vis.append(self.summ_writer.summ_oned('', diff_memXAI_all[I], bev=True, max_along_y=True, norm=False, only_return=True))
                cumdiff_memXAI_vis.append(self.summ_writer.summ_oned('', cumdiff_memXAI_all[I].clamp(0, 1), bev=True, max_along_y=True, norm=False, only_return=True))
                # diff_memXAI_vis.append(self.summ_writer.summ_oned('', torch.max(diff_memXAI_all[I], dim=3, keepdim=True)[0], bev=True, norm=False, only_return=True))
            self.summ_writer.summ_rgbs('3D_feats/diff_memXAI', diff_memXAI_vis)
            self.summ_writer.summ_rgbs('3D_feats/cumdiff_memXAI', cumdiff_memXAI_vis)
            
            for I in list(range(self.S-5)):
                
                # feat_memXAI_vis.append(self.summ_writer.summ_feat('', feat_memXAI_all[I], pca=True, only_return=True))

                occ_memXAI_g = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,I], self.Z, self.Y, self.X)
                occ_memXAI_g = F.interpolate(occ_memXAI_g, scale_factor=0.5, mode='trilinear')
                occ_memXAI_g = self.crop_feat(occ_memXAI_g)
                # occ_memXAI_g_vis.append(self.summ_writer.summ_occ('', occ_memXAI_g, only_return=True))
                # occ_memXAI_vis.append(self.summ_writer.summ_occ('', occ_memXAI, only_return=True))
                # occrel_memXAI_vis.append(self.summ_writer.summ_oned('', occrel_memXAI, bev=True, norm=False, only_return=True))

                # a = feat_memXAI_input_vis[I]
                # # b = feat_memXAI_vis[I]
                # b = occ_memXAI_vis[I]
                # c = diff_memXAI_vis[I]

                # a = self.summ_writer.summ_feat('', feat_memXAI_all[I], pca=True, only_return=True)
                a = self.summ_writer.summ_occ('', occ_memXAI_g, only_return=True)
                b = self.summ_writer.summ_occ('', occ_memXAI_all[I], only_return=True)
                c = self.summ_writer.summ_oned('', occrel_memXAI_all[I], bev=True, norm=False, only_return=True)
                d = diff_memXAI_vis[I]
                e = cumdiff_memXAI_vis[I]

                # these are B x C x H x W
                cat = torch.cat([a, b, c, d, e], dim=3)
                # abcd = torch.cat([a, b, c, d], dim=2)
                super_vis.append(cat)
            self.summ_writer.summ_rgbs('3D_feats/super_memXAI', super_vis)

        self.K = 32

        use_box_cache = False
        if hyp.do_use_cache and use_box_cache:
            data_ind = feed['data_ind']
            box_cache_fn = 'cache/%s_%06d_s%d_box.npz' % (self.set_data_name, data_ind, self.S)
            # check if the thing exists
            if os.path.isfile(box_cache_fn):
                print('found box cache at %s; we will use this' % med_cache_fn)
                cache = np.load(box_cache_fn, allow_pickle=True)['save_dict'].item()
                # cache = cache['save_dict']
                have_boxes = True

                lrtlist_camXIs = torch.from_numpy(cache['lrtlist_camXIs']).cuda().unbind(1)
                # connlist_memXAIs = torch.from_numpy(cache['connlist_memXAIs']).cuda().unbind(1)
                scorelist_s = [s for s in torch.from_numpy(cache['scorelist_s']).cuda().unbind(1)]
                box_vis = torch.from_numpy(cache['box_vis']).unbind(1)
                box_vis_bev = torch.from_numpy(cache['box_vis_bev']).unbind(1)
                
            else:
                print('could not find box cache at %s; we will write this' % box_cache_fn)

        if not have_boxes:
            box_vis_bev = []
            box_vis = []
            blue_vis = []
            conn_vis = []
            lrtlist_camXIs = []
            # connlist_memXAIs = []
            scorelist_s = []
            
            for I in list(range(self.S-5)):
                # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                #     self.pad_feat(diff_memXAI_all[I]).squeeze(1), self.K)
                # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                #     cumdiff_memXAI_all[I].squeeze(1), self.K)
                # flow05_mag = torch.norm(flow05_all[I], dim=1)
                # flow05_mag = torch.norm(consistent_flow05_all[I], dim=1)
                # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                #     flow05_mag, self.K)
                boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                    cumdiff_memXAI_all[I].squeeze(1), self.K)
                x, y, z, lx, ly, lz, rx, ry, rz = boxlist_memXAI.unbind(2)
                z = z + self.crop_guess[0]
                y = y + self.crop_guess[1]
                x = x + self.crop_guess[2]
                boxlist_memXAI = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=2)
                
                lrtlist_memXAI = utils_geom.convert_boxlist_to_lrtlist(boxlist_memXAI)
                lrtlist_camXAI = self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_memXAI, self.Z2, self.Y2, self.X2)
                lrtlist_camXI = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camXAs[:,I], lrtlist_camXAI)
                lrtlist_camXIs.append(lrtlist_camXI)
                # connlist_memXAIs.append(connlist)

                scorelist_e[scorelist_e > 0.0] = 1.0
                occ_memXAI = occ_memXAI_all[I]
                diff_memXAI = diff_memXAI_all[I]
                for n in list(range(self.K)):
                    # mask_1 = self.vox_util.assemble_padded_obj_masklist(
                    #     lrtlist_camXAI[:,n:n+1], scorelist_e[:,n:n+1],
                    #     self.Z2, self.Y2, self.X2, coeff=0.8).squeeze(1)
                    # mask_3 = self.vox_util.assemble_padded_obj_masklist(
                    #     lrtlist_camXAI[:,n:n+1], scorelist_e[:,n:n+1],
                    #     self.Z2, self.Y2, self.X2, coeff=1.8).squeeze(1)
                    # mask_1 = self.crop_feat(mask_1)
                    # mask_3 = self.crop_feat(mask_3)

                    mask_1 = connlist[:,n:n+1]
                    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                    mask_3 = (F.conv3d(mask_1, weights, padding=1)).clamp(0, 1)

                    center_mask = mask_1.clone()
                    surround_mask = (mask_3-mask_1).clamp(0,1)

                    center_ = utils_basic.reduce_masked_mean(occ_memXAI, center_mask, dim=[2,3,4])
                    surround_ = utils_basic.reduce_masked_mean(occ_memXAI, surround_mask, dim=[2,3,4])
                    score_ = center_ - surround_
                    score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)
                    score_[score_ < 0.51] = 0.0
                    scorelist_e[:,n] = score_

                    # ok good job, now, 
                
                    
                    
                scorelist_s.append(scorelist_e)

                # self.summ_writer.summ_rgb('proposals/anchor_frame', diff_memXAI_vis[self.anchor])
                # self.summ_writer.summ_rgb('proposals/get_boxes', boxes_image)
                blue_vis.append(boxes_image)

                # conn_vis.append(self.summ_writer.summ_occ('', torch.sum(connlist, dim=1, keepdims=True).clamp(0, 1), only_return=True))

                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,I],
                    # torch.ones_like(scorelist_e),
                    # # scorelist_e, 
                    # tidlist,
                    torch.cat([self.full_lrtlist_camXs[:,I], lrtlist_camXI], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_e], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(tidlist).long()], dim=1),
                    self.pix_T_cams[:,0], frame_id=I, only_return=True))

                # self.summ_writer.summ_lrtlist_bev(
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.pad_feat(occ_memXAI_all[I]),
                    # '', self.pad_feat(diff_memXAI_all[I]),
                    torch.cat([self.full_lrtlist_camXAs[:,I], lrtlist_camXAI], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_e], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(tidlist).long()], dim=1),
                    # torch.ones_like(scorelist_e),
                    # tidlist,
                    self.vox_util, frame_id=I, only_return=True))
                
            if hyp.do_use_cache and use_box_cache:
                # save this, so that we have it all next time
                save_dict = {}
                save_dict['lrtlist_camXIs'] = torch.stack(lrtlist_camXIs, dim=1).detach().cpu().numpy()
                # save_dict['connlist_memXAIs'] = torch.stack(connlist_memXAIs, dim=1).detach().cpu().numpy()
                save_dict['scorelist_s'] = torch.stack(scorelist_s, dim=1).detach().cpu().numpy()
                save_dict['box_vis'] = torch.stack(box_vis, dim=1).detach().cpu().numpy()
                save_dict['box_vis_bev'] = torch.stack(box_vis_bev, dim=1).detach().cpu().numpy()
                np.savez(box_cache_fn, save_dict=save_dict)
                print('saved boxes to %s cache, for next time' % box_cache_fn)
                
        print('got boxes; time %.2f' % (time.time() - start_time))
        if box_vis_bev is not None:
            self.summ_writer.summ_rgbs('proposals/all_boxes_bev', box_vis_bev)
            self.summ_writer.summ_rgbs('proposals/all_boxes', box_vis)
        # self.summ_writer.summ_rgbs('proposals/all_boxes_blue', blue_vis)
        # self.summ_writer.summ_rgbs('proposals/all_conn', conn_vis)

        best_score = torch.zeros_like(scorelist_s[0][:,0])
        best_lrt = torch.zeros_like(lrtlist_camXIs[0][:,0])
        best_occ = torch.zeros_like(occ_memXAI_all[0])
        best_I = 0

        # print('best_score', best_score.shape)
        # print('best_lrt', best_lrt.shape)
        # print('best_occ', best_occ.shape)
        
        for b in list(range(self.B)):
            for I in list(range(self.S-5)):
                scorelist = scorelist_s[I][b]
                lrtlist_camXI = lrtlist_camXIs[I][b]
                occ_memXAI = occ_memXAI_all[I][b]
                # print('scorelist', scorelist.shape)
                ind = torch.argmax(scorelist, dim=0)
                best_score_here = scorelist[ind]
                best_lrt_here = lrtlist_camXI[ind]

                # print('best_score_here', best_score_here.shape)
                # print('best_lrt_here', best_lrt_here.shape)
                if torch.squeeze(best_score_here) > best_score[b]:
                    best_score[b] = best_score_here
                    best_lrt[b] = best_lrt_here
                    best_occ[b] = occ_memXAI
                    best_I = I

        best_iou = np.zeros([self.B], np.float32)
        for b in list(range(self.B)):
            ious = np.zeros((self.N), dtype=np.float32)
            for i in list(range(self.N)):
                if self.full_scorelist_s[b,best_I,i] > 0.5:
                    ious[i] = np.squeeze(utils_geom.get_iou_from_corresponded_lrtlists(
                        best_lrt[b:b+1].unsqueeze(1), self.full_lrtlist_camXs[b:b+1,best_I,i:i+1])[0,0])
            best_iou[b] = np.max(ious)
        best_iou = torch.from_numpy(best_iou).float().cuda()

        self.summ_writer.summ_lrtlist_bev(
            'proposals/best_box_bev', self.pad_feat(best_occ),
            best_lrt.unsqueeze(1),
            # best_score.unsqueeze(1),
            best_iou.unsqueeze(1),
            torch.ones_like(best_score.unsqueeze(1)).long(),
            self.vox_util, frame_id=best_I)

        lrtlist_camXIs = torch.stack(lrtlist_camXIs, dim=1)
        scorelist_s = torch.stack(scorelist_s, dim=1)
        # this is B x S x N x 19
        # and note B==1
        lrtlist_camXIs_ = __p(lrtlist_camXIs)
        scorelist_s_ = __p(scorelist_s)
        full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
        full_scorelist_s_ = __p(self.full_scorelist_s)
        
        # iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # best_maps = np.zeros([self.S, len(iou_thresholds)])
        # for s in list(range(self.S-1)):
        #     _, lrtlist_g, _, scorelist_g = utils_eval.drop_invalid_lrts(
        #         best_lrt.unsqueeze(1), full_lrtlist_camXs_[s:s+1], best_score.unsqueeze(1), full_scorelist_s_[s:s+1])
        #     maps = utils_eval.get_mAP_from_lrtlist(best_lrt.unsqueeze(1), best_score.unsqueeze(1), lrtlist_g, iou_thresholds)
        #     best_maps[s] = maps
        # for ind, overlap in enumerate(iou_thresholds):
        #     maps = best_maps[:,ind]
        #     maps = maps[maps > 0]
        #     if len(maps):
        #         map_val = np.mean(maps)
        #     else:
        #         map_val = 0.0
        #     self.summ_writer.summ_scalar('best_ap/%.2f_iou' % overlap, map_val)
        # print('got "best" eval; time %.2f' % (time.time() - start_time))
                    
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # all_maps = []
        all_maps = np.zeros([self.S, len(iou_thresholds)])
        for s in list(range(self.S-5)):

            lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                lrtlist_camXIs_[s:s+1], full_lrtlist_camXs_[s:s+1], scorelist_s_[s:s+1], full_scorelist_s_[s:s+1])
            # maps = utils_eval.get_mAP_from_lrtlist(lrtlist_e[s:s+1], scorelist_e[s:s+1], lrtlist_g[s:s+1], iou_thresholds)
            maps = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
            all_maps[s] = maps
            if s==int(self.S/2):
                print('lrtlist_e', lrtlist_e[:,:,:3])
                print('scorelist_e', scorelist_e)
                print('lrtlist_g', lrtlist_g[:,:,:3])
                print('scorelist_g', scorelist_g)
                print('maps', maps)
        for ind, overlap in enumerate(iou_thresholds):
            maps = all_maps[:,ind]
            maps = maps[maps > 0]
            if len(maps):
                map_val = np.mean(maps)
            else:
                map_val = 0.0
            self.summ_writer.summ_scalar('ap/%.2f_iou' % overlap, map_val)
        print('got eval; time %.2f' % (time.time() - start_time))
        
        return total_loss, results, False

        if False:
            self.summ_writer.summ_feats('3D_feats/feat_memX0', [feat_memX00, feat_memX01], pca=True)
            self.summ_writer.summ_occs('3D_feats/occ_memX0', [occ_memX00, occ_memX01])
            self.summ_writer.summ_occs('3D_feats/occ_memX0_g', [self.occ_memX00, self.occ_memX01])
            
            feat_memX00_shuf = feat_memX00_shuf[occ_memX00_shuf*occ_memX01_shuf > 0.5]
            feat_memX01_shuf = feat_memX01_shuf[occ_memX00_shuf*occ_memX01_shuf > 0.5]

            perm = np.random.permutation(feat_memX00_shuf.shape[0])
            feat_memX00_shuf = feat_memX00_shuf[perm]
            feat_memX01_shuf = feat_memX01_shuf[perm]

            feat_memX00_shuf = feat_memX00_shuf.detach().cpu().numpy()
            feat_memX01_shuf = feat_memX01_shuf.detach().cpu().numpy()
            rand_diff = np.linalg.norm(feat_memX00_shuf - feat_memX01_shuf, axis=1)

            # print('rand_diff', rand_diff.shape)

            # print('diff_pool has %d items' % len(self.diff_pool))
            self.diff_pool.update(rand_diff)
            # print('diff_pool has %d items' % len(self.diff_pool))

            rand_diff_mean = np.mean(self.diff_pool.fetch())
            rand_diff_std = np.sqrt(np.var(self.diff_pool.fetch()))
            # print('rand_diff mean, std:', rand_diff_mean, rand_diff_std)

            feat_diff = torch.norm(feat_memX00 - feat_memX01, dim=1, keepdim=True)

            # print('feat_diff mean:', torch.mean(feat_diff).detach().cpu().numpy())

            feat_diff_occ = feat_diff * occ_memX00

            # large_diff_occ = (feat_diff_occ > rand_diff_mean).float()
            large_diff_occ = (feat_diff_occ > (rand_diff_mean + rand_diff_std)).float()
            small_diff_occ = (feat_diff_occ < (rand_diff_mean - rand_diff_std)).float()

            self.summ_writer.summ_feat('3D_feats/feat_diff', feat_diff, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_diff_occ', feat_diff_occ, pca=False)
            self.summ_writer.summ_feat('3D_feats/large_diff_occ', large_diff_occ, pca=False)
            self.summ_writer.summ_feat('3D_feats/small_diff_occ', small_diff_occ, pca=False)

        if hyp.do_mot:
            occ_memX0s, free_memX0s, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z, self.Y, self.X,
                agg=False)

            # if it is occ at time0 AND free at time1, something changed
            cha_memX00 = occ_memX0s[:,0] * free_memX0s[:,1]
            # the other way is good too, but note this is anchored on X01
            cha_memX01 = occ_memX0s[:,1] * free_memX0s[:,0]

            # if it is occ at time0 AND occ at time1, or same for free, then nothing changed
            sam_memX00 = (occ_memX0s[:,0] * occ_memX0s[:,1] + free_memX0s[:,0] * free_memX0s[:,1]).clamp(0, 1)
            sam_memX01 = (occ_memX0s[:,0] * occ_memX0s[:,1] + free_memX0s[:,0] * free_memX0s[:,1]).clamp(0, 1) # same

            cha_memX00 = F.interpolate(cha_memX00, scale_factor=0.5, mode='trilinear').round()
            sam_memX00 = F.interpolate(sam_memX00, scale_factor=0.5, mode='trilinear').round()
            cha_memX01 = F.interpolate(cha_memX01, scale_factor=0.5, mode='trilinear').round()
            sam_memX01 = F.interpolate(sam_memX01, scale_factor=0.5, mode='trilinear').round()
            
            cha_memX00 = self.crop_feat(cha_memX00)
            sam_memX00 = self.crop_feat(sam_memX00)
            cha_memX01 = self.crop_feat(cha_memX01)
            sam_memX01 = self.crop_feat(sam_memX01)
            
            self.summ_writer.summ_feat('3D_feats/cha_memX00', cha_memX00, pca=False)
            self.summ_writer.summ_feat('3D_feats/sam_memX00', sam_memX00, pca=False)

            # ensure voxels are not marked as both labels
            cha_memX00 = cha_memX00 * (1.0 - sam_memX00)
            cha_memX01 = cha_memX01 * (1.0 - sam_memX01)
            valid_memX00 = (cha_memX00 + sam_memX00).clamp(0, 1)
            valid_memX01 = (cha_memX01 + sam_memX01).clamp(0, 1)

            obj_match = torch.eq(self.mask_memX00, cha_memX00).float()
            bkg_match = torch.eq(1.0-self.mask_memX00, sam_memX00).float()
            acc_obj = utils_basic.reduce_masked_mean(obj_match, cha_memX00)
            acc_bkg = utils_basic.reduce_masked_mean(bkg_match, sam_memX00)
            if torch.sum(cha_memX00) > 0:
                self.summ_writer.summ_scalar('unscaled_mot/acc_obj', acc_obj.cpu().item())
            if torch.sum(sam_memX00) > 0:
                self.summ_writer.summ_scalar('unscaled_mot/acc_bkg', acc_bkg.cpu().item())

            # run motnet with X00 and X01 stacked on the batch dim
            mot_loss, mot_memX00_pred = self.motnet(
                # torch.cat([feat_memX00_input, feat_memX01_input], dim=0), 
                torch.cat([feat_memX00, feat_memX01], dim=0), 
                torch.cat([cha_memX00, cha_memX01], dim=0), 
                torch.cat([sam_memX00, sam_memX01], dim=0), 
                torch.cat([valid_memX00, valid_memX01], dim=0), 
                self.summ_writer)
            total_loss += mot_loss
            # slice out
            mot_memX00_pred = mot_memX00_pred[:self.B]
            occ_memX00_g = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z2, self.Y2, self.X2)
            occ_memX00_g = self.crop_feat(occ_memX00_g)
            pred_match = torch.eq(self.mask_memX00, mot_memX00_pred.round()).float()
            acc_pred = torch.mean(pred_match)
            acc_pred_obj = utils_basic.reduce_masked_mean(pred_match, self.mask_memX00)
            acc_pred_bkg = utils_basic.reduce_masked_mean(pred_match, (1.0-self.mask_memX00))
            acc_pred_bal = (acc_pred_obj + acc_pred_bkg)/2.0
            self.summ_writer.summ_scalar('unscaled_mot/acc_pred', acc_pred.cpu().item())
            self.summ_writer.summ_scalar('unscaled_mot/acc_pred_bal', acc_pred_bal.cpu().item())
            self.summ_writer.summ_scalar('unscaled_mot/acc_pred_obj', acc_pred_obj.cpu().item())
            self.summ_writer.summ_scalar('unscaled_mot/acc_pred_bkg', acc_pred_bkg.cpu().item())
            
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
            if set_name=='train' or set_name=='val':
                return self.run_train(feed)
            elif set_name=='test':
                return self.run_sub(feed)

        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)
