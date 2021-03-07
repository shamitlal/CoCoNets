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
# from nets.occnet import OccNet
# from nets.mocnet import MocNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet

from nets.vq3dnet import Vq3dNet
from nets.occnet import OccNet
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



def project_l2_ball_py(z):
    # project the vectors in z onto the l2 unit norm ball
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1, keepdims=True)), 1)

# def project_l2_ball_pt(z):
#     # project the vectors in z onto the l2 unit norm ball
#     return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)
        

# the idea of this mode is to overfit to a few examples and prove to myself that i can generate propose outputs

class CARLA_PROPOSE(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaProposeModel()
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
            
class CarlaProposeModel(nn.Module):
    def __init__(self):
        super(CarlaProposeModel, self).__init__()

        if hyp.do_feat2D:
            self.featnet2D = FeatNet2D()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            
        if hyp.do_feat3D:
            self.crop_guess = (18,18,18)
            # self.crop_guess = (19,19,19)
            self.featnet3D = FeatNet3D(in_dim=4, crop=self.crop_guess)
            
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
            
        if hyp.do_center:
            self.centernet = CenterNet()
            
        if hyp.do_seg:
            self.num_seg_labels = 13 # note label0 is "none"
            # we will predict all 12 valid of these, plus one "air" class
            self.segnet = SegNet(self.num_seg_labels)

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
            
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils_improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
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

        if self.set_name=='test':
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

            # full_lrtlist_camX0s_ =

            # boxlist_e = self.box_camRs[0:1].detach().cpu().numpy()
            # boxlist_g = self.box_camRs[0:1].detach().cpu().numpy()
            # scorelist_e = self.score_s[0:1].detach().cpu().numpy()
            # scorelist_g = self.score_s[0:1].detach().cpu().numpy()
            # boxlist_e, boxlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_boxes(
            #     boxlist_e, boxlist_g, scorelist_e, scorelist_g)

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
                # scene_centroid_x = np.random.uniform(-8.0, 8.0)
                # scene_centroid_y = np.random.uniform(-1.5, 3.0)
                # scene_centroid_z = np.random.uniform(10.0, 26.0)
                scene_centroid_x = 0.0
                scene_centroid_y = 1.0
                scene_centroid_z = 18.0
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
        # self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        # self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())


            
        full_scorelist_s_ = utils_misc.rescore_lrtlist_with_inbound(
            full_lrtlist_camX0s_, __p(full_tidlist_s), self.Z, self.Y, self.X, self.vox_util)
        # full_scorelist_s = __u(full_scorelist_s_)
        self.full_scorelist_s = __u(full_scorelist_s_)
        self.full_tidlist_s = full_tidlist_s
        self.full_lrtlist_camX0s = __u(full_lrtlist_camX0s_)

        for b in list(range(self.B)):
            if torch.sum(self.full_scorelist_s[b,0]) == 0:
                print('returning early, since there are zero objects inbound at frame0')
                return False
            else:
                print('this many objects on frame 0:', torch.sum(self.full_scorelist_s[b,0]).detach().cpu().numpy())
        
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
        
        
        
        

        if hyp.do_feat3D:
            feat_memX0s_input = torch.cat([
                self.occ_memX0s,
                self.rgb_memX0s*self.occ_memX0s,
            ], dim=2)
            feat_memX0_input = utils_basic.reduce_masked_mean(
                feat_memX0s_input[:,1:],
                self.occ_memX0s[:,1:].repeat(1, 1, 4, 1, 1, 1),
                dim=1)
            feat3D_loss, feat_memX0, valid_memX0 = self.featnet3D(
                feat_memX0_input,
                self.summ_writer,
            )
            total_loss += feat3D_loss
            self.summ_writer.summ_feat('3D_feats/feat_memX0_input', feat_memX0_input, pca=True)
            self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, pca=True)
            # print('feat total_loss', total_loss.detach().cpu().numpy())

            if hyp.do_emb3D:
                _, altfeat_memX0, altvalid_memX0 = self.featnet3D_slow(feat_memX0s_input[:,0])
                
                # self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
                self.summ_writer.summ_feat('3D_feats/altfeat_memX0_input', feat_memX0s_input[:,0], pca=True)
                self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, pca=True)
                # self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)

            _, _, Z2, Y2, X2 = list(feat_memX0.shape)
            Z_crop = int((self.Z2 - Z2)/2)
            Y_crop = int((self.Y2 - Y2)/2)
            X_crop = int((self.X2 - X2)/2)
            crop = (Z_crop, Y_crop, X_crop)
            if not (crop==self.crop_guess):
                print('crop', crop)
            assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
            
            # self.summ_writer.summ_feat('3D_feats/feat_memX0_up', feat_memX0, pca=True)

                
        # if hyp.do_up3D:
        #     up3D_loss, feat_memX0 = self.upnet3D(feat_memX0, self.summ_writer)
        #     total_loss += up3D_loss
        #     print('up total_loss', total_loss.detach().cpu().numpy())

        #     valid_memX0 = torch.ones_like(feat_memX0[:,0:1])

        #     _, _, Z2, Y2, X2 = list(feat_memX0.shape)
        #     Z_crop = int((self.Z - Z2)/2)
        #     Y_crop = int((self.Y - Y2)/2)
        #     X_crop = int((self.X - X2)/2)
        #     crop = (Z_crop, Y_crop, X_crop)
        #     if not (crop==self.crop_guess):
        #         print('crop', crop)
        #     assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
        #     self.summ_writer.summ_feat('3D_feats/feat_memX0_up', feat_memX0, pca=True)

        #     # if hyp.do_emb3D:
        #     #     _, altfeat_memX0, altvalid_memX0, _ = self.featnet3D_slow(feat_memX0s_input[:,0])
        #     #     self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
        #     #     self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)
            
        #     feat_memXs_input = torch.cat([
        #         self.occ_memXs,
        #         self.unp_memXs*self.occ_memXs,
        #     ], dim=2)
        #     feat_memRs_input = torch.cat([
        #         self.occ_memRs,
        #         self.unp_memRs*self.occ_memRs,
        #     ], dim=2)
            
        #     feat3D_loss, feat_memXs_ = self.featnet3D(
        #         __p(feat_memXs_input[:,1:]),
        #         self.summ_writer,
        #     )
        #     feat_memXs = __u(feat_memXs_)
        #     valid_memXs = torch.ones_like(feat_memXs[:,:,0:1])
        #     total_loss += feat3D_loss
            
        #     # warp things to R
        #     feat_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], feat_memXs)
        #     valid_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:, 1:], valid_memXs)

        #     feat_memR = utils_basic.reduce_masked_mean(
        #         feat_memRs,
        #         valid_memRs.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
        #         dim=1)
        #     valid_memR = torch.sum(valid_memRs, dim=1).clamp(0, 1)

        #     _, altfeat_memR = self.featnet3D(feat_memRs_input[:,0])
            
        #     self.summ_writer.summ_feat('3D_feats/feat_memR', feat_memR, valid=valid_memR, pca=True)
        #     self.summ_writer.summ_feat('3D_feats/altfeat_memR', altfeat_memR, pca=True)


        # if hyp.do_vq3d:
        #     # overwrite zi with its quantized version
        #     vq3d_loss, zi, _ = self.vq3dnet(self.zi, self.summ_writer)
        #     total_loss += vq3d_loss
        #     self.summ_writer.summ_feat('feat3D/zi_quantized', zi, pca=True)
        #     # print('vq total_loss', total_loss.detach().cpu().numpy())
        
        # if hyp.do_up3D:
        #     up3D_loss, zi_up = self.upnet3D(zi, self.summ_writer)
        #     total_loss += up3D_loss
        #     # print('up total_loss', total_loss.detach().cpu().numpy())

        #     _, _, Z2, Y2, X2 = list(zi_up.shape)
        #     Z_crop = int((self.Z2 - Z2)/2)
        #     Y_crop = int((self.Y2 - Y2)/2)
        #     X_crop = int((self.X2 - X2)/2)
        #     crop = (Z_crop, Y_crop, X_crop)
        #     if not (crop==self.crop_guess):
        #         print('crop', crop)
        #     assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
        # else:
        #     zi_up = self.zi

        if hyp.do_occ:
            occ_memX0_sup, free_memX0_sup, occ_memXs, free_memXs = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                # self.Z4, self.Y4, self.X4,
                self.Z2, self.Y2, self.X2,
                agg=True)
            vis_memX0 = (occ_memXs[:,0] + free_memXs[:,0]).clamp(0, 1)
            occ_memX0_sup = occ_memX0_sup[:,:,
                                          crop[0]:-crop[0],
                                          crop[1]:-crop[1],
                                          crop[2]:-crop[2]]
            free_memX0_sup = free_memX0_sup[:,:,
                                            crop[0]:-crop[0],
                                            crop[1]:-crop[1],
                                            crop[2]:-crop[2]]
            vis_memX0 = vis_memX0[:,:,
                                  crop[0]:-crop[0],
                                  crop[1]:-crop[1],
                                  crop[2]:-crop[2]]
            occ_loss, occ_memX0_pred = self.occnet(
                feat_memX0, 
                occ_memX0_sup,
                free_memX0_sup,
                # altvalid_memX0, 
                valid_memX0, 
                self.summ_writer)
            total_loss += occ_loss

            # occ_memi_sup, free_memi_sup, _, _ = self.vox_util.prep_occs_supervision(
            #     self.camis_T_camXs,
            #     self.xyz_camXs,
            #     self.Z2, self.Y2, self.X2,
            #     agg=True)
            # # occ_memi_sup = occ_memi_sup[:,:,
            # #                             self.crop_guess[0]:-self.crop_guess[0],
            # #                             self.crop_guess[1]:-self.crop_guess[1],
            # #                             self.crop_guess[2]:-self.crop_guess[2]]
            # # free_memi_sup = free_memi_sup[:,:,
            # #                               self.crop_guess[0]:-self.crop_guess[0],
            # #                               self.crop_guess[1]:-self.crop_guess[1],
            # #                               self.crop_guess[2]:-self.crop_guess[2]]
            # occ_loss, occ_memi_pred = self.occnet(
            #     zi_up, 
            #     occ_memi_sup,
            #     free_memi_sup,
            #     torch.ones_like(free_memi_sup),
            #     self.summ_writer)

            
            # print('occ total_loss', total_loss.detach().cpu().numpy())


        if hyp.do_emb3D:
            # compute 3D ML
            
            # print('feat_memX0', feat_memX0.shape)
            # print('altfeat_memX0', altfeat_memX0.shape)
            # print('valid_memX0', valid_memX0.shape)
            # print('altvalid_memX0', altvalid_memX0.shape)
            
            emb_loss_3D = self.embnet3D(
                feat_memX0,
                altfeat_memX0,
                valid_memX0.round(),
                (vis_memX0*altvalid_memX0).round(),
                self.summ_writer)
            total_loss += emb_loss_3D
                
        # lrtlist_camX0 = self.lrtlist_camX0s[:,0]
        # lrtlist_memX0 = self.vox_util.apply_mem_T_ref_to_lrtlist(
        #     lrtlist_camX0, self.Z, self.Y, self.X)
        # lrtlist_camX0_new = self.vox_util.apply_ref_T_mem_to_lrtlist(
        #     lrtlist_memX0, self.Z, self.Y, self.X)
        # print('lrtlist_camX0', lrtlist_camX0[0,0])
        # print('lrtlist_memX0', lrtlist_memX0[0,0])
        # print('now back:')
        # print('lrtlist_camX0', lrtlist_camX0_new[0,0])
        # print('diff', torch.sum(torch.abs(lrtlist_camX0-lrtlist_camX0_new)).detach().cpu().numpy())
            
        if hyp.do_center:
            # this net achieves the following:
            # objectness: put 1 at each object center and 0 everywhere else
            # orientation: at the object centers, classify the orientation into a rough bin
            # size: at the object centers, regress to the object size

            lrtlist_camX0 = self.lrtlist_camX0s[:,0]
            lrtlist_memX0 = self.vox_util.apply_mem_T_ref_to_lrtlist(
                lrtlist_camX0, self.Z, self.Y, self.X)
            scorelist = self.scorelist_s[:,0]
            
            center_loss, lrtlist_camX0_e, scorelist_e = self.centernet(
                feat_memX0, 
                crop,
                self.vox_util, 
                self.center_mask,
                lrtlist_camX0,
                lrtlist_memX0,
                scorelist, 
                self.summ_writer)
            total_loss += center_loss
            print('cen total_loss', total_loss.detach().cpu().numpy())

            if lrtlist_camX0_e is not None:
                # lrtlist_camX_e = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camX0[:,0], lrtlist_camX0_e)
                # lrtlist_camR_e = utils_geom.apply_4x4_to_lrtlist(self.camRs_T_camXs[:,0], lrtlist_camXs_e)
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_e',
                    self.rgb_camXs[0:1,0],
                    lrtlist_camX0_e[0:1], 
                    scorelist_e[0:1],
                    torch.arange(50).reshape(1, 50).long().cuda(), # tids
                    self.pix_T_cams[0:1,0])
                self.summ_writer.summ_lrtlist(
                    'center/boxlist_g',
                    self.rgb_camXs[0:1,0],
                    self.lrtlist_camXs[0:1,0],
                    self.scorelist_s[0:1,0],
                    self.tidlist_s[0:1,0],
                    self.pix_T_cams[0:1,0])
            
            
        if hyp.do_seg:
            seg_camXs = feed['seg_camXs']
            self.summ_writer.summ_seg('seg/seg_camX0', seg_camXs[:,0])
            seg_memX0 = utils_misc.parse_seg_into_mem(
                seg_camXs, self.num_seg_labels, self.occ_memX0s,
                self.pix_T_cams, self.camX0s_T_camXs, self.vox_util)
            seg_memX0 = seg_memX0[:,
                                  crop[0]:-crop[0],
                                  crop[1]:-crop[2],
                                  crop[2]:-crop[2]]
            seg_memX0_vis = torch.max(seg_memX0, dim=2)[0]
            # self.summ_writer.summ_seg('seg/seg_memX0_vis', seg_memX0_vis)

            # occ_memX0 = torch.max(self.occ_memXs, dim=1)[0]
            # occ_memX0 = occ_memX0[:,:,
            #                       crop[0]:-crop[0],
            #                       crop[1]:-crop[2],
            #                       crop[2]:-crop[2]]
            
            seg_loss, seg_memX0_pred = self.segnet(
                feat_memX0, 
                seg_memX0,
                # occ_memX0,
                occ_memX0_sup,
                free_memX0_sup,
                self.summ_writer)
            total_loss += seg_loss
            
            print('seg total_loss', total_loss.detach().cpu().numpy())
            
        if hyp.do_view:
            # assert(hyp.do_feat3D)
            # let's try to predict view1

            # Z_crop, Y_crop, X_crop = self.crop_guess
            # zi_up_pad = F.pad(zi_up, (Z_crop, Z_crop, Y_crop, Y_crop, X_crop, X_crop), 'constant', 0)
            
            # # decode the feat volume into an image
            # view_loss, rgb_e, view_camX0 = self.viewnet(
            #     self.pix_T_cams[:,0],
            #     self.camX0s_T_camXs[:,0],
            #     # feat_memX0_pad,
            #     feat_memX0,
            #     self.rgb_camXs[:,0],
            #     self.vox_util,
            #     valid=self.valid_camXs[:,0],
            #     summ_writer=self.summ_writer)
            # total_loss += view_loss

            # self.zi = utils_basic.l2_normalize(self.zi, dim=1)
            # self.summ_writer.summ_feat('3D_feats/zi', self.zi, pca=True)
            # self.summ_writer.summ_histogram('3D_feats/zi_hist', self.zi)
            
            # decode zi into an image
            view_loss, rgb_e, view_camX0 = self.viewnet(
                self.pix_T_cams[:,0],
                # self.camX0s_T_camXs[:,0],
                self.camXs_T_camis[:,0],
                self.zi,
                # zi_up,
                # zi_up_pad,
                self.rgb_camXs[:,0],
                self.vox_util,
                # valid=(0.5+ 0.5*self.valid_camXs[:,0]),
                valid=self.valid_camXs[:,0],
                summ_writer=self.summ_writer)
            total_loss += view_loss
            
        # if hyp.do_emb2D:
        #     assert(hyp.do_view)
        #     # create an embedding image, representing the bottom-up 2D feature tensor
        #     emb_loss_2D, _ = self.embnet2D(
        #         view_camX0,
        #         feat_camX0,
        #         torch.ones_like(view_camX0[:,0:1]),
        #         self.summ_writer)
        #     total_loss += emb_loss_2D
            
        # if hyp.do_emb3D:
        #     # compute 3D ML
        #     emb_loss_3D = self.embnet3D(
        #         feat_memR,
        #         altfeat_memR,
        #         valid_memR.round(),
        #         torch.ones_like(valid_memR),
        #         self.summ_writer)
        #     total_loss += emb_loss_3D

        # if hyp.do_linclass:
            
        #     masklist_memR = self.vox_util.assemble_padded_obj_masklist(
        #         self.lrtlist_camRs[:,0], self.scorelist_s[:,0], self.Z2, self.Y2, self.X2, coeff=1.2)
        #     mask_memR = torch.sum(masklist_memR, dim=1)
        #     self.summ_writer.summ_oned('obj/mask_memR', mask_memR.clamp(0, 1), bev=True)

        #     occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
        #     occ_memR = torch.sum(occ_memRs, dim=1).clamp(0,1)
            
        #     obj_memR = (mask_memR * occ_memR).view(-1)
        #     bkg_memR = ((1.0 - mask_memR) * occ_memR).view(-1)
        #     obj_inds = torch.nonzero(obj_memR, as_tuple=False)
        #     # obj_inds = obj_inds.detach().cpu().numpy()
        #     bkg_inds = torch.nonzero(bkg_memR, as_tuple=False)
        #     # bkg_inds = bkg_inds.detach().cpu().numpy()

        #     # print('altfeat_memR', altfeat_memR.shape)
        #     # print('mask_memR', mask_memR.shape)
        #     # print('occ_memR', occ_memR.shape)
        #     # print('%d obj_inds; %d bkg_inds' % (len(obj_inds), len(bkg_inds)))
        #     # input()
            
        #     code_vec = altfeat_memR.detach().permute(0,2,3,4,1).reshape(-1, hyp.feat3D_dim)
        #     obj_inds = obj_inds.reshape([-1])
        #     bkg_inds = bkg_inds.reshape([-1])

        #     if len(obj_inds) and len(bkg_inds):

        #         # print('obj_inds', obj_inds.shape)
        #         # print('bkg_inds', bkg_inds.shape)
        #         # print('codes_flat', codes_flat.shape)

        #         linclass_loss = self.linclassnet(
        #             code_vec, obj_inds, bkg_inds, self.summ_writer)

        #         # print('feat_memR', feat_memR.shape)
        #         # print('mask_memR', mask_memR.shape)
                
        #         total_loss += linclass_loss


        if hyp.do_render:
            # assert(hyp.do_feat)
            # we warped the features into the canonical view
            # now we resample to the target view anode

            # Z_crop, Y_crop, X_crop = self.crop_guess
            # zi_up_pad = F.pad(zi_up, (Z_crop, Z_crop, Y_crop, Y_crop, X_crop, X_crop), 'constant', 0)
            # occ_pad = F.pad(occ_memi_pred, (Z_crop, Z_crop, Y_crop, Y_crop, X_crop, X_crop), 'constant', 0)

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
            assert(sy==0.5 or sy==1.0)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            # assert(S==2) # else we should warp each feat in 1:

            feat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], zi_up[:,1:],
                # projpix_T_cams[:,0], self.camXs_T_camis[:,0], zi_up,
                hyp.view_depth, PH, PW, noise_amount=1.0)
            occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], zi_up[:,0:1],
                hyp.view_depth, PH, PW, grid_z_vec=dists)

            if sx==0.5:
                rgb_X00 = utils_basic.downsample(self.rgb_camXs[:,0], 2)
                valid_X00 = utils_basic.downsample(self.valid_camXs[:,0], 2)
            else:
                rgb_X00 = self.rgb_camXs[:,0]
                valid_X00 = self.valid_camXs[:,0]
            
            # print('dists', dists.detach().cpu().numpy())

            # dists = torch.linspace(
            #     self.vox_util.ZMIN,
            #     self.vox_util.ZMAX,
            #     steps=hyp.view_depth,
            #     dtype=torch.float32,
            #     device=torch.device('cuda'))
            
            # decode the perspective volume into an image
            render_loss, rgb_e = self.rendernet(
                feat_proj,
                occ_proj,
                rgb_X00,
                dists, 
                valid=valid_X00,
                summ_writer=self.summ_writer)
            total_loss += render_loss
        
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        assert(hyp.do_feat3D)
        def crop_feat(feat_pad):
            Z_pad, Y_pad, X_pad = self.crop_guess
            feat = feat_pad[:,:,
                            Z_pad:-Z_pad,
                            Y_pad:-Y_pad,
                            X_pad:-X_pad].clone()
            return feat
        def pad_feat(feat):
            Z_pad, Y_pad, X_pad = self.crop_guess
            feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
            return feat_pad
        
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.num_tries = 4
        t_noise = torch.randn([self.B, self.num_tries, 3]).float().cuda()
        t_noise[:,0] = 0.0
        t_noise[:,:,1] = 0.0 # set the Y part to be zero, so we only propose on the road

        tr_ious = torch.zeros([self.B, self.num_tries, self.S]).float().cuda()
        tr_diff_3ds = torch.zeros([self.B, self.num_tries, self.S]).float().cuda()
        
        self.render_centroid = utils_geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0] + t_noise[:,1]*0.1

        for tr in list(range(self.num_tries)):
            
            # add noise to the gt
            lrt_proposal = self.lrt_camXs[:,0]
            l, rt = utils_geom.split_lrt(lrt_proposal)
            r, t = utils_geom.split_rt(rt)
            t = t + t_noise[:,tr]
            rt = utils_geom.merge_rt(r, t)
            lrt_proposal = utils_geom.merge_lrt(l, rt)

            # center on the proposal, so that it does not fall out of bounds
            self.scene_centroid = utils_geom.get_clist_from_lrtlist(lrt_proposal.unsqueeze(1))[:,0]
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

            self.original_centroid = self.scene_centroid.clone()

            self.rgb_memXs0 = self.vox_util.unproject_rgb_to_mem(
                self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            self.rgb_memX0s0= self.vox_util.apply_4x4_to_vox(self.camX0s_T_camXs[:,0], self.rgb_memXs0)
            self.occ_memX0s0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z, self.Y, self.X)

            feat_memX0_input = torch.cat([
                self.occ_memX0s0,
                self.rgb_memX0s0*self.occ_memX0s0,
            ], dim=1)
            _, feat_memX0, valid_memX0 = self.featnet3D(feat_memX0_input)


            feat_memX0 = feat_memX0.detach()
            B, C, Z_, Y_, X_ = list(feat_memX0.shape)
            S = self.S
            # print('feat_memX0', feat_memX0.shape)

            # feat_memX0 is at half res, and it's cropped a bit
            # instead of cropping everything from now on,
            # let's just pad the feat
            ## at this point, i take that back, let's crop appropriately
            Z_pad, Y_pad, X_pad = self.crop_guess
            Z = Z_ + Z_pad*2
            Y = Y_ + Y_pad*2
            X = X_ + X_pad*2

            obj_mask_memX0s0 = self.vox_util.assemble_padded_obj_masklist(
                lrt_proposal.unsqueeze(1),
                self.score_s[:,0:1],
                Z, Y, X).squeeze(1)
            # # only take the occupied voxels
            # occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], Z, Y, X)
            # obj_mask_memX0s0 = obj_mask_memX0s0
            # occ_obj_mask_memX0 = obj_mask_memX0s0 * occ_memX0
            occ_obj_mask_memX0 = obj_mask_memX0s0.clone()
            print('using full obj mask, in case E too few points')

            # occ_memX0 = occ_memX0[:,:,
            #                       Z_pad:-Z_pad,
            #                       Y_pad:-Y_pad,
            #                       X_pad:-X_pad]
            occ_obj_mask_memX0 = occ_obj_mask_memX0[:,:,
                                                    Z_pad:-Z_pad,
                                                    Y_pad:-Y_pad,
                                                    X_pad:-X_pad]
            
            # for b in list(range(self.B)):
            #     if torch.sum(occ_obj_mask_memX0[b]) <= 8:
            #         print('returning early, since there are not enough valid object points')
            #         return total_loss, results, True

            self.summ_writer.summ_rgb('2D_inputs/rgb_camX0_%d' % tr, self.rgb_camXs[:,0])
            self.summ_writer.summ_feat('3D_feats/feat_memX0_input_%d' % tr, feat_memX0_input, pca=True)
            self.summ_writer.summ_feat('3D_feats/feat_memX0_%d' % tr, feat_memX0, pca=True)

            # compute a slightly wider mask, to help us delete the object
            obj_mask_memX0 = self.vox_util.assemble_padded_obj_masklist(
                lrt_proposal.unsqueeze(1),
                torch.ones((self.B, 1), dtype=torch.float32).cuda(),
                self.Z, self.Y, self.X, coeff=1.2).squeeze(1)
            # compute a very wide mask, to help us focus on the object
            obj_widemask_memX0 = self.vox_util.assemble_padded_obj_masklist(
                lrt_proposal.unsqueeze(1),
                torch.ones((self.B, 1), dtype=torch.float32).cuda(),
                self.Z, self.Y, self.X, coeff=1.2, additive_coeff=0.5).squeeze(1)
            bkg_memX0_input = feat_memX0_input * (1.0 - obj_mask_memX0)
            _, bkg_memX0, valid_memX0 = self.featnet3D(bkg_memX0_input)
            self.summ_writer.summ_feat('3D_feats/bkg_memX0_input_%d' % tr, bkg_memX0_input, pca=True)
            self.summ_writer.summ_feat('3D_feats/bkg_memX0_%d' % tr, bkg_memX0, pca=True)

            obj_memX0 = feat_memX0.clone()
            obj_memX0_pad = pad_feat(obj_memX0)
            # # sharp-edge version:
            # obj_mask_halfmemX0_pad = F.interpolate(obj_widemask_memX0, scale_factor=0.5, mode='trilinear')
            # blurry-edge version:
            obj_mask_minimemX0_pad = F.interpolate(obj_widemask_memX0, scale_factor=0.125, mode='trilinear')
            obj_mask_halfmemX0_pad = F.interpolate(obj_mask_minimemX0_pad, scale_factor=4.0, mode='trilinear')

            obj_mask_halfmemX0 = crop_feat(obj_mask_halfmemX0_pad)
            valid_mask_halfmemX0 = torch.ones_like(obj_mask_halfmemX0)
            valid_mask_halfmemX0_pad = pad_feat(valid_mask_halfmemX0)

            proposed_feat_memX0 = bkg_memX0 * (1.0 - obj_mask_halfmemX0) + obj_mask_halfmemX0 * obj_memX0
            self.summ_writer.summ_feat('3D_feats/proposed_feat_memX0_%d' % tr, proposed_feat_memX0, pca=True)




            feat0_vec = feat_memX0.reshape(B, hyp.feat3D_dim, -1)
            # this is B x C x huge
            feat0_vec = feat0_vec.permute(0, 2, 1)
            # this is B x huge x C

            obj_mask0_vec = occ_obj_mask_memX0.reshape(B, -1).round()
            # this is B x huge

            orig_xyz = utils_basic.gridcloud3D(B, Z, Y, X)
            # this is B x huge x 3
            orig_xyz = orig_xyz.reshape(B, Z, Y, X, 3)
            orig_xyz = orig_xyz[:,
                                Z_pad:-Z_pad,
                                Y_pad:-Y_pad,
                                X_pad:-X_pad]
            orig_xyz = orig_xyz.reshape(B, -1, 3)

            obj_lengths, cams_T_obj0 = utils_geom.split_lrtlist(lrt_proposal.unsqueeze(1))
            # this is B x S x 4 x 4
            obj_length = obj_lengths[:,0]
            cam0_T_obj = cams_T_obj0[:,0]
            # obj_T_cam0 = utils_geom.safe_inverse(cam0_T_obj)
            obj_T_cam0 = cam0_T_obj.inverse()

            mem_T_cam = self.vox_util.get_mem_T_ref(B, Z, Y, X)
            cam_T_mem = self.vox_util.get_ref_T_mem(B, Z, Y, X)

            lrt_camIs_g = self.lrt_camX0s.clone()
            lrt_camIs_e = torch.zeros_like(self.lrt_camX0s)
            mem0_T_memIs_e = torch.zeros((B, self.S, 4, 4), dtype=torch.float32).cuda()
            mem0_T_memIs_g = torch.zeros((B, self.S, 4, 4), dtype=torch.float32).cuda()
            # we will fill this up

            ious = torch.zeros([B, S]).float().cuda()
            diff_2ds = torch.zeros([B, S]).float().cuda()
            diff_3ds = torch.zeros([B, S]).float().cuda()
            point_counts = np.zeros([B, S])
            inb_counts = np.zeros([B, S])

            feat_vis = []
            occ_vis = []

            top3d_vis = []
            bot3d_vis = []
            top3d_occ_vis = []
            bot3d_occ_vis = []

            for s in list(range(self.S)):
                torch.cuda.empty_cache()
                # print('working on tr %d; s %d' % (tr, s))
                if not (s==0):
                    # remake the vox util and all the mem data
                    self.scene_centroid = utils_geom.get_clist_from_lrtlist(lrt_camIs_e[:,s-1:s])[:,0]
                    delta = self.scene_centroid - self.original_centroid
                    self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                        self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

                    self.occ_memX0s0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,s], self.Z, self.Y, self.X)
                    self.rgb_memXs0 = self.vox_util.unproject_rgb_to_mem(
                        self.rgb_camXs[:,s], self.Z, self.Y, self.X, self.pix_T_cams[:,s])
                    self.rgb_memX0s0 = self.vox_util.apply_4x4_to_vox(self.camX0s_T_camXs[:,s], self.rgb_memXs0)
                else:
                    delta = torch.zeros([B, 3]).float().cuda()
                occ_vis.append(self.summ_writer.summ_occ('', self.occ_memX0s0, only_return=True))

                inb = self.vox_util.get_inbounds(self.xyz_camX0s[:,s], self.Z2, self.Y2, self.X2, already_mem=False)
                num_inb = torch.sum(inb.float(), axis=1)
                inb_counts[:, s] = num_inb.cpu().numpy()

                feat_memI_input = torch.cat([
                    self.occ_memX0s0,
                    self.rgb_memX0s0*self.occ_memX0s0,
                ], dim=1)
                _, feat_memI, valid_memI = self.featnet3D(feat_memI_input)

                feat_memI = feat_memI.detach()

                feat_vec = feat_memI.reshape(B, hyp.feat3D_dim, -1)
                # this is B x C x huge
                feat_vec = feat_vec.permute(0, 2, 1)
                # this is B x huge x C

                memI_T_mem0 = utils_geom.eye_4x4(B)
                # we will fill this up

                # to simplify the impl, we will iterate over the batchmin
                for b in list(range(B)):
                    feat_vec_b = feat_vec[b]
                    feat0_vec_b = feat0_vec[b]
                    obj_mask0_vec_b = obj_mask0_vec[b]
                    orig_xyz_b = orig_xyz[b]
                    # these are huge x C

                    obj_inds_b = torch.where(obj_mask0_vec_b > 0)
                    obj_vec_b = feat0_vec_b[obj_inds_b]
                    xyz0 = orig_xyz_b[obj_inds_b]
                    # these are med x C

                    obj_vec_b = obj_vec_b.permute(1, 0)
                    # this is is C x med

                    corr_b = torch.matmul(feat_vec_b.detach(), obj_vec_b.detach())
                    # this is huge x med

                    heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z_, Y_, X_)
                    # we need to pad this, because we are about to take the argmax and interpret it as xyz
                    heat_b = F.pad(heat_b, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
                    # this is med x 1 x Z x Y x X

                    # for numerical stability, we sub the max, and mult by the resolution
                    heat_b_ = heat_b.reshape(-1, Z*Y*X)
                    heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    heat_b = heat_b - heat_b_max
                    heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

                    xyzI = utils_basic.argmax3D(heat_b*float(Z*10), hard=False, stack=True)
                    # this is med x 3

                    xyzI_cam = self.vox_util.Mem2Ref(xyzI.unsqueeze(1), Z, Y, X)
                    xyzI_cam += delta
                    xyzI = self.vox_util.Ref2Mem(xyzI_cam, Z, Y, X).squeeze(1)

                    memI_T_mem0[b] = utils_track.rigid_transform_3D(xyz0, xyzI)

                    # record #points, since ransac depends on this
                    point_counts[b, s] = len(xyz0)
                # done stepping through batch

                mem0_T_memI = utils_geom.safe_inverse(memI_T_mem0)
                cam0_T_camI = utils_basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

                # eval
                camI_T_obj = utils_basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
                # this is B x 4 x 4
                lrt_camIs_e[:,s] = utils_geom.merge_lrt(obj_length, camI_T_obj)

                mem0_T_memIs_e[:,s] = mem0_T_memI
                memX0_T_memY0 = mem0_T_memIs_e[:,s]
                memY0_T_memX0 = utils_geom.safe_inverse(memX0_T_memY0)


                # reset vox util to use the render centroid, for the top/bottom rendering
                self.scene_centroid = self.render_centroid.clone()
                self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
                                                  self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                # compute the bottom-up scene tensor with this data
                rgb_memX0i = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_camXs[:,s], self.Z, self.Y, self.X, self.pix_T_cams[:,s])
                self.rgb_memX0i = self.vox_util.apply_4x4_to_vox(self.camX0s_T_camXs[:,s], self.rgb_memXs0)
                self.occ_memX0i = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,s], self.Z, self.Y, self.X)
                feat_memX0i_input = torch.cat([
                    self.occ_memX0i,
                    self.rgb_memX0i*self.occ_memX0i,
                ], dim=1)
                _, feat_memX0i, _ = self.featnet3D(feat_memX0i_input)
                bot3d_vis.append(self.summ_writer.summ_feat('', feat_memX0i, pca=True, only_return=True))
                _, occ_memX0i_pred = self.occnet(feat_memX0i)
                occ_memX0i = F.sigmoid(occ_memX0i_pred)
                bot3d_occ_vis.append(self.summ_writer.summ_occ('', occ_memX0i, only_return=True))


                # compute top-down with the argmax
                scene_memY0, obj_mask_memY0 = utils_misc.assemble_hypothesis(
                    memY0_T_memX0, bkg_memX0, obj_memX0, obj_mask_halfmemX0, self.vox_util, self.crop_guess)
                top3d_vis.append(self.summ_writer.summ_feat('', scene_memY0, pca=True, only_return=True))

                _, occ_memY0_pred = self.occnet(scene_memY0)
                occ_memY0 = F.sigmoid(occ_memY0_pred)
                top3d_occ_vis.append(self.summ_writer.summ_occ('', occ_memY0, only_return=True))

                # diff_3d = utils_basic.reduce_masked_mean(torch.norm(feat_memX0i - scene_memY0, dim=1).unsqueeze(1), acc2d_bot)

                # # full scene diff
                diff_3d = torch.mean(torch.norm(feat_memX0i - scene_memY0, dim=1))
                # print(scene_memY0.shape)
                # print(obj_mask_memY0.shape)
                # diff_3d = utils_basic.reduce_masked_mean(torch.norm(feat_memX0i - scene_memY0, dim=1, keepdim=True), obj_mask_memY0)
                diff_3ds[:,s] = diff_3d

                ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)
                # print('iou is', ious[:,s].detach().cpu().numpy())
            results['ious'] = ious
            tr_ious[:,tr] = ious
            tr_diff_3ds[:,tr] = diff_3ds
            
            self.summ_writer.summ_rgbs('track/top3d_vis_%d' % tr, top3d_vis)
            self.summ_writer.summ_rgbs('track/bot3d_vis_%d' % tr, bot3d_vis)
            self.summ_writer.summ_rgbs('track/bot3d_occ_vis_%d' % tr, bot3d_occ_vis)
            self.summ_writer.summ_rgbs('track/top3d_occ_vis_%d' % tr, top3d_occ_vis)

            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%d_%02d' % (tr, s), torch.mean(ious[:,s]).cpu().item())
                self.summ_writer.summ_scalar('track/mean_diff_2d_%d_%02d' % (tr, s), torch.mean(diff_2ds[:,s]).cpu().item())
                self.summ_writer.summ_scalar('track/mean_diff_3d_%d_%03d' % (tr, s), torch.mean(diff_3ds[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou_%d' % tr, torch.mean(ious).cpu().item())
            self.summ_writer.summ_scalar('track/point_counts_%d' % tr, np.mean(point_counts))
            self.summ_writer.summ_scalar('track/inb_counts_%d' % tr, np.mean(inb_counts))

            results['diff_2ds'] = diff_2ds
            results['diff_3ds'] = diff_3ds

            lrt_camX0s_e = lrt_camIs_e.clone()
            lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)

            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_e_%d' % (tr, s), self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e_%d' % tr, visX_e)
            
            if tr==0:
                visX_g = []
                for s in list(range(self.S)):
                    visX_g.append(self.summ_writer.summ_lrtlist(
                        'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                        self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
                self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)


            obj_clist_camX0_e = utils_geom.get_clist_from_lrtlist(lrt_camX0s_e)
            obj_clist_camX0_g = utils_geom.get_clist_from_lrtlist(self.lrt_camX0s)

            dists = torch.norm(obj_clist_camX0_e - obj_clist_camX0_g, dim=2)
            # this is B x S
            mean_dist = utils_basic.reduce_masked_mean(dists, self.score_s)
            median_dist = utils_basic.reduce_masked_median(dists, self.score_s)
            # this is []
            self.summ_writer.summ_scalar('track/centroid_dist_mean_%d' % tr, mean_dist.cpu().item())
            self.summ_writer.summ_scalar('track/centroid_dist_median_%d' % tr, median_dist.cpu().item())

            self.summ_writer.summ_traj_on_occ('track/traj_e_%d' % tr,
                                              obj_clist_camX0_e, 
                                              self.occ_memX0s0,
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)
            if tr==0:
                self.summ_writer.summ_traj_on_occ('track/traj_g',
                                                  obj_clist_camX0_g,
                                                  self.occ_memX0s0,
                                                  self.vox_util, 
                                                  already_mem=False,
                                                  sigma=2)
            total_loss += mean_dist # we won't backprop, but it's nice to plot and print this anyway

        # print('tr_diff_3ds', tr_diff_3ds.detach().cpu().numpy())
        # print('tr_ious', tr_ious.detach().cpu().numpy())

        # tr_diff_3ds is B x N x S 
        # tr_ious is B x N x S
        
        tr_mean_diffs = torch.mean(tr_diff_3ds, dim=2)
        # this is B x N
        tr_mean_ious = torch.mean(tr_ious, dim=2)
        # B x N

        print('tr_mean_diffs', tr_mean_diffs.detach().cpu().numpy())
        print('tr_mean_ious', tr_mean_ious.detach().cpu().numpy())
        
        tr_mean_ious = tr_mean_ious[0] # N
        tr_mean_diffs = tr_mean_diffs[0] # N

        min_ind = torch.argmin(tr_mean_diffs, dim=0)
        best_ind = torch.argmax(tr_mean_ious, dim=0)
        worst_ind = torch.argmin(tr_mean_ious, dim=0)
        
        selected_iou = tr_mean_ious[min_ind]
        best_iou = tr_mean_ious[best_ind]
        worst_iou = tr_mean_ious[worst_ind]
        avg_iou = torch.mean(tr_mean_ious)

        results['selected_iou'] = selected_iou
        results['best_iou'] = best_iou
        results['worst_iou'] = worst_iou
        results['avg_iou'] = avg_iou
        
        
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_propose(self, feed):
        assert(hyp.do_feat3D)
        assert(hyp.do_occ)
        assert(self.B==1)

        scene_centroid_x = 0.0
        scene_centroid_y = 1.5 # 1.0 is a bit too high up
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)


        def crop_feat(feat_pad):
            Z_pad, Y_pad, X_pad = self.crop_guess
            feat = feat_pad[:,:,
                            Z_pad:-Z_pad,
                            Y_pad:-Y_pad,
                            X_pad:-X_pad].clone()
            return feat
        def pad_feat(feat):
            Z_pad, Y_pad, X_pad = self.crop_guess
            feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
            return feat_pad
        
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)


        self.rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
            self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
        self.rgb_memX0 = self.vox_util.apply_4x4_to_vox(self.camX0s_T_camXs[:,0], self.rgb_memX0)
        self.occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z, self.Y, self.X)
        self.occ_halfmemX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z2, self.Y2, self.X2)


        # for a quick filtering step, how about:
        # for all boxes, if the centroid is unoccupied, drop it
        # the clean way to do this is probably: if the centroid is unoccupied, don't even propose it
        
        feat_memX0_input = torch.cat([
            self.occ_memX0,
            self.rgb_memX0*self.occ_memX0,
        ], dim=1)
        _, feat_memX0, _ = self.featnet3D(
            feat_memX0_input,
            self.summ_writer,
        )
        self.summ_writer.summ_feat('3D_feats/feat_memX0_input', feat_memX0_input, pca=True)
        self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, pca=True)

        occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
            self.camX0s_T_camXs[:,0:1],
            self.xyz_camXs[:,0:1],
            self.Z2, self.Y2, self.X2,
            agg=True)
        # occ_memX0_sup = crop_feat(occ_memX0_sup)
        # free_memX0_sup = crop_feat(free_memX0_sup)
        _, occ_memX0_pred = self.occnet(feat_memX0)
        occ_memX0 = F.sigmoid(occ_memX0_pred)
        # occ_memX0 = (occ_memX0 + occ_memX0_sup).clamp(0,1) * (1.0 - free_memX0_sup)
        # # occ_memX0 = (occ_memX0 + occ_memX0_sup).clamp(0,1)
        self.summ_writer.summ_occ('3D_feats/occ_memX0', occ_memX0)

        # occ_memX0 = occ_memX0_sup.clone()


        # # a thing i should try here is to get boxes via connected components
        # # ok = utils_misc.get_boxes_from_flow_mag(occ_memX0, 10)
        # image, boxlist_memX0, scorelist, tidlist, _ = utils_misc.get_boxes_from_flow_mag(occ_memX0.squeeze(1), 10)
        # # ok = get_boxes_from_flow_mag(flow_mag, N)
        # print('image', image.shape)

        # lrtlist_mem = utils_geom.convert_boxlist_to_lrtlist(boxlist_memX0)
        # # print('lrtlist_mem', lrtlist_mem.shape)
        # lrtlist_cam = self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, self.Z2, self.Y2, self.X2)

        # self.summ_writer.summ_lrtlist(
        #     'proposals/all', self.rgb_camXs[:,0],
        #     lrtlist_cam,
        #     torch.ones_like(scorelist),
        #     tidlist,
        #     self.pix_T_cams[:,0])
        # self.summ_writer.summ_lrtlist_bev(
        #     'proposals/all_bev',
        #     occ_memX0, 
        #     # self.occ_halfmemX0,
        #     lrtlist_cam,
        #     torch.ones_like(scorelist),
        #     tidlist,
        #     self.vox_util)
        
        # self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        # return total_loss, results, False
        
        
        ok_memX0 = occ_memX0 > 0.7
        # ok_memX0_pad = pad_feat(ok_memX0)
        # # now figure out the inds
        
        self.summ_writer.summ_occ('3D_feats/ok_memX0', ok_memX0)

        if torch.sum(ok_memX0) == 0.0:
            print('returning early')
            return total_loss, results, True

        xyz_halfmem = utils_basic.meshgrid3D(self.B, self.Z2, self.Y2, self.X2, stack=True)
        xyz_halfmem = xyz_halfmem.permute(0, 4, 1, 2, 3)
        # xyz_halfmem = crop_feat(xyz_halfmem)
        ok_halfmem = ok_memX0.reshape(1, -1)

        centroids_mem = xyz_halfmem.reshape(self.B, 3, -1)
        centroids_mem = centroids_mem.permute(0, 2, 1)
        # this is B x N x 3
        centroids_mem = centroids_mem[ok_halfmem]
        centroids_mem = centroids_mem.reshape(self.B, -1, 3)
        print('trimmed centroids_mem', centroids_mem.shape)

        perm = np.random.permutation(centroids_mem.shape[1])
        centroids_mem = centroids_mem[:,perm]
        # centroids_mem = centroids_mem[:,::32]
        print('trimmed2 centroids_mem', centroids_mem.shape)

        if centroids_mem.shape[1] > 100:
            centroids_mem = centroids_mem[:,:100]
            print('trimmed3 centroids_mem', centroids_mem.shape)

        centroids = self.vox_util.Mem2Ref(centroids_mem, self.Z2, self.Y2, self.X2)

        # N = self.Z2*self.Y2*self.X2
        # now i have a centroid at each voxel that i can featurize well
        # let's turn these into boxes
        # lens = torch.ones_like(centroids)*8.0 # my guess is the objects are 8 voxels big

        # the mean is something like this, in meters:
        # 2.1934016, 2.1815984, 4.6390057

        lx = torch.ones_like(centroids[:,:,0])*2.2
        ly = torch.ones_like(centroids[:,:,0])*2.2
        lz = torch.ones_like(centroids[:,:,0])*4.6
        lens = torch.stack([lx, ly, lz], dim=2)
        
        # rots = torch.zeros_like(centroids)
        # rots = torch.zeros_like(centroids)

        rx = torch.zeros_like(centroids[:,:,0])
        ry = torch.zeros_like(centroids[:,:,0]) + np.pi/2.0
        rz = torch.zeros_like(centroids[:,:,0])
        rots = torch.stack([rx, ry, rz], dim=2)

        rots = rots * 0.0

        # centroids = torch.cat([centroids, centroids, centroids, centroids, centroids], dim=1)
        # lens = torch.cat([lens, lens, lens, lens, lens], dim=1)
        # rots = torch.cat([rots*0.0, rots*0.25, rots*0.5, rots*0.75, rots], dim=1)
        
        # centroids = torch.cat([centroids, centroids], dim=1)
        # lens = torch.cat([lens, lens*0.75], dim=1)
        # rots = torch.cat([rots, rots], dim=1)
        
        # rots = utils_geom.eye_4x4(self.B*self.N)
        # rots = utils_geom.eye_4x4(self.B*self.N)
        # rots = rots.reshape(self.B, self.N, 4, 4)
        boxlist_cam = torch.cat([centroids, lens, rots], dim=2)

        N = boxlist_cam.shape[1]
        
        # print('boxlist_mem', boxlist_mem.shape)
        # this is B x N x 9
        scorelist = torch.ones_like(boxlist_cam[:,:,0])
        tidlist = torch.ones_like(boxlist_cam[:,:,0])
        tidlist = torch.arange(N).reshape(1, N).long().cuda()
        
        # lrtlist_mem = utils_geom.convert_boxlist_to_lrtlist(boxlist_mem)
        # print('lrtlist_mem', lrtlist_mem.shape)
        # lrtlist_cam = self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, self.Z2, self.Y2, self.X2)

        lrtlist_cam = utils_geom.convert_boxlist_to_lrtlist(boxlist_cam)
        
        # masklist_1 = self.vox_util.assemble_padded_obj_masklist(
        #     lrtlist_cam, scorelist, self.Z2, self.Y2, self.X2, coeff=0.8)
        # # this is B x K x 1 x Z2 x Y2 x X2
        # masklist_ = torch.sum(masklist_1).clamp(0,1)
        # self.summ_writer.summ_occ('proposals/masklist_', masklist_)

        scores = torch.zeros_like(scorelist)
        for n in list(range(N)):
            mask_1 = self.vox_util.assemble_padded_obj_masklist(
                lrtlist_cam[:,n:n+1], scorelist[:,n:n+1],
                self.Z2, self.Y2, self.X2, coeff=0.8).squeeze(1)
            # mask_2 = self.vox_util.assemble_padded_obj_masklist(
            #     lrtlist_cam[:,n:n+1], scorelist[:,n:n+1],
            #     self.Z2, self.Y2, self.X2, coeff=1.2).squeeze(1)
            mask_3 = self.vox_util.assemble_padded_obj_masklist(
                lrtlist_cam[:,n:n+1], scorelist[:,n:n+1],
                self.Z2, self.Y2, self.X2, coeff=1.8).squeeze(1)
            
            # center_mask = crop_feat(mask_1)
            # surround_mask = crop_feat(mask_3-mask_1).clamp(0,1)

            center_mask = mask_1.clone()
            surround_mask = (mask_3-mask_1).clamp(0,1)

            center_ = utils_basic.reduce_masked_mean(occ_memX0, center_mask, dim=[2,3,4])
            surround_ = utils_basic.reduce_masked_mean(occ_memX0, surround_mask, dim=[2,3,4])
            score_ = center_ - surround_
            score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)
            scores[:,n] = score_

            # print('mask_1', mask_1.shape)
            # mask_ = torch.sum(mask_1, dim=1).clamp(0,1)
            # print('mask_', mask_.shape)
            # if n < 10:
            #     self.summ_writer.summ_occ('proposals/mask_%d', mask_)

        # indices = np.argsort(pred_scores)[::-1]
        K = 10
        scorelist, indlist = torch.topk(scores[0], K)
        # scorelist = scorelist.unsqueeze(0)
        lrtlist_cam = lrtlist_cam[0,indlist]
        tidlist = tidlist[0,indlist]

        ## hard drop
        indlist = scorelist>0.52
        scorelist = scorelist[indlist]
        tidlist = tidlist[indlist]
        lrtlist_cam = lrtlist_cam[indlist]

        scorelist = scorelist.unsqueeze(0)
        tidlist = tidlist.unsqueeze(0)
        lrtlist_cam = lrtlist_cam.unsqueeze(0)
        
        
        # masklist_1 = utils_vox.assemble_padded_obj_masklist(
        #     lrtlist_camR, scorelist, Z2, Y2, X2, coeff=0.8)
        # masklist_2 = utils_vox.assemble_padded_obj_masklist(
        #     lrtlist_camR, scorelist, Z2, Y2, X2, coeff=1.2)
        # masklist_3 = utils_vox.assemble_padded_obj_masklist(
        #     lrtlist_camR, scorelist, Z2, Y2, X2, coeff=1.8)
        

        self.summ_writer.summ_lrtlist(
            'proposals/all', self.rgb_camXs[:,0],
            lrtlist_cam,
            torch.ones_like(scorelist),
            tidlist,
            self.pix_T_cams[:,0])
        self.summ_writer.summ_lrtlist_bev(
            'proposals/all_bev',
            occ_memX0, 
            # self.occ_halfmemX0,
            lrtlist_cam,
            torch.ones_like(scorelist),
            tidlist,
            self.vox_util)

        self.summ_writer.summ_lrtlist(
            'proposals/all_gt', self.rgb_camXs[:,0],
            self.full_lrtlist_camX0s[:,0],
            self.full_scorelist_s[:,0],
            self.full_tidlist_s[:,0],
            self.pix_T_cams[:,0])
        
        self.summ_writer.summ_lrtlist_bev(
            'proposals/all_gt_bev',
            self.occ_halfmemX0,
            self.full_lrtlist_camX0s[:,0],
            self.full_scorelist_s[:,0],
            self.full_tidlist_s[:,0],
            self.vox_util)
        
        lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
            lrtlist_cam, self.full_lrtlist_camX0s[:,0], scorelist, self.full_scorelist_s[:,0])

        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        maps = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
        for ind, overlap in enumerate(iou_thresholds):
            self.summ_writer.summ_scalar('ap/%.2f_iou' % overlap, np.squeeze(maps[ind]))

        # maps = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, ious)
        # # maps = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
        # print('maps', maps)
        # for ind, overlap in enumerate(ious):
        #     self.summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])
            

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
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
                return self.run_propose(feed)
                # return self.run_test(feed)

        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)
