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
from nets.feat3dnet import Feat3dNet
# from nets.embnet3D import EmbNet3D
from nets.occnet import OccNet
from nets.localdecodernet import LocalDecoderParent
# from nets.mocnet import MocNet
# from nets.viewnet import ViewNet
# from nets.mocnet3D import MocNet3D
import ipdb 
st = ipdb.set_trace
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# from utils.moc import MocTrainer
# from utils.basic import *
# import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.py
import utils.misc
import utils.track
import utils.vox

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_ZOOM(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaZoomModel()
        if hyp.do_feat3d and hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

        # if hyp.do_emb3D:
        #     # freeze the slow model
        #     self.model.feat3dnet_slow.eval()
        #     self.set_requires_grad(self.model.feat3dnet_slow, False)
            
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

        set_nums = []
        set_names = []
        set_batch_sizes = []
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

                    feed, _ = next(set_loader)
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
                    # st()
                    loss_py = loss.cpu().item()

                    if hyp.do_test and (not returned_early):
                        ious = results['ious']
                        # st()
                        ious = ious[0].cpu().numpy()

                        all_ious[test_count] = ious
                        test_count += 1
                        
                        # print('all_ious', all_ious[:test_count])

                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        print('mean_ious across %d tests' % test_count, mean_ious)

                        # boxlist_e = torch.stack([height_e, width_e, y_e, x_e], dim=2)
                        # boxlist_g = torch.stack([height_g, width_g, y_g, x_g], dim=2)
                        
                        boxlist_e = results['boxlist_e'].detach().cpu().numpy()
                        boxlist_g = results['boxlist_g'].detach().cpu().numpy()
                        
                        # print('boxlist_e', boxlist_e)
                        # print('boxlist_g', boxlist_g)
                        
                        # these contain height, width, y, x
                        # now let me write a txt file

                        boxlist_e = np.reshape(boxlist_e, [-1, 4])
                        boxlist_g = np.reshape(boxlist_g, [-1, 4])
                        
                        # output_filename = '%s_boxes2d' % hyp.name
                        
                        import os

                        fn_e = os.path.join('./outs/', hyp.name + '_%06d_boxes2d_e.json' % test_count)
                        fn_g = os.path.join('./outs/', hyp.name + '_%06d_boxes2d_g.json' % test_count)

                        def write_json(boxlist_e, fn_e, gt=False):
                            S = boxlist_e.shape[0]
                            with open(fn_e, 'w') as f:
                                # f.write('%s\n' % filename)
                                f.write('[\n')
                                f.write('    {\n')
                                f.write('        "frames": [\n')
                                for s in list(range(S)):
                                    f.write('            {\n')
                                    f.write('                "timestamp": %.1f,\n' % s)
                                    f.write('                "num": %d,\n' % s)
                                    f.write('                "class": "frame",\n')
                                    if gt:
                                        f.write('                "annotations": [\n')
                                    else:
                                        f.write('                "hypotheses": [\n')
                                    f.write('                    {\n')
                                    if gt:
                                        f.write('                        "dco": true,\n')
                                    f.write('                        "height": %.2f,\n' % boxlist_e[s,0])
                                    f.write('                        "width": %.2f,\n' % boxlist_e[s,1])
                                    f.write('                        "id": "XXX",\n' % boxlist_e[s,1])
                                    f.write('                        "y": %.2f,\n' % boxlist_e[s,2])
                                    f.write('                        "x": %.2f\n' % boxlist_e[s,3])
                                    f.write('                    }\n')
                                    f.write('                ]\n')
                                    if not s==S-1:
                                        f.write('            },\n')
                                    else:
                                        f.write('            }\n')
                                # f.write('            },\n')
                                f.write('        ],\n')
                                f.write('        "class": "video",\n')
                                f.write('        "filename": "%06d"\n' % test_count)
                                f.write('    }\n')
                                f.write(']\n')        
                        write_json(boxlist_e, fn_e, gt=False)
                        write_json(boxlist_g, fn_g, gt=True)

                                

                # "timestamp": 0.054, 
                # "num": 0, 
                # "class": "frame", 
                # "annotations": [
                #     {
                #         "dco": true, 
                #         "height": 31.0, 
                #         "width": 31.0, 
                #         "id": "sheldon", 
                #         "y": 105.0, 
                #         "x": 608.0 
                #     }
                # ]
                            

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0)):
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # if hyp.do_emb3D:
                    #     def update_slow_network(slow_net, fast_net, beta=0.999):
                    #         param_k = slow_net.state_dict()
                    #         param_q = fast_net.named_parameters()
                    #         for n, q in param_q:
                    #             if n in param_k:
                    #                 param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                    #         slow_net.load_state_dict(param_k)
                    #     update_slow_network(self.model.feat3dnet_slow, self.model.feat3dnet)
                        
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
            mean_all_ious = np.mean(mean_ious)
            print('all_mean_ious', mean_all_ious)
            st()
            print("done")
            
class CarlaZoomModel(nn.Module):
    def __init__(self):
        super(CarlaZoomModel, self).__init__()

        self.crop_guess = (18,18,18)
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_feat3docc: 
            self.feat3dnet_occ = Feat3dNet(in_dim=4)
            self.localdecodernet_render_occ = LocalDecoderParent()

        if hyp.do_occ:
            self.occnet = OccNet()
        # st()
        if hyp.do_localdecoder:
            self.localdecodernet = LocalDecoderParent()
        # if hyp.do_emb3D:
        #     self.embnet3D = EmbNet3D()
        #     # make a slow net
        #     self.feat3dnet_slow = Feat3dNet(in_dim=4)
        #     # init slow params with fast params
        #     self.feat3dnet_slow.load_state_dict(self.feat3dnet.state_dict())
            
    def crop_feat(self, feat_pad, crop):
        # return feat_pad
        Z_pad, Y_pad, X_pad = crop
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat

    def crop_pointcloud(self, xyz_camX, crop):
        # xyz_camX is BxNx3
        Z_pad, Y_pad, X_pad = crop
        
        xyz_memX = self.vox_util.Ref2Mem(xyz_camX, self.Z2, self.Y2, self.X2)
        B, N, _ = xyz_memX.shape
        xyz_memX = xyz_memX.reshape(-1, 3)
        x, y, z = torch.unbind(xyz_memX, dim=-1)
        
        x_valid = (x > X_pad).byte() & (x < self.X2 - X_pad).byte()
        y_valid = (y > Y_pad).byte() & (y < self.Y2 - Y_pad).byte()
        z_valid = (z > Z_pad).byte() & (z < self.Z2 - Z_pad).byte()

        inbounds = x_valid.byte() & y_valid.byte() & z_valid.byte()
        masklist = inbounds.float()
        # print(masklist.shape)
        masklist = masklist.reshape(B, N)
        return masklist

    
    def pad_feat(self, feat, crop):
        # return feat
        Z_pad, Y_pad, X_pad = crop
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad
    
    # def crop_feat(self, feat_pad):
    #     Z_pad, Y_pad, X_pad = self.crop_guess
    #     feat = feat_pad[:,:,
    #                     Z_pad:-Z_pad,
    #                     Y_pad:-Y_pad,
    #                     X_pad:-X_pad].clone()
    #     return feat
    # def pad_feat(self, feat):
    #     Z_pad, Y_pad, X_pad = self.crop_guess
    #     feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
    #     return feat_pad
    
    def prepare_common_tensors(self, feed, prep_summ=True):
        results = dict()
        
        if prep_summ:
            self.summ_writer = utils.improc.Summ_writer(
                writer=feed['writer'],
                global_step=feed['global_step'],
                log_freq=feed['set_log_freq'],
                fps=8,
                just_gif=feed['just_gif'],
            )
        else:
            self.summ_writer = None

        self.include_vis = True

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z1, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)
        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0 = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(utils.geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_camXs)))
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camX0_T_camR0 = utils.basic.matmul2(self.camX0s_T_camXs[:,0], self.camXs_T_camRs[:,0])
        self.camR0s_T_camXs = utils.basic.matmul2(self.camR0s_T_camRs, self.camRs_T_camXs)
        self.camX0s_T_camRs = utils.basic.matmul2(self.camX0_T_camR0, self.camR0s_T_camRs)

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils.geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        if hyp.pseudo_traj:
            if hyp.make_dense:
            # self.rgb_camRs = feed["rgb_camRs"]
                all_origin_T_camXs_ = feed['all_origin_T_camXs']
                all_xyz_camXs_ = feed['all_xyz_camXs']
                B,S,V,N,D = all_xyz_camXs_.shape

                all_xyz_camXs = all_xyz_camXs_.reshape(B,S*V,N,D)
                self.all_origin_T_camXs = all_origin_T_camXs_.reshape(B,S*V,4,4)
                # st()
                self.all_xyz_camOrigins_ = __u(utils.geom.apply_4x4(__p(self.all_origin_T_camXs), __p(all_xyz_camXs)))
                self.all_xyz_camOrigins__ = self.all_xyz_camOrigins_.reshape(B,S,V,N,D)
                self.all_xyz_camOrigins =  self.all_xyz_camOrigins__.reshape(B,S,V*N,D)

                self.camRs_T_origin = __u(utils.geom.safe_inverse(__p(self.origin_T_camRs)))
                self.camXs_T_origin = __u(utils.geom.safe_inverse(__p(self.origin_T_camXs)))
                self.all_xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_origin), __p(self.all_xyz_camOrigins)))
                self.all_xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camRs), __p(self.all_xyz_camRs)))
                self.all_xyz_camXs = __u(utils.geom.apply_4x4(__p(self.camXs_T_camRs), __p(self.all_xyz_camRs)))
                # st()
                
            
            # st()
            # self.all_xyz_camX0s = 
        if self.set_name=='test':
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            if hyp.pseudo_traj and not hyp.use_lrt_not_box:
                self.box_camRs = feed["box_traj_camR"]
                self.lrt_camRs = utils.misc.parse_boxes(self.box_camRs, self.origin_T_camRs)
            else:
                self.lrt_camRs = feed["lrt_traj_camR"] 
            self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils.geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            self.lrt_camR0s = utils.geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)
        
        if self.set_name=='test':
            # center on an object, so that it does not fall out of bounds
            # print('setting scene centroid to the object c')
            self.scene_centroid = utils.geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
            self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, 
                self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        else:
            # # center randomly
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            # center on a random non-outlier point
            # x_camXs = xyz_camXs[:,:,0].cpu().numpy()
            # y_camXs = xyz_camXs[:,:,1].cpu().numpy()
            # z_camXs = xyz_camXs[:,:,2].cpu().numpy()

            # xyz = xyz_camXs[:,:,0].cpu().numpy()


            all_ok = False
            num_tries = 0
            while not all_ok:
                scene_centroid_x = np.random.uniform(-8.0, 8.0)
                scene_centroid_y = np.random.uniform(-1.5, 3.0)
                scene_centroid_z = np.random.uniform(10.0, 26.0)
                scene_centroid = np.array([scene_centroid_x,
                                           scene_centroid_y,
                                           scene_centroid_z]).reshape([1, 3])
                self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
                num_tries += 1

                # try to vox
                self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X,
                                                   self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                all_ok = True

                # we want to ensure this gives us a few points inbound for each batch el
                inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X, already_mem=False))
                num_inb = torch.sum(inb.float(), axis=2)
                if torch.min(num_inb) < 100:
                    all_ok = False


                if num_tries > 100:
                    return False
                    

            self.summ_writer.summ_scalar('zoom_sampling/num_tries', num_tries)
            self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb).cpu().item())
            # print('all set')
            # input()
            
            
            # perm = np.random.permutation(num_exms)
            # # np.var(a)
            
            # scene_centroid_x = np.random.uniform(-8.0, 8.0)
            # scene_centroid_y = np.random.uniform(-1.5, 3.0)
            # scene_centroid_z = np.random.uniform(10.0, 26.0)
            # scene_centroid = np.array([scene_centroid_x,
            #                            scene_centroid_y,
            #                            scene_centroid_z]).reshape([1, 3])
            # self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            

        if hyp.pseudo_traj:
            if hyp.make_dense:
                self.all_occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.all_xyz_camRs), self.Z, self.Y, self.X))
                self.all_occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.all_xyz_camX0s), self.Z, self.Y, self.X))
        # st()
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        self.occ_memR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s), self.Z, self.Y, self.X))
        self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        self.occ_memRs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z2, self.Y2, self.X2))
        # self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))
        self.occ_memR0s_quar = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s), self.Z4, self.Y4, self.X4))

        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
        self.unp_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs, self.unp_memXs)
        self.unp_memR0s = self.vox_util.apply_4x4s_to_voxs(self.camR0s_T_camXs, self.unp_memXs)

        self.fix_adam = True
        self.fix_adam2 = False

        if prep_summ and self.include_vis:
            # self.summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(self.depth_camXs, dim=1))
            # self.summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(self.valid_camXs, dim=1))
            # self.summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
            # if hyp.pseudo_traj:
            #     self.summ_writer.summ_rgbs('2d_inputs/rgb_camRs', torch.unbind(self.rgb_camRs, dim=1))
            self.summ_writer.summ_rgbs('2d_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            self.summ_writer.summ_occs('3d_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3d_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
            self.summ_writer.summ_occs('3d_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
            self.summ_writer.summ_rgb('2d_inputs/rgb_camX_0',self.rgb_camXs[0,:1])
            if hyp.make_dense:
                T,S,D,H,W = feed['all_rgb_camXs'][0].shape
                all_rgb_camXs_reshaped = feed['all_rgb_camXs'].reshape(T*S,D,H,W)

                unp_all_memXs = self.vox_util.unproject_rgb_to_mem(all_rgb_camXs_reshaped , self.Z, self.Y, self.X, self.pix_T_cams[0,:1].repeat(T*S,1,1))
                unp_all_memOrigin = self.vox_util.apply_4x4_to_vox(self.all_origin_T_camXs[0], unp_all_memXs)
                unp_all_memR = self.vox_util.apply_4x4_to_vox(utils.geom.safe_inverse(self.origin_T_camRs[0:1,0].repeat(T*S,1,1)), unp_all_memOrigin)
                unp_all_memR= unp_all_memR.reshape([T,S,D,self.Z,self.Y,self.X])
                unp_all_memR = torch.mean(unp_all_memR,dim=1)
                # st()
                all_occ_memRs_ex = self.all_occ_memRs[:1,0]
                self.all_feat_memR_input = torch.cat([all_occ_memRs_ex,all_occ_memRs_ex*unp_all_memR[:1]],dim=1)
                self.summ_writer.summ_feat('3D_inputs//all_feat_memR_input_t0', self.all_feat_memR_input, pca=True)

                self.summ_writer.summ_occs('3d_inputs/all_occ_memRs', torch.unbind(self.all_occ_memRs, dim=1))
                self.summ_writer.summ_occs('3d_inputs/all_occ_mem0s', torch.unbind(self.all_occ_memX0s, dim=1))
            
            
            # self.summ_writer.summ_unps('3d_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
            self.summ_writer.summ_occs('3d_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))

            self.summ_writer.summ_rgb('2d_inputs/rgb_camX0', self.rgb_camXs[:,0])
            # st()
            sx,sy = (0.25,0.25)
            self.projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            self.H,self.W = (int(self.H*sy),int(self.W*sx))
            self.all_depth_camXs_, self.all_valid_camXs_ = utils.geom.create_depth_image(self.projpix_T_cams[0,:], self.xyz_camX0s[0,:], self.H, self.W)
            self.depth_camXs_, self.valid_camXs_ = (self.all_depth_camXs_[:1], self.all_valid_camXs_[:1])
            # we need to go to X0 to see what will be inbounds
            self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, self.projpix_T_cams[0,:1])
            self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camXs_, self.Z, self.Y, self.X).float()
            self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*1, 1, self.H, self.W])
            self.depth_camXs = self.depth_camXs_
            self.valid_camXs = self.valid_camXs_ * self.inbound_camXs_
            # st()
            self.summ_writer.summ_oneds('2D_inputs/depth_camX0', torch.unbind((self.all_depth_camXs_*self.all_valid_camXs_).unsqueeze(1),dim=0), maxval=32.0)
            self.summ_writer.summ_oneds('2D_inputs/valid_camX0', torch.unbind(self.all_valid_camXs_.unsqueeze(1),dim=0), norm=False)

            # self.summ_writer.summ_oned('2d_inputs/depth_camX0', self.depth_camXs[:,0], maxval=20.0)
            # self.summ_writer.summ_oned('2d_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        return True

    def preprocess_commonblack(self,tensor_image):
        B,D,H,W = tensor_image.shape
        hashed_tensor = tensor_image.sum(dim=1).squeeze(0)
        hashed_tensor = hashed_tensor.reshape([-1])
        mode_value = torch.mode(hashed_tensor).values
        indices = torch.where(hashed_tensor==mode_value)
        tensor_image = tensor_image.reshape([1, 3,-1])
        tensor_image[:,:,indices[0]] = torch.tensor([[[255],[255],[255]]]).to(tensor_image.dtype)
        tensor_image = tensor_image.reshape([B,D,H,W])
        return tensor_image

    def run_test(self, feed):
        results = dict()
        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        # st()
        self.obj_clist_camR0 = utils.geom.get_clist_from_lrtlist(self.lrt_camR0s)
        self.obj_clist_camX0 = utils.geom.get_clist_from_lrtlist(self.lrt_camX0s)

        self.original_centroid = self.scene_centroid.clone()

        assert(hyp.do_feat3d)

        feat_memX0_input = torch.cat([
            self.occ_memX0s[:,0],
            self.unp_memX0s[:,0]*self.occ_memX0s[:,0],
        ], dim=1)
        _, feat_chalfmemX0, _ = self.feat3dnet(feat_memX0_input)
        
        if hyp.summ_pca_points_3d or hyp.summ_pca_points_2d:
            # pca_3ds = []
            # pca_2ds = []
            feats_implicit_e_pca_only_list = []
            all_xyz_memX0s_list = []

            for i in range(self.occ_memX0s.shape[1]):
                feat_memXRan_input = torch.cat([
                    self.occ_memX0s[:,i],
                    self.unp_memX0s[:,i]*self.occ_memX0s[:,i],
                ], dim=1)
                _, feat_memXRan, _ = self.feat3dnet(feat_memXRan_input)            
                
                if hyp.make_dense:
                    all_xyz_camRs = self.all_xyz_camRs
                    all_xyz_camX0s = self.all_xyz_camX0s
                    all_pix_T_cams = self.projpix_T_cams
                    all_xyz_camXs = self.all_xyz_camXs
                else:
                    all_xyz_camX0s = self.xyz_camX0s
                    all_pix_T_cams = self.projpix_T_cams
                    all_xyz_camXs = self.xyz_camXs

                _,_,Z, Y, X = feat_memXRan.shape 
                dimensions_to_use = [Z, Y, X]
                all_xyz_memX0s = self.vox_util.Ref2Mem(all_xyz_camX0s[0,i:i+1], dimensions_to_use[0],dimensions_to_use[1],dimensions_to_use[2])
                all_xyz_memX0s = torch.clamp(all_xyz_memX0s, 0, Z-1)
                feats_implicit_e_pca_only = self.localdecodernet(None, feat_memXRan, all_xyz_memX0s, summ_writer = self.summ_writer)
                scale_mul = 2
                resolution = [i*scale_mul for i in dimensions_to_use]
                all_xyz_memX0s = self.vox_util.Ref2Mem(all_xyz_camX0s[0,i:i+1], resolution[0],resolution[1],resolution[2])
                all_xyz_memX0s = torch.clamp(all_xyz_memX0s, 0, Z*scale_mul-1)

                feats_implicit_e_pca_only_list.append(feats_implicit_e_pca_only)
                all_xyz_memX0s_list.append(all_xyz_memX0s)

            if hyp.make_dense:
                self.summ_writer.summ_pca3d_implicit_valid_temporal("pca/pca3d", all_xyz_memX0s_list, feats_implicit_e_pca_only_list, resolution, self.vox_util, only_return= True)
            else:
                self.summ_writer.summ_pca3d_implicit_valid_temporal("pca/pca3d", all_xyz_memX0s_list, feats_implicit_e_pca_only_list, resolution, self.vox_util, only_return= True)
            # st()
            self.summ_writer.summ_pca2d_implicit_temporal("pca/pca2d", all_xyz_camXs, feats_implicit_e_pca_only_list, all_pix_T_cams, dimensions_to_use,(self.H,self.W ), only_return= True)

            # self.summ_writer.summ_rgbs('pcas/pca_3d', pca_3ds)
            # self.summ_writer.summ_rgbs('pcas/pca_2d', pca_2ds)
        B, C, Z_, Y_, X_ = list(feat_chalfmemX0.shape)
        S = self.S

        print('feat_memX0_input', feat_memX0_input.shape)
        print('feat_chalfmemX0', feat_chalfmemX0.shape)

        if hyp.make_dense:
            self.custom_xyz_camX0s = self.all_xyz_camX0s
            self.custom_xyz_camX0s = self.xyz_camX0s
        else:
            self.custom_xyz_camX0s = self.xyz_camX0s

        
        obj_mask_halfmemX0s = self.vox_util.assemble_padded_obj_masklist(
            self.lrt_camX0s,
            self.score_s,
            self.Z2, self.Y2, self.X2).squeeze(1)

        obj_mask_halfmemX0s_pcd = self.vox_util.assemble_padded_obj_masklist_xyz(
            self.lrt_camX0s,
            self.score_s,
            self.Z2, self.Y2, self.X2, self.custom_xyz_camX0s).squeeze(1)
        
        ## Visualize obj_mask_halfmemX0s_pcd    
        xyz_camX0_view0 = self.custom_xyz_camX0s[:, 0]
        mask_X0 = obj_mask_halfmemX0s_pcd[:,0]
        object_xyz_camX0 = xyz_camX0_view0[mask_X0>0]
        self.summ_writer.summ_pointcloud_on_rgb("Object_pointcloud", object_xyz_camX0.unsqueeze(0), self.rgb_camXs[:,0], self.pix_T_cams[:,0])
        ## Vis ends

        # only take the occupied voxels
        self.summ_writer.summ_oned("obj_mask_halfmemX0s", obj_mask_halfmemX0s[0], bev=True)
        occ_halfmemX0 = self.vox_util.voxelize_xyz(self.custom_xyz_camX0s[:,0], self.Z2, self.Y2, self.X2)
        obj_mask_halfmemX0 = obj_mask_halfmemX0s[:,0] * occ_halfmemX0
        obj_mask_chalfmemX0 = self.crop_feat(obj_mask_halfmemX0, self.crop_guess)

        obj_mask_chalfmemX0_pcd = self.crop_pointcloud(self.custom_xyz_camX0s[:, 0], self.crop_guess)
        obj_mask_chalfmemX0_pcd = obj_mask_chalfmemX0_pcd*obj_mask_halfmemX0s_pcd[:, 0]

        if hyp.pointfeat_ransac:
            for b in list(range(self.B)):
                print("Valid points: ", torch.sum(obj_mask_chalfmemX0_pcd[b]))
                if torch.sum(obj_mask_chalfmemX0_pcd[b]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True
        else:
            for b in list(range(self.B)):
                print("Valid points: ", torch.sum(obj_mask_chalfmemX0[b]))
                if torch.sum(obj_mask_chalfmemX0[b]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True
        if self.fix_adam or self.fix_adam2:
            # st()
            feat_chalfmemX0_cropped = self.crop_feat(feat_chalfmemX0, self.crop_guess)
        else:
            feat_chalfmemX0_cropped = feat_chalfmemX0
        # for b in list(range(self.B)):
        #     dist = torch.norm(self.obj_clist_camX0[:,0] - self.obj_clist_camX0[:,-1], dim=1)
        #     if dist > 1.0:
        #         # print('returning early, since the object moved less than 1m')
        #         print('returning early, since the object moved more than 1m')
        #         return total_loss, results, True
        feat0_vec = feat_chalfmemX0_cropped.view(B, hyp.feat3d_dim, -1)
        # this is B x C x huge
        feat0_vec = feat0_vec.permute(0, 2, 1)
        # this is B x huge x C

        print('feat0_vec', feat0_vec.shape)


        obj_mask0_vec = obj_mask_chalfmemX0.reshape(B, -1).round()
        # this is B x huge

        print('obj_mask0_vec', obj_mask0_vec.shape)


        orig_xyz_halfmem = utils.basic.gridcloud3d(B, self.Z2, self.Y2, self.X2)
        # this is B x -1 x 3
        orig_xyz_halfmem = orig_xyz_halfmem.reshape(B, self.Z2, self.Y2, self.X2, 3)
        orig_xyz_halfmem = orig_xyz_halfmem.permute(0, 4, 1, 2, 3)
        orig_xyz_chalfmem = self.crop_feat(orig_xyz_halfmem, self.crop_guess)
        orig_xyz_vec = orig_xyz_chalfmem.reshape(B, 3, -1).permute(0,2,1)
        # this is B x huge x 3
        # orig_xyz_chalfmem_pcd = self.crop_pointcloud(self.xyz_camX0s, self.crop_guess)

        print('orig_xyz_vec', orig_xyz_vec.shape)

        obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(self.lrt_camX0s)
        obj_length = obj_lengths[:,0]
        cam0_T_obj = cams_T_obj0[:,0]
        # this is B x S x 4 x 4

        mem_T_cam = self.vox_util.get_mem_T_ref(B, self.Z2, self.Y2, self.X2)
        cam_T_mem = self.vox_util.get_ref_T_mem(B, self.Z2, self.Y2, self.X2)

        lrt_camIs_g = self.lrt_camX0s.clone()
        lrt_camIs_e = torch.zeros_like(self.lrt_camX0s)
        # we will fill this up

        ious = torch.zeros([B, S]).float().cuda()
        point_counts = np.zeros([B, S])
        inb_counts = np.zeros([B, S])

        # Only use this precomputed step to get object feats (s=0). For scene feats (s>0), compute this 
        # again since bounds are changed at every step.
        xyz_memX0 = __u(self.vox_util.Ref2Mem(__p(self.custom_xyz_camX0s), self.Z2, self.Y2, self.X2))
        xyz_memX0_inbounds = __u(self.vox_util.get_inbounds(__p(xyz_memX0), self.Z2, self.Y2, self.X2, already_mem=True))

        if hyp.do_tsdf_implicit_occ:
            mult = 4
            xyz_tsdf_memX0 = utils.basic.gridcloud3d(1, self.Z2*mult, self.Y2*mult, self.X2*mult)/mult

        for s in range(self.S):
            if not (s==0):
                # remake the vox util and all the mem data
                self.scene_centroid = utils.geom.get_clist_from_lrtlist(lrt_camIs_e[:,s-1:s])[:,0]
                delta = self.scene_centroid - self.original_centroid
                self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, 
                    self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                # self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
                self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

                self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
                    __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
                self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
                self.summ_writer.summ_occ('track/reloc_occ_%d' % s, self.occ_memX0s[:,s])
            else:
                self.summ_writer.summ_occ('track/init_occ_%d' % s, self.occ_memX0s[:,s])
                delta = torch.zeros([B, 3]).float().cuda()
            # print('scene centroid:', self.scene_centroid.detach().cpu().numpy())

            inb = self.vox_util.get_inbounds(self.custom_xyz_camX0s[:,s], self.Z4, self.Y4, self.X4, already_mem=False)
            num_inb = torch.sum(inb.float(), axis=1)
            # print('num_inb', num_inb, num_inb.shape)
            # num_inb = torch.sum(self.occ_memX0s.float(), axis=[2, 3, 4])
            inb_counts[:, s] = num_inb.cpu().numpy()

            feat_memI_input = torch.cat([
                self.occ_memX0s[:,s],
                self.unp_memX0s[:,s]*self.occ_memX0s[:,s],
            ], dim=1)
            _, feat_chalfmemI, _ = self.feat3dnet(feat_memI_input)

            self.summ_writer.summ_feat('3d_feats/feat_%d_input' % s, feat_memI_input, pca=True)
            self.summ_writer.summ_feat('3d_feats/feat_%d' % s, feat_chalfmemI, pca=True)
            if (not hyp.pointfeat_ransac) and self.fix_adam2:
                feat_vec = self.crop_feat(feat_chalfmemI, self.crop_guess).view(B, hyp.feat3d_dim, -1)
            else:
                feat_vec = feat_chalfmemI.view(B, hyp.feat3d_dim, -1)

            if hyp.do_tsdf_implicit_occ:
                _, feat_chalfmemI_occ, _ = self.feat3dnet_occ(feat_memI_input)
                self.summ_writer.summ_feat('3d_feats/feat_occ_%d' % s, feat_chalfmemI_occ, pca=True)
                

            # this is B x C x huge
            feat_vec = feat_vec.permute(0, 2, 1)
            # this is B x huge x C

            memI_T_mem0 = utils.geom.eye_4x4(B)
            # we will fill this up

            # Compute again as bounds have changed
            # forS means for iteration S or bounds S
            if hyp.make_dense:
                xyz_memX0_forS = __u(self.vox_util.Ref2Mem(__p(self.all_xyz_camX0s), self.Z2, self.Y2, self.X2))
                # sampled = xyz_memX0_forS[:,:,::8]
                # more_points =sampled + torch.randn(sampled.shape).cuda()
                # xyz_memX0_forS = torch.cat((xyz_memX0_forS,more_points),dim=2)            
            else:
                xyz_memX0_forS = __u(self.vox_util.Ref2Mem(__p(self.custom_xyz_camX0s), self.Z2, self.Y2, self.X2))

            # to simplify the impl, we will iterate over the batch dim
            for b in list(range(B)):
                assert self.X2 == self.Y2
                assert self.X2 == self.Z2 
                feat_chalfmemI_b = feat_chalfmemI[b]
                feat_chalfmemX0_b = feat_chalfmemX0[b]
                obj_mask_chalfmemX0_pcd_b = obj_mask_chalfmemX0_pcd[b]
                xyz_memX0_inbounds_b0 = xyz_memX0_inbounds[b, 0]
                feat_vec_b = feat_vec[b]
                feat0_vec_b = feat0_vec[b]
                obj_mask0_vec_b = obj_mask0_vec[b]
                orig_xyz_b = orig_xyz_vec[b]
                
                # these are huge x C
                if hyp.do_localdecoder:
                    # Get pointfeats for scene                
                    xyz_memX0_bs = xyz_memX0_forS[b,s]
                    xyz_memX0_bs_inbounds = self.vox_util.get_inbounds_single(xyz_memX0_bs, self.Z2, self.Y2, self.X2, already_mem=True)
                    xyz_memX0_bs = xyz_memX0_bs[xyz_memX0_bs_inbounds]
                    feat_vec_b_pcd, _ = self.localdecodernet.localdecoder_feats(None, xyz_memX0_bs.unsqueeze(0), feat_chalfmemI_b.unsqueeze(0))
                    feat_vec_b_pcd = feat_vec_b_pcd[0]

                    if hyp.do_tsdf_implicit_occ:
                        feat_chalfmemI_occ_b = feat_chalfmemI_occ[b]
                        logits, _ = self.localdecodernet_render_occ.localdecoder_occ(None, xyz_tsdf_memX0, feat_chalfmemI_occ_b.unsqueeze(0))
                        probs = F.sigmoid(logits)[0]
                        xyz_tsdf_camX0 = self.vox_util.Mem2Ref(xyz_tsdf_memX0, self.Z2, self.Y2, self.X2)
                        
                        occupied_idxs = torch.where(probs>=0.99)[0]
                        unoccupied_idxs = torch.where(probs<0.99)[0]

                        occupied_memX0 = xyz_tsdf_memX0[:, occupied_idxs]

                        occupied_camX0 = xyz_tsdf_camX0[:, occupied_idxs]
                        unoccupied_camX0 = xyz_tsdf_camX0[:, unoccupied_idxs]

                        occupied_camR0 = utils.geom.apply_4x4(self.camRs_T_camXs[b:b+1, s], occupied_camX0)
                        unoccupied_camR0 = utils.geom.apply_4x4(self.camRs_T_camXs[b:b+1, s], unoccupied_camX0)
                        
                        rgb_camRs = feed['rgb_camRs']
                        self.summ_writer.summ_pointcloud_on_rgb("TSDF/occupied", occupied_camX0, self.rgb_camXs[:,s], self.pix_T_cams[:,s])
                        self.summ_writer.summ_pointcloud_on_rgb("TSDF/unoccupied", unoccupied_camX0, self.rgb_camXs[:,s], self.pix_T_cams[:,s])
                        
                        vis = self.summ_writer.summ_pointcloud_on_rgb("TSDF/occupied", occupied_camR0, rgb_camRs[:,s], self.pix_T_cams[:,s], only_return=True)
                        vis = torch.cat([vis, rgb_camRs[b, s]], dim=-1)
                        self.summ_writer.summ_rgb("TSDF/occupied_camR", vis.unsqueeze(0))
                        
                        vis = self.summ_writer.summ_pointcloud_on_rgb("TSDF/unoccupied", unoccupied_camR0, rgb_camRs[:,s], self.pix_T_cams[:,s], only_return=True)
                        vis = torch.cat([vis, rgb_camRs[b, s]], dim=-1)
                        self.summ_writer.summ_rgb("TSDF/occupied_camR", vis.unsqueeze(0))

                        if hyp.make_dense:
                            all_xyz_camXs = utils.geom.apply_4x4(self.camXs_T_origin[b:b+1, s], self.all_xyz_camOrigins[b:b+1, s])
                            all_xyz_memXs = self.vox_util.Ref2Mem(all_xyz_camXs, self.Z2, self.Y2, self.X2)
                            gt_occs_memXs = self.vox_util.get_occupancy(all_xyz_memXs, self.Z2, self.Y2, self.X2)
                            pred_occs_memXs = self.vox_util.get_occupancy(occupied_memX0, self.Z2, self.Y2, self.X2)
                            # self.summ_writer.summ_oned("TSDF/GT_BEV_Occupancy", gt_occs_memXs, bev=True)
                            self.summ_writer.summ_occ("TSDF/GT_BEV_Occupancy", gt_occs_memXs)
                            # self.summ_writer.summ_oned("TSDF/Pred_BEV_Occupancy", pred_occs_memXs, bev=True)
                            self.summ_writer.summ_occ("TSDF/Pred_BEV_Occupancy", pred_occs_memXs)
                            self.summ_writer.summ_occ("TSDF/Pred_BEV_Occupancy*Pred_BEV_GT", pred_occs_memXs*gt_occs_memXs)




                    # Get point feats for object
                    obj_xyzmem0_pcd =  xyz_memX0[b, 0]
                    obj_mask_chalfmemX0_pcd_b_inbounds = obj_mask_chalfmemX0_pcd_b*xyz_memX0_inbounds_b0
                    obj_xyzmem0_pcd = obj_xyzmem0_pcd[obj_mask_chalfmemX0_pcd_b_inbounds.long()>0]
                    obj_vec_b_pcd, _ = self.localdecodernet.localdecoder_feats(None, obj_xyzmem0_pcd.unsqueeze(0), feat_chalfmemX0_b.unsqueeze(0))
                    obj_vec_b_pcd = obj_vec_b_pcd[0]
                    obj_vec_b_pcd = obj_vec_b_pcd.permute(1,0)
                    # this is C x med
                else:
                    obj_inds_b = torch.where(obj_mask0_vec_b > 0)
                    obj_vec_b = feat0_vec_b[obj_inds_b]
                    xyz0 = orig_xyz_b[obj_inds_b]
                    # these are med x C
                    obj_vec_b = obj_vec_b.permute(1, 0)
                    # this is is C x med


                if hyp.pointfeat_ransac:
                    corr_b = torch.matmul(feat_vec_b_pcd, obj_vec_b_pcd)
                    corr_b = corr_b/(feat_vec_b_pcd.norm(dim=-1).unsqueeze(1) + 1e-4)
                    corr_b = corr_b/(obj_vec_b_pcd.norm(dim=0).unsqueeze(0) + 1e-4)
                    # this is huge x med
                    heat_b = corr_b.permute(1, 0)
                    heat_b = F.relu(heat_b)
                    # this is med x huge

                    # xyzI = utils.basic.argmax3d(heat_b, hard=False, stack=True)
                    try:
                        heatmap_b_argmax = torch.argmax(heat_b, dim=-1)
                    except Exception as e:
                        # st()
                        aa=1
                        return False, False, False
                    xyzI = xyz_memX0_bs[heatmap_b_argmax]
                    # This is med x 3

                    xyzI_cam = self.vox_util.Mem2Ref(xyzI.unsqueeze(1), self.Z2, self.Y2, self.X2)
                    xyzI_cam += delta
                    xyzI = self.vox_util.Ref2Mem(xyzI_cam, self.Z2, self.Y2, self.X2).squeeze(1)

                    memI_T_mem0[b] = utils.track.rigid_transform_3d(obj_xyzmem0_pcd, xyzI)

                    # record #points, since ransac depends on this
                    point_counts[b, s] = len(obj_xyzmem0_pcd)

                else:
                    corr_b = torch.matmul(feat_vec_b, obj_vec_b)
                    # this is huge x med

                    heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z_, Y_, X_)
                    # this is med x 1 x Z_ x Y_ x X_
                    if self.fix_adam:
                        pass
                    elif self.fix_adam2:
                        heat_b = self.pad_feat(heat_b, self.crop_guess)
                    else:
                        heat_b = self.pad_feat(heat_b, self.crop_guess)
                    # this is med x 1 x Z2 x Y2 x X2
                    heat_b = F.relu(heat_b)

                    # heat_b_ = heat_b.reshape(-1, Z_*Y_*X_)
                    # heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
                    # heat_b = heat_b - heat_b_max
                    heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

                    xyzI = utils.basic.argmax3d(heat_b, hard=False, stack=True)
                    # this is med x 3
                    # print('got soft argmax xyzI:', xyzI.detach().cpu().numpy())
                    xyzI_ = utils.basic.argmax3d(heat_b, hard=True, stack=True)

                    # print('got hard argmax xyzI:', xyzI_.detach().cpu().numpy())

                    # xyzI_cam = self.vox_util.Mem2Ref(xyzI.unsqueeze(1), Z_, Y_, X_)
                    # xyzI_cam += delta
                    # xyzI = self.vox_util.Ref2Mem(xyzI_cam, Z_, Y_, X_).squeeze(1)

                    xyzI_cam = self.vox_util.Mem2Ref(xyzI.unsqueeze(1), self.Z2, self.Y2, self.X2)
                    xyzI_cam += delta
                    xyzI = self.vox_util.Ref2Mem(xyzI_cam, self.Z2, self.Y2, self.X2).squeeze(1)

                    memI_T_mem0[b] = utils.track.rigid_transform_3d(xyz0, xyzI)

                    # record #points, since ransac depends on this
                    point_counts[b, s] = len(xyz0)
                
            # done stepping through batch
            mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
            cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

            # print('cam0_T_obj', cam0_T_obj)

            # eval
            camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
            # this is B x 4 x 4
            lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
            iou, _ = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1])
            # print('iou', iou)
            ious[:,s] = iou.squeeze(1)
        results['ious'] = ious
        # print('ious', ious)
        # input()

        for s in range(self.S):
            self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())

        self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
        self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))
        # self.summ_writer.summ_scalar('track/inb_counts', torch.mean(inb_counts).cpu().item())
        self.summ_writer.summ_scalar('track/inb_counts', np.mean(inb_counts))

        lrt_camX0s_e = lrt_camIs_e.clone()
        lrt_camXs_e = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)

        if self.include_vis:
            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_e' % s, self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
            visX_g = []
            for s in list(range(self.S)):
                visX_g.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)

        obj_clist_camX0_e = utils.geom.get_clist_from_lrtlist(lrt_camX0s_e)

        dists = torch.norm(obj_clist_camX0_e - self.obj_clist_camX0, dim=2)
        # this is B x S
        mean_dist = utils.basic.reduce_masked_mean(dists, self.score_s)
        median_dist = utils.basic.reduce_masked_median(dists, self.score_s)
        # this is []
        self.summ_writer.summ_scalar('track/centroid_dist_mean', mean_dist.cpu().item())
        self.summ_writer.summ_scalar('track/centroid_dist_median', median_dist.cpu().item())

        # if self.include_vis:
        if (True):
            self.summ_writer.summ_traj_on_occ('track/traj_e',
                                              obj_clist_camX0_e, 
                                              self.occ_memX0s[:,0],
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)
            self.summ_writer.summ_traj_on_occ('track/traj_g',
                                              self.obj_clist_camX0,
                                              self.occ_memX0s[:,0],
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)
        total_loss += mean_dist # we won't backprop, but it's nice to plot and print this anyway


        # i need these values, for g and e:
        # "height": 31.0, 
        # "width": 31.0, 
        # "id": "sheldon", 
        # "y": 105.0, 
        # "x": 608.0

        # no problem
        # my data is this:
        # ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)

        # corners_e = utils.geom.get_xyzlist_from_lrtlist(lrt_camIs_e)
        # corners_g = utils.geom.get_xyzlist_from_lrtlist(lrt_camIs_g)
        # # these are B x S x 8 x 3

        # corners_e_ = __p(corners_e)
        # corners_g_ = __p(corners_g)
        # pix_T_cam_ = __p(self.pix_T_cams)

        # corners_e_ = utils.geom.apply_pix_T_cam(pix_T_cam_, corners_e_)
        # corners_g_ = utils.geom.apply_pix_T_cam(pix_T_cam_, corners_g_)

        # corners_e = __u(corners_e_)
        # corners_g = __u(corners_g_)
        # # these are B x S x 8 x 2

        # heigh_e =
        boxlist2d_e = utils.geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], lrt_camIs_e, self.H, self.W)
        boxlist2d_g = utils.geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,0], lrt_camIs_g, self.H, self.W)

        height_e_, width_e_ = utils.geom.get_size_from_box2d(__p(boxlist2d_e))
        y_e_, x_e_ = utils.geom.get_centroid_from_box2d(__p(boxlist2d_e))
        height_e, width_e = __u(height_e_), __u(width_e_)
        y_e, x_e = __u(y_e_), __u(x_e_)

        height_g_, width_g_ = utils.geom.get_size_from_box2d(__p(boxlist2d_g))
        y_g_, x_g_ = utils.geom.get_centroid_from_box2d(__p(boxlist2d_g))
        height_g, width_g = __u(height_g_), __u(width_g_)
        y_g, x_g = __u(y_g_), __u(x_g_)
        # these are B x S

        boxlist_e = torch.stack([height_e, width_e, y_e, x_e], dim=2)
        boxlist_g = torch.stack([height_g, width_g, y_g, x_g], dim=2)
        results['boxlist_e'] = boxlist_e
        results['boxlist_g'] = boxlist_g

        # height_g_, width_g_ = utils.geom.get_size_from_box2D(__p(boxlist2d_g))
        # height_g, width_g = __u(height_g_), __u(width_g_)
        # y, x = utils.geom.get_centroid_from_box2D(box2D)

        # corners_e = torch.reshape(corners_e_, [self.B, self.S, 8, 2])
        # corners_g = torch.reshape(corners_g_, [self.B, self.S, 8, 2])






        # if not run featnet: 
        # ious = torch.zeros([self.B, self.S]).float().cuda()
        # for s in list(range(self.S)):
        #     # lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
        #     ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,0:1], self.lrt_camRs[:,s:s+1]).squeeze(1)
        # results['ious'] = ious
        # for s in range(self.S):
        #     self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
        # self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())

        # lrt_camX0s_e = self.lrt_camX0s[:,0:1].repeat(1, self.S, 1)
        # obj_clist_camX0_e = utils.geom.get_clist_from_lrtlist(lrt_camX0s_e)
        # self.summ_writer.summ_traj_on_occ('track/traj_e',
        #                                   obj_clist_camX0_e, 
        #                                   self.occ_memX0s[:,0],
        #                                   self.vox_util, 
        #                                   already_mem=False,
        #                                   sigma=2)
        # self.summ_writer.summ_traj_on_occ('track/traj_g',
        #                                   self.obj_clist_camX0,
        #                                   self.occ_memX0s[:,0],
        #                                   self.vox_util, 
        #                                   already_mem=False,
        #                                   sigma=2)


        
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_new_test(self, feed):
        st()
        assert False
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.obj_clist_camR0 = utils.geom.get_clist_from_lrtlist(self.lrt_camR0s)
        self.obj_clist_camX0 = utils.geom.get_clist_from_lrtlist(self.lrt_camX0s)

        self.original_centroid = self.scene_centroid.clone()
        
        # self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        # return total_loss, results, False
        
        lrt_camXAIs, scores = utils.track.track_proposal(
            self.B, self.S, 
            self.lrt_camX0s[:,0],
            self.pix_T_cams,
            self.rgb_camXs,
            self.xyz_camX0s,
            self.camX0s_T_camXs,
            self.feat3dnet,
            self.vox_util,
            self.crop_feat,
            self.pad_feat,
            self.crop_guess,
            summ_writer=self.summ_writer)

        # self.summ_writer.summ_oneds('track/top3d_mask_vis_%d' % super_iter, mask_memXAI_all, bev=True, norm=False)

        box_vis_bev = []
        box_vis = []
        for I in list(range(self.S)):
            lrt_ = lrt_camXAIs[:,I:I+1]
            score_ = scores[:,I:I+1]
            box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                '', self.pad_feat(self.occ_memXAI_all[I]),
                torch.cat([self.full_lrtlist_camXAs[:,I], lrt_], dim=1),
                torch.cat([self.full_scorelist_s[:,I], score_], dim=1),
                # torch.cat([self.full_scorelist_s[:,I], torch.ones_like(lrt_[:,:,0])], dim=1),
                torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(lrt_[:,:,0]).long()], dim=1),
                self.vox_util, frame_id=I, only_return=True))
            lrt_ = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camXAs[:,I], lrt_)
            box_vis.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,I],
                torch.cat([self.full_lrtlist_camXs[:,I], lrt_], dim=1),
                # torch.cat([self.full_scorelist_s[:,I], torch.ones_like(lrt_[:,:,0])], dim=1),
                torch.cat([self.full_scorelist_s[:,I], score_], dim=1),
                torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(lrt_[:,:,0]).long()], dim=1),
                self.pix_T_cams[:,I],
                frame_id=I, only_return=True))
        self.summ_writer.summ_rgbs('track/all_boxes_bev_%d' % super_iter, box_vis_bev)
        self.summ_writer.summ_rgbs('track/all_boxes_%d' % super_iter, box_vis)
        
    
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
            return self.run_train(feed)
        elif set_name=='test':
            ok = self.prepare_common_tensors(feed)
            if not ok:
                total_loss = torch.tensor(0.0).cuda()
                return total_loss, None, True
            # return self.run_new_test(feed)
            return self.run_test(feed)
        
        # arriving at this line is bad
        print('weird set_name:', set_name)
        assert(False)

