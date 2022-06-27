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
from nets.emb3dnet import Emb3dNet
from nets.occnet import OccNet
from nets.rendernet import RenderNet
from nets.rgbnet import RgbNet
from nets.localdecodernet import LocalDecoderParent


from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch import distributions as dist
import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.py
import utils.misc
import utils.track
import utils.vox
import utils.implicit
import ipdb
st = ipdb.set_trace
np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_MOC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaMocModel()
        if hyp.do_feat3d and hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

        if hyp.do_emb3d:
            # freeze the slow model
            self.model.feat3dnet_slow.eval()
            self.set_requires_grad(self.model.feat3dnet_slow, False)
            
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
        # st()
        if hyp.do_emb3d:
            self.model.feat3dnet_slow.load_state_dict(self.model.feat3dnet.state_dict())
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

                if log_this or set_do_backprop:
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
                    
                    if hyp.pseudo_multiview:
                        set_seqlen = 2


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

                    if ((not returned_early) and 
                        (set_do_backprop) and 
                        (hyp.lr > 0)):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if hyp.do_emb3d:
                        def update_slow_network(slow_net, fast_net, beta=0.999):
                            param_k = slow_net.state_dict()
                            param_q = fast_net.named_parameters()
                            for n, q in param_q:
                                if n in param_k:
                                    param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                            slow_net.load_state_dict(param_k)
                        update_slow_network(self.model.feat3dnet_slow, self.model.feat3dnet)
                        
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

            
class CarlaMocModel(nn.Module):
    def __init__(self):
        super(CarlaMocModel, self).__init__()

        # self.crop_guess = (18,18,18)
        # self.crop_guess = (2,2,2)
        self.crop = (18,18,18)
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_occ:
            self.occnet = OccNet()

        if hyp.point_contrast_og:
            # st()
            self.pointnet2 = Pointnet2(hyp.feat3d_dim)
            self.pointnet2.cuda()
        
        if hyp.do_render:
            self.rendernet = RenderNet()

        if hyp.do_rgb:
            self.rgbnet = RgbNet()

        if hyp.do_emb3d:
            self.emb3dnet = Emb3dNet()
            # make a slow net
            self.feat3dnet_slow = Feat3dNet(in_dim=4)
            # init slow params with fast params
            self.feat3dnet_slow.load_state_dict(self.feat3dnet.state_dict())

        if hyp.do_localdecoder_render or hyp.do_implicit_occ or hyp.do_tsdf_implicit_occ:
            self.localdecodernet_render_occ = LocalDecoderParent()

        if hyp.do_localdecoder:
            self.localdecodernet = LocalDecoderParent()
            
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
    
    def prepare_common_tensors(self, feed, prep_summ=True):
        results = dict()
        # st()
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

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        
        # self.rgb_camRs = feed["rgb_camRs"]
        self.rgb_camXs = feed["rgb_camXs"].float()
        self.pix_T_cams = feed["pix_T_cams"].float()

        self.origin_T_camRs = feed["origin_T_camRs"].float()
        self.origin_T_camXs = feed["origin_T_camXs"].float()




        self.camXs_T_origin = __u(utils.geom.safe_inverse(__p(self.origin_T_camXs)))
        self.camX0_T_origin = self.camXs_T_origin[:, 0]

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0 = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(utils.geom.safe_inverse(__p(self.origin_T_camRs)), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_camXs)))
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camX0_T_camR0 = utils.basic.matmul2(self.camX0s_T_camXs[:,0], self.camXs_T_camRs[:,0])
        self.camR0s_T_camXs = utils.basic.matmul2(self.camR0s_T_camRs, self.camRs_T_camXs)

        if hyp.pseudo_multiview:
            if hyp.make_dense:
                all_origin_T_camXs = feed['all_origin_T_camXs']
                all_xyz_camXs = feed['all_xyz_camXs']
                self.all_origin_T_camXs = all_origin_T_camXs
                self.all_camX0_T_camXs = utils.geom.get_camM_T_camXs(self.all_origin_T_camXs, ind=0)

                self.all_xyz_camOrigins_ = __u(utils.geom.apply_4x4(__p(all_origin_T_camXs), __p(all_xyz_camXs)))

                B,V,N,D = self.all_xyz_camOrigins_.shape
                self.all_xyz_camOrigins = self.all_xyz_camOrigins_.reshape([B,V*N,D])
                assert((self.origin_T_camRs[:,0] ==self.origin_T_camRs[:,1]).all())
                origin_T_camRs_unique = self.origin_T_camRs[:,0]
                self.all_xyz_camRs = utils.geom.apply_4x4(utils.geom.safe_inverse(origin_T_camRs_unique),self.all_xyz_camOrigins)
                self.all_xyz_camX0 = utils.geom.apply_4x4(utils.geom.safe_inverse(self.camX0_T_camR0),self.all_xyz_camRs)
                
            
            if hyp.eval_boxes:
                self.full_box_camRs = feed["full_boxlist_camR"].squeeze(1)
                self.full_scorelist = feed['full_scorelist'].squeeze(1)
                self.full_tidlist = feed["full_tidlist"].squeeze(1)
                assert (self.origin_T_camRs[:,0] == self.origin_T_camRs[:,1]).all()
                N_val = self.full_scorelist.shape[1]
                origin_T_camRs_tiled = self.origin_T_camRs[:,:1].repeat(1,N_val,1,1)
                self.full_lrt_camRs = utils.misc.parse_boxes(self.full_box_camRs, origin_T_camRs_tiled)

                self.camX0_T_camRs =  self.camXs_T_camRs[:,0:1]
                self.camX0_T_camRs_tiled = self.camX0_T_camRs.repeat(1,N_val,1,1)
                self.full_lrt_camX0 = utils.geom.apply_4x4s_to_lrts(self.camX0_T_camRs_tiled, self.full_lrt_camRs)
                self.summ_writer.summ_lrtlist('eval_boxes/box_camX0_g', self.rgb_camXs[:,0], self.full_lrt_camX0,self.full_scorelist, self.full_tidlist, self.pix_T_cams[:,0])


        if feed['set_name']=='test':
            self.box_camRs = feed["box_traj_camR"]
            # box_camRs is B x S x 9
            self.score_s = feed["score_traj"]
            self.tid_s = torch.ones_like(self.score_s).long()
            self.lrt_camRs = utils.geom.convert_boxlist_to_lrtlist(self.box_camRs)
            self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            self.lrt_camX0s = utils.geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, self.lrt_camXs)
            self.lrt_camR0s = utils.geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, self.lrt_camRs)
        
        if feed['set_name']=='test':
            # center on an object, so that it does not fall out of bounds
            scene_centroid = utils.geom.get_clist_from_lrtlist(self.lrt_camXs)[:,0]
        else:
            # center randomly 
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            
            if hyp.do_tsdf_implicit_occ:
                scene_centroid_z = np.random.uniform(0.0, 15) # this is carla mc mod specific.

            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

        
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW
        self.K = hyp.K
        self.Z, self.Y, self.X = hyp.Z_train, hyp.Y_train, hyp.X_train
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, feed['set_name'], scene_centroid=scene_centroid, assert_cube=True)
        
        self.xyz_camXs = feed["xyz_camXs"].float()
        
        if hyp.point_contrast_og:
            xyz_camXs = self.xyz_camXs
            depth_camXs, valid_camXs = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(xyz_camXs), self.H, self.W)
            xyz_camXs = __u(utils.geom.depth2pointcloud(depth_camXs, __p(self.pix_T_cams)))
            self.xyz_camXs = xyz_camXs.float()            
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils.geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))
        
        if hyp.do_tsdf_implicit_occ:
            self.occupied_camX0 = utils.geom.apply_4x4(self.camX0_T_origin, feed['tsdf_inside'])
            outside_camX0 = utils.geom.apply_4x4(self.camX0_T_origin, feed['tsdf_outside'])
            sdf1_camX0 = utils.geom.apply_4x4(self.camX0_T_origin, feed['tsdf_sdf1'])

            occupied_memX0 = self.vox_util.Ref2Mem(self.occupied_camX0, self.Z2, self.Y2, self.X2)
            occupancies_occupied_memX0 = self.vox_util.get_occupancy(occupied_memX0, self.Z2, self.Y2, self.X2)
            self.summ_writer.summ_occ("Implicit/Occupied_all", occupancies_occupied_memX0)

            self.occupied_camX0 = self.vox_util.get_sampled_inbound_points(self.occupied_camX0, num_pts=2*hyp.num_tsdf_to_sample)
            outside_camX0 = self.vox_util.get_sampled_inbound_points(outside_camX0, num_pts=hyp.num_tsdf_to_sample)
            sdf1_camX0 = self.vox_util.get_sampled_inbound_points(sdf1_camX0, num_pts=hyp.num_tsdf_to_sample)
            self.unoccupied_camX0 = torch.cat([outside_camX0, sdf1_camX0], dim=1)

        if hyp.pseudo_multiview:
            if hyp.do_implicit_occ:
                # sttime = time.time()
                # self.vox_util.get_free_and_occupied_points_better(all_xyz_camXs, self.all_camX0_T_camXs)
                self.occupied_camX0, self.unoccupied_camX0 = self.vox_util.get_free_and_occupied_points(self.all_xyz_camX0)
                # print("Time taken to get free and occ: ", time.time() - sttime)
            if hyp.make_dense:
                self.all_occ_memX0 = self.vox_util.voxelize_xyz(self.all_xyz_camX0, self.Z2, self.Y2, self.X2)
                # st()
        # we need to go to X0 to see what will be inbounds
        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.PH, self.PW)
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camXs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.PH, self.PW])
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        self.summ_writer.summ_oned('2D_inputs/depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('2D_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        self.summ_writer.summ_oned('2D_inputs/depth_camX1', self.depth_camXs[:,1]*self.valid_camXs[:,1], maxval=32.0)
        self.summ_writer.summ_oned('2D_inputs/valid_camX1', self.valid_camXs[:,1], norm=False)

        all_ok = True
        return all_ok

    # def find_nearby_indices(self,intersect_points,points_to_compare):


    def run_train(self, feed):

        results = dict()
        
        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        if hyp.debug:
            self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
            occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
            unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(__p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
            # occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
            
            # freespace = self.vox_util.get_freespace(__p(self.xyz_camXs),__p(occ_memXs))
            unp_all_memXs = self.vox_util.unproject_rgb_to_mem(feed['all_rgb_camXs'][0], self.Z, self.Y, self.X, self.pix_T_cams[0].repeat(2,1,1))
            unp_all_memOrigin = self.vox_util.apply_4x4_to_vox(self.all_origin_T_camXs[0], unp_all_memXs)
            unp_all_memR = self.vox_util.apply_4x4_to_vox(utils.geom.safe_inverse(self.origin_T_camRs[0:1,0].repeat(4,1,1)), unp_all_memOrigin)
            unp_all_memR = torch.mean(unp_all_memR,dim=0,keepdim=True)
            # st()

            if hyp.make_dense:        
                all_occ_memRs = self.vox_util.voxelize_xyz(self.all_xyz_camRs, self.Z, self.Y, self.X)[0:1]
                self.summ_writer.summ_occ('3D_inputs/all_occRs', all_occ_memRs)
                self.summ_writer.summ_occ('3D_inputs/all_occRs', self.all_occ_memX0)
                
            self.all_feat_memR_input = torch.cat([all_occ_memRs,all_occ_memRs*unp_all_memR],dim=1)
            self.summ_writer.summ_feat('3D_inputs//all_feat_memR_input', self.all_feat_memR_input, pca=True)
            # st()

            # st()
            self.summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occ_memXs, dim=1))
            self.summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occ_memRs, dim=1))
            self.summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unp_memXs, dim=1), torch.unbind(occ_memXs, dim=1))
            # st()

        # since we are using multiview data, all Rs are aligned
        # we'll encode X0 with the fast net, then warp to R
        # we'll use the X0 version for occ loss, to get max labels
        # we'll encode X1/R1 with slow net, and use this for emb loss
        feats_implicit_g_detach, feats_implicit_e, feats_implicit_n_detach = None, None, None
        if hyp.do_feat3d:
            assert(self.S==2)
            occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camXs[:,0], self.Z, self.Y, self.X)
            unp_memX0 = self.vox_util.unproject_rgb_to_mem(
                self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
            if hyp.do_rgb:
                occ_memX2_0 = self.vox_util.voxelize_xyz(self.xyz_camXs[:,0], self.Z2, self.Y2, self.X2)
                unp_memX2_0 = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_camXs[:,0], self.Z2, self.Y2, self.X2, self.pix_T_cams[:,0])                
            feat_memX0_input = torch.cat([occ_memX0, occ_memX0*unp_memX0], dim=1)
            feat3d_loss, feat_memX0, _ = self.feat3dnet(
                feat_memX0_input,
                self.summ_writer,
            )

            total_loss += feat3d_loss
            
            valid_memX0 = torch.ones_like(feat_memX0[:,0:1])
            # warp things to R0, for loss
            feat_memR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], feat_memX0)
            valid_memR = self.vox_util.apply_4x4_to_vox(self.camR0s_T_camXs[:, 0], valid_memX0)

            # self.summ_writer.summ_feat('feat3d/feat_memX1_input', feat_memX1_input, pca=True)
            self.summ_writer.summ_feat('feat3d/feat_memX0', feat_memX0, valid=valid_memX0, pca=True)
            self.summ_writer.summ_feat('feat3d/feat_memR', feat_memR, valid=valid_memR, pca=True)
            self.summ_writer.summ_oned('feat3d/valid_memR', valid_memR, bev=True, norm=False)

            if hyp.do_emb3d:
                occ_memR = self.vox_util.voxelize_xyz(self.xyz_camRs[:,1], self.Z, self.Y, self.X)
                unp_memX1 = self.vox_util.unproject_rgb_to_mem(self.rgb_camXs[:,1], self.Z, self.Y, self.X, self.pix_T_cams[:,1])
                unp_memR = self.vox_util.apply_4x4_to_vox(self.camRs_T_camXs[:,1], unp_memX1)
                feat_memR_input = torch.cat([occ_memR, occ_memR*unp_memR], dim=1)
                _, altfeat_memR, _ = self.feat3dnet_slow(feat_memR_input)
                altvalid_memR = torch.ones_like(altfeat_memR[:,0:1])
                self.summ_writer.summ_feat('feat3d/feat_memR_input', feat_memR_input, pca=True)
                self.summ_writer.summ_feat('feat3d/altfeat_memR', altfeat_memR, valid=altvalid_memR, pca=True)
                self.summ_writer.summ_oned('feat3d/altvalid_memR', altvalid_memR, bev=True, norm=False)

        if hyp.do_occ:
            # i think it's necessary to train feat3dnet with the occ sup directly; the other guy is frozen. this is maybe obvious
            # _, _, Z_, Y_, X_ = list(feat_memR.shape)
            occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camX0s_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2,
                agg=True)
            
            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_memX0_sup = 1.0 - (F.conv3d(1.0 - free_memX0_sup, weights, padding=1)).clamp(0, 1)

            # # crop to match feat 
            # occ_memX0_sup = self.crop_feat(occ_memX0_sup, self.crop)
            # free_memX0_sup = self.crop_feat(free_memX0_sup, self.crop)

            # feat_memX0 = self.pad_feat(feat_memX0, self.crop)
            # feat_memX0 = F.interpolate(feat_memX0, scale_factor=2, mode='trilinear')
            # valid_memX0 = self.pad_feat(valid_memX0, self.crop)
            # valid_memX0 = F.interpolate(valid_memX0, scale_factor=2, mode='trilinear')

            # feat_memX0 = self.crop_feat(feat_memX0, [crop*2+1 for crop in self.crop])
            # valid_memX0 = self.crop_feat(valid_memX0, [crop*2+1 for crop in self.crop])
            # occ_memX0_sup = self.crop_feat(occ_memX0_sup, [crop*2+1 for crop in self.crop])
            # free_memX0_sup = self.crop_feat(free_memX0_sup, [crop*2+1 for crop in self.crop])

            occ_loss, occ_memX0_pred, _ = self.occnet(
                feat_memX0, 
                occ_g=occ_memX0_sup,
                free_g=free_memX0_sup,
                valid=valid_memX0,
                summ_writer=self.summ_writer)
            total_loss += occ_loss
            
        
        if hyp.do_localdecoder and hyp.do_emb3d:
            # we have feat_memR and altfeat_memR. We'll use self.xyz_camRs[:,1] corresponding to altfeat_memR.
            # altfeat_memR will be embg and feat will be emge
            _, _, Z, Y, X = feat_memR.shape
            assert X == Y
            assert X == Z 

            xyz_camR_g = self.xyz_camRs[:,1]
            xyz_memR_g = self.vox_util.Ref2Mem(xyz_camR_g, Z, Y, X)
            c = torch.clamp(xyz_memR_g, 0, Z-1)
            occ_memR_g = occ_memRs[:,1]
            # st()
            # xyz_memR_g_int = xyz_memR_g.to(torch.int)


            xyz_camR_e = self.xyz_camRs[:,0]
            xyz_memR_e = self.vox_util.Ref2Mem(xyz_camR_e, Z, Y, X)
            xyz_memR_e = torch.clamp(xyz_memR_e, 0, Z-1)
            occ_memR_e = occ_memRs[:,0]

            xyz_mem_intesect_batches = []
            # st()
            if hyp.point_contrast:
                for batch in range(self.B):
                    ex_indices = torch.where(occ_memR_g[batch] * occ_memR_e[batch])
                    _,Z_inds,Y_inds,X_inds = ex_indices

                    xyz_mem_intesect = torch.stack((X_inds,Y_inds,Z_inds),dim=-1)
                    # to get the resolution the same
                    xyz_mem_intesect = xyz_mem_intesect/2.0
                    xyz_mem_intesect_batches.append(xyz_mem_intesect)
                
                minimum_points = min([i.shape[0] for i in xyz_mem_intesect_batches])
                xyz_mem_intesect_batches = [i[:minimum_points] for i in xyz_mem_intesect_batches]
                # st()
                xyz_mem_intesects = torch.stack(xyz_mem_intesect_batches)

                if xyz_mem_intesects.shape[1] == 0:
                    return torch.tensor(0.0), None, True
                # nearby_indices = self.find_nearby_indices(xyz_mem_intesects, xyz_memR_g)

                #generate points inside the visible voxels in both views:
                intercepts = torch.FloatTensor(*list(xyz_mem_intesects.shape)).uniform_(0, 0.5).cuda()

                xyz_mem_intesects = xyz_mem_intesects + intercepts

                xyz_memR_g = xyz_mem_intesects
                xyz_memR_g = torch.clamp(xyz_memR_g, 0, Z-1)



            # xyz_memR_sim = torch.matmul(xyz_memR_g_normalized.cpu(),xyz_memR_e_normalized_p.cpu())

            # torch.matmul(xyz_memR_e_normalized,xyz_memR_g_normalized)

            emb3d_g_implicit = altfeat_memR
            emb3d_g_implicit_detach = emb3d_g_implicit.detach()
            emb3d_e_implicit = feat_memR
            if hyp.implicit_camX:
                xyz_cam0_g = self.xyz_camX0s[:,1]
                xyz_mem0_g = self.vox_util.Ref2Mem(xyz_cam0_g, Z, Y, X)
                xyz_mem0_g = torch.clamp(xyz_mem0_g, 0, Z-1)                
                feats_implicit_g, feats_implicit_e, feats_implicit_n = self.localdecodernet(emb3d_g_implicit_detach, feat_memX0, xyz_memR_g, xyz_mem0_g=xyz_mem0_g, summ_writer = self.summ_writer)
            else:
                feats_implicit_g, feats_implicit_e, feats_implicit_n = self.localdecodernet(emb3d_g_implicit_detach, emb3d_e_implicit, xyz_memR_g, summ_writer = self.summ_writer)
            
            feats_implicit_g_detach = feats_implicit_g.detach()
            feats_implicit_n_detach = feats_implicit_n.detach()
            # st()
        elif hyp.point_contrast_og:
            # st()
            xyz_mem_intesect_batches = []
            xyz_camR_g = self.xyz_camRs[:,1]
            rgb_camR_g = self.rgb_camXs[:,1].reshape(self.B, 3,-1).permute(0,2,1)
            xyz_memR_g = self.vox_util.Ref2Mem(xyz_camR_g, self.Z, self.Y, self.X)
            xyz_memR_g = torch.clamp(xyz_memR_g, 0, self.Z-1)
            xyz_memR_g_normed = xyz_memR_g/self.Z
            xyz_feat_inp_g = torch.cat([xyz_memR_g_normed,xyz_memR_g_normed,rgb_camR_g],dim=-1)
            # st()
            # occ_memR_g = occ_memRs[:,1]
            # xyz_memR_g_int = xyz_memR_g.to(torch.int)


            xyz_camR_e = self.xyz_camRs[:,0]
            xyz_memR_e = self.vox_util.Ref2Mem(xyz_camR_e, self.Z, self.Y, self.X)
            xyz_memR_e = torch.clamp(xyz_memR_e, 0, self.Z-1)
            rgb_camR_e = self.rgb_camXs[:,0].reshape(self.B, 3,-1).permute(0,2,1)
            xyz_memR_e_normed = xyz_memR_e/self.Z
            xyz_feat_inp_e = torch.cat([xyz_memR_e_normed,xyz_memR_e_normed,rgb_camR_e],dim=-1)

            # xyz_feat_inp_ = __p(xyz_feat_inp)
            # st()
            xyz_feat_e = self.pointnet2(xyz_feat_inp_e)
            xyz_feat_g = self.pointnet2(xyz_feat_inp_g)

            total_num_inds = xyz_feat_g.shape[1]

            total_inds = torch.randperm(total_num_inds)
            # xyz_feat = torch.stack([xyz_feat_inp_g,xyz_feat_inp_e],dim=1)
            # self.pointnet2()
            indexes_e,indexes_g = utils.geom.find_closest(xyz_memR_g_normed,xyz_memR_e_normed)
            
            if indexes_e.shape[1]==0:
                return torch.tensor(0.0), None, True
            # xyz_memR_e_int = xyz_memR_e.to(torch.int)
            f_xyz_feat_e = []
            f_xyz_feat_g = []

            num_inds = indexes_e.shape[1]
            f_xyz_feat_n = xyz_feat_g[:,total_inds[:num_inds]]
            
            for i in range(self.B):
                f_xyz_feat_e.append(xyz_feat_e[i:i+1,indexes_e[i]])
                f_xyz_feat_g.append(xyz_feat_g[i:i+1,indexes_g[i]])

            f_xyz_feat_g = torch.cat(f_xyz_feat_g)
            f_xyz_feat_e = torch.cat(f_xyz_feat_e)
            # st()
            feats_implicit_g = f_xyz_feat_g.permute(0,2,1)
            feats_implicit_e = f_xyz_feat_e.permute(0,2,1)
            feats_implicit_n = f_xyz_feat_n.permute(0,2,1)


            feats_implicit_g_detach = feats_implicit_g.detach()
            feats_implicit_n_detach = feats_implicit_n.detach()
            # occ_memR_e = occ_memRs[:,0]            

            # for batch in range(self.B):
            #     ex_indices = torch.where(occ_memR_g[batch] * occ_memR_e[batch])
            #     _,Z_inds,Y_inds,X_inds = ex_indices
            #     xyz_mem_intesect = torch.stack((X_inds,Y_inds,Z_inds),dim=-1)
            #     # to get the resolution the same
            #     xyz_mem_intesect_batches.append(xyz_mem_intesect)
            
            # minimum_points = min([i.shape[0] for i in xyz_mem_intesect_batches])
            # xyz_mem_intesect_batches = [i[:minimum_points] for i in xyz_mem_intesect_batches]
            # xyz_mem_intesects = torch.stack(xyz_mem_intesect_batches)
            # utils.geom.find_closest(xyz_mem_intesects,xyz_memR_g_int)


        if hyp.summ_pca_points_3d or hyp.summ_pca_points_2d:
            if hyp.make_dense:
                all_xyz_camRs = self.all_xyz_camRs
                all_xyz_camX0s = self.all_xyz_camX0
                all_pix_T_cams = self.pix_T_cams[:,1]
            else:
                all_xyz_camRs = self.xyz_camRs[:,1]
                all_pix_T_cams = self.pix_T_cams[:,1]

            dimensions_to_use = [Z, Y, X]
            all_xyz_memRs = self.vox_util.Ref2Mem(all_xyz_camRs, dimensions_to_use[0],dimensions_to_use[1],dimensions_to_use[2])
            all_xyz_memRs = torch.clamp(all_xyz_memRs, 0, Z-1)
            feats_implicit_e_pca_only = self.localdecodernet(None, emb3d_e_implicit, all_xyz_memRs, summ_writer = self.summ_writer)
            # st()
            scale_mul = 2
            resolution = [i*scale_mul for i in dimensions_to_use]
            all_xyz_memRs = self.vox_util.Ref2Mem(all_xyz_camRs, resolution[0],resolution[1],resolution[2])
            all_xyz_memRs = torch.clamp(all_xyz_memRs, 0, Z*scale_mul-1)
            self.summ_writer.summ_pca3d_implicit("pca3d", all_xyz_memRs, feats_implicit_e_pca_only, resolution, self.vox_util,concat=self.all_feat_memR_input)

            self.summ_writer.summ_pca2d_implicit("pca2d", all_xyz_camRs, feats_implicit_e_pca_only, all_pix_T_cams, dimensions_to_use)

        
        if hyp.do_implicit_occ or hyp.do_tsdf_implicit_occ:
            # st()
            # feat_memX0, self.occupied_camX0, self.unoccupied_camX0
            _,_,Z,Y,X = feat_memX0.shape
            self.summ_writer.summ_pointcloud_on_rgb('Implicit/occupied', self.occupied_camX0, self.rgb_camXs[:, 0], self.pix_T_cams[:, 0])
            self.summ_writer.summ_pointcloud_on_rgb('Implicit/unoccupied', self.unoccupied_camX0, self.rgb_camXs[:, 0], self.pix_T_cams[:, 0])
            occupied_memX0 = self.vox_util.Ref2Mem(self.occupied_camX0, Z, Y, X)
            unoccupied_memX0 = self.vox_util.Ref2Mem(self.unoccupied_camX0, Z, Y, X)
            gt_occupancy_labels = torch.cat([torch.ones_like(occupied_memX0[:,:,0]), torch.zeros_like(unoccupied_memX0[:,:,0])], dim=1)
            gt_occupancy_memX0 = torch.cat([occupied_memX0, unoccupied_memX0], dim=1)
            logits, _ = self.localdecodernet_render_occ.localdecoder_occ(None, gt_occupancy_memX0, feat_memX0)
            p_r = dist.Bernoulli(logits=logits)

            loss_i = F.binary_cross_entropy_with_logits(logits, gt_occupancy_labels, reduction='none')
            loss = loss_i.mean()

            occ_e = (p_r.probs >= 0.5)
            iou = utils.implicit.compute_iou(occ_e.cpu().detach().numpy(), gt_occupancy_labels.cpu().detach().numpy()).mean()

            self.summ_writer.summ_scalar("Implicit/IOU", iou)
            total_loss = utils.misc.add_loss('Implicit/implicit_loss_occ', total_loss, loss, 1, self.summ_writer)

        if hyp.do_localdecoder_render:
            _, _, Z, Y, X = feat_memR.shape
            assert X == Y
            assert X == Z 

            xyz_camR_g = self.xyz_camRs[:,1]
            xyz_memR_g = self.vox_util.Ref2Mem(xyz_camR_g, Z, Y, X)
            xyz_memR_g = torch.clamp(xyz_memR_g, 0, Z-1)

            xyz_camX_colors_target, xyz_camX_proj = utils.geom.get_colors_for_pointcloud(self.xyz_camXs[:,1], self.rgb_camXs[:,1], self.pix_T_cams[:,1])
            rgb_g = xyz_camX_colors_target.permute(0,2,1)
            self.summ_writer.summ_pointcloud_on_rgb("imp_rendernet/pointcloud_on_rgb", self.xyz_camXs[:,1], self.rgb_camXs[:,1], self.pix_T_cams[:,1])
            rgb_loss, rgb_e = self.localdecodernet_render_occ(None, feat_memR, xyz_memR_g, summ_writer = self.summ_writer, rgb_g=rgb_g)
            # st()
            total_loss += rgb_loss
            rgb_vis_g = torch.zeros((1, 3, self.rgb_camXs[:,1].shape[2], self.rgb_camXs[:,1].shape[3])).cuda() - 0.5
            rgb_vis_e = torch.clone(rgb_vis_g)
            rgb_vis_g[0,:,xyz_camX_proj[0,:,1], xyz_camX_proj[0,:,0]] = rgb_g[0].permute(1,0)
            rgb_vis_e[0,:,xyz_camX_proj[0,:,1], xyz_camX_proj[0,:,0]] = rgb_e[0].permute(1,0)
            rgb_vis = torch.cat([rgb_vis_g, rgb_vis_e], dim=-1)
            self.summ_writer.summ_rgb("imp_rendernet/SDF_rendering", rgb_vis)                      
            # print(valid_memR)
            # feats_implicit_n, _ = self.localdecodernet(emb3d_g_implicit_detach, emb3d_e_implicit, xyz_memR_g, summ_writer = self.summ_writer)
        
        if hyp.do_emb3d:
            # compute 3D ML
            # st()
            if hyp.point_contrast_og:
                altvalid_memR, valid_memR, altfeat_memR, feat_memR = (None , None, None, None)
            else:
                valid_memR = valid_memR.round()
                altvalid_memR = altvalid_memR.round()

            emb3d_loss = self.emb3dnet(
                feat_memR,
                altfeat_memR,
                valid_memR,
                altvalid_memR,
                feats_implicit_g_detach,
                feats_implicit_e,
                feats_implicit_n_detach,
                self.summ_writer,[self.B,hyp.feat3d_dim,self.Z,self.Y,self.X])

            total_loss += emb3d_loss
        
        
        if hyp.do_rgb:
            rgb_memX2_0_g = unp_memX2_0* occ_memX2_0
            # st()
            rgb_loss, rgb_memX0_pred = self.rgbnet(
                feat_memX0,
                rgb_memX2_0_g,
                occ_memX2_0,#.repeat(1, 3, 1, 1, 1),
                self.summ_writer)
            total_loss += rgb_loss
        
        if hyp.do_render:
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
            assert(sy==0.5 or sy==1.0)
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            # assert(S==2) # else we should warp each feat in 1:
            feat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,1], self.camXs_T_camX0s[:,1], rgb_memX0_pred,
                hyp.view_depth, PH, PW, noise_amount=2.0)
            occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,1], self.camXs_T_camX0s[:,1], self.all_occ_memX0,
                hyp.view_depth, PH, PW, grid_z_vec=dists)

            if sx==0.5:
                rgb_X1 = utils.basic.downsample(self.rgb_camXs[:,1], 2)
                # valid_X1 = utils.basic.downsample(self.valid_camXs[:,1], 2)
            else:
                rgb_X1 = self.rgb_camXs[:,1]

            depth_X1 = self.depth_camXs[:,1]
            valid_X1 = self.valid_camXs[:,1]
            # st()
            render_loss, rgb_e, _, _ = self.rendernet(
                feat_proj,
                occ_proj,
                dists,
                rgb_g=rgb_X1,
                depth_g=depth_X1,
                valid=valid_X1,
                summ_writer=self.summ_writer)

            total_loss += render_loss

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.obj_clist_camR0 = utils.geom.get_clist_from_lrtlist(self.lrt_camR0s)
        self.obj_clist_camX0 = utils.geom.get_clist_from_lrtlist(self.lrt_camX0s)

        if self.include_vis:
            visX_g = []
            for s in list(range(self.S)):
                visX_g.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)

        if hyp.do_feat3d:
            feat_memX0s_input = torch.cat([
                self.occ_memX0s,
                self.unp_memX0s*self.occ_memX0s,
            ], dim=2)
            _, feat_memX0s_, _ = self.feat3dnet(
                __p(feat_memX0s_input),
                self.summ_writer,
            )
            feat_memX0s = __u(feat_memX0s_)
            # valid_memX0s = __u(valid_memX0s_)
            self.summ_writer.summ_feats('feat3d/feat_memX0s', torch.unbind(feat_memX0s, dim=1), pca=True)

            _, _, Z_, Y_, X_ = list(feat_memX0s[:,0].shape)
            obj_mask_memX0s = self.vox_util.assemble_padded_obj_masklist(
                self.lrt_camX0s,
                self.score_s,
                Z_, Y_, X_).squeeze(1)
            # only take the occupied voxels
            occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), Z_, Y_, X_))
            obj_mask_memX0s = obj_mask_memX0s * occ_memX0s

            for b in list(range(self.B)):
                if torch.sum(obj_mask_memX0s[b,0]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return total_loss, results, True
            self.summ_writer.summ_feats('track/obj_mask_memX0s', torch.unbind(obj_mask_memX0s, dim=1), pca=False)

            lrt_camX0s_e, point_counts, ious = utils.track.track_via_inner_products(
                self.lrt_camX0s, obj_mask_memX0s, feat_memX0s, self.vox_util)   

            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))
                
            lrt_camXs_e = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)
            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    'track/box_camX%d_e' % s, self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
                    self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
            
            obj_clist_camX0_e = utils.geom.get_clist_from_lrtlist(lrt_camX0s_e)

            dists = torch.norm(obj_clist_camX0_e - self.obj_clist_camX0, dim=2)
            # this is B x S
            mean_dist = utils.basic.reduce_masked_mean(dists, self.score_s)
            median_dist = utils.basic.reduce_masked_median(dists, self.score_s)
            # this is []
            self.summ_writer.summ_scalar('track/centroid_dist_mean', mean_dist.cpu().item())
            self.summ_writer.summ_scalar('track/centroid_dist_median', median_dist.cpu().item())
            
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
        else:
            
            # self.lrt_camXs = utils.geom.apply_4x4s_to_lrts(self.camXs_T_camRs, self.lrt_camRs)
            
            ious = torch.zeros([self.B, self.S]).float().cuda()
            for s in list(range(self.S)):
                # lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
                ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,0:1], self.lrt_camRs[:,s:s+1]).squeeze(1)
            for s in range(self.S):
                self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())
            self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, None, False

    def forward(self, feed):
        
        set_name = feed['set_name']
        if set_name=='test':
            just_gif = True
        else:
            just_gif = False
        feed['just_gif'] = just_gif
        
        ok = self.prepare_common_tensors(feed)

        if ok:
            if set_name=='train' or set_name=='val':
                return self.run_train(feed)
            elif set_name=='test':
                return self.run_test(feed)
        else:
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, False
        
        # # arriving at this line is bad
        # print('weird set_name:', set_name)
        # assert(False)
