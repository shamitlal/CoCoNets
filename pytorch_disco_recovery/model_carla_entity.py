import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.matchnet import MatchNet
from nets.occnet import OccNet
from nets.rendernet import RenderNet
from nets.rgbnet import RgbNet
from backend import saverloader, inputs

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc
import utils.track

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

# the idea here is to optimize into a set of templates
# along with a detector net.
# together, they should output:
# what the background is, and where each of the objects are
# we use a fixed number of objects
# they do not all have to be present


class CARLA_ENTITY(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaEntityModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        # if hyp.do_freeze_entity:
        #     self.model.entitynet.eval()
        #     self.set_requires_grad(self.model.entitynet, False)
            
    # take over go() from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

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

        set_latent_lists = []
        set_latent_optim_lists = []
        set_origin_T_cami_lists = []
        for set_input, set_b in zip(set_inputs, set_batch_sizes):
            set_len = len(set_input.dataset)
            print('set_len:', set_len)
            set_latent_list = [None]*set_len
            set_latent_optim_list = []
            for si in list(range(set_len)):
                # set_latent_optim_list.append(torch.optim.SGD([self.model.zi], lr=hyp.lr*2.0))
                set_latent_optim_list.append(torch.optim.Adam([self.model.zi], lr=hyp.lr*100.0))
            
            set_origin_T_cami_list = [None]*set_len
            set_latent_lists.append(set_latent_list)
            set_latent_optim_lists.append(set_latent_optim_list)
            set_origin_T_cami_lists.append(set_origin_T_cami_list)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': hyp.lr},
            # {'params': self.model.bkg, 'lr': hyp.lr}
        ])
        
        model_state_dict = self.model.state_dict()
        for k in model_state_dict.keys():
            print('key', k)
        
        self.start_iter = saverloader.load_weights(self.model, None)
        if hyp.latents_init:
            latent_list, latent_optim_list, origin_T_cami_list = saverloader.load_latents(hyp.latents_init)
            ind = set_names.index('train')
            print('putting these into ind', ind)
            set_latent_lists[ind] = latent_list
            set_latent_optim_lists[ind] = latent_optim_list
            set_origin_T_cami_lists[ind] = origin_T_cami_list
            
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
                 set_latent_list,
                 set_latent_optim_list,
                 set_origin_T_cami_list,
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
                set_latent_lists,
                set_latent_optim_lists,
                set_origin_T_cami_lists,
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

                    # zi_np = np.random.randn(set_batch_size, 4, int(hyp.Z/2), int(hyp.Y/2), int(hyp.X/2)).astype(np.float32)
                    zi_np = np.random.randn(set_batch_size, 4, int(hyp.Z), int(hyp.Y), int(hyp.X)).astype(np.float32)
                    origin_T_cami_np = np.zeros((set_batch_size, 4, 4), np.float32)
                    for b in list(range(set_batch_size)):
                        data_ind_b = data_ind[b]
                        # print('data_ind_b', data_ind_b)
                        # print('len set', len(set_latent_list))

                        zi_np_b = set_latent_list[data_ind_b]
                        origin_T_cami_np_b = set_origin_T_cami_list[data_ind_b]
                        if (zi_np_b is not None) and (origin_T_cami_np_b is not None):
                            # then this var must have been saved/optimized before
                            # use it
                            # print('using data from', data_ind_b)
                            zi_np[b] = zi_np_b
                            origin_T_cami_np[b] = origin_T_cami_np_b
                            # print('init with zi_np_b[0,0,:5,:5]', zi_np_b[0,0,:5,:5])
                        else:
                            print('this is the first time encountering index %d; initializing with random normal, and origin_T_camX0' % data_ind_b)
                            origin_T_camXs = feed["origin_T_camXs"]
                            origin_T_cami_np[b] = origin_T_camXs[b,0].cpu().numpy()
                    # print('origin_T_cami_np', origin_T_cami_np.shape)
                            
                    feed_cuda["origin_T_cami"] = torch.from_numpy(origin_T_cami_np).cuda()
                    feed_cuda["zi"] = torch.from_numpy(zi_np).cuda()
                    
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
                            for b in list(range(set_batch_size)):
                                zi_optim = set_latent_optim_list[data_ind[b]]
                                zi_optim.zero_grad()

                            loss.backward()

                            self.optimizer.step()
                            for b in list(range(set_batch_size)):
                                zi_optim = set_latent_optim_list[data_ind[b]]
                                zi_optim.step()


                            # ok, self.model.zi is holding new updated data, which we want to store
                            zi_np = self.model.zi.data.detach().cpu().numpy()
                            # zi_np = project_l2_ball_py(zi_np)
                            
                            # print('zi_np', zi_np)
                            for b in list(range(set_batch_size)):
                                # zi_cuda_b = zi_cuda[b]
                                # zi_np_b = zi_np[b]
                                # print('zi_cuda_b', zi_cuda_b.
                                # zi_np_b = set_latent_list[data_inds[b]]
                                # origin_T_cami[data_inds[b]] = zi_np_b
                                # origin_T_cami_np_b = set_origin_T_cami_list[data_inds[b]]
                                prev = set_latent_list[data_ind[b]]
                                new = zi_np[b]

                                set_latent_list[data_ind[b]] = zi_np[b]
                                set_origin_T_cami_list[data_ind[b]] = origin_T_cami_np[b]

                        # if hyp.do_emb3D:
                        #     def update_slow_network(slow_net, fast_net, beta=0.999):
                        #         param_k = slow_net.state_dict()
                        #         param_q = fast_net.named_parameters()
                        #         for n, q in param_q:
                        #             if n in param_k:
                        #                 param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
                        #         slow_net.load_state_dict(param_k)
                        #     update_slow_network(self.model.featnet3D_slow, self.model.featnet3D)
                        
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

                ind = set_names.index('train')
                saverloader.save_latents(set_latent_lists[ind],
                                         set_latent_optim_lists[ind],
                                         set_origin_T_cami_lists[ind],
                                         self.checkpoint_dir,
                                         step)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

class CarlaEntityModel(nn.Module):
    def __init__(self):
        super(CarlaEntityModel, self).__init__()
        
        self.crop = (18,18,18)
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)

        if hyp.do_render:
            self.rendernet = RenderNet()

        if hyp.do_occ:
            self.occnet = OccNet()

        if hyp.do_rgb:
            self.rgbnet = RgbNet()
            
        self.zi = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z), int(hyp.Y), int(hyp.X)]).float().cuda()
        self.zi = torch.autograd.Variable(self.zi, requires_grad=True)

        # self.bkg = torch.zeros([hyp.batch_sizes['train'], 4, int(hyp.Z), int(hyp.Y), int(hyp.X)]).float().cuda()
        # self.bkg = torch.autograd.Variable(self.bkg, requires_grad=True)

    def place_scene_at_dr(self, rgb_mem, xyz_cam, dr, Z, Y, X, vox_util):
        # this function voxelizes the scene with some rotation delta

        # dr is B x 3, containing drx, dry, drz (the rotation delta)
        # Z, Y, X are the resolution of the zoom
        # sz, sy, sx are the metric size of the zoom
        B, N, D = list(xyz_cam.shape)
        assert(D==3)

        # to help us create some mats:
        rot0 = utils.geom.eye_3x3(B)
        t0 = torch.zeros(B, 3).float().cuda()

        camr_T_cam = utils.geom.merge_rt(utils.geom.eul2rotm(dr[:,0], dr[:,1], dr[:,2]), t0)

        xyz_camr = utils.geom.apply_4x4(camr_T_cam, xyz_cam)
        occ_memr = vox_util.voxelize_xyz(xyz_camr, Z, Y, X)
        rgb_memr = vox_util.apply_4x4_to_vox(camr_T_cam, rgb_mem)
        return occ_memr, rgb_memr
            
    def crop_feat(self, feat_pad):
        Z_pad, Y_pad, X_pad = self.crop
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    
    def pad_feat(self, feat):
        Z_pad, Y_pad, X_pad = self.crop
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad
    
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        self.zi.data = feed["zi"]
        # self.zi = torch.autograd.Variable(feed["zi"], requires_grad=True)
        # self.zi.data = feed["zi"]
        self.origin_T_cami = feed["origin_T_cami"]
        # print('origin_T_cami', self.origin_T_cami.shape)
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        # if self.set_name=='test':
        #     self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        # elif self.set_name=='val':
        #     self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        # else:
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camis_T_camXs = utils.geom.get_cami_T_camXs(self.origin_T_cami, self.origin_T_camXs)
        self.camXs_T_camis = __u(utils.geom.safe_inverse(__p(self.camis_T_camXs)))
        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camis = __u(utils.geom.apply_4x4(__p(self.camis_T_camXs), __p(self.xyz_camXs)))

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_camXs = feed['rgb_camXs']
        self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))


        st()
        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camXs_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        self.summ_writer.summ_oned('inputs/depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        # self.summ_writer.summ_oned('inputs/valid_camX0_after', self.valid_camXs[:,0], norm=False)




        boxlists = feed["boxlists"]
        self.scorelist_s = feed["scorelists"]
        self.tidlist_s = feed["tidlists"]
        boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
        origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
        lrtlist_camRs_, _ = utils.misc.parse_boxes(boxlists_, origin_T_camRs_)
        self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        self.lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camX0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        self.summ_writer.summ_lrtlist(
            'inputs/boxlist_g',
            self.rgb_camXs[0:1,0],
            self.lrtlist_camXs[0:1,0],
            self.scorelist_s[0:1,0],
            self.tidlist_s[0:1,0],
            self.pix_T_cams[0:1,0])
        
        obj_masklist_memX0 = self.vox_util.assemble_padded_obj_masklist(
            self.lrtlist_camX0s[:,0],
            self.scorelist_s[:,0],
            self.Z, self.Y, self.X)
        # this is B x N x 1 x Z x Y x X
        self.obj_mask_memX0 = torch.sum(obj_masklist_memX0, dim=1).clamp(0, 1)
        self.summ_writer.summ_oned('inputs/obj_mask_memX0', self.obj_mask_memX0, bev=True, max_along_y=True, norm=False)

        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        
        self.rgb_memis = self.vox_util.apply_4x4s_to_voxs(self.camis_T_camXs, self.rgb_memXs)
        self.occ_memis = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camis), self.Z, self.Y, self.X))

        if hyp.do_feat3d:

            self.rgb_memis = self.vox_util.apply_4x4s_to_voxs(self.camis_T_camXs, self.rgb_memXs)
            self.occ_memis = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camis), self.Z, self.Y, self.X))

            feat_memi_input = torch.cat([
                self.occ_memis[:,0],
                self.occ_memis[:,0]*self.rgb_memis[:,0]], dim=1)

            self.lrtlist_camis = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camis_T_camXs), __p(self.lrtlist_camXs)))
            obj_masklist_memi = self.vox_util.assemble_padded_obj_masklist(
                self.lrtlist_camis[:,0],
                self.scorelist_s[:,0],
                self.Z, self.Y, self.X, additive_coeff=1.0)
            obj_mask_memi = torch.sum(obj_masklist_memi, dim=1).clamp(0, 1)

            self.summ_writer.summ_feat('3d_feats/feat_memi_input_unmasked', feat_memi_input, pca=True)
            feat_memi_input = feat_memi_input * (1.0 - obj_mask_memi)
            self.summ_writer.summ_feat('3d_feats/feat_memi_input', feat_memi_input, pca=True)
            feat_loss, feat_halfmemi = self.feat3dnet(
                feat_memi_input, norm=False, summ_writer=None)
            self.summ_writer.summ_feat('3d_feats/feat_halfmemi', feat_halfmemi, pca=True)

            rgb_memi = feat_halfmemi[:,1:]
            occ_memi = feat_halfmemi[:,0:1]
            # occ_memi = F.sigmoid(feat_halfmemi[:,0:1])
            self.summ_writer.summ_occ('3d_feats/occ_halfmemi', F.sigmoid(occ_memi))

            rgb_memi = self.pad_feat(rgb_memi)
            occ_memi = self.pad_feat(occ_memi)

            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5 or sx==1.0) # else we need a fancier downsampler
            assert(sy==0.5 or sy==1.0)
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))
            feat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], rgb_memi,
                hyp.view_depth, PH, PW, noise_amount=2.0)
            occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], occ_memi,
                hyp.view_depth, PH, PW, grid_z_vec=dists)

            if sx==0.5:
                rgb_X00 = utils.basic.downsample(self.rgb_camXs[:,0], 2)
                valid_X00 = utils.basic.downsample(self.valid_camXs[:,0], 2)
            else:
                rgb_X00 = self.rgb_camXs[:,0]
                depth_X00 = self.depth_camXs[:,0]
                valid_X00 = self.valid_camXs[:,0]
            _, rgb_e, _, _ = self.rendernet(
                feat_proj,
                occ_proj,
                dists)
            self.summ_writer.summ_rgb('3d_feats/rgb_e', rgb_e)
            
            
        
        # sy = float(self.PH)/float(self.H)
        # sx = float(self.PW)/float(self.W)
        # pix_T_cam0 = utils.geom.scale_intrinsics(self.pix_T_cams[:,0], sx, sy)
        
        # rgb_proj = self.vox_util.apply_pixX_T_memR_to_voxR(
        #     pix_T_cam0, self.camis_T_camXs[:,0], self.rgb_memis[:,0],
        #     hyp.view_depth, self.PH, self.PW, logspace_slices=False)
        # occ_proj = self.vox_util.apply_pixX_T_memR_to_voxR(
        #     pix_T_cam0, self.camis_T_camXs[:,0], self.occ_memis[:,0],
        #     hyp.view_depth, self.PH, self.PW, logspace_slices=False)
        # # this is B x C x D x Y x X
        # rgb_vis = utils.basic.reduce_masked_mean(rgb_proj, occ_proj.repeat(1, 3, 1, 1, 1), dim=2)
        self.summ_writer.summ_rgb('inputs/rgb_camX0', self.rgb_camXs[:,0])
        # self.summ_writer.summ_rgb('2D_inputs/rgb_vis', rgb_vis)

        if hyp.do_occ:
            occ_memi_sup, free_memi_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camis_T_camXs,
                self.xyz_camXs,
                self.Z, self.Y, self.X,
                agg=True)

            # be more conservative with "free"
            weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
            free_memi_sup = 1.0 - (F.conv3d(1.0 - free_memi_sup, weights, padding=1)).clamp(0, 1)
            free_memi_sup = 1.0 - (F.conv3d(1.0 - free_memi_sup, weights, padding=1)).clamp(0, 1)

            occ_loss, occ_memi_pred, _, _ = self.occnet(
                self.zi[:,0:1],
                occ_memi_sup,
                free_memi_sup,
                torch.ones_like(free_memi_sup),
                self.summ_writer)
            total_loss += occ_loss
        
        if hyp.do_rgb:
            rgb_memi_g = utils.basic.reduce_masked_mean(
                self.rgb_memis, self.occ_memis.repeat(1, 1, 3, 1, 1, 1),
                dim=1)
            rgb_loss, rgb_memi_pred = self.rgbnet(
                self.zi[:,1:4],
                rgb_memi_g,
                self.occ_memis[:,0],#.repeat(1, 3, 1, 1, 1),
                self.summ_writer)
            total_loss += rgb_loss
        
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
            projpix_T_cams = __u(utils.geom.scale_intrinsics(__p(self.pix_T_cams), sx, sy))

            # assert(S==2) # else we should warp each feat in 1:

            feat_proj, dists = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], self.zi[:,1:],
                hyp.view_depth, PH, PW, noise_amount=2.0)
            occ_proj, _ = self.vox_util.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], self.camXs_T_camis[:,0], self.zi[:,0:1],
                hyp.view_depth, PH, PW, grid_z_vec=dists)

            if sx==0.5:
                rgb_X00 = utils.basic.downsample(self.rgb_camXs[:,0], 2)
                valid_X00 = utils.basic.downsample(self.valid_camXs[:,0], 2)
            else:
                rgb_X00 = self.rgb_camXs[:,0]
                depth_X00 = self.depth_camXs[:,0]
                valid_X00 = self.valid_camXs[:,0]
            
            # print('dists', dists.detach().cpu().numpy())
            # print('rgb_X00', rgb_X00.shape)
            # print('feat_proj', feat_proj.shape)
            # print('occ_proj', occ_proj.shape)
            # print('dists', dists.shape)

            # dists = torch.linspace(
            #     self.vox_util.ZMIN,
            #     self.vox_util.ZMAX,
            #     steps=hyp.view_depth,
            #     dtype=torch.float32,
            #     device=torch.device('cuda'))
            
            # decode the perspective volume into an image

            render_loss, rgb_e, _, _ = self.rendernet(
                feat_proj,
                occ_proj,
                dists,
                rgb_g=rgb_X00,
                depth_g=depth_X00,
                valid=valid_X00,
                summ_writer=self.summ_writer)
            total_loss += render_loss

            



        # self.zi.data = utils.basic.l2_normalize(self.zi.data, dim=1)
        self.summ_writer.summ_feat('3d_feats/zi', self.zi, pca=True)
        # self.summ_writer.summ_feat('3D_feats/zi', self.zi.data, pca=False)
        # # self.summ_writer.summ_histogram('3D_feats/zi_hist', self.zi.data)

        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(self.zi, absolute=True)
        smooth_vox = torch.mean(dz+dy+dx, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        # total_loss = utils.misc.add_loss('feat3D/smooth_loss', total_loss, smooth_loss, hyp.feat3D_smooth_coeff, self.summ_writer)
        total_loss = utils.misc.add_loss('feat3d/smooth_loss', total_loss, smooth_loss, 0.1, self.summ_writer)
        # total_loss = utils.misc.add_loss('feat3d/smooth_loss', total_loss, smooth_loss, 0.0, self.summ_writer)

            

        # # assert(self.S==2)

        # origin_T_cam0 = self.origin_T_cams[:, 0]
        # origin_T_cam1 = self.origin_T_cams[:, 1]
        # cam0_T_cam1 = utils.basic.matmul2(utils.geom.safe_inverse(origin_T_cam0), origin_T_cam1)

        # # let's immediately discard the true motion and make some fake motion

        # xyz0_cam0 = self.xyz_cams[:,0]
        # xyz1_cam0 = utils.geom.apply_4x4(cam0_T_cam1, self.xyz_cams[:,1])

        # # camX_T_cam0 = utils.geom.get_random_rt(
        # xyz_cam_g, rx, ry, rz = utils.geom.get_random_rt(
        #     self.B,
        #     r_amount=4.0,
        #     t_amount=2.0,
        #     sometimes_zero=False,
        #     return_pieces=True)
        # rot = utils.geom.eul2rotm(rx*0.1, ry, rz*0.1)
        # camX_T_cam0 = utils.geom.merge_rt(rot, xyz_cam_g)
        
        # cam0_T_camX = utils.geom.safe_inverse(camX_T_cam0)
        # xyz1_camX = utils.geom.apply_4x4(camX_T_cam0, xyz1_cam0)

        # occ0_mem0 = self.vox_util.voxelize_xyz(xyz0_cam0, self.Z, self.Y, self.X)
        # occ1_memX = self.vox_util.voxelize_xyz(xyz1_camX, self.Z, self.Y, self.X)

        # rgb0_mem0 = self.vox_util.unproject_rgb_to_mem(
        #     self.rgb_cams[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
        # rgb1_mem1 = self.vox_util.unproject_rgb_to_mem(
        #     self.rgb_cams[:,1], self.Z, self.Y, self.X, self.pix_T_cams[:,1])
        
        # rgb1_memX = self.vox_util.apply_4x4_to_vox(
        #     utils.basic.matmul2(camX_T_cam0, cam0_T_cam1), rgb1_mem1)
        
        # self.summ_writer.summ_occs('inputs/occ_mems', [occ0_mem0, occ1_memX])
        # self.summ_writer.summ_unps('inputs/rgb_mems', [rgb0_mem0, rgb1_memX], [occ0_mem0, occ1_memX])

        # if hyp.do_feat3d:
        #     feat_mem0_input = torch.cat([occ0_mem0, occ0_mem0*rgb0_mem0], dim=1)
        #     feat_memX_input = torch.cat([occ1_memX, occ1_memX*rgb1_memX], dim=1)
        #     feat_loss0, feat_halfmem0 = self.feat3dnet(feat_mem0_input, self.summ_writer)
        #     feat_loss1, feat_halfmemX = self.feat3dnet(feat_memX_input, self.summ_writer)
        #     total_loss += feat_loss0 + feat_loss1

        # # if hyp.do_entity:
        # #     assert(hyp.do_feat3d)
        # #     entity_loss, cam0_T_cam1_e, _ = self.entitynet(
        # #         feat_halfmem0,
        # #         feat_halfmemX,
        # #         cam0_T_camX,
        # #         self.vox_util,
        # #         self.summ_writer)
        # #     total_loss += entity_loss

        # if hyp.do_match:
        #     assert(hyp.do_feat3d)

        #     occ_rs = []
        #     rgb_rs = []
        #     feat_rs = []
        #     feat_rs_trimmed = []
        #     for ind, rad in enumerate(self.radlist):
        #         rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
        #         occ_r, rgb_r = self.place_scene_at_dr(
        #             rgb0_mem0, self.xyz_cams[:,0], rad_,
        #             self.Z, self.Y, self.X, self.vox_util)
        #         occ_rs.append(occ_r)
        #         rgb_rs.append(rgb_r)

        #         inp_r = torch.cat([occ_r, occ_r*rgb_r], dim=1)
        #         _, feat_r = self.feat3dnet(inp_r)
        #         feat_rs.append(feat_r)
        #         feat_r_trimmed = feat_r[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
        #         # print('feat_r_trimmed', feat_r_trimmed.shape)
        #         feat_rs_trimmed.append(feat_r_trimmed)
                
        #     self.summ_writer.summ_occs('entity/occ_rs', occ_rs)
        #     self.summ_writer.summ_unps('entity/rgb_rs', rgb_rs, occ_rs)
        #     self.summ_writer.summ_feats('entity/feat_rs', feat_rs, pca=True)
        #     self.summ_writer.summ_feats('entity/feat_rs_trimmed', feat_rs_trimmed, pca=True)

        #     match_loss, camX_T_cam0_e, cam0_T_camX_e = self.matchnet(
        #         torch.stack(feat_rs_trimmed, dim=1), # templates
        #         feat_halfmemX, # search region
        #         self.vox_util,
        #         xyz_cam_g=xyz_cam_g,
        #         rad_g=ry,
        #         summ_writer=self.summ_writer)
        #     total_loss += match_loss

        #     occ1_mem0_e = self.vox_util.apply_4x4_to_vox(cam0_T_camX_e, occ1_memX)
        #     occ1_mem0_g = self.vox_util.apply_4x4_to_vox(cam0_T_camX, occ1_memX)

        #     self.summ_writer.summ_occs('entity/occ_mems_0', [occ0_mem0, occ1_memX])
        #     self.summ_writer.summ_occs('entity/occ_mems_e', [occ0_mem0, occ1_mem0_e.round()])
        #     self.summ_writer.summ_occs('entity/occ_mems_g', [occ0_mem0, occ1_mem0_g.round()])
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)
        # data_ok = False
        
        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            elif self.set_name=='test':
                return self.run_sfm(feed)
                # return self.run_orb(feed)
                # return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
