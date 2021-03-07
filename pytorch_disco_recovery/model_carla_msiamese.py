import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np

from tensorboardX import SummaryWriter
from backend import saverloader, inputs

from model_base import Model
from nets.featnet3D import FeatNet3D
from nets.matchnet import MatchNet
from nets.occnet import OccNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.embnet2D import EmbNet2D
from nets.embnet3D import EmbNet3D

import torch.nn.functional as F

# import utils_vox
import vox_util
import utils_py
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net
from imageio import imsave
import skimage

np.set_printoptions(precision=2)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_MSIAMESE(Model):
    # take over __init__() from base
    def __init__(self, checkpoint_dir, log_dir):

        print('------ CREATING NEW MODEL ------')
        print(hyp.name)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.all_inputs = inputs.get_inputs()
        print("------ Done getting inputs ------")
        
        self.device = torch.device("cuda")
        
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaMsiameseModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
            

    # take over go() from base
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
            test_count = 0
            if hyp.do_export_stats:
                all_confs = np.zeros([hyp.max_iters, hyp.S_test], np.float32)
            
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
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_name'] = set_name
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

                    if hyp.do_test and (not returned_early):
                        ious = results['ious']
                        confs = results['confs']
                        ious = ious[0].cpu().numpy()
                        confs = confs[0].cpu().numpy()
                        # print('ious', ious)
                        all_ious[test_count] = ious
                        all_confs[test_count] = confs
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        mean_confs = np.mean(all_confs[:test_count], axis=0)
                        print('mean_ious', mean_ious)
                        print('mean_confs', mean_confs)
                        
                        if hyp.do_export_vis:
                            visX_e = results['visX_e']
                            # these are lists, where each item is shaped 1 x 3 x 128 x 384
                            vis_e = [utils_improc.back2color(im).cpu().numpy()[0] for im in visX_e]
                            vis_e = [np.transpose(vis, [1, 2, 0]) for vis in vis_e]

                            vis_bev_e = results['vis_bev_e']
                            # these are lists, where each item is shaped 1 x 3 x Z x X
                            vis_bev = [utils_improc.back2color(im).cpu().numpy()[0] for im in vis_bev_e]
                            utils_py.print_stats('vis_bev[0]', vis_bev[0])
                            vis_bev = [np.transpose(vis, [1, 2, 0]) for vis in vis_bev]

                            for fr, (vis, vis2) in enumerate(zip(vis_e, vis_bev)):
                                out_path = 'outs/%s_vis_both_%04d_%02d.png' % (hyp.name, test_count, fr)
                                # print('vis', vis.shape)
                                # print('vis2', vis2.shape)
                                # utils_py.print_stats('vis', vis)
                                # utils_py.print_stats('vis2 before resize', vis2)
                                vis2 = vis2.astype(np.float32)/255.0
                                vis2 = skimage.transform.resize(vis2, (384, 384))
                                vis2 = (vis2*255.0).astype(np.uint8)
                                # utils_py.print_stats('vis2 after resize', vis2)
                                # print('vis', vis.shape)
                                # print('vis2', vis2.shape)
                                vis = np.concatenate([vis, vis2], axis=0)
                                imsave(out_path, vis)
                                # print('saved %s' % out_path)

                    if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

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

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

        if hyp.do_export_stats:

            out_fn = '%s_output_dict.npy' % (hyp.name)
            np.save(out_fn, {
                'all_ious': all_ious,
                'all_confs': all_confs,
            })
            print('saved %s' % out_fn)
            

class CarlaMsiameseModel(nn.Module):
    def __init__(self):
        super(CarlaMsiameseModel, self).__init__()
        in_dim = 1
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=in_dim)
        if hyp.do_match:
            self.matchnet = MatchNet()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=in_dim)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
            
        self.iou_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        self.conf_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

    def place_object_at_delta(self, center_cam, rot_cam, xyz_cam, dt, dr, Z, Y, X, sz, sy, sx, vox_util):
        # this function places the object at the center of a mem tensor, plus some delta

        # dt is B x 3, containing dx, dy, dz (the translation delta)
        # dr is B x 3, containing drx, dry, drz (the rotation delta)
        # Z, Y, X are the resolution of the zoom
        # sz, sy, sx are the metric size of the zoom
        B, C = list(center_cam.shape)
        B, D = list(rot_cam.shape)
        assert(C==3)
        assert(D==3)

        # to help us create some mats:
        rot0 = utils_geom.eye_3x3(B)
        t0 = torch.zeros(B, 3).float().cuda()

        # this takes us from cam to a system where the object is in the middle
        objcenter_T_cam = utils_geom.merge_rt(rot0, -center_cam)
        norot_T_objcenter = utils_geom.merge_rt(utils_geom.eul2rotm(rot_cam[:,0], rot_cam[:,1], rot_cam[:,2]), t0)
        # now, actually, we do want some small rotation, given in dr
        yesrot_T_norot = utils_geom.merge_rt(utils_geom.eul2rotm(dr[:,0], dr[:,1], dr[:,2]), t0)
        # and finally we want some small displacement, given in dt
        final_T_yesrot = utils_geom.merge_rt(rot0, dt)

        final_T_cam = utils_basic.matmul4(final_T_yesrot,
                                          yesrot_T_norot,
                                          norot_T_objcenter,
                                          objcenter_T_cam)
        # now, we want this "final" centroid to be in the middle of the tensor,
        # so we subtract the midpoint of the metric bounds
        mid_xyz = np.array([sx/2.0, sy/2.0, sz/2.0]).reshape(1, 3)
        mid_xyz = torch.from_numpy(mid_xyz).float().cuda().repeat(B, 1)
        midmem_T_final = utils_geom.merge_rt(rot0, mid_xyz)
        midmem_T_cam = utils_basic.matmul2(midmem_T_final, final_T_cam)

        xyz_midmem = utils_geom.apply_4x4(midmem_T_cam, xyz_cam)
        occ_midmem, lrt_midmem = vox_util.voxelize_near_xyz(xyz_midmem, mid_xyz, Z, Y, X, sz=sz, sy=sy, sx=sx)
        lrt_cam = utils_geom.apply_4x4_to_lrtlist(midmem_T_cam.inverse(), lrt_midmem.unsqueeze(1)).squeeze(1)
        return occ_midmem, lrt_cam

        

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

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()                                                                 
        self.vox_util = vox_util.Vox_util(
            self.Z, self.Y, self.X, 
            self.set_name, scene_centroid=self.scene_centroid,
            assert_cube=True)

        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        
        self.rgb_camXs = feed["rgb_camXs"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(__p(self.camR0s_T_camRs).inverse())
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        
        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))


        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        self.K = 1 # the traj data only has one object inside each ex
        
        if set_data_format=='traj':
            box_camRs = feed["box_traj_camR"]
            score_s = feed["score_traj"]
            tid_s = torch.ones_like(score_s).long()
            # box_camRs is B x S x 9
            lrt_camRs = utils_geom.convert_boxlist_to_lrtlist(box_camRs)
            lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs)
            # lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0_T_camXs, lrt_camXs)
            lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, lrt_camRs)
            # these are B x S x 19
            self.box_camRs = box_camRs
            self.lrt_camXs = lrt_camXs
            self.lrt_camRs = lrt_camRs
            self.lrt_camR0s = lrt_camR0s

            self.obj_clist_camXs = utils_geom.get_clist_from_lrtlist(lrt_camXs)
            self.obj_clist_camRs = utils_geom.get_clist_from_lrtlist(lrt_camRs)
            self.obj_clist_camR0s = utils_geom.get_clist_from_lrtlist(lrt_camR0s)

            self.obj_rlist_camR0s = utils_geom.get_rlist_from_lrtlist(lrt_camR0s)
            # print('obj_rlist_camR0s', self.obj_rlist_camR0s)
            self.obj_rlist_camXs = utils_geom.get_rlist_from_lrtlist(lrt_camXs)
            # print('obj_rlist_camXs', self.obj_rlist_camXs)

            # ok, i have the orientation for each step
            # 

            self.obj_scorelist = score_s

            # # we'll do separate voxelization; use this as the vox size
            # self.vox_size = 0.125
            
            # _, lrt_0 = self.place_object_at_delta(
            #     self.obj_clist_camR0s[:,0],
            #     self.obj_rlist_camR0s[:,0],
            #     self.xyz_camR0s[:,0],
            #     torch.zeros([1, 3]).float().cuda(), # dt
            #     torch.zeros([1, 3]).float().cuda(), # dr
            #     self.ZZ, self.ZY, self.ZX,
            #     sz, sy, sx,
            #     self.vox_util)
            
            # we want to generate a "crop" at the same resolution as the regular voxels
            self.obj_occR0s_, self.template_lrts_ = self.vox_util.voxelize_near_xyz(
                __p(self.xyz_camR0s),
                __p(self.obj_clist_camR0s),
                self.ZZ,
                self.ZY,
                self.ZX,
                sz=(self.ZZ*self.vox_size_Z),
                sy=(self.ZY*self.vox_size_Y),
                sx=(self.ZX*self.vox_size_X))
            self.obj_occR0s = __u(self.obj_occR0s_)
            self.template_lrts = __u(self.template_lrts_)
            # print('template_lrts', self.template_lrts)

            obj_mask_templates = self.vox_util.assemble_padded_obj_masklist_within_region(
                self.lrt_camR0s[:,0:1], self.obj_scorelist[:,0:1], self.template_lrts[:,0],
                int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2), coeff=1.0)
            self.obj_mask_template = obj_mask_templates[:,0] # first template

            for b in list(range(self.B)):
                if torch.sum(self.obj_mask_template[b]) <= 8:
                    print('returning early, since there are not enough valid object points')
                    return False # NOT OK
        else:
            # center randomly 
            scene_centroid_x = np.random.uniform(-8.0, 8.0)
            scene_centroid_y = np.random.uniform(-1.5, 3.0)
            scene_centroid_z = np.random.uniform(10.0, 26.0)
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()                                                                 
            self.vox_util = vox_util.Vox_util(
                self.Z, self.Y, self.X, 
                self.set_name, scene_centroid=self.scene_centroid,
                assert_cube=True)
            self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

            self.vox_size_X = self.vox_util.default_vox_size_X
            self.vox_size_Y = self.vox_util.default_vox_size_Y
            self.vox_size_Z = self.vox_util.default_vox_size_Z
            

            

        #####################
        ## visualize what we got
        #####################
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)

        # self.deglist = [-10, -5, 0, 5, 10]
        # self.deglist = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
        # self.deglist = [-10, 0, 10]
        # self.deglist = [-5, 0, 5]
        # self.deglist = [-4, -2, 0, 2, 4]
        # self.deglist = [-6, -4, -2, 0, 2, 4, 6]
        # self.deglist = [-5, -2.5, 0, 2.5, 5]
        # self.deglist = [-8, -4, 0, 4, 8]
        # self.deglist = [-8, 0, 8]
        # self.deglist = [-12, 0, 12]
        self.deglist = [-6, 0, 6]
        # self.deglist = [-12, -6, 0, 6, 12]
        # self.deglist = [-8, -4, 0, 4, 8]
        self.trim = 5
        # self.deglist = [0]
        self.radlist = [utils_geom.deg2rad(deg) for deg in self.deglist]

        # print('set_name', self.set_name)
        # print('vox_size_X', self.vox_size_X)
        # print('vox_size_Y', self.vox_size_Y)
        # print('vox_size_Z', self.vox_size_Z)
        
        return True # OK

    def train_multiview(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        if hyp.do_feat3D:
            # feat_memX0s_input = torch.cat([
            #     self.occ_memX0s,
            #     # self.unp_memX0s*self.occ_memX0s,
            # ], dim=2)
            feat_memX0s_input = self.occ_memX0s.clone()
            feat3D_loss, feat_memX0s_, valid_memX0s_ = self.featnet3D(
                __p(feat_memX0s_input[:,1:]),
                self.summ_writer,
            )
            feat_memX0s = __u(feat_memX0s_)
            valid_memX0s = __u(valid_memX0s_)
            total_loss += feat3D_loss

            feat_memX0 = utils_basic.reduce_masked_mean(
                feat_memX0s,
                valid_memX0s.repeat(1, 1, hyp.feat3D_dim, 1, 1, 1),
                dim=1)
            valid_memX0 = torch.sum(valid_memX0s, dim=1).clamp(0, 1)
            self.summ_writer.summ_feat('3D_feats/feat_memX0', feat_memX0, valid=valid_memX0, pca=True)
            self.summ_writer.summ_feat('3D_feats/valid_memX0', valid_memX0, pca=False)

            if hyp.do_emb3D:
                _, altfeat_memX0, altvalid_memX0 = self.featnet3D_slow(feat_memX0s_input[:,0])
                self.summ_writer.summ_feat('3D_feats/altfeat_memX0', altfeat_memX0, valid=altvalid_memX0, pca=True)
                self.summ_writer.summ_feat('3D_feats/altvalid_memX0', altvalid_memX0, pca=False)

        # if hyp.do_occ:
        #     _, _, Z_, Y_, X_ = list(feat_memX0.shape)
        #     if encode_in_X0:
        #         occ_memX0_sup, free_memX0_sup, _, free_memX0s = self.vox_util.prep_occs_supervision(
        #             self.camX0s_T_camXs,
        #             self.xyz_camXs,
        #             Z_, Y_, X_,
        #             agg=True)
        #         self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memX0_sup)
        #         self.summ_writer.summ_occ('occ_sup/free_sup', free_memX0_sup)
        #         self.summ_writer.summ_occs('occ_sup/freeX0s_sup', torch.unbind(free_memX0s, dim=1))
        #         self.summ_writer.summ_occs('occ_sup/occX0s_sup', torch.unbind(self.occ_memX0s_half, dim=1))
        #         occ_loss, occ_memX0_pred = self.occnet(
        #             altfeat_memX0, 
        #             occ_memX0_sup,
        #             free_memX0_sup,
        #             altvalid_memX0, 
        #             self.summ_writer)
        #     else:
        #         occ_memR_sup, free_memR_sup, _, free_memRs = self.vox_util.prep_occs_supervision(
        #             self.camRs_T_camXs,
        #             self.xyz_camXs,
        #             Z_, Y_, X_,
        #             agg=True)
        #         self.summ_writer.summ_occ('occ_sup/occ_sup', occ_memR_sup)
        #         self.summ_writer.summ_occ('occ_sup/free_sup', free_memR_sup)
        #         self.summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(free_memRs, dim=1))
        #         self.summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(self.occ_memRs_half, dim=1))

        #         occ_loss, occ_memR_pred = self.occnet(
        #             altfeat_memR, 
        #             occ_memR_sup,
        #             free_memR_sup,
        #             altvalid_memR,
        #             self.summ_writer)
                
        #     total_loss += occ_loss
                    
        if hyp.do_emb3D:
            # compute 3D ML
            
            emb_loss_3D = self.embnet3D(
                feat_memX0,
                altfeat_memX0,
                valid_memX0.round(),
                altvalid_memX0.round(),
                self.summ_writer)
            
            total_loss += emb_loss_3D

        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
    
    def train_matcher(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        assert(self.S==2)

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)
        
        if hyp.do_test:
            # return early
            return total_loss, results, True

        # ok, my idea here is:
        # extract frame0 at some rotation, which is norot0+delta0
        # extract frame1 at some other rotation, which is norot1+delta0+delta1
        # randomize delta0 very widely; randomize delta1 to be in some reasonable window

        # vox_size = 0.25

        
        sz = self.ZZ*self.vox_size_Z
        sy = self.ZY*self.vox_size_Y
        sx = self.ZX*self.vox_size_X

        # for frame0, place the object exactly in the middle, but at some random orientation
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        rand_rz = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/1.0, high=np.pi/1.0, size=[self.B])
        rand_rx = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_r = np.stack([rand_rx, rand_ry, rand_rz], axis=1) # this is B x 3
        dr_0 = torch.from_numpy(rand_r).float().cuda()*0.1

        occ_0s = []
        lrt_0s = []
        for ind, rad in enumerate(self.radlist):
            rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
            occ_0, lrt_0 = self.place_object_at_delta(
                self.obj_clist_camR0s[:,0],
                self.obj_rlist_camR0s[:,0],
                self.xyz_camR0s[:,0],
                dt_0, dr_0 + rad_,
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)
            self.summ_writer.summ_occ('3D_inputs/occ_0_%d' % ind, occ_0)
            occ_0s.append(occ_0)
            lrt_0s.append(lrt_0)
        # occ_0s = torch.stack(occ_0s, dim=1)
        # lrt_0s = torch.stack(lrt_0s, dim=1)

        # for frame1, place the object somewhere NEAR dt_0
        # dt_1 = np.random.uniform(low=-0.5*sz/2.0, high=0.5*sz/2.0, size=[self.B, 3])
        dt_1 = np.random.uniform(low=-0.5*sz/2.0, high=0.5*sz/2.0, size=[self.B, 3])*0.5
        # print('sx, sy, sz', sx, sy, sz)
        # print('dt_1', dt_1)
        dt_1 = dt_0 + torch.from_numpy(dt_1).float().cuda()
        # dt_1 = dt_0.clone()
        # place it at some orientation NEAR dr_0
        rand_rz = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/16.0, high=np.pi/16.0, size=[self.B]) # 11.25 degs
        rand_rx = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_r = np.stack([rand_rx, rand_ry, rand_rz], axis=1) # this is B x 3
        # print('rand_ry', rand_ry)
        # print('rand_r0', rand_r)
        dr_1 = dr_0 + torch.from_numpy(rand_r).float().cuda()*0.5 # down to 5.62 here
        # dr_1 = dr_0.clone()
        # print('dr_0', dr_0.detach().cpu().numpy())
        # print('dr_1', dr_1.detach().cpu().numpy())
        
        dr_1_ = dr_0 + (self.obj_rlist_camR0s[:,1] - self.obj_rlist_camR0s[:,0])
        dr_1 = (dr_1 + dr_1_)/2.0
        
        
        occ_1, lrt_1 = self.place_object_at_delta(
            self.obj_clist_camR0s[:,1],
            self.obj_rlist_camR0s[:,1],
            self.xyz_camR0s[:,1],
            dt_1, dr_1,
            self.ZZ, self.ZY, self.ZX,
            sz, sy, sx,
            self.vox_util)
        mask_1 = self.vox_util.assemble_padded_obj_masklist_within_region(
            self.lrt_camR0s[:,1:2], self.obj_scorelist[:,1:2], lrt_1,
            int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2), coeff=1.0).squeeze(1)
        self.summ_writer.summ_occ('3D_inputs/occ_1', occ_1)
        self.summ_writer.summ_occ('3D_inputs/mask_1', mask_1)
        
        if hyp.do_feat3D:
            feat_0s = []
            feat_0s_trimmed = []
            for ind, occ_0 in enumerate(occ_0s):
                # print('working on occ_0_%d' % ind)
                # featurize the object in frame0
                _, feat_0, _ = self.featnet3D(
                    occ_0,
                    None)
                feat_0s.append(feat_0)

                # trim out
                feat_0_trimmed = feat_0[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                occ_0_trimmed = occ_0[:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
                feat_0s_trimmed.append(feat_0_trimmed)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input' % ind, occ_0, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output' % ind, feat_0, pca=True)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input_trimmed' % ind, occ_0_trimmed, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output_trimmed' % ind, feat_0_trimmed, pca=True)

                # print('feat_0', feat_0.shape)
                # print('feat_0_trimmed', feat_0_trimmed.shape)
                
            # occ_0s_ = __p(occ_0s)
            # _, feat_0s_, _ = self.featnet3D(
            #     occ_0s_,
            #     None)
            # feat_0s = __u(feat_0s_)
            # # trim out
            # feat_0s_trimmed = feat_0s[:,:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
            # occ_0s_trimmed = occ_0s[:,:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
            # occ_0s = torch.unbind(occ_0s, dim=1)
            # occ_0s_trimmed = torch.unbind(occ_0s_trimmed, dim=1)
            # feat_0s = torch.unbind(feat_0s, dim=1)
            # feat_0s_trimmed = torch.unbind(feat_0s_trimmed, dim=1)
            # self.summ_writer.summ_feats('3D_feats/feat_0s_input', occ_0s, pca=False)
            # self.summ_writer.summ_feats('3D_feats/feat_0s_output', feat_0s, pca=True)
            # self.summ_writer.summ_feats('3D_feats/feat_0s_input_trimmed', occ_0s_trimmed, pca=False)
            # self.summ_writer.summ_feats('3D_feats/feat_0s_output_trimmed', feat_0s_trimmed, pca=True)

            
            SZ, SY, SX = self.ZZ, self.ZY, self.ZX
            
            feat_loss, feat_1, validR1 = self.featnet3D(
                occ_1,
                self.summ_writer)

            # print('search_lrt', search_lrt.cpu().numpy())
            total_loss += feat_loss
            # self.summ_writer.summ_feat('3D_feats/feat_1_input', search_occR1, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_1_input', occ_1, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_1_output', feat_1, pca=True)

        if hyp.do_match:
            assert(hyp.do_feat3D)
            
            obj_loc_search = self.vox_util.Ref2Zoom(
                self.obj_clist_camR0s[:,1].unsqueeze(1), lrt_1, SX, SY, SZ, additive_pad=0.0).squeeze(1)
            # print('obj_loc_halfmem', obj_loc_halfmem.cpu().numpy())
            self.summ_writer.summ_traj_on_occ('match/obj_loc_g',
                                              obj_loc_search.unsqueeze(1),
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
                
            corrs, rad_e, obj_loc_search_e, _, match_loss = self.matchnet(
                feat_0s_trimmed, # list of templates
                self.radlist, 
                feat_1, # search region
                obj_loc_search*0.5, # gt position in halfsearch coords
                dr_1[:,1] - dr_0[:,1], # just the yaw delta
                use_window=False,
                summ_writer=self.summ_writer)
            total_loss += match_loss

            self.summ_writer.summ_traj_on_occ('match/obj_loc_e',
                                              obj_loc_search_e.unsqueeze(1)*2.0,
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
            # corr, xyz_offset, match_loss = self.matchnet(
            # # the middle of the halfsearch coords is obj_loc_ref from the previous iter
            # search_mid = np.array([SX/4.0, SY/4.0, SZ/4.0], np.float32).reshape([1, 3])
            # search_mid = torch.from_numpy(search_mid).float().to('cuda')
            # obj_loc_halfmem = self.vox_util.Ref2Mem(obj_loc_ref.unsqueeze(1), self.Z2, self.Y2, self.X2).squeeze(1)
            # obj_loc_halfmem = (obj_loc_search - search_mid) + obj_loc_halfmem
            # obj_loc_mem = obj_loc_halfmem * 2.0
            # # this is B x 3, and in mem coords

            # # obj_loc_ref = self.vox_util.Mem2Ref(obj_loc_mem.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
            # obj_clist_memR0s_e = torch.stack(peak_xyzs_mem, dim=1)
            # obj_rlist_memR0s_e = torch.stack(peak_rots_mem, dim=1)
            # self.summ_writer.summ_traj_on_occ('track/estim_traj',
            #                                   obj_clist_memR0s_e,
            #                                   self.occ_memR0s[:,0],
            #                                   self.vox_util,
            #                                   already_mem=True,
            #                                   sigma=2)

            # utils_basic.print_stats_py('train corr', corr.detach().cpu().numpy())
            # self.summ_writer.summ_histogram('corr_train', corr)
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def track_over_seq(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # return total_loss, results, False
        assert(hyp.do_feat3D and (hyp.do_match))

        # for frame0, place the object exactly in the middle, totally straight
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        dr_0 = torch.zeros(self.B, 3).float().cuda()

        sz = self.ZZ*self.vox_size_Z
        sy = self.ZY*self.vox_size_Y
        sx = self.ZX*self.vox_size_X

        occ_0s = []
        lrt_0s = []
        for ind, rad in enumerate(self.radlist):
            occ_0, lrt_0 = self.place_object_at_delta(
                self.obj_clist_camR0s[:,0],
                self.obj_rlist_camR0s[:,0],
                self.xyz_camR0s[:,0],
                dt_0, dr_0 + rad,
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)
            self.summ_writer.summ_occ('3D_inputs/occ_0_%d' % ind, occ_0)
            occ_0s.append(occ_0)
            lrt_0s.append(lrt_0)

        if hyp.do_feat3D:
            feat_0s = []
            feat_0s_trimmed = []
            for ind, occ_0 in enumerate(occ_0s):
                # featurize the object in frame0
                _, feat_0, _ = self.featnet3D(
                    occ_0,
                    None)
                feat_0s.append(feat_0)

                # trim out
                feat_0_trimmed = feat_0[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                occ_0_trimmed = occ_0[:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
                feat_0s_trimmed.append(feat_0_trimmed)

                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input' % ind, occ_0, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output' % ind, feat_0, pca=True)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input_trimmed' % ind, occ_0_trimmed, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output_trimmed' % ind, feat_0_trimmed, pca=True)
        
        # # featurize the object in frame0
        # _, obj_featR0, _ = self.featnet3D(
        #     self.obj_occR0s[:,0],
        #     None)
        # self.summ_writer.summ_feat('3D_feats/obj_featR0_output', obj_featR0, pca=True)
        # obj_featR0_trimmed = obj_featR0[:,:,trim:-trim:,trim:-trim:,trim:-trim:]
        # self.summ_writer.summ_feat('3D_feats/search_feaR0_trimmed', obj_featR0_trimmed, pca=True)

        # init the obj location with gt of frame0
        obj_loc_ref_prev_prev = self.obj_clist_camR0s[:,0]
        obj_loc_ref_prev = self.obj_clist_camR0s[:,0]
        obj_loc_ref = self.obj_clist_camR0s[:,0]
        obj_r = self.obj_rlist_camR0s[:,0]

        # track in frames 0 to N
        # (really only 1 to N is necessary, but 0 is good for debug)
        search_occR0s = []
        search_featR0s = []
        peak_xyzs_mem = []
        peak_rots_mem = []

        # for robustnet
        lrt_camR0s_e = []

        # make the search size 2x the zoom size
        # SZ, SY, SX = self.ZZ*2, self.ZY*2, self.ZX*2
        SZ, SY, SX = self.ZZ, self.ZY, self.ZX

        # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # self.occ_memX0s_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z2, self.Y2, self.X2))
        # self.occ_memXs_half = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z2, self.Y2, self.X2))
        # self.occ_memRs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camRs), self.Z, self.Y, self.X))
        self.occ_memR0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camR0s), self.Z, self.Y, self.X))

        # self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0_T_camXs, self.unp_memXs)
        # self.unp_memXs_half = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z2, self.Y2, self.X2, __p(self.pix_T_cams)))
        # self.unp_memX0s_half = self.vox_util.apply_4x4s_to_voxs(self.camX0_T_camXs, self.unp_memXs_half)
        
        confs = torch.zeros([self.B, self.S]).float().cuda()
        for s in list(range(0, self.S)):
            # vel = obj_loc_ref - obj_loc_ref_prev
            # obj_loc_ref_search = obj_loc_ref + vel
            # obj_loc_ref_search = obj_loc_ref.clone()
            # search_occRi, search_lrt = self.vox_util.voxelize_near_xyz(
            #     self.xyz_camR0s[:,s],
            #     obj_loc_ref_search,
            #     SZ, SY, SX,
            #     sz=(SZ*self.vox_size_Z),
            #     sy=(SY*self.vox_size_Y),
            #     sx=(SX*self.vox_size_X))
            # search_occR0s.append(search_occRi)

            # _, search_featRi, _ = self.featnet3D(
            #     search_occRi,
            #     None)
            # search_featR0s.append(search_featRi)

            occ_1, lrt_1 = self.place_object_at_delta(
                obj_loc_ref,
                obj_r,
                self.xyz_camR0s[:,s],
                torch.zeros([1, 3]).float().cuda(), # dt
                torch.zeros([1, 3]).float().cuda(), # dr
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)

            feat_loss, feat_1, validR1 = self.featnet3D(
                occ_1,
                self.summ_writer)
            
            if hyp.do_match:
                corrs, rad_e, obj_loc_halfsearch_e, conf, match_loss = self.matchnet(
                    feat_0s_trimmed, # list of templates
                    self.radlist, 
                    feat_1, # search region
                    torch.zeros(self.B, 3).float().cuda(), # gt position in halfsearch coords
                    torch.zeros(self.B).float().cuda(), # gt yaw
                )

                # conflist = []
                # for corr in corrlist:
                #     conf = utils_samp.sample3D(corr, xyz_e, Z, Y, X)
                # print('corrlist', torch.stack(corrlist, dim=1).shape)
                # print('xyz_e', xyz_e.unsqueeze(1).shape)
                # conf = utils_samp.bilinear_sample3D(torch.stack(corrlist, dim=1), xyz_e.unsqueeze(1))#, Z, Y, X)
                # print('conf', conf.detach().cpu().numpy())
                # print('rad_e', rad_e.detach().cpu().numpy())
                # assume the max here is a good approx of the
                # print('conf', conf.shape)
                # max_conf = torch.max(conf, dim=1)n
                # print('max_conf', max_conf.shape)
                # utils_misc.add_loss('match/max_conf', 0, max_conf, 0, summ_writer)

                
                # corrs = corrs.reshape(self.B, -1)
                # corrs = torch.nn.functional.softmax(corrs, dim=1)
                # corrs = corrs.reshape(self.B, -1, int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2))

                # conf = utils_samp.bilinear_sample3D(corrs, obj_loc_halfsearch_e.unsqueeze(1))
                # # assume the max here is a good approx of the sample
                # # print('conf', conf.shape)
                # # max_conf = torch.max(conf, dim=1)
                # # print('max_conf', max_conf.shape)
                # # utils_misc.add_loss('match/max_conf', 0, max_conf, 0, summ_writer)
                # max_conf = torch.max(conf, dim=1)[0]
                # # # print('max_conf', max_conf.shape)
                # # # self.summ_writer.summ_scalar('track/max_conf', max_conf.cpu().item())
                # # self.summ_writer.summ_scalar('track/max_conf_%02d' % s, max_conf.cpu().item())

                confs[:,s] = conf
                self.conf_pools[s].update(conf.cpu().numpy())
                mean = self.conf_pools[s].mean()
                self.summ_writer.summ_scalar('track/max_conf_%02d' % s, mean)
                
                
                
                
                # corr, rad_e, obj_loc_halfsearch, _ = self.matchnet(
                #     obj_featR0_trimmed, # template
                #     self.radlist, 
                #     search_featRi, # search region
                #     torch.zeros(self.B, 3).float().cuda(), # gt position in search coords
                #     use_window=False,
                # )
                # corr is B x 1 x Z x Y x X
                # obj_loc_search is B x 3


                # obj_loc_search = obj_loc_halfsearch * 2.0
                # obj_loc_ref = self.vox_util.Zoom2Ref(obj_loc_search,

                # print('obj_loc_halfsearch', obj_loc_halfsearch.detach().cpu().numpy())
                # obj_loc_halfsearch = torch.ones_like(obj_loc_halfsearch)*15.5
                # print('obj_loc_halfsearch updated:', obj_loc_halfsearch.detach().cpu().numpy())
                
                obj_loc_ref = self.vox_util.Zoom2Ref(obj_loc_halfsearch_e.unsqueeze(1),
                                                     lrt_1,
                                                     # search_lrt,
                                                     int(SZ/2), int(SY/2), int(SX/2))
                obj_loc_mem = self.vox_util.Ref2Mem(obj_loc_ref, self.Z, self.Y, self.X)

                obj_loc_ref = obj_loc_ref.squeeze(1)
                obj_loc_mem = obj_loc_mem.squeeze(1)
                # print('obj_loc_halfsearch', obj_loc_halfsearch.detach().cpu().numpy())

                # # the middle of the halfsearch coords is obj_loc_ref from the previous iter
                # search_mid = np.array([SX/4.0, SY/4.0, SZ/4.0], np.float32).reshape([1, 3])
                # search_mid = torch.from_numpy(search_mid).float().to('cuda')
                # obj_loc_halfmem = self.vox_util.Ref2Mem(obj_loc_ref.unsqueeze(1), self.Z2, self.Y2, self.X2).squeeze(1)
                # obj_loc_halfmem = (obj_loc_halfsearch - search_mid) + obj_loc_halfmem
                # obj_loc_mem = obj_loc_halfmem * 2.0
                # # this is B x 3, and in mem coords

                # print('obj_loc_mem', obj_loc_mem.detach().cpu().numpy())

                # just take the time0 rotation 
                # obj_r = self.obj_rlist_camR0s[:,0]

                obj_dr = torch.stack([torch.zeros_like(rad_e), rad_e, torch.zeros_like(rad_e)], dim=1)
                obj_r = obj_r + obj_dr

                peak_xyzs_mem.append(obj_loc_mem)
                peak_rots_mem.append(obj_r)

                # # update the obj loc for the next step
                # obj_loc_ref = self.vox_util.Mem2Ref(obj_loc_mem.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
                
        obj_clist_memR0s_e = torch.stack(peak_xyzs_mem, dim=1)
        obj_rlist_memR0s_e = torch.stack(peak_rots_mem, dim=1)
        self.summ_writer.summ_traj_on_occ('track/estim_traj',
                                          obj_clist_memR0s_e,
                                          self.occ_memR0s[:,0],
                                          self.vox_util,
                                          already_mem=True,
                                          sigma=2)
        self.summ_writer.summ_traj_on_occ('track/true_traj',
                                          self.obj_clist_camR0s*self.obj_scorelist.unsqueeze(2),
                                          self.occ_memR0s[:,0],
                                          self.vox_util, 
                                          already_mem=False,
                                          sigma=2)

        obj_clist_camR0s_e = self.vox_util.Mem2Ref(obj_clist_memR0s_e, self.Z, self.Y, self.X)
        # this is B x S x 3
        dists = torch.norm(obj_clist_camR0s_e - self.obj_clist_camR0s, dim=2)
        # this is B x S
        dist = utils_basic.reduce_masked_mean(dists, self.obj_scorelist)
        # this is []
        self.summ_writer.summ_scalar('track/centroid_dist', dist.cpu().item())

        obj_lens = self.box_camRs[:,:,3:6]

        box_camR0s_e = torch.cat([obj_clist_camR0s_e, obj_lens, obj_rlist_memR0s_e], dim=2)
        # print('box_camR0s_e', box_camR0s_e.detach().cpu().numpy())
        lrt_camR0s_e = utils_geom.convert_boxlist_to_lrtlist(box_camR0s_e)

        # lrt_camRs_e = utils_geom.convert_boxlist_to_lrtlist(box_camRs_e)
        # this is B x S x 19
        lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s, lrt_camR0s_e)
        lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs_e)
               

        means = []
        ious = torch.zeros([self.B, self.S]).float().cuda()
        for s in list(range(self.S)):
            ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,s:s+1], lrt_camRs_e[:,s:s+1]).squeeze(1)
            self.iou_pools[s].update(ious[:,s].detach().cpu().numpy())
            # print('iou_pools[%d]' % s, self.iou_pools[s].fetch())
            # print('iou_pools[%d].mean()' % s, self.iou_pools[s].mean())
            # input()
            mean = self.iou_pools[s].mean()
            means.append(mean)
            self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, mean)
        results['ious'] = ious
        results['confs'] = confs
        self.summ_writer.summ_scalar('track/mean_iou', np.mean(means))

        if self.include_image_summs:
            
            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,s],
                    torch.cat([self.lrt_camXs[:,s:s+1], lrt_camXs_e[:,s:s+1]], dim=1),
                    torch.cat([torch.ones([self.B,1]).float().cuda(), ious[:,s:s+1]], dim=1),
                    # torch.ones([self.B,1]).float().cuda(),
                    torch.arange(1,3).reshape(self.B, 2).long().cuda(),
                    # torch.ones([self.B,2]).long().cuda(),
                    # self.obj_scorelist[:,s:s+1], tid_s[:,s:s+1],
                    self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
            results['visX_e'] = visX_e

            vis_bev_e = []
            self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
            for s in list(range(self.S)):
                heightmap = self.summ_writer.summ_occ('', self.occ_memXs[:,s], only_return=True)
                # print('heightmap', heightmap.shape)
                # colormap = self.summ_writer.summ_oned('', heightmap, bev=True, maxval=self.Y, norm=False, only_return=True)
                vis_bev_e.append(self.summ_writer.summ_lrtlist_bev(
                    # '', self.unp_memXs[:,s], self.occ_memXs[:,s],
                    # '', colormap, self.occ_memXs[:,s],
                    # '', heightmap, self.occ_memXs[:,s],
                    '', self.occ_memXs[:,s],
                    torch.cat([self.lrt_camXs[:,s:s+1], lrt_camXs_e[:,s:s+1]], dim=1),
                    torch.cat([torch.ones([self.B,1]).float().cuda(), ious[:,s:s+1]], dim=1),
                    # torch.ones([self.B,1]).float().cuda(),
                    torch.arange(1,3).reshape(self.B, 2).long().cuda(), # tids
                    # torch.ones([self.B,2]).long().cuda(),
                    # self.obj_scorelist[:,s:s+1], tid_s[:,s:s+1],
                    self.vox_util,
                    only_return=True))
            self.summ_writer.summ_rgbs('track/box_bev_e', vis_bev_e)
            results['vis_bev_e'] = vis_bev_e

        return total_loss, results, False
        
    def forward(self, feed):
        set_name = feed['set_name']
        # print('-'*100)
        # print('working on %s' % set_name)

        ok = self.prepare_common_tensors(feed)
        total_loss = torch.tensor(0.0).cuda()
        
        if not ok:
            print('not enough object points; returning early')
            return total_loss, None, True

        if set_name=='train':
            return self.train_matcher(feed)
        elif set_name=='val':
            return self.train_matcher(feed)
            # # return total_loss, None, True
            # return self.train_multiview(feed)
        elif set_name=='test':
            # return total_loss, None, True
            return self.track_over_seq(feed)
        else:
            print('weird set_name:', set_name)

            assert(False)
        
