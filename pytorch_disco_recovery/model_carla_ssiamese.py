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
from nets.motionregnet import Motionregnet

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

class CARLA_SSIAMESE(Model):
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
        self.model = CarlaSsiameseModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)
            
        if hyp.do_freeze_match:
            self.model.matchnet.eval()
            self.set_requires_grad(self.model.matchnet, False)

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
            # if hyp.do_export_stats:
            #     all_confs = np.zeros([hyp.max_iters, hyp.S_test], np.float32)

                
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
                        # confs = results['confs']
                        ious = ious[0].cpu().numpy()
                        # confs = confs[0].cpu().numpy()
                        # print('ious', ious)
                        all_ious[test_count] = ious
                        # all_confs[test_count] = confs
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        # mean_confs = np.mean(all_confs[:test_count], axis=0)
                        print('mean_ious', mean_ious)
                        # print('mean_confs', mean_confs)
                        
                        if hyp.do_export_vis:
                            visX_e = results['visX_e']
                            # these are lists, where each item is shaped 1 x 3 x 128 x 384
                            # vis_e = [utils_improc.back2color(im).cpu().numpy()[0] for im in visX_e]
                            vis_e = [im.cpu().numpy()[0] for im in visX_e]
                            vis_e = [np.transpose(vis, [1, 2, 0]) for vis in vis_e]

                            vis_bev_e = results['vis_bev_e']
                            # these are lists, where each item is shaped 1 x 3 x Z x X
                            vis_bev = [im.cpu().numpy()[0] for im in vis_bev_e]
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
                # 'all_confs': all_confs,
            })
            print('saved %s' % out_fn)
            

class CarlaSsiameseModel(nn.Module):
    def __init__(self):
        super(CarlaSsiameseModel, self).__init__()
        in_dim = 1
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=in_dim)
        if hyp.do_match:
            self.deglist = [-6, 0, 6]
            self.radlist = [utils_geom.deg2rad(deg) for deg in self.deglist]
            self.trim = 5
            self.matchnet = MatchNet(self.radlist)
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=in_dim)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
        if hyp.do_motionreg:
            self.motionregnet = Motionregnet()
            
        self.iou_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        # self.conf_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

    def place_object_at_delta(self, lrt_cam, xyz_cam, dt, dr, Z, Y, X, sz, sy, sx, vox_util):
        # this function places the object at the center of a mem tensor, plus some delta

        # dt is B x 3, containing dx, dy, dz (the translation delta)
        # dr is B x 3, containing drx, dry, drz (the rotation delta)
        # Z, Y, X are the resolution of the zoom
        # sz, sy, sx are the metric size of the zoom
        # B, C = list(center_cam.shape)
        # B, D = list(rot_cam.shape)
        # assert(C==3)
        # assert(D==3)
        B, D = list(lrt_cam.shape)

        # to help us create some mats:
        rot0 = utils_geom.eye_3x3(B)
        t0 = torch.zeros(B, 3).float().cuda()

        # # this takes us from cam to a system where the object is in the middle
        # objcenter_T_cam = utils_geom.merge_rt(rot0, -center_cam)
        # print('rot_cam', rot_cam.detach().cpu().numpy())
        # # norot_T_objcenter = utils_geom.merge_rt(utils_geom.eul2rotm(-rot_cam[:,0], rot_cam[:,1], -rot_cam[:,2]), t0)
        # norot_T_objcenter = utils_geom.merge_rt(utils_geom.eul2rotm(rot_cam[:,0], rot_cam[:,1], rot_cam[:,2]), t0)

        _, cam_T_norot = utils_geom.split_lrt(lrt_cam)
        norot_T_cam = utils_geom.safe_inverse(cam_T_norot)
        # now, actually, we do want some small rotation, given in dr
        yesrot_T_norot = utils_geom.merge_rt(utils_geom.eul2rotm(dr[:,0], dr[:,1], dr[:,2]), t0)
        # and finally we want some small displacement, given in dt
        final_T_yesrot = utils_geom.merge_rt(rot0, dt)

        # final_T_cam = utils_basic.matmul4(final_T_yesrot,
        #                                   yesrot_T_norot,
        #                                   norot_T_objcenter,
        #                                   objcenter_T_cam)
        final_T_cam = utils_basic.matmul3(final_T_yesrot,
                                          yesrot_T_norot,
                                          norot_T_cam)
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
        self.set_data_format = feed['set_data_format']

        
        # print('S = %d' % self.S)
        
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
        
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(__p(self.camR0s_T_camRs).inverse())
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        

        self.K = 1 # the traj data only has one object inside each ex

        box_camRs = feed["box_traj_camR"]
        score_s = feed["score_traj"]
        tid_s = torch.ones_like(score_s).long()
        # box_camRs is B x S x 9
        lrt_camRs = utils_misc.parse_boxes(box_camRs, self.origin_T_camRs)
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
        self.obj_scorelist = score_s
        
        #####################
        ## visualize what we got
        #####################
        # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)

        # print('set_name', self.set_name)
        # print('vox_size_X', self.vox_size_X)
        # print('vox_size_Y', self.vox_size_Y)
        # print('vox_size_Z', self.vox_size_Z)
        
        return True # OK

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
        self.rgb_camXs = feed["rgb_camXs"]
        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))


        sz = self.ZZ*self.vox_size_Z
        sy = self.ZY*self.vox_size_Y
        sx = self.ZX*self.vox_size_X

        # for frame0, place the object exactly in the middle, and nearly straight up
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        rand_rz = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/1.0, high=np.pi/1.0, size=[self.B])
        rand_rx = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_r = np.stack([rand_rx, rand_ry, rand_rz], axis=1) # this is B x 3
        dr_0 = torch.from_numpy(rand_r).float().cuda()*0.05 # this makes it near-straight

        occ_0s = []
        for ind, rad in enumerate(self.radlist):
            rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
            occ_0, lrt_0 = self.place_object_at_delta(
                self.lrt_camR0s[:,0],
                self.xyz_camR0s[:,0],
                dt_0, dr_0 + rad_,
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)
            self.summ_writer.summ_occ('3D_inputs/occ_0_%d' % ind, occ_0)
            occ_0s.append(occ_0)



            
        # for frame1, place the object somewhere NEAR dt_0
        dt_1 = np.random.uniform(low=-0.5*sz/2.0, high=0.5*sz/2.0, size=[self.B, 3])*0.5
        dt_1 = dt_0 + torch.from_numpy(dt_1).float().cuda()
        # place it at some orientation NEAR dr_0
        rand_rz = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/16.0, high=np.pi/16.0, size=[self.B]) # 11.25 degs
        rand_rx = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_r = np.stack([rand_rx, rand_ry, rand_rz], axis=1) # this is B x 3
        dr_1 = dr_0 + torch.from_numpy(rand_r).float().cuda()*0.5 # down to 5.62 here
        
        occ_1, lrt_1 = self.place_object_at_delta(
            self.lrt_camR0s[:,1],
            self.xyz_camR0s[:,1],
            dt_1, dr_1,
            self.ZZ, self.ZY, self.ZX,
            sz, sy, sx,
            self.vox_util)
        self.summ_writer.summ_occ('3D_inputs/occ_1', occ_1)
        
        if hyp.do_feat3D:
            feat_0s = []
            feat_0s_trimmed = []
            for ind, occ_0 in enumerate(occ_0s):
                # print('working on occ_0_%d' % ind)
                # featurize the object in frame0
                _, feat_0, _ = self.featnet3D(occ_0)
                feat_0s.append(feat_0)

                # trim out
                feat_0_trimmed = feat_0[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                occ_0_trimmed = occ_0[:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
                feat_0s_trimmed.append(feat_0_trimmed)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input' % ind, occ_0, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output' % ind, feat_0, pca=True)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_input_trimmed' % ind, occ_0_trimmed, pca=False)
                self.summ_writer.summ_feat('3D_feats/feat_0_%d_output_trimmed' % ind, feat_0_trimmed, pca=True)
            feat_0s_trimmed = torch.stack(feat_0s_trimmed, dim=1)
                
            feat_loss, feat_1, validR1 = self.featnet3D(occ_1, self.summ_writer)

            # print('search_lrt', search_lrt.cpu().numpy())
            total_loss += feat_loss
            # self.summ_writer.summ_feat('3D_feats/feat_1_input', search_occR1, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_1_input', occ_1, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_1_output', feat_1, pca=True)

        if hyp.do_match:
            assert(hyp.do_feat3D)
            
            obj_loc_search = self.vox_util.Ref2Zoom(
                self.obj_clist_camR0s[:,1].unsqueeze(1), lrt_1, self.ZZ, self.ZY, self.ZX).squeeze(1)
            self.summ_writer.summ_traj_on_occ('match/obj_loc_g',
                                              obj_loc_search.unsqueeze(1),
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
                
            corrs, rad_e, obj_loc_halfsearch_e, match_loss = self.matchnet(
                feat_0s_trimmed, # list of templates
                feat_1, # search region
                obj_loc_search*0.5, # gt position in halfsearch coords
                dr_1[:,1] - dr_0[:,1], # just the yaw delta
                use_window=False,
                summ_writer=self.summ_writer)
            total_loss += match_loss

            self.summ_writer.summ_traj_on_occ('match/obj_loc_e',
                                              obj_loc_halfsearch_e.unsqueeze(1)*2.0,
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
            
        self.summ_writer.summ_scalar('loss', total_loss)#.cpu().item())
        return total_loss, results, False

    def train_forecaster(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)


        # self.rgb_camX0 = feed["rgb_camX0"]
        # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camX0)

        # self.xyz_camX0 = feed["xyz_camX0"]
        # self.xyz_camX0 = self.xyz_camXs[:,0]
        # self.xyz_camX0 = feed["xyz_camX0"]
        # self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        # self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))
        # self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        # self.xyz_camR0 = utils_geom.apply_4x4(self.camRs_T_camXs[:,0], self.xyz_camX0)

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        
        # sz = (hyp.ZMAX-hyp.ZMIN)*2.0
        # sy = (hyp.YMAX-hyp.YMIN)*2.0
        # sx = hyp.XMAX-hyp.XMIN

        sz, sy, sx = 32, 32, 32

        # for frame0, place the object NEARLY in the middle, and NEARLY straight up
        rand_t = np.random.uniform(low=-0.5, high=0.5, size=[self.B, 3])
        dt_0 = torch.from_numpy(rand_t).float().cuda()
        rand_r = np.random.uniform(low=-np.pi/64.0, high=np.pi/64.0, size=[self.B, 3])
        dr_0 = torch.from_numpy(rand_r).float().cuda() # this makes it near-straight
        
        # dt_0 = torch.zeros(self.B, 3).float().cuda()
        # dr_0 = torch.zeros(self.B, 3).float().cuda()

        # xyz_agg = self.xyz_camR0s[:,min(0,hyp.motionreg_t_past-1-2):max(self.S,hyp.motionreg_t_past-1+2)].reshape(self.B, -1, 3)
        xyz_agg = self.xyz_camR0s[:,min(0,hyp.motionreg_t_past-1-2):hyp.motionreg_t_past-1].reshape(self.B, -1, 3)
        occ_0, lrt_0 = self.place_object_at_delta(
            self.lrt_camR0s[:,hyp.motionreg_t_past-1], # if t_past=4, we want ind3 to be our anchor
            # self.xyz_camR0s[:,hyp.motionreg_t_past-1],
            xyz_agg,
            dt_0, dr_0,
            self.ZZ, self.ZY, self.ZX,
            sz, sy, sx,
            self.vox_util)
        self.summ_writer.summ_occ('3D_inputs/occ_0', occ_0)

        clist_0 = self.vox_util.Ref2Zoom(self.obj_clist_camR0s, lrt_0, self.ZZ, self.ZY, self.ZX)
        clist_camR0 = self.obj_clist_camR0s.clone()
        # print('clist_orig', self.obj_clist_camR0s.detach().cpu().numpy(), self.obj_clist_camR0s.shape)
        # print('clist_0', clist_0.detach().cpu().numpy(), clist_0.shape)
        self.summ_writer.summ_traj_on_occ('3D_inputs/obj_traj_g',
                                          clist_0,
                                          occ_0, 
                                          self.vox_util, 
                                          already_mem=True,
                                          sigma=1)

        # print('clist_camR0[:,-1]', clist_camR0[:,-1].detach().cpu().numpy())
        # print('original clist_0', clist_0[:,-1].detach().cpu().numpy())
        flip = torch.rand(1)
        # flip = 1.0
        # print('flip', flip.detach().numpy())
        if flip > 0.5:
            # print('flip!', flip.detach().numpy())
            # flip the x part
            # occ_0 is B x C x Z x Y x X
            occ_0 = occ_0.flip(4)
            clist_0[:,:,0] = (self.ZX-1) - clist_0[:,:,0]
        # print('flipped clist_0', clist_0[:,-1].detach().cpu().numpy())

        if hyp.do_feat3D:
            _, feat_0, _ = self.featnet3D(occ_0)
            self.summ_writer.summ_feat('3D_feats/feat_0_input', occ_0, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_0_output', feat_0, pca=True)

        if hyp.do_motionreg:
            clist_past_0_g = clist_0[:,:hyp.motionreg_t_past].clone()
            clist_futu_0_g = clist_0[:,hyp.motionreg_t_past:].clone()
            clist_futu_camR0_g = clist_camR0[:,hyp.motionreg_t_past:].clone()
            clist_past_mask = torch.ones_like(clist_past_0_g[:,:,0:1])
            
            motionreg_loss, clist_futus_0_e = self.motionregnet(
                feat_0,
                clist_past_0_g,
                clist_past_mask, 
                clist_futu_0_g,
                is_test=(self.set_name=='test'), 
                summ_writer=self.summ_writer)
            total_loss += motionreg_loss

            if flip > 0.5:
                # print('flipping back')
                # flip back
                occ_0 = occ_0.flip(4)
                # print('(self.ZX-1) - clist_0[0,-1,0]', (self.ZX-1), clist_0[0,-1,0])
                clist_0[:,:,0] = (self.ZX-1) - clist_0[:,:,0]
                # print('after sub, clist_0[0,-1,0]:', clist_0[0,-1,0])
                clist_futu_0_g[:,:,0] = (self.ZX-1) - clist_futu_0_g[:,:,0]
                clist_futus_0_e[:,:,:,0] = (self.ZX-1) - clist_futus_0_e[:,:,:,0]
                
            # print('reset clist_0', clist_0[:,-1].detach().cpu().numpy())
            # input()

            # clists_0_e is B x K x S x 3
            clist_futus_0_e_ = clist_futus_0_e.reshape(self.B, hyp.motionreg_num_slots*hyp.motionreg_t_futu, 3)
            clist_futus_camR0_e_ = self.vox_util.Zoom2Ref(clist_futus_0_e_, lrt_0, self.ZZ, self.ZY, self.ZX)
            clist_futus_camR0_e = clist_futus_camR0_e_.reshape(self.B, hyp.motionreg_num_slots, hyp.motionreg_t_futu, 3)
            dists = torch.norm(clist_futus_camR0_e - clist_futu_camR0_g.unsqueeze(1), dim=3)
            # this is B x K x hyp.motionreg_t_futu

            mean_dist = torch.mean(dists)
            min_dist = torch.mean(torch.min(dists, dim=1)[0])
            self.summ_writer.summ_scalar('unscaled_motionreg/min_centroid_dist', min_dist)#.cpu().item())
            self.summ_writer.summ_scalar('unscaled_motionreg/mean_centroid_dist', mean_dist)#.cpu().item())

            
            if self.summ_writer.save_this:
                vis_clean = torch.zeros(1, 3, self.ZZ, self.ZX).byte().cpu()
                for k in list(range(hyp.motionreg_num_slots)):
                    vis = self.summ_writer.summ_traj_on_occ(
                        '',
                        clist_futus_0_e[0:1,k],
                        occ_0[0:1],
                        self.vox_util,
                        traj_g=(clist_futu_0_g[0:1] if (k==0) else None),
                        show_bkg=(True if (k==0) else False),
                        already_mem=True,
                        only_return=True,
                        sigma=1)
                    vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                    vis_clean[vis_new_any>0] = vis[vis_new_any>0]
                self.summ_writer.summ_rgb('motionreg/clist_futus_0_eg', utils_improc.preprocess_color(vis_clean))

                

                # print('clist_futus_camR0_e[:,:,-1]', clist_futus_camR0_e[:,:,-1].detach().cpu().numpy())
                # print('dists_futus_camR0_e[:,:,-1]', clist_futus_camR0_e[:,:,-1].detach().cpu().numpy())
                

                # occ_memR0 = self.vox_util.voxelize_xyz(self.xyz_camR0s[:,hyp.motionreg_t_past-1], self.Z, self.Y, self.X)
                occ_v, lrt_v = self.vox_util.voxelize_near_xyz(self.xyz_camR0s[0:1,hyp.motionreg_t_past-1],
                                                               self.obj_clist_camR0s[0:1,hyp.motionreg_t_past-1],
                                                               self.Z, self.Y, self.X,
                                                               sz=sz, sy=sy, sx=sx)
                vis_clean = torch.zeros(1, 3, self.Z, self.X).byte().cpu()
                for k in list(range(hyp.motionreg_num_slots)):
                    vis = self.summ_writer.summ_traj_on_occ(
                        '',
                        self.vox_util.Ref2Zoom(clist_futus_camR0_e[0:1,k], lrt_v[0:1], self.Z, self.Y, self.X),
                        occ_v[0:1],
                        self.vox_util,
                        traj_g=(self.vox_util.Ref2Zoom(clist_futu_camR0_g[0:1], lrt_v[0:1], self.Z, self.Y, self.X) if (k==0) else None),
                        show_bkg=(True if (k==0) else False),
                        already_mem=True,
                        only_return=True,
                        sigma=1)
                    vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                    vis_clean[vis_new_any>0] = vis[vis_new_any>0]
                self.summ_writer.summ_rgb('motionreg/clist_futus_mem0_eg', utils_improc.preprocess_color(vis_clean))
                

            
        self.summ_writer.summ_scalar('loss', total_loss)#.cpu().item())
        return total_loss, results, False

    def track_over_seq(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        assert(hyp.do_feat3D and (hyp.do_match))

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        # init the obj lrt with gt of frame0
        obj_lrt = self.lrt_camR0s[:,0]
        # place the object exactly in the middle, totally straight
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        dr_0 = torch.zeros(self.B, 3).float().cuda()

        sz = self.ZZ*self.vox_size_Z
        sy = self.ZY*self.vox_size_Y
        sx = self.ZX*self.vox_size_X

        # create a few deltas around this
        occ_0s = []
        for ind, rad in enumerate(self.radlist):
            occ_0, lrt_0 = self.place_object_at_delta(
                obj_lrt,
                self.xyz_camR0s[:,0],
                dt_0, dr_0 + rad,
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)
            self.summ_writer.summ_occ('3D_inputs/occ_0_%d' % ind, occ_0)
            occ_0s.append(occ_0)

        feat_0s = []
        feat_0s_trimmed = []
        for ind, occ_0 in enumerate(occ_0s):
            # featurize the object in frame0
            _, feat_0, _ = self.featnet3D(occ_0)
            feat_0s.append(feat_0)

            # trim out
            feat_0_trimmed = feat_0[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
            occ_0_trimmed = occ_0[:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
            feat_0s_trimmed.append(feat_0_trimmed)

            self.summ_writer.summ_feat('3D_feats/feat_0_%d_input' % ind, occ_0, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_0_%d_output' % ind, feat_0, pca=True)
            self.summ_writer.summ_feat('3D_feats/feat_0_%d_input_trimmed' % ind, occ_0_trimmed, pca=False)
            self.summ_writer.summ_feat('3D_feats/feat_0_%d_output_trimmed' % ind, feat_0_trimmed, pca=True)
        feat_0s_trimmed = torch.stack(feat_0s_trimmed, dim=1)
        
        lrt_camR0s_e = torch.zeros([self.B, self.S, 19]).float().cuda()
        obj_loc_es = torch.zeros([self.B, self.S, 3, self.ZZ, self.ZX]).byte().cpu()
        obj_loc_gs = torch.zeros([self.B, self.S, 3, self.ZZ, self.ZX]).byte().cpu()
        # confs = torch.zeros([self.B, self.S]).float().cuda()
        
        # track in frames 0 to N
        # (really only 1 to N is necessary, but 0 is good for debug)
        for s in list(range(0, self.S)):
            # create a search region around the previous lrt
            occ_1, lrt_1 = self.place_object_at_delta(
                obj_lrt,
                self.xyz_camR0s[:,s],
                torch.zeros(self.B, 3).float().cuda(),
                torch.zeros(self.B, 3).float().cuda(),
                self.ZZ, self.ZY, self.ZX,
                sz, sy, sx,
                self.vox_util)

            # featurize
            feat_loss, feat_1, validR1 = self.featnet3D(occ_1, self.summ_writer)

            # locate the object
            _, rad_e, obj_loc_halfsearch_e, _ = self.matchnet(
                feat_0s_trimmed, # list of templates
                feat_1, # search region
                torch.zeros(self.B, 3).float().cuda(), # gt position in halfsearch coords
                torch.zeros(self.B).float().cuda(), # gt yaw
            )
            # # record our confidence
            # confs[:,s] = conf
            # self.conf_pools[s].update(conf.cpu().numpy())
            # mean = self.conf_pools[s].mean()
            # self.summ_writer.summ_scalar('track/max_conf_%02d' % s, mean)

            # convert the loc answer to ref coords
            obj_loc_ref = self.vox_util.Zoom2Ref(
                obj_loc_halfsearch_e.unsqueeze(1), lrt_1,
                int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2)).squeeze(1)

            # now pack this into a final "lrt" answer
            obj_len, obj_rt = utils_geom.split_lrt(obj_lrt)
            obj_r, obj_t = utils_geom.split_rt(obj_rt)
            obj_r_update = utils_geom.eul2rotm(
                torch.zeros_like(rad_e), rad_e, torch.zeros_like(rad_e))
            obj_r = torch.matmul(obj_r, obj_r_update)
            obj_rt = utils_geom.merge_rt(obj_r, obj_loc_ref)
            obj_lrt = utils_geom.merge_lrt(obj_len, obj_rt)
            lrt_camR0s_e[:,s] = obj_lrt

            # vis the loc ans on the search region
            obj_loc_es[:,s] = self.summ_writer.summ_traj_on_occ('match/obj_loc_e_%d' % s,
                                                                obj_loc_halfsearch_e.unsqueeze(1)*2.0,
                                                                occ_1,
                                                                self.vox_util,
                                                                already_mem=True,
                                                                sigma=2,
                                                                only_return=True)
            obj_loc_search_g = self.vox_util.Ref2Zoom(self.obj_clist_camR0s[:,s].unsqueeze(1), lrt_1, self.ZZ, self.ZY, self.ZX).squeeze(1)
            obj_loc_gs[:,s] = self.summ_writer.summ_traj_on_occ('match/obj_loc_g_%d' % s,
                                                                obj_loc_search_g.unsqueeze(1),
                                                                occ_1,
                                                                self.vox_util,
                                                                already_mem=True,
                                                                sigma=2,
                                                                only_return=True)
        # end loop over s
        
        self.summ_writer.summ_rgbs('track/obj_loc_es', obj_loc_es.unbind(1))
        self.summ_writer.summ_rgbs('track/obj_loc_gs', obj_loc_gs.unbind(1))

        lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s, lrt_camR0s_e)
        lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs_e)
                
        obj_clist_camR0s_e = utils_geom.get_clist_from_lrtlist(lrt_camR0s_e)
        self.occ_memR0 = self.vox_util.voxelize_xyz(self.xyz_camR0s[:,0], self.Z, self.Y, self.X)
        self.summ_writer.summ_traj_on_occ('track/track_estim',
                                          obj_clist_camR0s_e,
                                          self.occ_memR0,
                                          self.vox_util,
                                          already_mem=False,
                                          sigma=2)
        self.summ_writer.summ_traj_on_occ('track/track_true',
                                          self.obj_clist_camR0s,#*self.obj_scorelist.unsqueeze(2),
                                          self.occ_memR0,
                                          self.vox_util, 
                                          already_mem=False,
                                          sigma=2)
        
        dists = torch.norm(obj_clist_camR0s_e - self.obj_clist_camR0s, dim=2)
        # this is B x S
        dist = utils_basic.reduce_masked_mean(dists, self.obj_scorelist)
        # this is []
        self.summ_writer.summ_scalar('track/centroid_dist', dist)#.cpu().item())

        means = []
        ious = torch.zeros([self.B, self.S]).float().cuda()
        for s in list(range(self.S)):
            ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,s:s+1], lrt_camRs_e[:,s:s+1]).squeeze(1)
            self.iou_pools[s].update(ious[:,s].detach().cpu().numpy())
            # print('iou_pools[%d]' % s, self.iou_pools[s].fetch())
            # print('iou_pools[%d].mean()' % s, self.iou_pools[s].mean())
            mean = self.iou_pools[s].mean()
            means.append(mean)
            self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, mean)
        results['ious'] = ious
        # results['confs'] = confs
        self.summ_writer.summ_scalar('track/mean_iou', np.mean(means))

        # vis boxes in perspective and bev
        if self.include_image_summs:
            self.rgb_camXs = feed["rgb_camXs"]
            visX_e = []
            for s in list(range(self.S)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,s],
                    torch.cat([self.lrt_camXs[:,s:s+1], lrt_camXs_e[:,s:s+1]], dim=1),
                    torch.cat([torch.ones([self.B,1]).float().cuda(), ious[:,s:s+1]], dim=1),
                    torch.arange(1,3).reshape(self.B, 2).long().cuda(),
                    self.pix_T_cams[:,0], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
            results['visX_e'] = visX_e

            vis_bev_e = []
            self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
            for s in list(range(self.S)):
                vis_bev_e.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.occ_memXs[:,s],
                    torch.cat([self.lrt_camXs[:,s:s+1], lrt_camXs_e[:,s:s+1]], dim=1),
                    torch.cat([torch.ones([self.B,1]).float().cuda(), ious[:,s:s+1]], dim=1),
                    torch.arange(1,3).reshape(self.B, 2).long().cuda(), # tids
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
            return self.train_forecaster(feed)
        elif set_name=='test':
            # # see testset perf of forecaster
            # return self.train_forecaster(feed)
            return self.track_over_seq(feed)
        else:
            print('weird set_name:', set_name)

            assert(False)
        
            
