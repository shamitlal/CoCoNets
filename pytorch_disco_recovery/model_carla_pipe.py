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
from nets.confnet import ConfNet
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


class CARLA_PIPE(Model):
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
        self.model = CarlaPipeModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

        if hyp.do_emb3D:
            # freeze the slow model
            self.model.featnet3D_slow.eval()
            self.set_requires_grad(self.model.featnet3D_slow, False)
            
        if hyp.do_freeze_match:
            self.model.matchnet.eval()
            self.set_requires_grad(self.model.matchnet, False)
            
        if hyp.do_freeze_motionreg:
            self.model.motionregnet.eval()
            self.set_requires_grad(self.model.motionregnet, False)
            
        if hyp.do_freeze_conf:
            self.model.confnet.eval()
            self.set_requires_grad(self.model.confnet, False)

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
                        ious = results['ious'][0].detach().cpu().numpy()
                        confs = results['confs'][0].detach().cpu().numpy()
                        print('ious', ious)
                        all_ious[test_count,:len(ious)] = ious
                        all_confs[test_count,:len(confs)] = confs
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        mean_confs = np.mean(all_confs[:test_count], axis=0)
                        print('mean_ious', mean_ious)
                        # print('mean_confs', mean_confs)
                        
                        if hyp.do_export_vis and log_this:
                            vis_bev = results['vis_bev']
                            vis_per = results['vis_pers']
                            print(vis_bev[0].shape)
                            print(vis_per[0].shape)
                            vis_bev = [np.transpose(vis[0].detach().cpu().numpy(), [1, 2, 0]) for vis in vis_bev]
                            vis_per = [np.transpose(vis[0].detach().cpu().numpy(), [1, 2, 0]) for vis in vis_per]

                            for fr, (vis, vis2) in enumerate(zip(vis_per, vis_bev)):
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
            

class CarlaPipeModel(nn.Module):
    def __init__(self):
        super(CarlaPipeModel, self).__init__()
        in_dim = 1
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D(in_dim=in_dim)
        if hyp.do_match:
            self.deglist = [-6, 0, 6]
            self.radlist = [utils_geom.deg2rad(deg) for deg in self.deglist]
            self.trim = 5
            self.mid_deg_ind = int(np.floor(len(self.deglist)/2))
            self.matchnet = MatchNet(self.radlist)
        if hyp.do_conf:
            self.confnet = ConfNet()
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
            # make a slow net
            self.featnet3D_slow = FeatNet3D(in_dim=in_dim)
            # init slow params with fast params
            self.featnet3D_slow.load_state_dict(self.featnet3D.state_dict())
        if hyp.do_motionreg:
            self.motionregnet = Motionregnet()
            
        self.iou_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        self.conf_pools = [utils_misc.SimplePool(3, version='np') for i in list(range(hyp.S_test))]
        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

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

        # print('Z, Y, X', self.Z, self.Y, self.X)
            
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
        self.rgb_camX0 = self.rgb_camXs[:,0]
        # self.rgb_camX0 = feed["rgb_camX0"]
        self.pix_T_cams = feed["pix_T_cams"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(__p(self.camR0s_T_camRs).inverse())
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        

        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]
        
        if set_data_format=='traj' or set_data_format=='simpletraj':
            lrt_camRs, self.obj_lens = utils_misc.parse_boxes(feed["box_traj_camR"], self.origin_T_camRs)
            score_s = feed["score_traj"]
            tid_s = torch.ones_like(score_s).long()
            lrt_camXs = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs)
            lrt_camX0s = utils_geom.apply_4x4s_to_lrts(self.camX0s_T_camXs, lrt_camXs)
            lrt_camR0s = utils_geom.apply_4x4s_to_lrts(self.camR0s_T_camRs, lrt_camRs)
            # these are B x S x 19
            # self.box_camRs = box_camRs
            # self.obj_lens = box_camRs[:,:,3:6]
            # self.obj_angles = box_camRs[:,:,6:9]
            
            self.lrt_camXs = lrt_camXs
            self.lrt_camRs = lrt_camRs
            self.lrt_camR0s = lrt_camR0s
            self.lrt_camX0s = lrt_camX0s

            # print('raw angles', self.box_camRs[0,:,6:])

            self.obj_clist_camXs = utils_geom.get_clist_from_lrtlist(lrt_camXs)
            self.obj_clist_camRs = utils_geom.get_clist_from_lrtlist(lrt_camRs)
            self.obj_clist_camR0s = utils_geom.get_clist_from_lrtlist(lrt_camR0s)
            self.obj_clist_camX0s = utils_geom.get_clist_from_lrtlist(lrt_camX0s)

            self.obj_rlist_camR0s = utils_geom.get_rlist_from_lrtlist(lrt_camR0s)
            self.obj_rlist_camX0s = utils_geom.get_rlist_from_lrtlist(lrt_camX0s)
            # print('obj_rlist_camR0s', self.obj_rlist_camR0s)
            self.obj_rlist_camXs = utils_geom.get_rlist_from_lrtlist(lrt_camXs)
            # print('obj_rlist_camXs', self.obj_rlist_camXs)

            # ok, i have the orientation for each step
            # 

            self.obj_scorelist = score_s

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
        self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camX0)
        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)

        # self.deglist = [-6, 0, 6]
        # self.trim = 5
        # self.radlist = [utils_geom.deg2rad(deg) for deg in self.deglist]


        if self.include_image_summs:
            self.rgb_camXs = feed["rgb_camXs"]
            visX_e = []
            for s in list(range(0, self.S, 2)):
                visX_e.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,s],
                    torch.cat([self.lrt_camXs[:,s:s+1]], dim=1),
                    torch.cat([torch.ones([self.B,1]).float().cuda()], dim=1),
                    torch.arange(1,3).reshape(self.B, 2).long().cuda(),
                    self.pix_T_cams[:,s], only_return=True))
            self.summ_writer.summ_rgbs('track/box_camXs_g', visX_e)

        # print('set_name', self.set_name)
        # print('vox_size_X', self.vox_size_X)
        # print('vox_size_Y', self.vox_size_Y)
        # print('vox_size_Z', self.vox_size_Z)
        
        return True # OK

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
        # print('lrt_cam', lrt_cam, lrt_cam.shape)
        B, D = list(lrt_cam.shape)

        # to help us create some mats:
        rot0 = utils_geom.eye_3x3(B)
        t0 = torch.zeros(B, 3).float().cuda()

        _, cam_T_norot = utils_geom.split_lrt(lrt_cam)
        norot_T_cam = utils_geom.safe_inverse(cam_T_norot)
        # now, actually, we do want some small rotation, given in dr
        yesrot_T_norot = utils_geom.merge_rt(utils_geom.eul2rotm(dr[:,0], dr[:,1], dr[:,2]), t0)
        # and finally we want some small displacement, given in dt
        final_T_yesrot = utils_geom.merge_rt(rot0, dt)

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

    def collapse_past_and_refresh_future(self, k):
        # trajectory k wins!
        # so, up to step s, we need hypothesis in every slot,
        # and we need to refresh the future

        # print('self.step', self.step)
        # print('winner', k)

        ## collapse past
        winner = self.hypotheses[:,k,:self.step+1]
        self.hypotheses[:,:,:self.step+1] = winner.unsqueeze(1)
        # self.confidences[:,:,:self.step+1] = 1.0

        # print(self.hypotheses.shape)
        # print('-')

        # print('this hyp:', self.hypotheses[:,k,self.step])

        # print('collapsing')
        # print('self.hypotheses[0,:,:5,0]', self.hypotheses[0,:,:5,0])

        if self.K > 1 and self.step < self.S_cap:
            ## refresh future
            xyz_agg = self.xyz_camR0s[:,min(0,self.step-2):self.step].reshape(self.B, -1, 3)
            occ_1, lrt_1 = self.place_object_at_delta(
                self.hypotheses[:,k,self.step],
                xyz_agg,
                torch.zeros(self.B, 3).float().cuda(),
                torch.zeros(self.B, 3).float().cuda(),
                self.ZZ, self.ZY, self.ZX,
                hyp.ZMAX-hyp.ZMIN,
                hyp.YMAX-hyp.YMIN,
                hyp.XMAX-hyp.XMIN,
                self.vox_util)
            # featurize
            feat_loss, feat_1, _ = self.featnet3D(occ_1)

            # next i need to create the clist_past for the forecaster
            # turn the lrtlist into a clist
            clist_camR0s_e = utils_geom.get_clist_from_lrtlist(self.hypotheses[:,k])
            # transform this into our context region coords
            clist_1_e = self.vox_util.Ref2Zoom(clist_camR0s_e, lrt_1, self.ZZ, self.ZY, self.ZX)

            clist_past_1 = torch.zeros([self.B, hyp.motionreg_t_past, 3]).float().cuda()
            clist_mask_1 = torch.zeros([self.B, hyp.motionreg_t_past, 1]).float().cuda()
            for s_past in list(range(hyp.motionreg_t_past)):
                # print('working on s_past %d' % s_past)
                ind = self.step - s_past
                if ind >= 0:
                    # print('ok, we will take ind %d from clist_camR0s_e' % ind)
                    clist_past_1[:,hyp.motionreg_t_past-s_past-1] = clist_1_e[:,ind]
                    clist_mask_1[:,hyp.motionreg_t_past-s_past-1] = 1.0
            # print('clist_past_1', clist_past_1.detach().cpu().squeeze().numpy())
            # print('clist_mask_1', clist_mask_1.detach().cpu().squeeze().numpy())

            # forecast the future centroids
            _, clist_futus_1_e = self.motionregnet(
                feat_1,
                clist_past_1,
                clist_mask_1,
                clist_futu=None,
                is_test=True, # do not dropout
                summ_writer=None,
            )
            clist_futus_1_e_ = clist_futus_1_e.reshape(self.B, -1, 3)
            clist_futus_camR0_e_ = self.vox_util.Zoom2Ref(clist_futus_1_e_, lrt_1, self.ZZ, self.ZY, self.ZX)

            # infer the lrts from the centroids
            future_lrts_ = utils_geom.convert_clist_to_lrtlist(clist_futus_camR0_e_, self.obj_lens[:,0])
            future_lrts = future_lrts_.reshape(self.B, hyp.motionreg_num_slots, hyp.motionreg_t_futu, 19)

            # update the hypotheses 
            usable_len = min(hyp.motionreg_t_futu, self.S_cap-self.step-1)
            self.hypotheses[:,1:,self.step+1:self.step+1+usable_len] = future_lrts[:,:,:usable_len]

            # if some length remains, just replicate the end of the traj
            self.hypotheses[:,1:,self.step+1+usable_len:] = future_lrts[:,:,-1].unsqueeze(2)

        # manually set hypothesis0 to be zero motion
        self.hypotheses[:,0,self.step+1:] = self.hypotheses[:,k,self.step:self.step+1]
        
    def visualize_hypotheses(self):
        if hyp.do_export_vis and self.summ_writer.save_this:
            occ_v, lrt_v = self.vox_util.voxelize_near_xyz(
                self.xyz_camR0s[0:1,self.step],
                # utils_geom.get_clist_from_lrtlist(self.hypotheses[0:1,0,self.step:self.step+1]).squeeze(1),
                self.obj_clist_camR0s[0:1,self.step],
                # utils_geom.get_clist_from_lrtlist(self.hypotheses[0:1,0,self.step:self.step+1]).squeeze(1),
                self.Z, self.Y, self.X,
                sz=32, sy=32, sx=32)
            vis_clean = torch.zeros(1, 3, self.Z, self.X).byte().cpu()
            for k in list(range(self.K)):
                s_a = self.step
                s_b = min(self.step+30,self.S_cap)
                vis = self.summ_writer.summ_traj_on_occ(
                    '',
                    self.vox_util.Ref2Zoom(
                        utils_geom.get_clist_from_lrtlist(self.hypotheses[0:1,k,s_a:s_b]),
                        lrt_v[0:1], self.Z, self.Y, self.X),
                    occ_v[0:1],
                    self.vox_util,
                    traj_g=(self.vox_util.Ref2Zoom(self.obj_clist_camR0s[0:1,:self.S_cap], lrt_v[0:1], self.Z, self.Y, self.X) if (k==0) else None),
                    show_bkg=(True if (k==0) else False),
                    already_mem=True,
                    only_return=True,
                    sigma=1)
                vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
                vis_clean[vis_new_any>0] = vis[vis_new_any>0]

            # box past
            vis = self.summ_writer.summ_lrtlist_bev(
                '',
                torch.zeros_like(occ_v[0:1]),
                self.hypotheses[0:1,0,:self.step],
                torch.ones(1,self.step).float().cuda(), # scores
                torch.zeros(1,self.step).long().cuda(), # tids
                self.vox_util,
                lrt=lrt_v,
                only_return=True,
                frame_id=self.step)
            vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            vis_clean[vis_new_any>0] = vis[vis_new_any>0]

            # gt box vis
            vis = self.summ_writer.summ_lrtlist_bev(
                '',
                torch.zeros_like(occ_v[0:1]),
                self.lrt_camR0s[0:1,self.step:self.step+1],
                torch.ones(1,1).float().cuda(), # scores
                torch.ones(1,1).long().cuda(), # tids
                self.vox_util,
                lrt=lrt_v,
                only_return=True,
                frame_id=self.step)
            vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            vis_clean[vis_new_any>0] = vis[vis_new_any>0]

            # box estimates
            vis = self.summ_writer.summ_lrtlist_bev(
                '',
                torch.zeros_like(occ_v[0:1]),
                self.hypotheses[0:1,:,self.step],
                self.confidences[0:1,:,self.step],
                torch.arange(2,self.K+2).reshape(1, self.K).long().cuda(), # tids
                self.vox_util,
                lrt=lrt_v,
                only_return=True,
                frame_id=self.step)
            vis_new_any = (torch.max(vis, dim=1, keepdims=True)[0]).repeat(1, 3, 1, 1)
            vis_clean[vis_new_any>0] = vis[vis_new_any>0]

            self.hypotheses_vis[0:1,self.step] = vis_clean

            # perspective vis
            # hypotheses_camXs = []
            # for k in list(range(self.K)):
            #     lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s[:,:self.S_cap], self.hypotheses[:,k])
            #     lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs[:,:self.S_cap], lrt_camRs_e)
            #     hypotheses_camXs.append(lrt_camXs_e)

            lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(
                self.camRs_T_camR0s[:,self.step:self.step+1].repeat(1, self.K, 1, 1),
                self.hypotheses[:,:,self.step])
            lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(
                self.camXs_T_camRs[:,self.step:self.step+1].repeat(1, self.K, 1, 1),
                lrt_camRs_e)

            self.hypotheses_vis_pers[0:1,self.step] = self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[0:1,self.step],
                torch.cat([self.lrt_camXs[0:1,self.step:self.step+1], lrt_camXs_e[0:1]], dim=1),
                torch.cat([torch.ones(1,1).float().cuda(), self.confidences[0:1,:,self.step]], dim=1), # scores
                # torch.ones(1,self.K+3).reshape(1, self.K+3).float().cuda(), # scores
                torch.arange(1,self.K+4).reshape(1, self.K+3).long().cuda(), # tids
                self.pix_T_cams[0:1,self.step],
                only_return=True,
                frame_id=self.step)

    def create_template(self, lrt, xyz):
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        dr_0 = torch.zeros(self.B, 3).float().cuda()

        occ_0s = []
        for ind, rad in enumerate(self.radlist):
            occ_0, lrt_0 = self.place_object_at_delta(
                lrt, xyz,
                dt_0, dr_0 + rad,
                self.ZZ, self.ZY, self.ZX,
                self.ZZ*self.vox_size_Z,
                self.ZY*self.vox_size_Y,
                self.ZX*self.vox_size_X,
                self.vox_util)
            # self.summ_writer.summ_occ('3D_inputs/occ_0_%d' % ind, occ_0)
            occ_0s.append(occ_0)
        ## featurize
        feat_0s_trimmed = []
        occ_0s_trimmed = []
        for ind, occ_0 in enumerate(occ_0s):
            # featurize the object in frame0
            _, feat_0, _ = self.featnet3D(occ_0)

            # trim out
            feat_0_trimmed = feat_0[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
            occ_0_trimmed = occ_0[:,:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:,self.trim*2:-self.trim*2:]
            feat_0s_trimmed.append(feat_0_trimmed)
            occ_0s_trimmed.append(occ_0_trimmed)
            # self.summ_writer.summ_feat('3D_feats/feat_0_%d_input' % ind, occ_0, pca=False)
            # self.summ_writer.summ_feat('3D_feats/feat_0_%d_output' % ind, feat_0, pca=True)
            # self.summ_writer.summ_feat('3D_feats/feat_0_%d_input_trimmed' % ind, occ_0_trimmed, pca=False)
            # self.summ_writer.summ_feat('3D_feats/feat_0_%d_output_trimmed' % ind, feat_0_trimmed, pca=True)
        feat_0s_trimmed = torch.stack(feat_0s_trimmed, dim=1)
        occ_0s_trimmed = torch.stack(occ_0s_trimmed, dim=1)
        return feat_0s_trimmed, occ_0s_trimmed

    def track_over_seq(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # return total_loss, results, False
        
        assert(hyp.do_feat3D and (hyp.do_match) and (hyp.do_motionreg))

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils_geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camR0s = __u(utils_geom.apply_4x4(__p(self.camR0s_T_camRs), __p(self.xyz_camRs)))

        ### CREATE TEMPLATES
        ## voxelize around the object

        template, template_occ = self.create_template(self.lrt_camR0s[:,0], self.xyz_camR0s[:,0])
        self.templates = [template]
        self.template_occs = [template_occ]
        self.template_confidences = [1.0]
        self.max_templates = 3
        self.template_updates = 0

        ## INITIALIZE HYPOTHESES 
        
        self.S_cap = 100

        self.hypotheses_vis = torch.zeros([1, self.S_cap, 3, self.Z, self.X]).byte().cpu()
        self.hypotheses_vis_pers = torch.zeros([1, self.S_cap, 3, self.H, self.W]).byte().cpu()

        self.K = hyp.motionreg_num_slots+1
        # self.K = 1
        # initialize the hypotheses with the zeroth frame's gt
        self.hypotheses = self.lrt_camR0s[:,0].reshape(self.B, 1, 1, 19).repeat(1, self.K, self.S_cap, 1).float().cuda()
        self.confidences = torch.zeros(self.B, self.K, self.S_cap).float().cuda()

        self.step = 0 # our current step in life
        conf_thresh = 0.7
        
        while (self.step < self.S_cap):
            # print('-'*10)
            # print('step = %d' % self.step)
            
            if self.step==0:
                self.collapse_past_and_refresh_future(0)
            else:
                obj_lrts = []
                clist = utils_geom.get_clist_from_lrtlist(self.hypotheses[:,:,self.step])
                rlist = utils_geom.get_rlist_from_lrtlist(self.hypotheses[:,:,self.step])
                for k in range(self.K):
                    obj_lrt = self.hypotheses[:,k,self.step]
                    skip = False
                    for k2 in range(k):
                        c_here = clist[:,k]
                        c_prev = clist[:,k2]
                        r_here = rlist[:,k]
                        r_prev = rlist[:,k2]
                        c_norm = torch.norm(c_here - c_prev, dim=1)
                        r_norm = utils_geom.angular_l1_norm(r_here, r_prev, dim=1)
                        # print('norm = %.2f' % norm[0].detach().cpu().numpy())
                        # print('c_norm = %.2f' % c_norm[0].detach().cpu().numpy())
                        # if k==(self.K-1):
                        #     print('r_norm = %.2f' % r_norm[0].detach().cpu().numpy())
                        if (not skip) and (c_norm[0] < 0.5) and (r_norm[0] < utils_geom.deg2rad(6)):
                            # we've already done something close to this
                            # let's just take the old answer and move on
                            self.hypotheses[:,k,self.step] = self.hypotheses[:,k2,self.step]
                            self.confidences[:,k,self.step] = self.confidences[:,k2,self.step]
                            # print('skipping')
                            skip = True
                            
                    if not skip:
                        ### MATCH
                        # create a search region around the current lrt estimate
                        occ_1, lrt_1 = self.place_object_at_delta(
                            obj_lrt,
                            self.xyz_camR0s[:,self.step],
                            torch.zeros(self.B, 3).float().cuda(),
                            torch.zeros(self.B, 3).float().cuda(),
                            self.ZZ, self.ZY, self.ZX,
                            self.ZZ*self.vox_size_Z,
                            self.ZY*self.vox_size_Y,
                            self.ZX*self.vox_size_X,
                            self.vox_util)
                        # featurize
                        _, feat_1, _ = self.featnet3D(occ_1)

                        confs_here = []
                        locs_here = []
                        rads_here = []
                        
                        for template in self.templates:
                            
                            # locate the object
                            corrs, rad_e, obj_loc_halfsearch_e, _ = self.matchnet(template, feat_1)
                            assert(self.B==1)

                            # estimate the iou
                            ious_e, _ = self.confnet(corrs, occ_1)
                            min_iou_e = torch.min(ious_e, dim=1)[0]
                            # take the min iou estimate as the confidence
                            conf = min_iou_e.squeeze()
                            
                            confs_here.append(conf)
                            locs_here.append(obj_loc_halfsearch_e)
                            rads_here.append(rad_e)
                            
                        conf, t = torch.max(torch.stack(confs_here, dim=0), dim=0)
                        # print('template %d wins' % t)
                        obj_loc_halfsearch_e = locs_here[t]
                        rad_e = rads_here[t]

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
                        obj_lrts.append(obj_lrt)

                        # bookkeep 
                        self.confidences[:,k,self.step] = conf
                        # it helps to update the traj with this new info
                        self.hypotheses[:,k,self.step] = obj_lrt
                    

                if (self.step+1) < self.S_cap:
                    # for hypo0 (the zeromot hypo), search, search at this argmax (like the baseline siamese tracker)
                    self.hypotheses[:,0,self.step+1] = self.hypotheses[:,0,self.step]
                        
                # end loop over k

                conf_max, k = torch.max(self.confidences[:,:,self.step], dim=1)
                conf_max = conf_max.squeeze()
                k = k.squeeze()
                conf_max_py = conf_max.detach().cpu().squeeze().numpy()

                # # hack: set non-max conf to 1.0, so that it does not print in the vis
                # for k_ in list(range(self.K)):
                #     if not k_==k:
                #         # eliminate this from the vis, by setting the value to 1.0
                #         self.confidences[:,k_,self.step] = 1.0

                
                # print('step %d, hyp %d has max; conf = %.2f' % (self.step, k, conf_max.detach().cpu().numpy()))
                if conf_max > conf_thresh:
                    # print('-'*10)
                    # print('step = %d' % self.step)
                    # print('confidences:', self.confidences[:,:,self.step])
                    # print('conf of %d looks good!' % k)

                    # self.hypotheses[:,k,self.step] = obj_lrts[k]
                    
                    # use the winner to overwrite the past
                    self.collapse_past_and_refresh_future(k)
                    # found_one = True

                    lrt_camRs = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s[:,:self.S_cap], self.hypotheses[:,0])
                    ious = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,:self.S_cap], lrt_camRs)
                    iou = ious[:,self.step].squeeze().detach().cpu().numpy()
                    
                    if len(self.templates) < self.max_templates:
                        # take this as a template
                        # template = self.create_template(self.hypotheses[:,k,self.step], self.confidences[:,k,self.step])
                        print('timestep %d; grabbing a new template, which has conf %.2f; actual iou is %.2f' % (self.step, conf_max_py, iou))
                        new_template, new_occ = self.create_template(self.hypotheses[:,k,self.step], self.xyz_camR0s[:,self.step])
                        self.templates.append(new_template)
                        self.template_occs.append(new_occ)
                        self.template_confidences.append(conf_max_py)
                        self.template_updates += 1
                        self.summ_writer.summ_feat('templates/template_%02d' % self.template_updates, new_template[:,self.mid_deg_ind], pca=True)
                        self.summ_writer.summ_occ('templates/template_occ_%02d' % self.template_updates, new_occ[:,self.mid_deg_ind])
                    else:
                        t = np.argmin(self.template_confidences)
                        worst_conf = self.template_confidences[t]
                        worst_template = self.templates[t]
                        if conf_max_py > worst_conf:
                            print('timestep %d; replacing template %d; new conf is %.2f; actual iou is %.2f' % (self.step, t, conf_max_py, iou))
                            new_template, new_occ = self.create_template(self.hypotheses[:,k,self.step], self.xyz_camR0s[:,self.step])
                            self.templates[t] = new_template
                            self.template_occs[t] = new_occ
                            self.template_confidences[t] = conf_max_py
                            self.template_updates += 1
                            self.summ_writer.summ_feat('templates/template_%02d' % self.template_updates, new_template[:,self.mid_deg_ind], pca=True)
                            self.summ_writer.summ_occ('templates/template_occ_%02d' % self.template_updates, new_occ[:,self.mid_deg_ind])
                            
                    
            self.visualize_hypotheses()

            self.step = self.step + 1
        # end while step
        self.summ_writer.summ_rgbs('track/hypotheses', self.hypotheses_vis.unbind(1))
        self.summ_writer.summ_rgbs('track/hypotheses_pers', self.hypotheses_vis_pers.unbind(1))

        ## vis the hypotheses 
        for k in range(self.K):
            occ_v, lrt_v = self.vox_util.voxelize_near_xyz(
                self.xyz_camR0s[0:1,0], self.obj_clist_camR0s[0:1,0],
                self.Z, self.Y, self.X,
                sz=32, sy=32, sx=32)
            vis_bev = self.summ_writer.summ_lrtlist_bev(
                '',
                occ_v[0:1],
                self.hypotheses[0:1,k,::2],
                torch.ones(1,self.S_cap).float().cuda(),
                torch.arange(0,self.S_cap).reshape(1, self.S_cap).long().cuda(), # tids
                self.vox_util,
                lrt=lrt_v,
                only_return=True)
            self.summ_writer.summ_rgb('track/boxes_bev_hypothesis_%d' % k, vis_bev)
        vis_bev = self.summ_writer.summ_lrtlist_bev(
            '',
            occ_v[0:1],
           self.lrt_camR0s[0:1,:self.S_cap:2],
            torch.ones(1,self.S_cap).float().cuda(),
            torch.arange(0,self.S_cap).reshape(1, self.S_cap).long().cuda(), # tids
            self.vox_util,
            lrt=lrt_v,
            only_return=True)
        self.summ_writer.summ_rgb('track/boxes_bev_g', vis_bev)

        # no matter what, at the last step, collapse the past with the most confident traj
        conf_max, k = torch.max(self.confidences[:,:,-1], dim=1)
        conf_max = conf_max.squeeze()
        k = k.squeeze()
        print('step %d, hyp %d has max; conf = %.2f' % (self.step, k, conf_max.detach().cpu().numpy()))
        self.collapse_past_and_refresh_future(k)

        # evaluate ious

        # i assume hypothesis0 holds the winner
        print('self.hypotheses[:,0]', self.hypotheses[:,0].shape)
        print('self.camRs_T_camR0s[:,:self.S_cap]', self.camRs_T_camR0s[:,:self.S_cap].shape)
        lrt_camRs = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s[:,:self.S_cap], self.hypotheses[:,0])
        ious = utils_geom.get_iou_from_corresponded_lrtlists(self.lrt_camRs[:,:self.S_cap], lrt_camRs)
        # print('ious', ious.detach().cpu().numpy(), ious.shape)
        # input()

        results['ious'] = ious
        results['confs'] = self.confidences[:,0]
        results['vis_bev'] = self.hypotheses_vis.unbind(1)
        results['vis_pers'] = self.hypotheses_vis_pers.unbind(1)
        
        # self.summ_writer.summ_rgb('track/hypotheses_0', self.hypotheses_vis[:,0])
        return total_loss, results, False

        
    def forward(self, feed):
        set_name = feed['set_name']
        # print('-'*100)
        # print('working on %s' % set_name)

        ok = self.prepare_common_tensors(feed)
        total_loss = torch.tensor(0.0).cuda()
        
        # return total_loss, None, True
        # if not ok:
        #     print('not enough object points; returning early')
        # return total_loss, None, True

        return self.track_over_seq(feed)
        
