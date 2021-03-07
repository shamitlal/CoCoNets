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
from nets.translationnet import TranslationNet
from nets.rigidnet import RigidNet
from nets.robustnet import RobustNet
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

class CARLA_VSIAMESE(Model):
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
        self.model = CarlaVsiameseModel().to(self.device)
        
        if hyp.do_freeze_feat3D:
            self.model.featnet3D.eval()
            self.set_requires_grad(self.model.featnet3D, False)

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
                        ious = ious[0].cpu().numpy()
                        # print('ious', ious)
                        all_ious[test_count] = ious
                        test_count += 1
                        # print('all_ious', all_ious[:test_count])
                        mean_ious = np.mean(all_ious[:test_count], axis=0)
                        print('mean_ious', mean_ious)
                        
                        if hyp.do_export_vis:
                            visX_e = results['visX_e']
                            visX_g = results['visX_g']
                            # these are lists, where each item is shaped 1 x 3 x 128 x 384
                            vis_e = [utils_improc.back2color(im).cpu().numpy()[0] for im in visX_e]
                            # vis_g = [utils_improc.back2color(im).cpu().numpy()[0] for im in visX_g]
                            # now they're 3 x 128 x 384
                            # vis_eg = [np.concatenate([e, g], axis=1) for (e,g) in zip(vis_e, vis_g)]
                            # this is a list of 3 x 256 x 384 items
                            # vis_eg = [np.transpose(vis, [1, 2, 0]) for vis in vis_eg]
                            
                            vis_e = [np.transpose(vis, [1, 2, 0]) for vis in vis_e]
                            # for fr, vis in enumerate(vis_e):
                            #     out_path = 'outs/%s_vis_eg_%04d_%02d.png' % (hyp.name, test_count, fr)
                            #     imsave(out_path, vis)
                            #     print('saved %s' % out_path)
                            

                            vis_bev_e = results['vis_bev_e']
                            # these are lists, where each item is shaped 1 x 3 x Z x X
                            vis_bev = [utils_improc.back2color(im).cpu().numpy()[0] for im in vis_bev_e]
                            utils_py.print_stats('vis_bev[0]', vis_bev[0])
                            # now they're 3 x Z x X
                            # print('vis_bev[0].shape', vis_bev[0].shape)
                            # vis_bevg = [np.transpose(vis, [1, 2, 0]) for vis in vis_bevg]
                            vis_bev = [np.transpose(vis, [1, 2, 0]) for vis in vis_bev]
                            # for fr, vis in enumerate(vis_bev):
                            #     out_path = 'outs/%s_vis_bev_%04d_%02d.png' % (hyp.name, test_count, fr)
                            #     imsave(out_path, vis)
                            #     print('saved %s' % out_path)


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

class CarlaVsiameseModel(nn.Module):
    def __init__(self):
        super(CarlaVsiameseModel, self).__init__()
        if hyp.do_feat3D:
            self.featnet3D = FeatNet3D()
        if hyp.do_match:
            self.matchnet = MatchNet()
        if hyp.do_translation:
            self.translationnet = TranslationNet()
        if hyp.do_rigid:
            self.rigidnet = RigidNet()
            self.iou_pools = [utils_misc.SimplePool(20) for i in list(range(hyp.S_test))]
            # self.iou_pools = [utils_misc.SimplePool(1) for i in list(range(hyp.S_test))]
        if hyp.do_robust:
            self.robustnet = RobustNet()
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
        # self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0_T_camXs), __p(self.xyz_camXs)))


        self.S = feed["set_seqlen"]
        self.K = 1 # the traj data only has one object inside each ex
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

        #####################
        ## visualize what we got
        #####################
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
        # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # self.summ_writer.summ_unps('3D_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        self.summ_writer.summ_occs('3D_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        self.summ_writer.summ_feat('3D_inputs/obj_mask', self.obj_mask_template, pca=False)
        return True # OK 
        
    def train_over_pair(self, feed):
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
        
        # clist_camR0 = self.obj_clist_camR0s
        # rxlist_camR0 = self.obj_rlist_camR0s[:,:,1] # rx,ry,rz exist on dim2
        # rylist_camR0 = self.obj_rlist_camR0s[:,:,1] # rx,ry,rz exist on dim2
        # yawlist_camR0 = self.obj_rlist_camR0s[:,:,1] # rx,ry,rz exist on dim2
        # this is B x S

        # rotlist = __u(utils_geom.eul2rotm(__p(yawlist_camR0), __p(yawlist_camR0),
        #                                   torch.zeros_like(yawlist_camR0))
        
        # rotlist_camR0 = __u(utils_geom.eul2rotm(__p(self.obj_rlist_camR0s[:,:,0]),
        #                                         __p(self.obj_rlist_camR0s[:,:,1]),
        #                                         __p(self.obj_rlist_camR0s[:,:,2])))
        # rot = rotlist[:,0]
        # # rot is B x 3 x 3

        # xyz0 = clist_camR0[:,0] + 0 # make the template perfectly centered
        # xyz1 = clist_camR0[:,1] + torch.from_numpy(rand).float().cuda() # make the search region off a bit

        
        def place_object_at_delta(center_cam, rot_cam, xyz_cam, dt, dr, Z, Y, X, sz, sy, sx, vox_util):
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
            
        sz = self.ZZ*self.vox_size_Z
        sy = self.ZY*self.vox_size_Y
        sx = self.ZX*self.vox_size_X
        # rand = np.random.uniform(low=-0.5*sz/2.0, high=0.5*sz/2.0, size=[3])
        # search_loc_ref = self.obj_clist_camR0s[:,1] + torch.from_numpy(rand).float().cuda()

        # for frame0, place the object exactly in the middle, but at some random orientation
        dt_0 = torch.zeros(self.B, 3).float().cuda()
        rand_rz = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/1.0, high=np.pi/1.0, size=[self.B])
        rand_rx = np.random.uniform(low=-np.pi/8.0, high=np.pi/8.0, size=[self.B])
        rand_r = np.reshape(np.array([rand_rx, rand_ry, rand_rz]), (self.B, 3))
        dr_0 = torch.from_numpy(rand_r).float().cuda()

        # occ_0, lrt_0 = self.vox_util.voxelize_at_xyz_with_delta(
        #     self.obj_clist_camR0s[:,0],
        #     self.obj_rlist_camR0s[:,0],
        #     self.xyz_camR0s[:,0],
        #     dt_0, dr_0,
        #     self.ZZ, self.ZY, self.ZX,
        #     sz, sy, sx)
        occ_0, lrt_0 = place_object_at_delta(
            self.obj_clist_camR0s[:,0],
            self.obj_rlist_camR0s[:,0],
            self.xyz_camR0s[:,0],
            dt_0, dr_0,
            self.ZZ, self.ZY, self.ZX,
            sz, sy, sx,
            self.vox_util)
        # print('occ_0', occ_0.shape)
        # print('lrt_0', lrt_0.shape)
        # print('lrt_camR0s[:,0:1]', lrt_camR0s[:,0:1].shape)
        mask_0 = self.vox_util.assemble_padded_obj_masklist_within_region(
            self.lrt_camR0s[:,0:1], self.obj_scorelist[:,0:1], lrt_0,
            int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2), coeff=1.0).squeeze(1)
        self.summ_writer.summ_occ('3D_inputs/occ_0', occ_0)
        self.summ_writer.summ_occ('3D_inputs/mask_0', mask_0)

        # for frame1, place the object somewhere NEAR dt_0
        dt_1 = np.random.uniform(low=-0.5*sz/2.0, high=0.5*sz/2.0, size=[self.B, 3])
        # print('sx, sy, sz', sx, sy, sz)
        # print('dt_1', dt_1)
        dt_1 = dt_0 + torch.from_numpy(dt_1).float().cuda()
        # place it at some orientation NEAR dr_0
        rand_rz = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_ry = np.random.uniform(low=-np.pi/16.0, high=np.pi/16.0, size=[self.B])
        rand_rx = np.random.uniform(low=-np.pi/32.0, high=np.pi/32.0, size=[self.B])
        rand_r = np.reshape(np.array([rand_rx, rand_ry, rand_rz]), (self.B, 3))
        dr_1 = dr_0 + torch.from_numpy(rand_r).float().cuda()*2.0
        # dr_1 = dr_0.clone()
        
        occ_1, lrt_1 = place_object_at_delta(
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
            
            # featurize the object in frame0
            _, obj_featR0, _ = self.featnet3D(
                occ_0,
                # self.obj_occR0s[:,0],
                None)
            # self.summ_writer.summ_feat('3D_feats/obj_featR0_input', self.obj_occR0s[:,0], pca=False)
            self.summ_writer.summ_feat('3D_feats/obj_featR0_input', occ_0, pca=False)
            self.summ_writer.summ_feat('3D_feats/obj_featR0_output', obj_featR0, pca=True)

            # we'll make a search region containing the object
            # we'll search midway between the objects time0 and time1 locs, plus some noise
            # search_loc_ref = (self.obj_clist_camR0s[:,0] + self.obj_clist_camR0s[:,1])*0.5
            # search_loc_ref += np.random.normal()
            # search_loc_ref = (self.obj_clist_camR0s[:,0]*0.25 + self.obj_clist_camR0s[:,1]*0.75)

            SZ, SY, SX = self.ZZ, self.ZY, self.ZX
            # SZ, SY, SX = self.ZZ*2, self.ZY*2, self.ZX*2
            # SZ, SY, SX = int(self.ZZ*1.5), int(self.ZY*1.5), int(self.ZX*1.5)
            
            # rand = np.random.normal([0,0,0])
            # rand = np.random.uniform(low=-0.95*SZ*self.vox_size_Z/2.0, high=0.95*SZ*self.vox_size_Z/2.0, size=[3])
            rand = np.random.uniform(low=-0.5*SZ*self.vox_size_Z/2.0, high=0.5*SZ*self.vox_size_Z/2.0, size=[3])
            # rand = np.random.uniform(low=-0.25*SZ*self.vox_size_Z/2.0, high=0.25*SZ*self.vox_size_Z/2.0, size=[3])
            search_loc_ref = self.obj_clist_camR0s[:,1] + torch.from_numpy(rand).float().cuda()
            # print('rand', rand)
            # search_loc_ref = 
            # print('search_loc_ref', search_loc_ref.cpu().numpy())
            
            # search_occR1, search_lrt = self.vox_util.voxelize_near_xyz(
            #     self.xyz_camR0s[:,1],
            #     search_loc_ref,
            #     SZ, SY, SX,
            #     sz=(SZ*self.vox_size_Z),
            #     sy=(SY*self.vox_size_Y),
            #     sx=(SX*self.vox_size_X))
            feat_loss, featR1, validR1 = self.featnet3D(
                # search_occR1,
                occ_1,
                self.summ_writer)

            # print('search_lrt', search_lrt.cpu().numpy())
            total_loss += feat_loss
            # self.summ_writer.summ_feat('3D_feats/featR1_input', search_occR1, pca=False)
            self.summ_writer.summ_feat('3D_feats/featR1_input', occ_1, pca=False)
            self.summ_writer.summ_feat('3D_feats/featR1_output', featR1, pca=True)

        # if hyp.do_match:
        #     assert(hyp.do_feat3D)
            
        #     obj_loc_halfmem = self.vox_util.Ref2Zoom(
        #         self.obj_clist_camR0s[:,1].unsqueeze(1), search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)

        #     # obj_loc_halfmem = self.vox_util.Ref2Zoom(
        #     #     self.obj_clist_camR0s[:,1].unsqueeze(1), search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
            
        #     # # check this visually (yes looks good)
        #     # self.summ_writer.summ_traj_on_occ('match/obj_loc',
        #     #                                   obj_loc_halfmem.unsqueeze(1)*2.0,
        #     #                                   search_occR1,
        #     #                                   self.vox_util,
        #     #                                   already_mem=True,
        #     #                                   sigma=2)
                
        #     corr, _, match_loss = self.matchnet(
        #         obj_featR0, # template
        #         featR1, # search region
        #         obj_loc_halfmem, # gt position in search coords
        #         self.summ_writer)
        #     total_loss += match_loss

        #     # utils_basic.print_stats_py('train corr', corr.detach().cpu().numpy())
        #     self.summ_writer.summ_histogram('corr_train', corr)
            
        # if hyp.do_translation:
        #     assert(hyp.do_feat3D)
            
        #     obj_loc_halfmem = self.vox_util.Ref2Zoom(
        #         self.obj_clist_camR0s[:,1].unsqueeze(1), search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
            
        #     _, translation_loss = self.translationnet(
        #         obj_featR0, # template
        #         featR1, # search region
        #         obj_loc_halfmem, # gt position in search coords
        #         self.summ_writer)
        #     total_loss += translation_loss
            
        if hyp.do_rigid:
            assert(hyp.do_feat3D)
            
            # obj_loc_search = self.vox_util.Ref2Zoom(
            #     self.obj_clist_camR0s[:,1].unsqueeze(1), search_lrt, SX, SY, SZ, additive_pad=0.0).squeeze(1)
            # # print('obj_loc_halfmem', obj_loc_halfmem.cpu().numpy())
            # self.summ_writer.summ_traj_on_occ('rigid/obj_loc_g',
            #                                   obj_loc_search.unsqueeze(1),
            #                                   search_occR1,
            #                                   self.vox_util,
            #                                   already_mem=True,
            #                                   sigma=2)
            
            obj_loc_search = self.vox_util.Ref2Zoom(
                self.obj_clist_camR0s[:,1].unsqueeze(1), lrt_1, SX, SY, SZ, additive_pad=0.0).squeeze(1)
            # print('obj_loc_halfmem', obj_loc_halfmem.cpu().numpy())
            self.summ_writer.summ_traj_on_occ('rigid/obj_loc_g',
                                              obj_loc_search.unsqueeze(1),
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)

            # for rotation, i want the relative rotation of the template to the current step
            # since i am using an axis-aligned crop at timestep0, 
            # this is just R0[:,i] - R0[:,0]
            obj_r_delta = self.obj_rlist_camR0s[:,1] - self.obj_rlist_camR0s[:,0]
            # very soon i will want to adjust this to use the higher dim space
            ## update: let's do this when it matters

            obj_r_delta = utils_geom.wrap2pi(obj_r_delta)
            
            # print('obj_r_delta', obj_r_delta.detach().cpu().numpy())
            
            # R = hyp.rigid_repeats
            # sampling_mask = (self.obj_mask_template).repeat(R, 1, 1, 1, 1)
            # sampled_corners, sampled_centers, sampling_failed = utils_misc.sample_eight_points(sampling_mask, max_tries=1000, random_center=False)
            R = hyp.rigid_repeats
            sampling_mask = (mask_0).repeat(R, 1, 1, 1, 1)
            sampled_corners, sampled_centers, sampling_failed = utils_misc.sample_eight_points(sampling_mask, max_tries=1000, random_center=False)
            if sampling_failed:
                # return early
                return total_loss, results, True
            # assert(not sampling_failed)
            # obj_loc_halfsearch, _, rigid_loss = self.rigidnet(
            #     obj_featR0, # template
            #     self.obj_mask_template, # mask of object within the template
            #     featR1, # search region
            #     obj_loc_search/2.0, # gt position in halfsearch coords (since it's featurized)
            #     obj_r_delta, # gt rotation in search coords
            #     sampled_corners,
            #     sampled_centers,
            #     vox_util=self.vox_util,
            #     summ_writer=self.summ_writer)
            
            obj_loc_halfsearch, _, rigid_loss = self.rigidnet(
                obj_featR0, # template
                mask_0, # mask of object within the template
                featR1, # search region
                obj_loc_search/2.0, # gt position in halfsearch coords (since it's featurized)
                dr_1-dr_0, # gt relative rotation in search coords
                sampled_corners,
                sampled_centers,
                vox_util=self.vox_util,
                summ_writer=self.summ_writer)
            total_loss += rigid_loss

            self.summ_writer.summ_traj_on_occ('rigid/obj_loc_e',
                                              obj_loc_halfsearch.unsqueeze(1)*2.0,
                                              occ_1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
            
        if hyp.do_robust:
            assert(hyp.do_feat3D)


            # i need to know the coordinates of the template
            # and the coordinates of the search region

            # i think the way to do this is:
            # get from template coords to search coords
            
            obj_loc_search = self.vox_util.Ref2Zoom(
                self.obj_clist_camR0s[:,1].unsqueeze(1), search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
            self.summ_writer.summ_traj_on_occ('robust/obj_loc_g',
                                              obj_loc_search.unsqueeze(1)*2.0,
                                              search_occR1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)

            # obj_loc_search, _, robust_loss = self.robustnet(
            lrt_camR1, robust_loss = self.robustnet(
                obj_featR0, # template
                featR1, # search region
                self.obj_mask_template, # mask of object within the template
                self.template_lrts[:,0],
                search_lrt,
                self.vox_util,
                self.lrt_camR0s,
                self.summ_writer)
            total_loss += robust_loss

            # print('ok, robust done')

            obj_clist_camR0_e = utils_geom.get_clist_from_lrtlist(lrt_camR1.unsqueeze(1))
            obj_loc_search = self.vox_util.Ref2Zoom(
                obj_clist_camR0_e, search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
            self.summ_writer.summ_traj_on_occ('robust/obj_loc_e',
                                              obj_loc_search.unsqueeze(1)*2.0,
                                              search_occR1,
                                              self.vox_util,
                                              already_mem=True,
                                              sigma=2)
            
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def track_over_seq(self, feed):
        results = dict()
        total_loss = torch.tensor(0.0, requires_grad=True).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        # return total_loss, results, False
        assert(hyp.do_feat3D and (hyp.do_match or hyp.do_translation or hyp.do_rigid or hyp.do_robust))

        # featurize the object in frame0
        _, obj_featR0, _ = self.featnet3D(
            self.obj_occR0s[:,0],
            None)
        self.summ_writer.summ_feat('3D_feats/obj_featR0_output', obj_featR0, pca=True)

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
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
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
        

        for s in list(range(0, self.S)):
            # vel = obj_loc_ref - obj_loc_ref_prev
            # obj_loc_ref_search = obj_loc_ref + vel
            obj_loc_ref_search = obj_loc_ref.clone()
            search_occRi, search_lrt = self.vox_util.voxelize_near_xyz(
                self.xyz_camR0s[:,s],
                obj_loc_ref_search,
                SZ, SY, SX,
                sz=(SZ*self.vox_size_Z),
                sy=(SY*self.vox_size_Y),
                sx=(SX*self.vox_size_X))
            search_occR0s.append(search_occRi)

            _, search_featRi, _ = self.featnet3D(
                search_occRi,
                None)
            search_featR0s.append(search_featRi)

            if hyp.do_translation or hyp.do_match:
                if hyp.do_translation:
                    obj_loc_search, _ = self.translationnet(
                        obj_featR0, # template
                        search_featRi, # search region
                        torch.zeros(self.B, 3).float().cuda(), # gt position in search coords
                    )
                elif hyp.do_match:
                    corr, xyz_offset, _ = self.matchnet(
                        obj_featR0, # template
                        search_featRi, # search region
                        torch.zeros(self.B, 3).float().cuda(), # gt position in search coords
                        use_window=True,
                    )
                    # corr is B x 1 x Z x Y x X
                    # xyz_offset is 1 x 3
                    obj_loc_search = utils_track.convert_corr_to_xyz(corr, xyz_offset)
                    # this is B x 3, and in search coords

                # the middle of the halfsearch coords is obj_loc_ref from the previous iter
                search_mid = np.array([SX/4.0, SY/4.0, SZ/4.0], np.float32).reshape([1, 3])
                search_mid = torch.from_numpy(search_mid).float().to('cuda')
                obj_loc_halfmem = self.vox_util.Ref2Mem(obj_loc_ref.unsqueeze(1), self.Z2, self.Y2, self.X2).squeeze(1)
                obj_loc_halfmem = (obj_loc_search - search_mid) + obj_loc_halfmem
                obj_loc_mem = obj_loc_halfmem * 2.0
                # this is B x 3, and in mem coords

                # just take the time0 rotation 
                obj_r = self.obj_rlist_camR0s[:,0]

                peak_xyzs_mem.append(obj_loc_mem)
                peak_rots_mem.append(obj_r)

                # update the obj loc for the next step
                obj_loc_ref = self.vox_util.Mem2Ref(obj_loc_mem.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
                
            elif hyp.do_rigid:

                xyz = utils_basic.gridcloud3D(self.B, int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2), norm=False)[0]
                mask = self.obj_mask_template[0]

                xyz = xyz.reshape(-1, 3)
                mask = mask.reshape(-1)                
                xyz = xyz[torch.where(mask > 0)]
                xyz = xyz.unsqueeze(0)
                # print('gathered up this thing:', xyz.detach().cpu().numpy().shape)
                # input()
                xyz_camR0_old = self.vox_util.Zoom2Ref(xyz, self.template_lrts[:,0], int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2))
                obj_lens = self.box_camRs[:,:,3:6]

                # vel = obj_loc_ref - obj_loc_ref_prev
                vel = obj_loc_ref - (obj_loc_ref_prev + obj_loc_ref_prev_prev)/2.0
                obj_loc_ref_next = obj_loc_ref + vel
                box_camR0s_e = torch.cat([obj_loc_ref_next.unsqueeze(1), obj_lens[:,0:1], obj_r.unsqueeze(1)], dim=2)
                lrt_camR0s_e = utils_geom.convert_boxlist_to_lrtlist(box_camR0s_e)

                _, rt_camR0_old = utils_geom.split_lrt(self.template_lrts[:,0])
                _, rt_camR0_new = utils_geom.split_lrt(lrt_camR0s_e[:,0])
                # each of these is a ref_T_obj
                newR0_T_oldR0 = torch.matmul(rt_camR0_new, rt_camR0_old.inverse())
                xyz_camR0_new = utils_geom.apply_4x4(newR0_T_oldR0, xyz_camR0_old)
                # ok, i think i've moved the points

                # maybe i can visualize this right away

                occ_old = self.vox_util.voxelize_xyz(xyz_camR0_old, self.Z, self.Y, self.X)
                occ_new = self.vox_util.voxelize_xyz(xyz_camR0_new, self.Z, self.Y, self.X)
                # self.summ_writer.summ_occs('track/obj_occ_old_new_%d' % s, [occ_old, occ_new])

                # ok great
                # now, i need to get to Xi coords, and figure out which points are inbounds

                xyz_camR = utils_geom.apply_4x4(self.camRs_T_camR0s[:,s], xyz_camR0_new)
                xyz_camX = utils_geom.apply_4x4(self.camXs_T_camRs[:,s], xyz_camR)
                # print('xyz_camX before', xyz_camX.shape)
                xy = utils_geom.apply_pix_T_cam(self.pix_T_cams[:,s], xyz_camX)
                z = xyz_camX[:,:,2]

                depth, valid = utils_geom.create_depth_image(self.pix_T_cams[:,s], xyz_camX, self.H, self.W)
                # self.summ_writer.summ_oned('track/obj_depth_%d' % s, depth)
                
                depth2, valid2 = utils_geom.create_depth_image(self.pix_T_cams[:,s], self.xyz_camXs[:,s], self.H, self.W)
                match = (torch.abs(depth2-depth) < 2.0).float()
                valid = valid*match*valid2

                # now i need to map my way back, and use the subset of points
                xyz_camX = utils_geom.depth2pointcloud(depth, self.pix_T_cams[:,s])

                xyz_camX = xyz_camX[0]
                valid = valid[0].reshape(-1)
                xyz_camX = xyz_camX[torch.where(valid)]
                xyz_camX = xyz_camX.unsqueeze(0)
                # xyz_camX = xyz_camX[xyz_camX[:,2]>0.01]

                xyz_camR = utils_geom.apply_4x4(self.camRs_T_camXs[:,s], xyz_camX)
                xyz_camR0 = utils_geom.apply_4x4(self.camR0s_T_camRs[:,s], xyz_camR)
                xyz_camR0 = utils_geom.apply_4x4(newR0_T_oldR0.inverse(), xyz_camR0)

                # # let's make sure this is a subset:
                # print('xyz_camX after', xyz_camX.shape)
                # input()

                # yes ok.
                # next i need to get to template coords

                # here, once again, it might be convenient as hell that i recently decided to use zoom==search coords

                # search_occRi_mask, _ = self.vox_util.voxelize_near_xyz(
                #     self.xyz_camR0s[:,s],
                #     obj_loc_ref,
                #     SZ, SY, SX,
                #     sz=(SZ*self.vox_size_Z),
                #     sy=(SY*self.vox_size_Y),
                #     sx=(SX*self.vox_size_X))
                # search_occR0s.append(search_occRi)
                # print('self.obj_clist_camR0s', self.obj_clist_camR0s.shape)
                # note that i want the mask of the object within the template!! 
                obj_occR0_mask, template_lrt = self.vox_util.voxelize_near_xyz(
                    xyz_camR0,
                    self.obj_clist_camR0s[:,0],
                    int(self.ZZ/2),
                    int(self.ZY/2),
                    int(self.ZX/2),
                    sz=(self.ZZ*self.vox_size_Z),
                    sy=(self.ZY*self.vox_size_Y),
                    sx=(self.ZX*self.vox_size_X))

                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                obj_occR0_mask = F.conv3d(obj_occR0_mask, weights, padding=1)
                obj_occR0_mask = torch.clamp(obj_occR0_mask, 0, 1)
                self.summ_writer.summ_occs('track/obj_mask_and_mask_%d' % s, [self.obj_mask_template, obj_occR0_mask])
                
                
                
                # # this is B x S x 19
                # lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s[:,s:s+1], lrt_camR0s_e)
                # lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs[:,s:s+1], lrt_camRs_e)
                # lrt_camX_e = lrt_camXs_e.squeeze(1)

                
                # obj_mask_templates = self.vox_util.assemble_padded_obj_masklist_within_region(
                #     self.lrt_camR0s[:,0:1], self.obj_scorelist[:,0:1], self.template_lrts[:,0],
                #     int(self.ZZ/2), int(self.ZY/2), int(self.ZX/2), coeff=1.0)
                

                # right around here, i should decide that if there aren't enough points in the template, i just push with const velocity, and not waste rigidnet's time

                # let's sample corners in python
                
                R = hyp.rigid_repeats
                sampling_mask = (self.obj_mask_template*obj_occR0_mask).repeat(R, 1, 1, 1, 1)
                # sampled_corners, sampled_centers, sampling_failed = utils_misc.sample_eight_points(sampling_mask, max_tries=2000)
                sampled_corners, sampled_centers, sampling_failed = utils_misc.sample_eight_points(sampling_mask, max_tries=2000, random_center=False)
                if sampling_failed:
                    print('sampling failed; returning const velocity estimate')
                    # update the obj loc for the next step
                    # constant velocity
                    vel = obj_loc_ref - obj_loc_ref_prev
                    obj_loc_ref_prev_prev = obj_loc_ref_prev.clone()
                    obj_loc_ref_prev = obj_loc_ref.clone()
                    obj_loc_ref = obj_loc_ref + vel
                    obj_loc_mem = self.vox_util.Ref2Mem(obj_loc_ref.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
                    peak_xyzs_mem.append(obj_loc_mem)
                    peak_rots_mem.append(obj_r)
                else:
                    obj_loc_halfsearch, obj_r_delta, _ = self.rigidnet(
                        obj_featR0, # template
                        self.obj_mask_template*obj_occR0_mask, # mask of object within the template
                        search_featRi, # search region
                        torch.zeros(self.B, 3).float().cuda(), # gt rotation delta in search coords
                        torch.zeros(self.B, 3).float().cuda(), # gt position in search coords
                        sampled_corners,
                        sampled_centers,
                    )
                    obj_r = self.obj_rlist_camR0s[:,0] + obj_r_delta

                    # print('obj_loc_search', obj_loc_search.detach().cpu().numpy())


                    # the middle of the halfsearch coords is obj_loc_ref from the previous iter
                    search_mid = np.array([SX/4.0, SY/4.0, SZ/4.0], np.float32).reshape([1, 3])
                    search_mid = torch.from_numpy(search_mid).float().to('cuda')
                    obj_loc_halfmem = self.vox_util.Ref2Mem(obj_loc_ref_search.unsqueeze(1), self.Z2, self.Y2, self.X2).squeeze(1)
                    obj_loc_halfmem = (obj_loc_halfsearch - search_mid) + obj_loc_halfmem
                    obj_loc_mem = obj_loc_halfmem * 2.0
                    # this is B x 3, and in mem coords


                    # print('obj_loc_mem', obj_loc_mem.detach().cpu().numpy())


                    ## swap the estimates for gt
                    # obj_clist_memR0s_e = self.vox_util.Ref2Mem(self.obj_clist_camR0s, self.Z, self.Y, self.X)
                    # obj_loc_mem = obj_clist_memR0s_e[:,s]
                    # obj_r = self.obj_rlist_camR0s[:,s] - self.obj_rlist_camR0s[:,0]

                    peak_xyzs_mem.append(obj_loc_mem)
                    peak_rots_mem.append(obj_r)

                    # update the obj loc for the next step
                    obj_loc_ref_prev_prev = obj_loc_ref_prev.clone()
                    obj_loc_ref_prev = obj_loc_ref.clone()
                    obj_loc_ref = self.vox_util.Mem2Ref(obj_loc_mem.unsqueeze(1), self.Z, self.Y, self.X).squeeze(1)
            elif hyp.do_robust:
                lrt_camRi, _ = self.robustnet(
                    obj_featR0, # template
                    search_featRi, # search region
                    self.obj_mask_template, # mask of object within the template
                    self.template_lrts[:,0],
                    search_lrt,
                    self.vox_util,
                    self.lrt_camR0s)
                lrt_camR0s_e.append(lrt_camRi)

                obj_clist_camR0_e = utils_geom.get_clist_from_lrtlist(lrt_camRi.unsqueeze(1))
                obj_loc_search = self.vox_util.Ref2Zoom(
                    obj_clist_camR0_e, search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
                self.summ_writer.summ_traj_on_occ('robust/obj_loc_e_%d' % s,
                                                  obj_loc_search.unsqueeze(1)*2.0,
                                                  search_occRi,
                                                  self.vox_util,
                                                  already_mem=True,
                                                  sigma=2)

                # obj_clist_camRi_e = utils_geom.get_clist_from_lrtlist(lrt_camRi.unsqueeze(1))
                # obj_loc_search = self.vox_util.Ref2Zoom(
                #     obj_clist_camR0_e, search_lrt, self.ZZ, self.ZY, self.ZX, additive_pad=0.0).squeeze(1)
                # self.summ_writer.summ_traj_on_occ('robust/obj_loc_e',
                #                                   obj_loc_search.unsqueeze(1)*2.0,
                #                                   search_occR1,
                #                                   self.vox_util,
                #                                   already_mem=True,
                #                                   sigma=2)

                # obj_loc_search, obj_r_delta, _ = self.robustnet(
                #     obj_featR0, # template
                #     search_featRi, # search region
                #     torch.zeros(self.B, 3).float().cuda(), # gt rotation delta in search coords
                #     torch.zeros(self.B, 3).float().cuda(), # gt position in search coords
                # )
                

        if not hyp.do_robust:
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
        else:
            lrt_camR0s_e = torch.stack(lrt_camR0s_e, dim=1)
            lrt_camRs_e = utils_geom.apply_4x4s_to_lrts(self.camRs_T_camR0s, lrt_camR0s_e)
            lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camRs, lrt_camRs_e)

            obj_clist_camR0s_e = utils_geom.get_clist_from_lrtlist(lrt_camR0s_e)
            self.summ_writer.summ_traj_on_occ('track/estim_traj',
                                              obj_clist_camR0s_e,
                                              self.occ_memR0s[:,0],
                                              self.vox_util,
                                              already_mem=False,
                                              sigma=2)
            self.summ_writer.summ_traj_on_occ('track/true_traj',
                                              self.obj_clist_camR0s*self.obj_scorelist.unsqueeze(2),
                                              self.occ_memR0s[:,0],
                                              self.vox_util, 
                                              already_mem=False,
                                              sigma=2)

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
            
        
            # visX_g = []
            # for s in list(range(self.S)):
            #     visX_g.append(self.summ_writer.summ_lrtlist(
            #         '', self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
            #         torch.ones([self.B,1]).float().cuda(),
            #         torch.ones([self.B,1]).long().cuda(),
            #         self.pix_T_cams[:,0], only_return=True))
            # results['visX_g'] = visX_g
            # self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)

        return total_loss, results, False
        
    def forward(self, feed):
        ok = self.prepare_common_tensors(feed)
        if not ok:
            print('not enough object points; returning early')
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True

        set_name = feed['set_name']
        if set_name=='train' or set_name=='val':
            return self.train_over_pair(feed)
        elif set_name=='test':
            return self.track_over_seq(feed)
        else:
            print('weird set_name:', set_name)
            assert(False)
        
