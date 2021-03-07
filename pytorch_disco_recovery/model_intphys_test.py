import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import os

from model_base import Model
from nets.featnet import FeatNet
from nets.detnet import DetNet
from nets.forecastnet import ForecastNet

import torch.nn.functional as F

import utils_vox
import utils_samp
import utils_geom
import utils_misc
import utils_improc
import utils_basic
import utils_track
import frozen_flow_net
import utils_eval
import utils_py

from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
EPS = 1e-6
MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class INTPHYS_TEST(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = IntphysTestModel()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_det:
            self.model.detnet.eval()
            self.set_requires_grad(self.model.detnet, False)
        if hyp.do_freeze_forecast:
            self.model.forecastnet.eval()
            self.set_requires_grad(self.model.forecastnet, False)

    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")
        if hyp.lr > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr)
            self.start_iter = saverloader.load_weights(self.model, self.optimizer)
            print("------ Done loading weights ------")
        else:
            self.start_iter = 0

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

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
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

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                # if log_this or set_do_backprop:
                          
                read_start_time = time.time()

                feed = next(set_loader)
                feed_cuda = {}
                for k in feed:
                    try:
                        feed_cuda[k] = feed[k].cuda(non_blocking=True)
                    except:
                        # some things are not tensors (e.g., filename)
                        feed_cuda[k] = feed[k]

                # feed_cuda = next(iter(set_input))
                read_time = time.time() - read_start_time

                feed_cuda['writer'] = set_writer
                feed_cuda['global_step'] = step
                feed_cuda['set_num'] = set_num
                feed_cuda['set_name'] = set_name

                iter_start_time = time.time()
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
                iter_time = time.time()-iter_start_time
                total_time = time.time()-self.start_time

                if not returned_early:
                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_py,
                                                                                        set_name))
            
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: # close writers, to flush cache into the file
            writer.close()
            
class IntphysTestModel(nn.Module):
    def __init__(self):
        super(IntphysTestModel, self).__init__()
            
        self.device = torch.device("cuda")

        self.include_image_summs = True
        # self.include_image_summs = False
        
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_det:
            self.detnet = DetNet()
        # if hyp.do_forecast:
        #     self.forecastnet = ForecastNet()
        
    def forward(self, feed):
        results = dict()
        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               fps=8,
                                               just_gif=True)
        global_step = feed['global_step']
        
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        K = hyp.K
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        ZZ, ZY, ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9

        pix_T_cams = feed["pix_T_cams"]
        rgb_cams = feed["rgb_camXs"]
        xyz_cams = feed["xyz_camXs"]
        boxlist_cams = feed["boxlist_camXs"]
        validlist_s = feed["validlist_s"] # coordinate-less and plural
        vislist_s = feed["vislist_s"] # coordinate-less and plural
        tidlist_s = feed["tidlist_s"] # coordinate-less and plural

        ## postproc
        lrtlist_cams = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_cams))).reshape(B, S, N, 19)
        (obj_lrtlist_cams,
         obj_validlist_s,
         obj_vislist_s,
        ) = utils_misc.collect_object_info(lrtlist_cams,
                                           tidlist_s,
                                           validlist_s,
                                           vislist_s,
                                           mod='X',
                                           do_vis=False,
                                           summ_writer=summ_writer)
        # obj_lrtlist_cams is N x B x S x 19
        occ_mems = __u(utils_vox.voxelize_xyz(__p(xyz_cams), Z, Y, X))
        occ_mems_half = __u(utils_vox.voxelize_xyz(__p(xyz_cams), Z2, Y2, X2))

        ## projected depth, and inbound mask
        depth_cams_, valid_cams_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_cams), H, W)
        dense_xyz_cams_ = utils_geom.depth2pointcloud(depth_cams_, __p(pix_T_cams))
        inbound_cams_ = utils_vox.get_inbounds(dense_xyz_cams_, Z, Y, X).float()
        inbound_cams_ = torch.reshape(inbound_cams_, [B*S, 1, H, W])
        depth_cams = __u(depth_cams_)
        valid_cams = __u(valid_cams_) * __u(inbound_cams_)

        if self.include_image_summs and summ_writer.save_this:
            for k in list(range(K)):
                obj_lrtlist_camX = obj_lrtlist_cams[k]
                # this is B x S x 19
                obj_validlist = obj_validlist_s[k]
                # this is B x S
                obj_clist = utils_geom.get_clist_from_lrtlist(obj_lrtlist_camX).squeeze(2)
                # this is B x S x 3
                obj_clist = obj_clist*obj_validlist.unsqueeze(2)
                summ_writer.summ_traj_on_occ('traj/obj%d_clist' % k,
                                             obj_clist, occ_mems[:,0], already_mem=False)
                
            lrt_vis = []
            for s in list(range(S)):
                depth_vis = summ_writer.summ_oned('', depth_cams[:,s],
                                                  logvis=True, maxval=20, norm=False, only_return=True)
                depth_vis = utils_improc.preprocess_color(depth_vis)
                o = summ_writer.summ_lrtlist('', depth_vis[0:1], lrtlist_cams[0:1,s],
                                             validlist_s[0:1,s], tidlist_s[0:1,s], pix_T_cams[0:1,s], only_return=True)
                lrt_vis.append(o)
            summ_writer.summ_rgbs('obj/lrtlist_cams_on_depthXs', lrt_vis)
                
            summ_writer.summ_rgbs('2D_inputs/rgb_cams', torch.unbind(rgb_cams, dim=1))
            summ_writer.summ_oneds('2D_inputs/depth_cams', torch.unbind(depth_cams, dim=1),
                                   logvis=True, maxval=20, norm=False)
            summ_writer.summ_oneds('2D_inputs/valid_cams', torch.unbind(valid_cams, dim=1))
            summ_writer.summ_occs('3D_inputs/occ_mems', torch.unbind(occ_mems, dim=1))

        if hyp.do_feat:
            # occ_mems is B x S x 1 x H x W x D

            feat_mems_input = occ_mems
            feat_mems_input_ = __p(feat_mems_input)
            feat_mems_, valid_mems_, _ = self.featnet(
                feat_mems_input_,
                summ_writer,
                comp_mask=__p(occ_mems),
                include_image_summs=self.include_image_summs,
            )
            feat_mems = __u(feat_mems_)
            valid_mems = __u(valid_mems_)
            if self.include_image_summs and summ_writer.save_this:
                summ_writer.summ_feats('3D_feats/feat_mems_input', torch.unbind(feat_mems_input, dim=1), pca=False)
                summ_writer.summ_feats('3D_feats/feat_mems_output', torch.unbind(feat_mems, dim=1),
                                       torch.unbind(valid_mems, dim=1), pca=True)
            
        if hyp.do_det:
            
            boxlist_mems = __u(utils_vox.convert_boxlist_camR_to_memR(__p(boxlist_cams), Z2, Y2, X2))
            axboxlist_mems = __u(utils_geom.inflate_to_axis_aligned_boxlist(__p(boxlist_mems)))

            _, boxlist_mems_e_, scorelist_s_e_, tidlist_s_e_, _, _ = self.detnet(
                __p(axboxlist_mems),
                __p(vislist_s),
                __p(feat_mems),
                summ_writer)
            boxlist_mems_e = __u(boxlist_mems_e_)
            scorelist_s_e = __u(scorelist_s_e_)
            tidlist_s_e = __u(tidlist_s_e_)
            # boxlist_mems_e is B x S x N x 19
            # scorelist_s_e is B x S x N
            # tidlist_s_e is B x S x N

            if boxlist_mems_e is not None:
                # note the returned data has batchsize=1
                
                boxlist_cams_e = __u(utils_vox.convert_boxlist_memR_to_camR(__p(boxlist_mems_e), Z2, Y2, X2))
                
                if self.include_image_summs and summ_writer.save_this:
                    lrtlist_cams_e = __u(utils_geom.convert_boxlist_to_lrtlist(__p(boxlist_cams_e)))
                    vis_e = []
                    for s in list(range(S)):
                        lrtlist_cams_e[:,s]
                        scorelist_s_e[:,s]
                        tidlist_s_e[:,s]
                        vis_e.append(summ_writer.summ_lrtlist('det/axboxlist_cam_e', rgb_cams[0:1,s], lrtlist_cams_e[:,s],
                                                              scorelist_s_e[:,s], tidlist_s_e[:,s], pix_T_cams[0:1,s], only_return=True))
                    summ_writer.summ_rgbs('det/axboxlist_cams_e', vis_e)

        # plan:
        # walk through the frames
        # when i detect an object, i need to start tracking it
        # (so actually i need a tracker right now)
        # once i track it for 3 steps, fore the firecaster

        # ok how do i get a tracker right now?
        # > turn the object info into flow, and supervise a flownet, and deploy the flow-based method
        # >> ok not too bad, though the flownet will take time to converge
        # >>> maybe not all that long, since it is supervised
        # >>> and plus it's productive, since flow would be a good input to the forecaster
        # > for every re-detection, do some hungarian assignment, based on feature distance
        # >> this is productive too, since i need this kind of thing to solve block2

        # ok let's try the second option first, since it requires less waiting time, and brings me closer to an eval. maybe, for instance, i can solve block2 right away.

        # but even that means using the estimated stuff
        # let me get closer to the finish line, by using gt
        # please please please
        # just relax, calvin, you've got a big bruise on your head

        # so now then, i need two slightly different trajectories of the object
        # ah: i need different data printed out for this:
        # > full seqs, starting at 0, with all objects
        # >> ok got it

        # now let's say the plan again:
        
        # plan:
        # walk through the frames
        # first track everything perfectly

        # wait wait wait
        # i do not necessarily need to "track" through occlusions
        # since i am in 3d, i can take full advantage of 3d nonintersection
        # and just do a couple sanity checks
        # like the feature distance

        # so ok:

        # the idea is this:
        # form tracklets. these start and end on visibility
        # from the first 3 steps of each tracklet, fire the forecaster
        # then, walk along the forecasted locations: 
        # > if the loc is unoccluded and the detector did not fire, raise a flag

        # simplified idea, for the full gt case:
        # get the true traj of each object
        # get the visible detections in the same coords
        # for each traj location:
        # > check extended raycast visibility
        # > check if there is a detection there

        # ok: let me assemble this at least

        assert(B==1) # so i can relax

        # free_mems = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems)))
        # vis_mems0 = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems)))
        # vis_mems1 = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems), ray_add=0.25))
        # vis_mems = torch.clamp(vis_mems0+vis_mems1, 0, 1)

        # free_mems0 = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems), ray_add=0.5))
        # free_mems1 = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems), ray_add=2.0))
        # free_mems = torch.clamp(free_mems0+free_mems1, 0, 1)
        # # free_mems = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems), ray_add=3.0))
        # vis_mems = torch.clamp(free_mems+occ_mems, 0, 1)

        free_mems = __u(utils_vox.get_freespace(__p(xyz_cams), __p(occ_mems), ray_add=1.5))
        vis_mems = torch.clamp(free_mems+occ_mems, 0, 1)
        # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
        # vis_mems = __u(F.conv3d(__p(vis_mems), weights, padding=1))
        # vis_mems = torch.clamp(vis_mems, 0, 1)
        
        # we need to do this bc right now the offset one misses the very early part of the scene
        
        if self.include_image_summs and summ_writer.save_this:
            summ_writer.summ_occs('3D_inputs/free_mems', torch.unbind(free_mems, dim=1))
            summ_writer.summ_occs('3D_inputs/vis_mems', torch.unbind(vis_mems, dim=1))
            summ_writer.summ_feats('3D_inputs/vis_mems_oned', torch.unbind(vis_mems, dim=1), pca=False)

            summ_writer.summ_occ('3D_inputs/occ_mem0', occ_mems[:,0])
            summ_writer.summ_feat('3D_inputs/vis_mem0_oned', vis_mems[:,0], pca=False)
            summ_writer.summ_occ('3D_inputs/vis_mem0', vis_mems[:,0])

        find_accs = []
        ray_accs = []
        for k in list(range(K)):
            # print('working on k = %d' % k)
            lrtlist_camX = obj_lrtlist_cams[k]
            # this is B x S x 19
            validlist = obj_validlist_s[k]
            # this is B x S
            vislist = obj_vislist_s[k]
            # this is B x S

            # this array holds values in 1,0 indicating whether we found an obj at the traj location or not
            foundlist = torch.zeros_like(validlist)

            # boxlist_mems = __u(utils_vox.convert_boxlist_camR_to_memR(__p(boxlist_cams), Z, Y, X))
            
            clist_cam = utils_geom.get_clist_from_lrtlist(lrtlist_camX).squeeze(2)
            # this is B x S x 3
            clist_mem = utils_vox.Ref2Mem(clist_cam, Z, Y, X)
            # this is B x S x 3

            foundlist = utils_misc.find_detections_corresponding_to_traj(
                clist_cam,
                validlist,
                # boxlist_mems[:,:,:,:3],
                # vislist_s,
                boxlist_cams_e[:,:,:,:3]*scorelist_s_e.unsqueeze(3),
                scorelist_s_e,
            )

            found_match = utils_basic.reduce_masked_mean((foundlist==validlist).float(), validlist)
            find_accs.append(found_match)

            # print('validlist', validlist[0,:10])
            # print('foundlist', foundlist[0,:10])
            # print('vislist', vislist[0,:10])
            # # input()

            # ok this makes sense so far
            # next step is to get visibilities using raycasting
            # then i match
            
            raylist = torch.zeros_like(vislist)
            for s in list(range(S)):
                vis_mem = vis_mems[:,s]
                x, y, z = torch.unbind(clist_mem[:,s:s+1], dim=2)
                # these are B x 1
                vis_here = utils_samp.bilinear_sample3D(vis_mem, x, y, z)
                raylist[:,s] = vis_here
            raylist = raylist*validlist
                            
            # print('raylist', raylist.round()[0,:10])

            ray_vis_match = utils_basic.reduce_masked_mean((raylist.round()==vislist).float(), validlist)
            ray_accs.append(ray_vis_match)
        ray_accs = torch.stack(ray_accs) # ray_accs is shaped K
        ray_accs = ray_accs.reshape(K, 1, 1).repeat(1, B, S) # K x B x S
        ray_acc = utils_basic.reduce_masked_mean(ray_accs, obj_validlist_s)
        # summ_writer.summ_scalar('ray_vis_match_%d' % k, ray_vis_match.cpu().item())
        summ_writer.summ_scalar('ray_vis_match_ray_acc', ray_acc.cpu().item())
        print('ray_acc', ray_acc.detach().cpu().numpy())

        find_accs = torch.stack(find_accs) # find_accs is shaped K
        find_accs = find_accs.reshape(K, 1, 1).repeat(1, B, S) # K x B x S
        find_acc = utils_basic.reduce_masked_mean(find_accs, obj_validlist_s)
        summ_writer.summ_scalar('find_match_acc', find_acc.cpu().item())
        print('find_acc', find_acc.detach().cpu().numpy())

        # note some of the missed stuff may be due to the object flying out of bounds


        # goal: get tracklets
        # all i need is 3 good frames
        # but they better be within the first 10
        # so here is my idea:
        # for each frame, starting at 1 and going to 9,
        # > for each object detected in that frame,
        # >> look into the frame ahead, and the frame behind:
        # >>> for each object in that frame, 
        # >>>> compute some custom feature distance (including 3d pos probably)
        # >>> if both frames have a good match,
        # >>>> call the triplet a tracklet
        # >>> i only need one triplet to work confidently for all three objects

                                
        # for s in list(range(1, 10)):
        #     scorelist_here = scorelist_s_e[s]

        max_reliable_count = 0 
        reliable_centers = []
        for s in list(range(1, 8)):
            
            # boxlist_cams_e is B x S x N x 19
            boxlist_cam_here = boxlist_cams_e[:,s]
            boxlist_cam_prev = boxlist_cams_e[:,s-1]
            boxlist_cam_next = boxlist_cams_e[:,s+1]

            scorelist_here = scorelist_s_e[:,s]
            scorelist_prev = scorelist_s_e[:,s-1]
            scorelist_next = scorelist_s_e[:,s+1]

            scorelist_here = scorelist_here.round()
            scorelist_prev = scorelist_prev.round()
            scorelist_next = scorelist_next.round()

            num_objects_here = torch.sum(scorelist_here).detach().cpu().numpy()
            num_objects_prev = torch.sum(scorelist_prev).detach().cpu().numpy()
            num_objects_next = torch.sum(scorelist_next).detach().cpu().numpy()

            if num_objects_here>0 and num_objects_here==num_objects_prev and num_objects_here==num_objects_next:
                print('triplet centered at %d looks pretty good; in each frame we saw %d objects' % (
                    s, num_objects_here))
                if num_objects_here > max_reliable_count:
                    max_reliable_count = num_objects_here
                    reliable_centers = []
                if num_objects_here == max_reliable_count:
                    reliable_centers.append(s)
        print('we are going with %d objects, with these frames guiding' % max_reliable_count, reliable_centers)


        best_tidlist = np.zeros([B, 3, N]).astype(np.int32)
        best_ind = 0
        for si, s in enumerate(reliable_centers):
            print('### working on centerframe %d ###' % s)
            # boxlist_cams_e is B x S x N x 19
            boxlist_cam_here = boxlist_cams_e[:,s]
            boxlist_cam_prev = boxlist_cams_e[:,s-1]
            boxlist_cam_next = boxlist_cams_e[:,s+1]

            scorelist_here = scorelist_s_e[:,s]
            scorelist_prev = scorelist_s_e[:,s-1]
            scorelist_next = scorelist_s_e[:,s+1]

            scorelist_here = scorelist_here.round()
            scorelist_prev = scorelist_prev.round()
            scorelist_next = scorelist_next.round()
            
            N = int(torch.sum(scorelist_here).detach().cpu().numpy())
            boxlist_cam_here = boxlist_cam_here[:,:N]
            boxlist_cam_prev = boxlist_cam_prev[:,:N]
            boxlist_cam_next = boxlist_cam_next[:,:N]
            scorelist_here = scorelist_here[:,:N]
            scorelist_prev = scorelist_prev[:,:N]
            scorelist_next = scorelist_next[:,:N]
            # clipping should not affect things
            assert(torch.sum(scorelist_here).detach().cpu().numpy()==N)
            assert(torch.sum(scorelist_prev).detach().cpu().numpy()==N)
            assert(torch.sum(scorelist_next).detach().cpu().numpy()==N)

            # claims = np.zeros([B, N, S])
            

            # assignments = np.zeros([B, N, S])
            # assignments = np.zeros([B, N, 3])
            # new_tidlist = np.zeros([B, 3, N]).astype(np.int32)
            claims = np.zeros([N, 3]).astype(np.int32)
            for n_here in list(range(N)):
                print('--- working on object %d ---' % n_here)
                box_cam_here = boxlist_cam_here[:,n_here]
                # this is B x 9
                # xyz_cam_here = box_cam_here[:,:3]
                # # this is B x 3

                # match_prev = torch.zeros_like(scorelist_here)
                # for n_prev in list(range(N)):
                #     # print('n_prev', n_prev)
                #     box_cam_prev = boxlist_cam_prev[:,n_prev]
                #     xyz_cam_prev = box_cam_prev[:,:3]
                #     # match_prev[:,n_prev] = torch.exp(-torch.norm(xyz_cam_here-xyz_cam_prev, dim=1))
                #     match_prev[:,n_prev] = torch.norm(xyz_cam_here-xyz_cam_prev, dim=1)
                # match_next = torch.zeros_like(scorelist_here)
                # for n_next in list(range(N)):
                #     # print('n_next', n_next)
                #     box_cam_next = boxlist_cam_next[:,n_next]
                #     xyz_cam_next = box_cam_next[:,:3]
                #     # match_next[:,n_next] = torch.exp(-torch.norm(xyz_cam_here-xyz_cam_next, dim=1))
                #     match_next[:,n_next] = torch.norm(xyz_cam_here-xyz_cam_next, dim=1)
                # print('match_prev', np.squeeze(match_prev.detach().cpu().numpy()))
                # print('match_next', np.squeeze(match_next.detach().cpu().numpy()))

                
                # constant-velocity matching

                xyz_here = box_cam_here[:,:3]
                # this is B x 3
                xyzlist_prev = boxlist_cam_prev[:,:,:3]
                xyzlist_next = boxlist_cam_next[:,:,:3]
                # these are B x N x 3

                best_dist = 10.0
                # we expect the object to be in the center of the two locations
                for n_prev in list(range(N)):
                    for n_next in list(range(N)):
                        xyz_prev = xyzlist_prev[:,n_prev]
                        xyz_next = xyzlist_next[:,n_next]
                        xyz_mid = (xyz_next+xyz_prev)/2.0
                        dist = torch.norm(xyz_mid-xyz_here, dim=1).detach().cpu().numpy()
                        print('if this is the answer, dist = %.2f' % dist)
                        if dist < best_dist:
                            best_dist = dist
                            match_prev = n_prev
                            match_next = n_next
                print('best_dist = %.2f' % best_dist)

                # this object claims n_prev and n_next as correspondences
                claims[n_here,0] = match_prev
                claims[n_here,1] = n_here
                claims[n_here,2] = match_next

            print('claims\n', claims)
            
            # new_tidlist[0, 0, match_prev] = n_here
            # new_tidlist[0, 1, n_here] = n_here
            #     # new_tidlist[0, 2, match_next] = n_here
            # print('new_tidlist:\n', new_tidlist.shape)

            # summ_writer.summ_lrtlist('assoc/boxes_%d' % ri,
            #                      rgb_cams[0:1,s],
            #                      # boxlist_cam_here,
            #                      utils_geom.convert_boxlist_to_lrtlist(boxlist_cam_here),
            #                      scorelist_here,
            #                      torch.from_numpy(new_tidlist[:,1]).long().cuda(),
            #                      pix_T_cams[0:1,s],
            # )
            # summ_writer.summ_lrtlist('assoc/boxes_%d' % ri,
            #                      rgb_cams[0:1,s-1],
            #                      # boxlist_cam_here,
            #                      utils_geom.convert_boxlist_to_lrtlist(boxlist_cam_prev),
            #                      scorelist_next,
            #                      torch.from_numpy(new_tidlist[:,1]).long().cuda(),
            #                      pix_T_cams[0:1,s],
            # )
            # summ_writer.summ_lrtlist('assoc/boxes_%d' % ri,
            #                      rgb_cams[0:1,s+1],
            #                      # boxlist_cam_here,
            #                      utils_geom.convert_boxlist_to_lrtlist(boxlist_cam_next),
            #                      scorelist_next,
            #                      torch.from_numpy(new_tidlist[:,2]).long().cuda(),
            #                      pix_T_cams[0:1,s],
            # )


        # vis = []

        # # new_tidlist = torch.from_numpy(new_tidlist.astype(np.int32)).cuda()

        # # tidlist_s_e[0,s-1,:N] = new_tidlist[0,0]
        # # tidlist_s_e[0,s,:N] = new_tidlist[0,1]
        # # tidlist_s_e[0,s+1,:N] = new_tidlist[0,2]

        # for s in list(range(S)):
        #     # # lrtlist_cams_e[:,s]
        #     # # scorelist_s_e[:,s]
        #     # # # tidlist_s_e[:,s]
        #     # print(lrtlist_cams_e.shape)
        #     # print(tidlist_s_e.shape)
        #     # print(new_tidlist.shape)
        #     vis_e.append(summ_writer.summ_lrtlist('det/axboxlist_cam_e', rgb_cams[0:1,s], lrtlist_cams_e[:,s,:N],
        #                                           scorelist_s_e[:,s,:N],
        #                                           tidlist_s_e[:,s],
        #                                           # new_tidlist,
        #                                           pix_T_cams[0:1,s], only_return=True))
        # summ_writer.summ_rgbs('assoc/axboxlist_cams_e_%d' % s, vis_e)
            
            
            
            
            

        
                    
        # summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

