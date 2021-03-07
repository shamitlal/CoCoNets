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

def propose_boxes_by_differencing(
        K, S,
        occ_memXAI_all,
        diff_memXAI_all,
        crop_zyx,
        set_data_name=None,
        data_ind=None,
        super_iter=None,
        use_box_cache=False,
        summ_writer=None):

    have_boxes = False
    if hyp.do_use_cache and use_box_cache:
        box_cache_fn = 'cache/%s_%06d_s%d_box_%d.npz' % (set_data_name, data_ind, S, super_iter)
        # check if the thing exists
        if os.path.isfile(box_cache_fn):
            print('found box cache at %s; we will use this' % box_cache_fn)
            cache = np.load(box_cache_fn, allow_pickle=True)['save_dict'].item()
            # cache = cache['save_dict']

            have_boxes = True
            lrtlist_memXAI_all = torch.from_numpy(cache['lrtlist_memXAI_all']).cuda().unbind(1)
            connlist_memXAI_all = torch.from_numpy(cache['connlist_memXAI_all']).cuda().unbind(1)
            scorelist_all = [s for s in torch.from_numpy(cache['scorelist_all']).cuda().unbind(1)]
            blue_vis = [s for s in torch.from_numpy(cache['blue_vis']).cuda().unbind(1)]
        else:
            print('could not find box cache at %s; we will write this' % box_cache_fn)

    if not have_boxes:
        blue_vis = []
        # conn_vis = []
        # lrtlist_camXIs = []
        lrtlist_memXAI_all = []
        connlist_memXAI_all = []
        scorelist_all = []

        for I in list(range(S)):
            # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
            #     self.pad_feat(diff_memXAI_all[I]).squeeze(1), self.K)
            # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
            #     cumdiff_memXAI_all[I].squeeze(1), self.K)
            # flow05_mag = torch.norm(flow05_all[I], dim=1)
            # flow05_mag = torch.norm(consistent_flow05_all[I], dim=1)
            # boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
            #     flow05_mag, self.K)

            diff_memXAI = diff_memXAI_all[I]
            # border = 1
            # diff_memXAI[:,:,0:border] = 0
            # diff_memXAI[:,:,:,0:border] = 0
            # diff_memXAI[:,:,:,:,0:border] = 0
            # diff_memXAI[:,:,-border:] = 0
            # diff_memXAI[:,:,:,-border:] = 0
            # diff_memXAI[:,:,:,:,-border:] = 0
            if summ_writer is not None:
                summ_writer.summ_oned('proposals/diff_iter_%d' % I, diff_memXAI, bev=True)

            boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                diff_memXAI.squeeze(1), K)
            x, y, z, lx, ly, lz, rx, ry, rz = boxlist_memXAI.unbind(2)
            ly = ly + 1.0
            z = z + crop_zyx[0]
            y = y + crop_zyx[1]
            x = x + crop_zyx[2]
            boxlist_memXAI = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=2)

            lrtlist_memXAI = utils_geom.convert_boxlist_to_lrtlist(boxlist_memXAI)
            lrtlist_memXAI_all.append(lrtlist_memXAI)
            connlist_memXAI_all.append(connlist)

            scorelist_e[scorelist_e > 0.0] = 1.0
            occ_memXAI = occ_memXAI_all[I]
            diff_memXAI = diff_memXAI_all[I]
            for n in list(range(K)):
                mask_1 = connlist[:,n:n+1]
                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                mask_3 = (F.conv3d(mask_1, weights, padding=1)).clamp(0, 1)

                center_mask = mask_1.clone()
                surround_mask = (mask_3-mask_1).clamp(0,1)

                center_ = utils_basic.reduce_masked_mean(occ_memXAI, center_mask, dim=[2,3,4])
                surround_ = utils_basic.reduce_masked_mean(occ_memXAI, surround_mask, dim=[2,3,4])
                score_ = center_ - surround_
                score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)
                score_[score_ < 0.55] = 0.0
                scorelist_e[:,n] = score_
            scorelist_all.append(scorelist_e)

            # self.summ_writer.summ_rgb('proposals/anchor_frame', diff_memXAI_vis[self.anchor])
            # self.summ_writer.summ_rgb('proposals/get_boxes', boxes_image)
            blue_vis.append(boxes_image)

            # conn_vis.append(self.summ_writer.summ_occ('', torch.sum(connlist, dim=1, keepdims=True).clamp(0, 1), only_return=True))

        if hyp.do_use_cache and use_box_cache:
            # save this, so that we have it all next time
            save_dict = {}

            save_dict['lrtlist_memXAI_all'] = torch.stack(lrtlist_memXAI_all, dim=1).detach().cpu().numpy()
            save_dict['connlist_memXAI_all'] = torch.stack(connlist_memXAI_all, dim=1).detach().cpu().numpy()
            save_dict['scorelist_all'] = torch.stack(scorelist_all, dim=1).detach().cpu().numpy()
            save_dict['blue_vis'] = torch.stack(blue_vis, dim=1).detach().cpu().numpy()
            np.savez(box_cache_fn, save_dict=save_dict)
            print('saved boxes to %s cache, for next time' % box_cache_fn)

    return lrtlist_memXAI_all, connlist_memXAI_all, scorelist_all, blue_vis


def track_one_step_via_inner_product(
        B, C,
        Z_, Y_, X_,
        lrt_camXAI,
        # lrt_prevprev,
        featI_vec,
        feat0_vec,
        obj_mask0_vec,
        obj_length,
        cam0_T_obj,
        orig_xyz,
        diff_memXAI, 
        vox_util,
        cropper,
        crop_param,
        delta,
        summ_writer=None,
        use_window=False):

    Z_pad, Y_pad, X_pad = crop_param
    Z = Z_ + Z_pad*2
    Y = Y_ + Y_pad*2
    X = X_ + X_pad*2
    crop_vec = torch.from_numpy(np.reshape(np.array([X_pad, Y_pad, Z_pad]), (1, 1, 3))).float().cuda()

    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
    diff_memXAI = (F.conv3d(diff_memXAI, weights, padding=1)).clamp(0, 1)
    # diff_memXAI = (F.conv3d(diff_memXAI, weights, padding=1)).clamp(0, 1)

    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)
    obj_T_cam0 = cam0_T_obj.inverse()

    # lrt_camXAI is the last known position of the object; it represents lrt_prev
    # lrt_prevprev is the second-last known position

    # # we can immediately compute a prior, based on const velocity
    # el, rt_prevprev = utils_geom.split_lrt(lrt_prevprev)
    # _, rt_prev = utils_geom.split_lrt(lrt_camXAI)
    # r_prevprev, t_prevprev = utils_geom.split_rt(rt_prevprev)
    # r_prev, t_prev = utils_geom.split_rt(rt_prev)
    # vel = t_prev - t_prevprev
    # t_curr = t_prev + vel
    # rt_curr = utils_geom.merge_rt(r_prev, t_curr)
    # lrt_curr = utils_geom.merge_lrt(el, rt_curr)
    lrt_curr = lrt_camXAI.clone()
    _, camI_T_obj = utils_geom.split_lrt(lrt_curr)
    memI_T_mem0 = utils_basic.matmul4(mem_T_cam, camI_T_obj, obj_T_cam0, cam_T_mem)
    mem0_T_memI = memI_T_mem0.inverse()
    # mem0_T_memI = utils_basic.matmul4(mem_T_cam, cam0_T_obj, camI_T_obj.inverse(), cam_T_mem)

    # print('t-2', t_prevprev.detach().cpu().numpy())
    # print('t-1', t_prev.detach().cpu().numpy())
    # print('t-0', t_curr.detach().cpu().numpy())

    
    # # clist_curr = utils_geom.get_clist_from_lrtlist(lrt_camXAI.unsqueeze(1))
    # clist_camXAI = utils_geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1))
    # clist_memXAI = vox_util.Ref2Mem(clist_camXAI, Z, Y, X)
    # inb = vox_util.get_inbounds(clist_memXAI-crop_vec, Z_, Y_, X_, already_mem=True, padding=4.0)
    # # print('inb', inb.detach().cpu().numpy())
    # if torch.sum(inb) == 0:
    #     # print('centroid predicted OOB; returning the prior')
    #     # return early; we won't find the object bc it's now oob
    #     # mem0_T_memI = utils_geom.eye_4x4(B)
    #     # score = torch.zeros_like(lrt_camXAI[:,0])
    #     score = torch.ones_like(lrt_camXAI[:,0])*0.5
    #     # return lrt_camXAI, score, mem0_T_memI
    #     return lrt_curr, score, mem0_T_memI

    

    # print('clist_memXAI', clist_memXAI.detach().cpu().numpy(), clist_memXAI.shape)
    # # we can use this to create a search window
    # if use_window:
    #     window = vox_util.xyz2circles(
    #         clist_camXAI, Z, Y, X, radius=4.0, soft=False, already_mem=False)
    #     window = cropper(window)
    #     # if summ_writer is not None:
    #     #     # summ_writer.summ_oned('track/window_%d' % s, window, bev=True)
    #     #     summ_writer.summ_oned('track/window', window, bev=True)

    # memI_T_mem0 = utils_geom.eye_4x4(B)
    # # we will fill this up

    feat0_map = feat0_vec.permute(0, 2, 1).reshape(B, -1, Z_, Y_, X_)
    featI_map = featI_vec.permute(0, 2, 1).reshape(B, -1, Z_, Y_, X_)

    xyzI_prior_full = utils_geom.apply_4x4(memI_T_mem0, orig_xyz)
    # xyzI_coords_full = utils_basic.gridcloud3D(B, Z, Y, X)

    
    # to simplify the impl, we will iterate over the batchdim
    for b in list(range(B)):
        featI_vec_b = featI_vec[b]
        feat0_vec_b = feat0_vec[b]
        featI_map_b = featI_map[b]
        feat0_map_b = feat0_map[b]
        obj_mask0_vec_b = obj_mask0_vec[b]
        orig_xyz_b = orig_xyz[b]
        # these are huge x C

        xyzI_prior_b = xyzI_prior_full[b]
        # xyzI_coords_b = xyzI_coords_full[b]


        obj_inds_b = torch.where(obj_mask0_vec_b > 0)
        obj_vec_b = feat0_vec_b[obj_inds_b]
        xyz0 = orig_xyz_b[obj_inds_b]
        xyzI_prior = xyzI_prior_b[obj_inds_b]
        # xyzI_coords = xyzI_coords_b[obj_inds_b]
        # these are med x C

        # now i want to make masks, shaped med x Z x Y x X
        # according to xyzI_prior and with the help of xyzI_coords

        # print('mean xyzI_prior', torch.mean(xyzI_prior, dim=0).detach().cpu().numpy(), xyzI_prior.shape)
        
        if use_window:
            # this window works ok at full res
            window = vox_util.xyz2circles(xyzI_prior.unsqueeze(0) - crop_vec,
                                          Z_, Y_, X_, radius=4.0, soft=False, already_mem=True)
            # window is 1 x med x Z x Y x X
            window = window.permute(1, 0, 2, 3, 4)
            # window is med x 1 x Z x Y x X

        # print('started at xyz0', xyz0[:10].detach().cpu().numpy())

        obj_vec_b = obj_vec_b.permute(1, 0)
        # this is is C x med

        corr_b = torch.matmul(featI_vec_b.detach(), obj_vec_b.detach())
        # this is huge x med

        heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z_, Y_, X_)
        # heat_b is med x 1 x Z x Y x X

        # window = vox_util.xyz2circles(xyz0.unsqueeze(0) - crop_vec,
        #                               Z_, Y_, X_, radius=6.0, soft=False, already_mem=True)
        # # window is 1 x med x Z x Y x X
        # window = window.permute(1, 0, 2, 3, 4)
        # # window is med x 1 x Z x Y x X
        
        # print('xyz0', xyz0.shape)
        # print('heat_b', heat_b.shape)
        # print('window', window.shape)
        # if summ_writer is not None:
        #     # summ_writer.summ_oned('track/window_%d' % s, window, bev=True)
        #     for n in list(range(len(xyz0))):
        #         summ_writer.summ_oned('track/window_%d' % n, window[n:n+1], bev=True)

        # heat_b = F.relu(heat_b * (1.0 - self.occ_memXAI_median[b:b+1]))
        # heat_b = F.relu(heat_b * (1.0 - self.occ_memXAI_median[b:b+1]))
        heat_b = F.relu(heat_b)
        if use_window:
            heat_b = heat_b * window
        # heat_b = F.relu(heat_b * diff_memXAI[b:b+1])

        # heat_b = heat_b*0.5 + heat_b*0.5*diff_memXAI[b:b+1]
        # window_b = window[b:b+1]
        # window_weight = 0.5
        # # heat_b = heat_b*(1.0-window_weight) + heat_b*window_b*window_weight
        # # heat_b = heat_b*(1.0-window_weight) + window*window_weight
        # heat_b = heat_b * window[b:b+1]

        # we need to pad this, because we are about to take the argmax and interpret it as xyz
        heat_b = F.pad(heat_b, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        # this is med x 1 x Z x Y x X

        # # this is 1 x med x Z x Y x X

        # heat_b = heat_b * window0

        # # for numerical stability, we sub the max, and mult by the resolution
        # heat_b_ = heat_b.reshape(-1, Z*Y*X)
        # heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
        # heat_b = heat_b - heat_b_max
        heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

        xyzI = utils_basic.argmax3D(heat_b, hard=True, stack=True)
        # this is med x 3
        # note that this is in the coords of padded featI

        # print('xyzI', xyzI.shape)
        # print('first argmax', xyzI_new[0].detach().cpu().numpy())
        

        # print('peaked at xyzI', xyzI[:10].detach().cpu().numpy())

        
        def get_cycle_dist(featI_map_b, xyzI,
                           feat0_vec_b, xyz0,
                           Z_pad, Y_pad, X_pad,
                           crop_vec):
            # so i can extract features there
            correspI = utils_samp.bilinear_sample3D(featI_map_b.unsqueeze(0), xyzI.unsqueeze(0) - crop_vec).squeeze(0)
            # this is C x med

            # print('correspI', correspI.shape)
            # print('first feat:', correspI[0].detach().cpu().numpy())

            # next i want to find the locations of these features in feat0_map
            # feat0_vec_b is huge x C
            reverse_corr_b = torch.matmul(feat0_vec_b.detach(), correspI.detach())
            # this is huge x med
            reverse_heat_b = reverse_corr_b.permute(1, 0).reshape(-1, 1, Z_, Y_, X_)
            # this is med x 1 x Z x Y x X
            reverse_heat_b = F.relu(reverse_heat_b)
            # we need to pad this, because we are about to take the argmax and interpret it as xyz

            if False:
                reverse_window = vox_util.xyz2circles(xyz0 - crop_vec,
                                                      Z_, Y_, X_, radius=4.0, soft=False, already_mem=True)
                # reverse_window is 1 x med x Z x Y x X
                reverse_window = reverse_window.permute(1, 0, 2, 3, 4)
                # reverse_window is med x 1 x Z x Y x X
                reverse_heat_b  = reverse_heat_b * reverse_window

            reverse_heat_b = F.pad(reverse_heat_b, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
            # this is med x 1 x Z x Y x X
            
            # # for numerical stability, we sub the max, and mult by the resolution
            # reverse_heat_b_ = reverse_heat_b.reshape(-1, Z*Y*X)
            # reverse_heat_b_max = (torch.max(reverse_heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
            # reverse_heat_b = reverse_heat_b - reverse_heat_b_max
            reverse_heat_b = reverse_heat_b * float(len(reverse_heat_b[0].reshape(-1)))
            reverse_xyzI = utils_basic.argmax3D(reverse_heat_b, hard=False, stack=True)
            # this is med x 3

            # corr_b = torch.matmul(featI_vec_b.detach(), obj_vec_b.detach())
            # # this is huge x med
            # heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z_, Y_, X_)

            # print('reversed to xyzI', reverse_xyzI[:10].detach().cpu().numpy())
            # now, if the correspondences are good, i probably landed in xyz0
            dist = torch.norm(reverse_xyzI - xyz0, dim=1)
            # this is med
            return dist
        
        def zero_out_bad_peaks(xyzI, dist, heat_b, thresh=4.0):
            for n in list(range(len(dist))):
                if dist[n] > thresh:
                    # this means the xyzI was bad, since his neighbor disagrees with him

                    # xyzI_bad = xyzI_round[dist > 5.0]
                    # x, y, z = xyzI_bad[n,0], xyzI_bad[n,1], xyzI_bad[n,2]

                    o = xyzI[n].round().long()
                    x, y, z = o[0], o[1], o[2]

                    # print('setting this to zero:', n, o.detach().cpu().numpy(), 'value', heat_b[n,:,z,y,x].detach().cpu().numpy())

                    # heat_b is med x 1 x Z x Y x X
                    heat_b[n,:,z,y,x] = 0
                    heat_b[n,:,z+1,y,x] = 0
                    heat_b[n,:,z-1,y,x] = 0
                    heat_b[n,:,z,y+1,x] = 0
                    heat_b[n,:,z,y-1,x] = 0
                    heat_b[n,:,z,y,x+1] = 0
                    heat_b[n,:,z,y,x-1] = 0
                    # heat_b[n,:,z+0,y+0,x+0] = 0
                    # heat_b[n,:,z+0,y+0,x+1] = 0
                    # heat_b[n,:,z+0,y+1,x+0] = 0
                    # heat_b[n,:,z+1,y+0,x+0] = 0
                    # heat_b[n,:,z-1,y-1,x-1] = 0
            return heat_b

        # dist = torch.ones_like(xyzI)*100
        # thresh = 3.0
        # for cyc in list(range(8)):
        #     if torch.max(dist) > thresh:
        #         dist = get_cycle_dist(featI_map_b, xyzI,
        #                               feat0_vec_b, xyz0, 
        #                               Z_pad, Y_pad, X_pad,
        #                               crop_vec)
        #         # utils_basic.print_stats('cycle dist %d' % cyc, dist)
        #         heat_b = zero_out_bad_peaks(xyzI, dist, heat_b, thresh=thresh)
        #     # note i recently switched to taking a hard argmax here
        #     xyzI = utils_basic.argmax3D(heat_b, hard=True, stack=True)


        # we need to get to cam coordinates to cancel the scene centroid delta
        xyzI_cam = vox_util.Mem2Ref(xyzI.unsqueeze(1), Z, Y, X)
        xyzI_cam += delta
        xyzI = vox_util.Ref2Mem(xyzI_cam, Z, Y, X).squeeze(1)

        memI_T_mem0[b] = utils_track.rigid_transform_3D(xyz0, xyzI)

        # record #points, since ransac depends on this
        # point_counts[b, s] = len(xyz0)
    # done stepping through batch

    mem0_T_memI = utils_geom.safe_inverse(memI_T_mem0)
    cam0_T_camI = utils_basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)
    # mem0_T_memIs_e[:,s] = mem0_T_memI

    # eval
    camI_T_obj = utils_basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
    # this is B x 4 x 4

    new_lrt_camXAI = utils_geom.merge_lrt(obj_length, camI_T_obj)
    score = torch.ones_like(lrt_camXAI[:,0])
    return new_lrt_camXAI, score, mem0_T_memI
    # lrt_camXAIs[:,s] = utils_geom.merge_lrt(obj_length, camI_T_obj)
    # # ious[:,s] = utils_geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)

def jitter_lrt(lrt_camXAI, score, occ_memXAI, vox_util):
    # right here, let's try a few alts, based on CS
    el, rt = utils_geom.split_lrt(lrt_camXAI)
    r, t = utils_geom.split_rt(rt)
    rx, ry, rz = utils_geom.rotm2eul(r)

    B, C, Z, Y, X = list(occ_memXAI.shape)

    if score > 0.5:
        # maybe we will trade up
        score_ = torch.ones([1, 1]).float().cuda()
        K = 12
        alt_scores = []
        alt_lrts = []
        for k in list(range(K)):
            if k==0:
                coeff = 0.0
            else:
                coeff = 1.0
                
            rx_noise = torch.randn(rx.shape).float().cuda() * 0.05 * coeff
            ry_noise = torch.randn(rx.shape).float().cuda() * 0.05 * coeff
            rz_noise = torch.randn(rx.shape).float().cuda() * 0.05 * coeff
            el_noise = torch.randn(el.shape).float().cuda() * 0.1 * coeff
            t_noise = torch.randn(t.shape).float().cuda() * 0.1 * coeff
                
            # print('t_noise', t_noise)
            r_ = utils_geom.eul2rotm(rx+rx_noise,
                                     ry+ry_noise,
                                     rz+rz_noise)
            t_ = t + t_noise
            el_ = (el + el_noise).clamp(min=0.5)
            rt_ = utils_geom.merge_rt(r_, t_)
            lrt_ = utils_geom.merge_lrt(el_, rt_)

            mask_1 = vox_util.assemble_padded_obj_masklist(
                lrt_.unsqueeze(1), score_,
                Z, Y, X, coeff=0.8).squeeze(1)
            mask_3 = vox_util.assemble_padded_obj_masklist(
                lrt_.unsqueeze(1), score_,
                Z, Y, X, coeff=1.2, additive_coeff=1.0).squeeze(1)
            # mask_1 = cropper(mask_1)
            # mask_3 = cropper(mask_3)

            center_mask = mask_1.clone()
            surround_mask = (mask_3-mask_1).clamp(0,1)

            center_ = utils_basic.reduce_masked_mean(occ_memXAI, center_mask, dim=[2,3,4])
            surround_ = utils_basic.reduce_masked_mean(occ_memXAI, surround_mask, dim=[2,3,4])
            score_ = center_ - surround_
            score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)

            alt_scores.append(score_.squeeze())
            alt_lrts.append(lrt_)
        best_ind = torch.argmin(torch.stack(alt_scores, dim=0))
        print('best_ind:', best_ind.detach().cpu().numpy())
        best_lrt = alt_lrts[best_ind]
        lrt_camXAI = best_lrt
    return lrt_camXAI

def track_proposal(B, S, 
                   lrt_camXAI, I,
                   feat_memXAI_all,
                   occ_memXAI_all,
                   diff_memXAI_all,
                   feat_memXAI_median, 
                   occ_memXAI_median,
                   pix_T_cams,
                   rgb_camXs,
                   xyz_camXAs,
                   camXAs_T_camXs,
                   featnet3d,
                   occrelnet,
                   scene_vox_util,
                   cropper,
                   padder,
                   crop_zyx,
                   summ_writer=None,
):
    # lrt is B x 19
    # I is an int, indicating the camera

    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)

    feat_memXAI = feat_memXAI_all[I]
    B, C, Z_, Y_, X_ = list(feat_memXAI.shape)
    Z_pad, Y_pad, X_pad = crop_zyx
    Z_scene = Z_ + Z_pad*2
    Y_scene = Y_ + Y_pad*2
    X_scene = X_ + X_pad*2
    lrt_camXAI = jitter_lrt(lrt_camXAI, torch.ones_like(lrt_camXAI[:,0]), occ_memXAI_all[I], scene_vox_util)

    # original_centroid = scene_vox_util.scene_centroid.clone()

    original_centroid = utils_geom.get_clist_from_lrtlist(lrt_camXAI.unsqueeze(1)).squeeze(1)
    Z_zoom, Y_zoom, X_zoom = hyp.Z_zoom, hyp.Y_zoom, hyp.X_zoom
    orig_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=original_centroid, assert_cube=True)

    # # jitter and recenter
    # occ_memXAI = orig_vox_util.voxelize_xyz(xyz_camXAs[:,I], Z_zoom, Y_zoom, X_zoom)
    # lrt_camXAI = jitter_lrt(lrt_camXAI, torch.ones_like(lrt_camXAI[:,0]), occ_memXAI, orig_vox_util)
    # original_centroid = utils_geom.get_clist_from_lrtlist(lrt_camXAI.unsqueeze(1)).squeeze(1)
    # orig_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=original_centroid, assert_cube=True)

    rgb_memXII = orig_vox_util.unproject_rgb_to_mem(
        rgb_camXs[:,I], Z_zoom, Y_zoom, X_zoom, pix_T_cams[:,I])
    rgb_memXAI = orig_vox_util.apply_4x4_to_vox(camXAs_T_camXs[:,I], rgb_memXII)
    occ_memXAI = orig_vox_util.voxelize_xyz(xyz_camXAs[:,I], Z_zoom, Y_zoom, X_zoom)
    feat_memXAI_input = torch.cat([
        occ_memXAI, rgb_memXAI*occ_memXAI,
    ], dim=1)
    _, feat_memXAI, _ = featnet3d(feat_memXAI_input)

    # feat_memXAI = feat_memXAI_all[I]
    B, C, Z_, Y_, X_ = list(feat_memXAI.shape)
    Z_pad, Y_pad, X_pad = crop_zyx
    Z = Z_ + Z_pad*2
    Y = Y_ + Y_pad*2
    X = X_ + X_pad*2

    # occ_memXAI = self.occ_memXAI_all[I]
    # occ_memXAI = vox_util_obj.voxelize_xyz(xyz_camXAs[:,I], Z, Y, X)
    occ_memXAI = orig_vox_util.voxelize_xyz(xyz_camXAs[:,I], Z, Y, X)

    lrt_camXAI_ = lrt_camXAI.unsqueeze(1)
    score_ = torch.ones_like(lrt_camXAI_[:,:,0])

    obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
        lrt_camXAI_, score_, Z, Y, X).squeeze(1)
    occ_obj_mask_memXAI = obj_mask_memXAI * occ_memXAI
    occ_obj_mask_memXAI = cropper(occ_obj_mask_memXAI)

    # if torch.sum(occ_obj_mask_memXAI) < 16:
    #     # we cannot track with so few voxels#
    #     # let's use the full mask insteead
    #     occ_obj_mask_memXAI = cropper(obj_mask_memXAI)
    
    # occ_obj_mask_memXAI = cropper(obj_mask_memXAI)
        
    # if torch.sum(occ_obj_mask_memXAI) < 16:
    #     # still too small!
    #     # pad by 1m on each side
    #     obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
    #         lrt_camXAI_, score_, Z, Y, X, additive_coeff=1.0).squeeze(1)
    #     occ_obj_mask_memXAI = obj_mask_memXAI * occ_memXAI
    #     # occ_obj_mask_memXAI = cropper(obj_mask_memXAI)
    #     # if it's still too small, the tracker will just return identity

    if torch.sum(occ_obj_mask_memXAI) < 8:
        print('using the full mask instead of just occ')
        # still too small!
        # use the full mask
        obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
            lrt_camXAI_, score_, Z, Y, X).squeeze(1)
        # pad by 1m on each side
        occ_obj_mask_memXAI = cropper(obj_mask_memXAI)

    # self.summ_writer.summ_occ('track/obj_mask_memXAI', obj_mask_memXAI)

    # i need the feats of this object
    # let's use 0 to mean source/original, instead of AI
    feat0_vec = feat_memXAI.reshape(B, C, -1)
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C

    obj_mask0_vec = occ_obj_mask_memXAI.reshape(B, -1).round()
    # this is B x huge

    orig_xyz = utils_basic.gridcloud3D(B, Z, Y, X)
    # this is B x huge x 3
    orig_xyz = orig_xyz.reshape(B, Z, Y, X, 3)
    orig_xyz = orig_xyz.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    # print('orig_xyz', orig_xyz.shape)
    orig_xyz = cropper(orig_xyz)
    # print('crpped orig_xyz', orig_xyz.shape)
    orig_xyz = orig_xyz.reshape(B, 3, -1)
    orig_xyz = orig_xyz.permute(0, 2, 1)
    # this is B x huge x 3
    # print('rehaped orig_xyz', orig_xyz.shape)

    obj_length_, camXAI_T_obj_ = utils_geom.split_lrtlist(lrt_camXAI_)
    obj_length = obj_length_[:,0]
    cam0_T_obj = camXAI_T_obj_[:,0]

    mem_T_cam = orig_vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = orig_vox_util.get_ref_T_mem(B, Z, Y, X)

    lrt_camXAIs = lrt_camXAI_.repeat(1, S, 1)
    scores = torch.ones_like(lrt_camXAIs[:,:,0])
    # now we need to write the non-I timesteps
    mem0_T_memIs_e = torch.zeros((B, S, 4, 4), dtype=torch.float32).cuda()

    # print('step %d is given' % I)

    # lrt_exists = np.zeros(S)
    # lrt_exists[I] = 1

    smooth = 3
    
    clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
    havelist = torch.zeros_like(clist[:,:,0])
    havelist[:,I] = 1.0
    
    # first let's go forward
    for s in range(I+1, S):
        
        prev_lrt = lrt_camXAIs[:,s-1]
        # if s==I+1 or s==1:
        #     # s-2 does not exist, so:
        #     prevprev_lrt = lrt_camXAIs[:,s-1]
        # else:
        #     prevprev_lrt = lrt_camXAIs[:,s-2]
        # print('using', s-1, s-2, 'to form the prior')

        clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
        vel = torch.zeros((B, 3), dtype=torch.float32).cuda()
        for b in list(range(B)):
            clist_b = clist[b]
            havelist_b = havelist[b]
            if torch.sum(havelist_b) > 2:
                clist_have = clist_b[havelist_b > 0].reshape(-1, 3)
                # print('clist_have', clist_have.shape)
                clist_a = clist_have[1:]
                clist_b = clist_have[:-1]
                # print('clist_a', clist_a.shape)
                # print('clist_b', clist_b.shape)
                vel[b] = torch.mean(clist_a - clist_b, dim=0)
        # print('vel seems to be', vel.detach().cpu().numpy())

        # # if True:
        # if s==I+1:
        #     vel = 0.0
        # else:
        #     # if 
        #     clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)

        #     # this is B x S x 3
            
        #     clist_a = clist[:,I+1:s]
        #     clist_b = clist[:,I:s-1]
        #     # print('clist_a', clist_a.detach().cpu().numpy())
        #     # print('clist_b', clist_b.detach().cpu().numpy())
        #     vel = torch.mean(clist_a-clist_b, dim=1)
        #     # print('mean vel', vel.detach().cpu().numpy())
            
        # # clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)

        # s_a = max(s-smooth, I+1)
        # s_b = min(s+smooth, S)
        # xyz0 = torch.mean(clist[:,s_a:s+1], dim=1)
        # xyz1 = torch.mean(clist[:,s:s_b+1], dim=1)
        # vel = xyz1-xyz0

        el, rt_prev = utils_geom.split_lrt(prev_lrt)
        r_prev, t_prev = utils_geom.split_rt(rt_prev)
        t_curr = t_prev + vel
        rt_curr = utils_geom.merge_rt(r_prev, t_curr)
        lrt_curr = utils_geom.merge_lrt(el, rt_curr)
        
        # print('t_prev', t_prev.detach().cpu().numpy())
        # print('t_curr', t_curr.detach().cpu().numpy())
        # input()
            
            # # we can immediately compute a prior, based on const velocity
            # el, rt_prevprev = utils_geom.split_lrt(prevprev_lrt)
            # _, rt_prev = utils_geom.split_lrt(prev_lrt)
            # r_prevprev, t_prevprev = utils_geom.split_rt(rt_prevprev)
            # r_prev, t_prev = utils_geom.split_rt(rt_prev)
            # vel = t_prev - t_prevprev
            # t_curr = t_prev + vel
            # rt_curr = utils_geom.merge_rt(r_prev, t_curr)
            # lrt_curr = utils_geom.merge_lrt(el, rt_curr)
        #     # later, do that t_curr stuff to find out the new position
        #     new_centroid = utils_geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1)).squeeze(1)
        # else:
        new_centroid = utils_geom.get_clist_from_lrtlist(lrt_camXAIs[:,s-1].unsqueeze(1)).squeeze(1)

        clist_camXAI = utils_geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1))
        clist_memXAI = scene_vox_util.Ref2Mem(clist_camXAI, Z_scene, Y_scene, X_scene)
        crop_vec = torch.from_numpy(np.reshape(np.array([X_pad, Y_pad, Z_pad]), (1, 1, 3))).float().cuda()
        inb = scene_vox_util.get_inbounds(clist_memXAI-crop_vec,
                                          Z_scene-Z_pad*2, Y_scene-Y_pad*2, X_scene-X_pad*2, already_mem=True, padding=1.0)
        if torch.sum(inb) == 0:
            # print('centroid predicted OOB; returning the prior')
            # return early; we won't find the object bc it's now oob
            mem0_T_memI = utils_geom.eye_4x4(B)
            # score = torch.zeros_like(lrt_camXAI[:,0])
            score = torch.ones_like(lrt_camXAI[:,0])*0.5
            # return lrt_camXAI, score, mem0_T_memI
            lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = lrt_curr, score, mem0_T_memI
        else:
            delta = new_centroid - original_centroid
            new_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=new_centroid, assert_cube=True)
            # the centroid is in XA coords, so i should put everything into XA coords

            rgb_memXII = new_vox_util.unproject_rgb_to_mem(
                rgb_camXs[:,s], Z_zoom, Y_zoom, X_zoom, pix_T_cams[:,s])
            rgb_memXAI = new_vox_util.apply_4x4_to_vox(camXAs_T_camXs[:,s], rgb_memXII)
            occ_memXAI = new_vox_util.voxelize_xyz(xyz_camXAs[:,s], Z_zoom, Y_zoom, X_zoom)
            feat_memXAI_input = torch.cat([
                occ_memXAI, rgb_memXAI*occ_memXAI,
            ], dim=1)
            _, feat_memXAI, _ = featnet3d(feat_memXAI_input)

            # summ_writer.summ_feat('track/feat_%d_input' % s, feat_memXAI_input, pca=True)
            # summ_writer.summ_feat('track/feat_%d' % s, feat_memXAI, pca=True)

            _, occrel_memXAI = occrelnet(feat_memXAI)
            obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
                lrt_curr.unsqueeze(1), score_, Z, Y, X).squeeze(1)
            # pad by 1m on each side
            obj_mask_memXAI = cropper(obj_mask_memXAI)
            if torch.sum(obj_mask_memXAI * occrel_memXAI) < 4:
                print('occrel is iffy at the pred location; returning the prior')
                mem0_T_memI = utils_geom.eye_4x4(B)
                # score = torch.zeros_like(lrt_camXAI[:,0])
                score = torch.ones_like(lrt_camXAI[:,0])*0.5
                # return lrt_camXAI, score, mem0_T_memI
                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = lrt_curr, score, mem0_T_memI
            else:
                # print('working on step %d' % s)
                # featI_vec = feat_memXAI_all[s].view(B, C, -1)
                featI_vec = feat_memXAI.view(B, C, -1)
                # this is B x C x huge
                featI_vec = featI_vec.permute(0, 2, 1)
                # this is B x huge x C

                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = track_one_step_via_inner_product(
                    B, hyp.feat3D_dim,
                    Z_, Y_, X_, 
                    prev_lrt,
                    # prevprev_lrt,
                    featI_vec,
                    feat0_vec,
                    obj_mask0_vec,
                    obj_length,
                    cam0_T_obj,
                    orig_xyz,
                    diff_memXAI_all[s],
                    new_vox_util,
                    cropper,
                    crop_zyx,
                    delta,
                    summ_writer=summ_writer,
                    use_window=False)
        # print('wrote ans for', s)
        havelist[:,s] = 1.0
        

        # lrt_camXAIs[:,s] = jitter_lrt(lrt_camXAIs[:,s], scores[:,s], occ_memXAI_all[s])

    # next let's go backward
    for s in range(I-1, -1, -1):
        # # print('working on step %d' % s)
        # featI_vec = feat_memXAI_all[s].view(B, hyp.feat3D_dim, -1)
        # # this is B x C x huge
        # featI_vec = featI_vec.permute(0, 2, 1)
        # # this is B x huge x C

        # clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)

        
        prev_lrt = lrt_camXAIs[:,s+1]
        # if s==S-2:
        #     # print('using', s+1, s+1, 'to form teh prior')
        #     # s+2 does not exist, so:
        #     prevprev_lrt = lrt_camXAIs[:,s+1]
        # else:
        #     # print('using', s+1, s+2, 'to form teh prior')
        #     prevprev_lrt = lrt_camXAIs[:,s+2]
        

        clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
        vel = torch.zeros((B, 3), dtype=torch.float32).cuda()
        for b in list(range(B)):
            clist_b = clist[b]
            havelist_b = havelist[b]
            # note this assumes that the havelist does not have holes
            if torch.sum(havelist_b) > 2:
                clist_b = clist[b]
                havelist_b = havelist[b]
                clist_have = clist_b[havelist_b > 0].reshape(-1, 3)
                clist_a = clist_have[1:]
                clist_b = clist_have[:-1]
                vel[b] = torch.mean(clist_a - clist_b, dim=0)
        vel = vel * -1
        # print('vel seems to be', vel.detach().cpu().numpy())

        # if s==I-1:
        #     vel = 0.0
        # else:
        #     clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
        #     # this is B x S x 3
            
        #     clist_a = clist[:,I-1:s]
        #     clist_b = clist[:,I:s-1]
        #     # print('clist_a', clist_a.detach().cpu().numpy())
        #     # print('clist_b', clist_b.detach().cpu().numpy())
        #     vel = torch.mean(clist_a-clist_b, dim=1)
        #     # print('mean vel', vel.detach().cpu().numpy())

        el, rt_prev = utils_geom.split_lrt(prev_lrt)
        r_prev, t_prev = utils_geom.split_rt(rt_prev)
        t_curr = t_prev + vel
        rt_curr = utils_geom.merge_rt(r_prev, t_curr)
        lrt_curr = utils_geom.merge_lrt(el, rt_curr)
        
        # print('t_prev', t_prev.detach().cpu().numpy())
        # print('t_curr', t_curr.detach().cpu().numpy())

        
        # # print('prev_c', prev_c.detach().cpu().numpy())
        # # print('prevprev_c', prevprev_c.detach().cpu().numpy())

        # if True:

        #     clist = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
            

        #     s_a = max(s-smooth, I-1)
        #     s_b = min(s+smooth, I-1)
            
        #     s_a = min(s-smooth, 0)
        #     s_b = max(s+smooth, I-1)
        #     print('s_a, s_b', s_a, s_b)
        #     # reverse this order! 
        #     xyz1 = torch.mean(clist[:,s_a:s+1], dim=1)
        #     xyz0 = torch.mean(clist[:,s:s_b+1], dim=1)
        #     vel = xyz1-xyz0
        #     print('xyz0', xyz0.detach().cpu().numpy())
        #     print('xyz1', xyz1.detach().cpu().numpy())
        #     print('vel', vel.detach().cpu().numpy())

        #     el, rt_prev = utils_geom.split_lrt(prev_lrt)
        #     r_prev, t_prev = utils_geom.split_rt(rt_prev)
        #     t_curr = t_prev + vel
        #     rt_curr = utils_geom.merge_rt(r_prev, t_curr)
        #     lrt_curr = utils_geom.merge_lrt(el, rt_curr)

            
        #     # # we can immediately compute a prior, based on const velocity
        #     # el, rt_prevprev = utils_geom.split_lrt(prevprev_lrt)
        #     # _, rt_prev = utils_geom.split_lrt(prev_lrt)
        #     # r_prevprev, t_prevprev = utils_geom.split_rt(rt_prevprev)
        #     # r_prev, t_prev = utils_geom.split_rt(rt_prev)
        #     # vel = t_prev - t_prevprev
        #     # t_curr = t_prev + vel
        #     # rt_curr = utils_geom.merge_rt(r_prev, t_curr)
        #     # lrt_curr = utils_geom.merge_lrt(el, rt_curr)
        #     # later, do that t_curr stuff to find out the new position
        # #     new_centroid = utils_geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1)).squeeze(1)
        # # else:
        
        new_centroid = utils_geom.get_clist_from_lrtlist(lrt_camXAIs[:,s+1].unsqueeze(1)).squeeze(1)

        clist_camXAI = utils_geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1))
        clist_memXAI = scene_vox_util.Ref2Mem(clist_camXAI, Z_scene, Y_scene, X_scene)
        crop_vec = torch.from_numpy(np.reshape(np.array([X_pad, Y_pad, Z_pad]), (1, 1, 3))).float().cuda()
        inb = scene_vox_util.get_inbounds(clist_memXAI-crop_vec,
                                          Z_scene-Z_pad*2, Y_scene-Y_pad*2, X_scene-X_pad*2, already_mem=True, padding=1.0)
        if torch.sum(inb) == 0:
            # print('centroid predicted OOB; returning the prior')
            # return early; we won't find the object bc it's now oob
            mem0_T_memI = utils_geom.eye_4x4(B)
            # score = torch.zeros_like(lrt_camXAI[:,0])
            score = torch.ones_like(lrt_camXAI[:,0])*0.5
            # return lrt_camXAI, score, mem0_T_memI
            lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = lrt_curr, score, mem0_T_memI
        else:
            delta = new_centroid - original_centroid
            new_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=new_centroid, assert_cube=True)

            rgb_memXII = new_vox_util.unproject_rgb_to_mem(
                rgb_camXs[:,s], Z_zoom, Y_zoom, X_zoom, pix_T_cams[:,s])
            rgb_memXAI = new_vox_util.apply_4x4_to_vox(camXAs_T_camXs[:,s], rgb_memXII)
            occ_memXAI = new_vox_util.voxelize_xyz(xyz_camXAs[:,s], Z_zoom, Y_zoom, X_zoom)
            feat_memXAI_input = torch.cat([
                occ_memXAI, rgb_memXAI*occ_memXAI,
            ], dim=1)
            _, feat_memXAI, _ = featnet3d(feat_memXAI_input)

            _, occrel_memXAI = occrelnet(feat_memXAI)
            obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
                lrt_curr.unsqueeze(1), score_, Z, Y, X).squeeze(1)
            # pad by 1m on each side
            obj_mask_memXAI = cropper(obj_mask_memXAI)
            
            # # if torch.sum(obj_mask_memXAI * occrel_memXAI) < 8:
            # tot_ = torch.sum(obj_mask_memXAI)
            
            if torch.sum(obj_mask_memXAI * occrel_memXAI) < 4:
                print('occrel is iffy at the pred location; returning the prior')
                mem0_T_memI = utils_geom.eye_4x4(B)
                # score = torch.zeros_like(lrt_camXAI[:,0])
                score = torch.ones_like(lrt_camXAI[:,0])*0.5
                # return lrt_camXAI, score, mem0_T_memI
                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = lrt_curr, score, mem0_T_memI
            else:
                # print('working on step %d' % s)
                # featI_vec = feat_memXAI_all[s].view(B, C, -1)
                featI_vec = feat_memXAI.view(B, C, -1)
                # this is B x C x huge
                featI_vec = featI_vec.permute(0, 2, 1)
                # this is B x huge x C

                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = track_one_step_via_inner_product(
                    B, hyp.feat3D_dim,
                    Z_, Y_, X_, 
                    prev_lrt,
                    # prevprev_lrt,
                    featI_vec,
                    feat0_vec,
                    obj_mask0_vec,
                    obj_length,
                    cam0_T_obj,
                    orig_xyz,
                    diff_memXAI_all[s],
                    new_vox_util,
                    cropper,
                    crop_zyx,
                    delta,
                    summ_writer=summ_writer,
                    use_window=False)
        havelist[:,s] = 1.0

            # # print('working on step %d' % s)
            # # featI_vec = feat_memXAI_all[s].view(B, C, -1)
            # featI_vec = feat_memXAI.view(B, C, -1)
            # # this is B x C x huge
            # featI_vec = featI_vec.permute(0, 2, 1)
            # # this is B x huge x C


            # lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = track_one_step_via_inner_product(
            #     B, hyp.feat3D_dim,
            #     Z_, Y_, X_, 
            #     prev_lrt,
            #     prevprev_lrt,
            #     featI_vec,
            #     feat0_vec,
            #     obj_mask0_vec,
            #     obj_length,
            #     cam0_T_obj,
            #     orig_xyz,
            #     diff_memXAI_all[s],
            #     new_vox_util,
            #     cropper,
            #     crop_zyx,
            #     delta,
            #     summ_writer=summ_writer,
            #     use_window=False)
            # # print('wrote ans for', s)

        # lrt_camXAIs[:,s] = jitter_lrt(lrt_camXAIs[:,s], scores[:,s], occ_memXAI_all[s])
        
    obj_clist_camXAI = utils_geom.get_clist_from_lrtlist(lrt_camXAIs)
    if summ_writer is not None:
        summ_writer.summ_traj_on_occ('track/traj',
                                     obj_clist_camXAI,
                                     # padder(cropper(occ_memXAI)), # crop and pad, so i can see the empty area
                                     padder(occ_memXAI_all[I]), 
                                     scene_vox_util, 
                                     already_mem=False,
                                     sigma=2)
        
    # find total travel distance, and discard stationary objects
    c0 = obj_clist_camXAI[:,0]
    cE = obj_clist_camXAI[:,-1]
    scores1_ = (scores == 1.0).float()
    diff_per_step = scores1_[:,:-1]*torch.norm(obj_clist_camXAI[:,1:] - obj_clist_camXAI[:,:-1], 2)
    # if torch.norm(c0-cE, dim=1) < 1.0:
    if (torch.sum(diff_per_step) < 2.0) or (torch.norm(c0-cE, dim=1) < 2.0):
        # not a real object
        scores = scores * 0.0
        # but note we will still supress the masks
        # note there is a small chance this will discard an object that returns to its startpoint

    top3d_vis = []
    top3d_occ_vis = []
    scene_memY0_all = []
    occ_memY0_all = []
    mask_memY0_all = []

    obj_mask_memXAI = scene_vox_util.assemble_padded_obj_masklist(
        lrt_camXAI_, score_, Z_scene, Y_scene, X_scene).squeeze(1)
    
    for s in list(range(S)):
        memX0_T_memY0 = mem0_T_memIs_e[:,s]
        memY0_T_memX0 = utils_geom.safe_inverse(memX0_T_memY0)

        # compute top-down with the tracklet
        scene_memY0, mask_memY0 = utils_misc.assemble_hypothesis(
            memY0_T_memX0,
            lrt_camXAIs[:,s],
            feat_memXAI_median,
            feat_memXAI_all[s],
            cropper(obj_mask_memXAI),
            scene_vox_util,
            crop_zyx)
        mask_memY0_all.append(mask_memY0)
        occ_memY0, mask_memY0 = utils_misc.assemble_hypothesis(
            memY0_T_memX0,
            lrt_camXAIs[:,s],
            occ_memXAI_median,
            occ_memXAI_all[s],
            cropper(obj_mask_memXAI),
            scene_vox_util,
            crop_zyx, norm=False)

        scene_memY0_all.append(scene_memY0)
        occ_memY0_all.append(occ_memY0)
        
        top3d_vis.append(summ_writer.summ_feat('', scene_memY0, pca=True, only_return=True))
        # _, occ_memY0_pred, _, _ = self.occnet(scene_memY0)
        # occ_memY0 = F.sigmoid(occ_memY0_pred)
        top3d_occ_vis.append(summ_writer.summ_occ('', occ_memY0, only_return=True))
    if summ_writer is not None:
        summ_writer.summ_rgbs('track/top3d_vis', top3d_vis)
        summ_writer.summ_rgbs('track/top3d_occ_vis', top3d_occ_vis)
        # summ_writer.summ_oneds('track/top3d_mask_vis', mask_memY0_all, bev=True, norm=False)
    return lrt_camXAIs, scores, mask_memY0_all, scene_memY0_all, occ_memY0_all

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
        

# the idea of this mode is to overfit to a few examples and prove to myself that i can generate explain outputs

class NUSCENES_EXPLAIN(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = NuscenesExplainModel()
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
            
        eval_track = False
        if hyp.do_test:
            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            num_ious = len(iou_thresholds)
        
            if not hyp.do_flow and eval_track:
                all_track_maps_3d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
                all_track_maps_2d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
                all_track_maps_pers = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
            all_proposal_maps_3d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
            all_proposal_maps_2d = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
            all_proposal_maps_pers = np.zeros([hyp.max_iters, hyp.S_test, num_ious], np.float32)
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

                        if hyp.do_test and (not returned_early):
                            proposal_maps_3d = results['all_proposal_maps_3d']
                            all_proposal_maps_3d[test_count] = proposal_maps_3d
                            proposal_maps_2d = results['all_proposal_maps_2d']
                            all_proposal_maps_2d[test_count] = proposal_maps_2d
                            proposal_maps_pers = results['all_proposal_maps_pers']
                            all_proposal_maps_pers[test_count] = proposal_maps_pers
                            
                            if not hyp.do_flow and eval_track:
                                track_maps_3d = results['all_track_maps_3d']
                                all_track_maps_3d[test_count] = track_maps_3d
                                track_maps_2d = results['all_track_maps_2d']
                                all_track_maps_2d[test_count] = track_maps_2d
                                track_maps_pers = results['all_track_maps_pers']
                                all_track_maps_pers[test_count] = track_maps_pers

                            test_count += 1

                            print('-'*10)
                            
                            mean_proposal_maps_3d = np.mean(all_proposal_maps_3d[:test_count], axis=0)
                            print('mean_proposal_maps_3d', np.mean(mean_proposal_maps_3d, axis=0))
                            
                            mean_proposal_maps_2d = np.mean(all_proposal_maps_2d[:test_count], axis=0)
                            print('mean_proposal_maps_2d', np.mean(mean_proposal_maps_2d, axis=0))
                            
                            mean_proposal_maps_pers = np.mean(all_proposal_maps_pers[:test_count], axis=0)
                            print('mean_proposal_maps_pers', np.mean(mean_proposal_maps_pers, axis=0))
                            
                            if not hyp.do_flow and eval_track:
                                mean_track_maps_3d = np.mean(all_track_maps_3d[:test_count], axis=0)
                                print('mean_track_maps_3d', np.mean(mean_track_maps_3d, axis=0))

                                mean_track_maps_2d = np.mean(all_track_maps_2d[:test_count], axis=0)
                                print('mean_track_maps_2d', np.mean(mean_track_maps_2d, axis=0))
                                
                                mean_track_maps_pers = np.mean(all_track_maps_pers[:test_count], axis=0)
                                print('mean_track_maps_pers', np.mean(mean_track_maps_pers, axis=0))
                            

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
            
class NuscenesExplainModel(nn.Module):
    def __init__(self):
        super(NuscenesExplainModel, self).__init__()

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
        
        # if global_step < 14:
        #     return False

        self.B = feed["set_batch_size"]
        self.set_name = feed['set_name']
        
        self.boxlist = feed["boxlist"]
        self.vehicle_T_cams = feed["vehicle_T_cams"]
        self.pix_T_cams = feed["cams_intrinsics"]
        self.vehicle_T_lidar = feed["vehicle_T_lidar"]
        self.cams_origin_T_vehicle = feed["cams_origin_T_vehicle"]
        self.lidar_origin_T_vehicle = feed["lidar_origin_T_vehicle"]
        self.rgb_cams = feed["rgb_cams"]
        self.xyzi_lidar = feed["xyzi_lidar"]
        self.us_T_lidar = feed["us_T_lidar"]

        print('rgb_cams', self.rgb_cams.shape)
        print('xyzi_lidar', self.xyzi_lidar.shape)
        print('boxlist', self.boxlist.shape)
        print('pix_T_cams', self.pix_T_cams.shape)
        print('cams_origin_T_vehicle', self.cams_origin_T_vehicle.shape)
        print('lidar_origin_T_vehicle', self.lidar_origin_T_vehicle.shape)
        print('us_T_lidar', self.us_T_lidar.shape)
        print('vehicle_T_lidar', self.vehicle_T_lidar.shape)

        _, self.S, _, self.H, self.W, _ = list(self.rgb_cams.shape)

        # let's assume boxlist is in camera coords

        rgb_cam0s = self.rgb_cams[:,:,0]
        pix_T_cam0 = self.pix_T_cams[:,0]
        # print('pix_T_cam0', pix_T_cam0.detach().cpu().numpy())
        # print('boxlist', self.boxlist[:,0].detach().cpu().numpy())

        us_T_origin = utils_basic.matmul2(self.us_T_lidar, self.lidar_origin_T_vehicle[:,0])

        print('us_T_origin', us_T_origin.shape)
        
        # this is B x S x H x W x 3
        rgb_cam0s = utils_improc.preprocess_color(rgb_cam0s.permute(0, 1, 4, 2, 3))
        self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_cam0s, dim=1))

        lrtlist_origin = utils_geom.convert_boxlist_to_lrtlist(self.boxlist)
        lrtlist_lidar = utils_geom.apply_4x4_to_lrtlist(us_T_origin, lrtlist_origin)
        lrtlist_vehicle = utils_geom.apply_4x4_to_lrtlist(self.vehicle_T_lidar[:,0], lrtlist_lidar)
        lrtlist_cam0 = utils_geom.apply_4x4_to_lrtlist(self.cams_origin_T_vehicle[:,0,0], lrtlist_vehicle)
        # lrtlist_cam0 = utils_geom.apply_4x4s_to_lrts(self.cams_origin_T_vehicle[:,:,0], lrtlist_origin)
        
        visX_e = []
        for s in list(range(self.S)):
            lrt_ = lrtlist_cam0[:,s:s+1]
            score_ = torch.ones_like(lrt_[:,0])
            tid_ = torch.ones_like(lrt_[:,0]).long()
            visX_e.append(self.summ_writer.summ_lrtlist(
                '',
                rgb_cam0s[:,s],
                lrt_,
                score_,
                tid_,
                pix_T_cam0, only_return=True))
        self.summ_writer.summ_rgbs('2D_inputs/box_camXs_g', visX_e)

        
        return False
        
        
        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.N = hyp.N
        
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
        set_data_format = feed['set_data_format']
        self.set_data_name = feed['set_data_name']
        self.S = feed["set_seqlen"]
        

        # self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        # self.camR0s_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        # self.camRs_T_camR0s = __u(utils_geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))
        # self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        # self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.cams_T_velos = feed["cams_T_velos"]
        self.camXs_T_camX0s = __u(utils_geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.xyz_veloXs = feed["xyz_veloXs"]
        self.xyz_camXs = __u(utils_geom.apply_4x4(__p(self.cams_T_velos), __p(self.xyz_veloXs)))
        self.xyz_camX0s = __u(utils_geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))


        # if self.set_name=='test':
        
        self.anchor = int(self.S/2)

        self.camXAs_T_camXs = utils_geom.get_camM_T_camXs(self.origin_T_camXs, ind=self.anchor)
        self.camXs_T_camXAs = __u(utils_geom.safe_inverse(__p(self.camXAs_T_camXs)))
        self.xyz_camXAs = __u(utils_geom.apply_4x4(__p(self.camXAs_T_camXs), __p(self.xyz_camXs)))
        
        # self.camRAs_T_camRs = utils_geom.get_camM_T_camXs(self.origin_T_camRs, ind=self.anchor)
        # self.camRs_T_camRAs = __u(utils_geom.safe_inverse(__p(self.camRAs_T_camRs)))
        # self.camRAs_T_camXs = __u(torch.matmul(__p(self.camRAs_T_camRs), __p(self.camRs_T_camXs)))
        # self.xyz_camRAs = __u(utils_geom.apply_4x4(__p(self.camRAs_T_camRs), __p(self.xyz_camRs)))
        # self.camRAs_T_camRs = utils_geom.get_camM_T_camRs(self.origin_T_camRs, ind=self.anchor)
        # _, self.scene_centroid = utils_geom.split_rt(self.origin_T_camRs[:,self.anchor])

        # origin_T_camX0 = utils_geom.split_rt(self.origin_T_camXs[:,0])
        # origin_T_camXE = self.origin_T_camXs[:,-1]
        _, camX0 = utils_geom.split_rt(self.origin_T_camXs[:,0])
        _, camXE = utils_geom.split_rt(self.origin_T_camXs[:,-1])
        # print('camX0', camX0, camX0.shape)
        # print('camXE', camXE, camXE.shape)
        dist = torch.norm(camX0 - camXE, dim=1)
        # print('dist', dist.detach().cpu().numpy())

        # if dist > 10:
        #     print('the camera moved too much')
        #     return False
        # if dist < 10:
        #     print('the camera is not moving')
        #     return False

        all_ok = False
        num_tries = 0
        while not all_ok:
            # scene_centroid_x = np.random.uniform(-8.0, 8.0)
            # scene_centroid_y = np.random.uniform(-1.5, 3.0)
            # scene_centroid_z = np.random.uniform(10.0, 26.0)

            # lrtlist_camXA = self.full_lrtlist_camXs[:,self.anchor]
            # clist_camXA = utils_geom.get_clist_from_lrtlist(lrtlist_camXA)
            # scene_centroid_y = clist_camXA[:,0,1]
            
            # scene_centroid_x = np.random.uniform(-4.0, 4.0)
            # scene_centroid_y = np.random.uniform(1, 2)
            # scene_centroid_z = np.random.uniform(16.0, 20.0)
            # scene_centroid = np.array([scene_centroid_x,
            #                            scene_centroid_y,
            #                            scene_centroid_z]).reshape([1, 3])

            scene_centroid_x = 0.0
            scene_centroid_y = 1.5 # 1.0 is a bit too high up
            scene_centroid_z = 18.0
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])

            
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            num_tries += 1
            all_ok = True
            self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

            occ_memXA0 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,0], self.Z2, self.Y2, self.X2)
            occ_memXAE = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,-1], self.Z2, self.Y2, self.X2)
            occ_memXA0 = self.crop_feat(occ_memXA0)
            occ_memXAE = self.crop_feat(occ_memXAE)

            # occ_memXAA = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,self.anchor], self.Z2, self.Y2, self.X2)
            # occ_memXAA = self.crop_feat(occ_memXAA)

            occ_memXA0_ = occ_memXA0.reshape(self.B, -1)
            occ_memXAE_ = occ_memXAE.reshape(self.B, -1)
            num_inb0 = torch.sum(occ_memXA0_, dim=1)
            num_inbE = torch.sum(occ_memXAE_, dim=1)
            # this is B
            min_pts = 1000
            # print('num_inb0', num_inb0.detach().cpu().numpy())
            # print('num_inbE', num_inbE.detach().cpu().numpy())
            
            # if torch.mean(num_inb0) < min_pts or torch.mean(num_inbE) < min_pts:
            #     print('num_inb0', num_inb0.detach().cpu().numpy())
            #     all_ok = False
            #     # give up immediately
            #     return False
                
            if num_tries > 20:
                print('cannot find a good centroid; returning early')
                return False
        # print('scene_centroid', scene_centroid)
        self.summ_writer.summ_scalar('zoom_sampling/num_tries', float(num_tries))
        self.summ_writer.summ_scalar('zoom_sampling/num_inb', torch.mean(num_inb0+num_inbE).cpu().item())
        # self.summ_writer.summ_occs('3D_feats/occ_memXAs', [occ_memXA0, occ_memXAE])

        
        # scene_centroid_x = 0.0
        # scene_centroid_y = 1.5 # 1.0 is a bit too high up
        # scene_centroid_z = 18.0
        # scene_centroid = np.array([scene_centroid_x,
        #                            scene_centroid_y,
        #                            scene_centroid_z]).reshape([1, 3])
        # self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        # self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

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

    def run_explain(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils_basic.pack_seqdim(x, self.B)
        __u = lambda x: utils_basic.unpack_seqdim(x, self.B)

        self.boxlist_camXs = feed["boxlists"]
        self.scorelist_s = feed["scorelists"]
        self.tidlist_s = feed["tidlists"]

        boxlist_camXs_ = __p(self.boxlist_camXs)
        scorelist_s_ = __p(self.scorelist_s)
        tidlist_s_ = __p(self.tidlist_s)
        boxlist_camXs_, tidlist_s_, scorelist_s_ = utils_misc.shuffle_valid_and_sink_invalid_boxes(
            boxlist_camXs_, tidlist_s_, scorelist_s_)
        self.boxlist_camXs = __u(boxlist_camXs_)
        self.scorelist_s = __u(scorelist_s_)
        self.tidlist_s = __u(tidlist_s_)

        self.lrtlist_camXs = __u(utils_geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camXs)))
        self.lrtlist_camX0s = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        print('lrtlist_camXs', self.lrtlist_camXs.shape)
        
        # self.lrtlist_camX0s = utils_geom.apply_4x4_to_lrtlist(self.camX0s_T_camXs, self.lrtlist_camXs)

        # self.lrt_camRAs = utils_geom.apply_4x4s_to_lrts(self.camRAs_T_camRs, self.lrt_camRs)
        
        # full_boxlist_camRs = feed["full_boxlist_camR"]
        # full_scorelist_s = feed["full_scorelist"]
        # full_tidlist_s = feed["full_tidlist"]

        # full_boxlist_camRs is B x S x N x 9
        # N = self.scorelist_s.shape[2]
        

        self.full_lrtlist_camXs = self.lrtlist_camXs
        self.full_lrtlist_camX0s = self.lrtlist_camX0s
        self.full_scorelist_s = self.scorelist_s
        self.full_tidlist_s = self.tidlist_s
        # self.full_lrtlist_camRs = __u(full_lrtlist_camRs_)
        # self.full_lrtlist_camR0s = __u(full_lrtlist_camR0s_)
        # self.full_lrtlist_camXs = __u(full_lrtlist_camXs_)
        # self.full_lrtlist_camX0s = __u(full_lrtlist_camX0s_)

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
                                iou_3d, _ = utils_geom.get_iou_from_corresponded_lrtlists(
                                    target[b:b+1].unsqueeze(1), self.full_lrtlist_camX0s[b:b+1,s1,i:i+1])
                                ious[i] = np.squeeze(iou_3d[0,0])
                        if float(np.max(ious)) < 0.99:
                            # the object must have moved
                            new_scorelist_s[b,s0,n] = 1.0
        self.full_scorelist_s = new_scorelist_s * self.full_scorelist_s

        print('objects detectable across the entire seq:', torch.sum(self.full_scorelist_s).detach().cpu().numpy())
        if torch.sum(self.full_scorelist_s) == 0:
            # return early, since no objects are inbound AND moving
            return total_loss, results, True

                
        # self.summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        vis = []
        # for s in list(range(0, self.S, 2)):
        for s in list(range(0, self.S)):
            vis.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,s],
                self.full_lrtlist_camXs[:,s],
                self.full_scorelist_s[:,s],
                self.full_tidlist_s[:,s],
                self.pix_T_cams[:,s],
                only_return=True))
        self.summ_writer.summ_rgbs('2D_inputs/lrtlist_camXs', vis)

        # return total_loss, results, True
        # return total_loss, results, True
        

        # # self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        # #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # # self.rgb_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.rgb_memXs)
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        
        # # self.summ_writer.summ_rgb('2D_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.summ_writer.summ_occs('3D_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3D_inputs/rgb_memX0s', torch.unbind(self.rgb_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))

        have_feats = False
        have_medians = False
        # have_boxes = False

        use_feat_cache = True
        if hyp.do_use_cache and use_feat_cache:
            data_ind = feed['data_ind']
            feat_cache_fn = 'cache/%s_%06d_s%d_feat.npz' % (self.set_data_name, data_ind, self.S)
            # check if the thing exists
            if os.path.isfile(feat_cache_fn):
                print('found feat cache at %s; we will use this' % feat_cache_fn)
                cache = np.load(feat_cache_fn, allow_pickle=True)['save_dict'].item()
                # cache = cache['save_dict']
                have_feats = True

                self.feat_memXAI_all = torch.from_numpy(cache['feat_memXAI_all']).cuda().unbind(1)
                self.occ_memXAI_all = torch.from_numpy(cache['occ_memXAI_all']).cuda().unbind(1)
                occrel_memXAI_all = torch.from_numpy(cache['occrel_memXAI_all']).cuda().unbind(1)
                vis_memXAI_all = torch.from_numpy(cache['vis_memXAI_all']).cuda().unbind(1)

                # feat_memXAI_input_vis = torch.from_numpy(cache['feat_memXAI_input_vis']).unbind(1)
                # feat_memXAI_vis = torch.from_numpy(cache['feat_memXAI_vis']).unbind(1)
                self.scene_centroid = torch.from_numpy(cache['scene_centroid']).cuda()
                self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)

                # make sure the shapes match what we expect
                _, _, Z2, Y2, X2 = list(self.feat_memXAI_all[0].shape)
                Z_crop = int((self.Z2 - Z2)/2)
                Y_crop = int((self.Y2 - Y2)/2)
                X_crop = int((self.X2 - X2)/2)
                crop = (Z_crop, Y_crop, X_crop)
                if not (crop==self.crop_guess):
                    print('crop', crop)
                    assert(crop==self.crop_guess) # otw we need to rewrite self.crop above
            else:
                print('could not find feat cache at %s; we will write this' % feat_cache_fn)
                
        if not have_feats and use_feat_cache:
            with torch.no_grad():
                vis_memXAI_all = []
                self.feat_memXAI_all = []
                self.occ_memXAI_all = []
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
                    # self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                    self.rgb_memXAI = self.vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,I], self.rgb_memXII)
                    self.occ_memXAI = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,I], self.Z, self.Y, self.X)

                    feat_memXAI_input = torch.cat([
                        self.occ_memXAI,
                        self.rgb_memXAI*self.occ_memXAI,
                    ], dim=1)
                    _, feat_memXAI, _ = self.featnet3D(feat_memXAI_input)
                    _, occ_memXAI_pred, _, _ = self.occnet(feat_memXAI)
                    occ_memXAI = F.sigmoid(occ_memXAI_pred)

                    # # these boosts seem to hurt:
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
                    self.feat_memXAI_all.append(feat_memXAI)
                    self.occ_memXAI_all.append(occ_memXAI)
                    occrel_memXAI_all.append(occrel_memXAI)

                if hyp.do_use_cache:
                    # save this, so that we have it all next time
                    save_dict = {}
                    save_dict['scene_centroid'] = self.scene_centroid.detach().cpu().numpy()
                    save_dict['vis_memXAI_all'] = torch.stack(vis_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['feat_memXAI_all'] = torch.stack(self.feat_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['occ_memXAI_all'] = torch.stack(self.occ_memXAI_all, dim=1).detach().cpu().numpy()
                    save_dict['occrel_memXAI_all'] = torch.stack(occrel_memXAI_all, dim=1).detach().cpu().numpy()
                    np.savez(feat_cache_fn, save_dict=save_dict)
                    print('saved feats to %s cache, for next time' % feat_cache_fn)
                    # return early, to not apply grads
                    return total_loss, results, True

        self.summ_writer.summ_feats('3D_feats/feat_memXAI', self.feat_memXAI_all, pca=True)
        self.summ_writer.summ_occs('3D_feats/occ_memXAI', self.occ_memXAI_all)

        if hyp.do_flow:
            print('computing flow...')
            # flow01_all = []
            # flow12_all = []
            # flow23_all = []
            # flow34_all = []
            # flow45_all = []
            # flow02_all = []
            # flow03_all = []
            # flow04_all = []
            # flow10_all = []
            # flow04_vis = []
            # flow05_new_vis = []
            # flow10_vis = []
            # consistent_flow05_all = []
            # consistent_flow05_vis = []
            # consistent_mask05_vis = []
            # feat_vis = []
            # flowmag_vis = []

            flow01_vis = []
            flow05_all = []
            flow05_vis = []
            occ0_vis = []
            
            clip = 1.0
            for s in list(range(self.S-5)):
                # vis0 = vis_memXAs[:,s]
                # print('working on s=%d' % s)
                # rel0 = occrel_memXAI_all[s]
                occ0 = self.occ_memXAI_all[s]
                occ1 = self.occ_memXAI_all[s+1]
                occ2 = self.occ_memXAI_all[s+2]
                occ3 = self.occ_memXAI_all[s+3]
                occ4 = self.occ_memXAI_all[s+4]
                occ5 = self.occ_memXAI_all[s+5]
                feat0 = self.feat_memXAI_all[s]
                feat1 = self.feat_memXAI_all[s+1]
                feat2 = self.feat_memXAI_all[s+2]
                feat3 = self.feat_memXAI_all[s+3]
                feat4 = self.feat_memXAI_all[s+4]
                feat5 = self.feat_memXAI_all[s+5]

                # occ0 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+0], self.Z2, self.Y2, self.X2)
                # occ1 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+1], self.Z2, self.Y2, self.X2)
                # occ2 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+2], self.Z2, self.Y2, self.X2)
                # occ3 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+3], self.Z2, self.Y2, self.X2)
                # occ4 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+4], self.Z2, self.Y2, self.X2)
                # occ5 = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s+5], self.Z2, self.Y2, self.X2)
                # occ0 = self.crop_feat(occ0)
                # occ1 = self.crop_feat(occ1)
                # occ2 = self.crop_feat(occ2)
                # occ3 = self.crop_feat(occ3)
                # occ4 = self.crop_feat(occ4)
                # occ5 = self.crop_feat(occ5)

                # weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                # occ0 = (F.conv3d(occ0, weights, padding=1)).clamp(0, 1)
                # occ1 = (F.conv3d(occ1, weights, padding=1)).clamp(0, 1)
                # occ2 = (F.conv3d(occ2, weights, padding=1)).clamp(0, 1)
                # occ3 = (F.conv3d(occ3, weights, padding=1)).clamp(0, 1)
                # occ4 = (F.conv3d(occ4, weights, padding=1)).clamp(0, 1)
                # occ5 = (F.conv3d(occ5, weights, padding=1)).clamp(0, 1)
                
                _, flow01 = self.flownet(feat0, feat1)
                _, flow12 = self.flownet(feat1, feat2)
                _, flow23 = self.flownet(feat2, feat3)
                _, flow34 = self.flownet(feat3, feat4)
                _, flow45 = self.flownet(feat4, feat5)

                _, flow54 = self.flownet(feat5, feat4)
                _, flow43 = self.flownet(feat4, feat3)
                _, flow32 = self.flownet(feat3, feat2)
                _, flow21 = self.flownet(feat2, feat1)
                _, flow10 = self.flownet(feat1, feat0)

                # _, flow01 = self.flownet(feat0, feat0)
                # _, flow12 = self.flownet(feat1, feat1)
                # _, flow23 = self.flownet(feat2, feat2)
                # _, flow34 = self.flownet(feat3, feat3)
                # _, flow45 = self.flownet(feat4, feat4)

                # _, flow54 = self.flownet(feat5, feat5)
                # _, flow43 = self.flownet(feat4, feat4)
                # _, flow32 = self.flownet(feat3, feat3)
                # _, flow21 = self.flownet(feat2, feat2)
                # _, flow10 = self.flownet(feat1, feat1)

                # flow01 = flow01 * occ0
                # flow12 = flow12 * occ1
                # flow23 = flow23 * occ2
                # flow34 = flow34 * occ3
                # flow45 = flow45 * occ4

                # flow54 = flow54 * occ5
                # flow43 = flow43 * occ4
                # flow32 = flow32 * occ3
                # flow21 = flow21 * occ2
                # flow10 = flow10 * occ1
                

                # flow01 = utils_basic.gaussian_blur_3d(flow01, kernel_size=3, sigma=2.0)
                # flow12 = utils_basic.gaussian_blur_3d(flow12, kernel_size=3, sigma=2.0)
                # flow23 = utils_basic.gaussian_blur_3d(flow23, kernel_size=3, sigma=2.0)
                # flow34 = utils_basic.gaussian_blur_3d(flow34, kernel_size=3, sigma=2.0)
                # flow45 = utils_basic.gaussian_blur_3d(flow45, kernel_size=3, sigma=2.0)

                # flow10 = utils_basic.gaussian_blur_3d(flow10, kernel_size=3, sigma=2.0)
                # flow21 = utils_basic.gaussian_blur_3d(flow21, kernel_size=3, sigma=2.0)
                # flow32 = utils_basic.gaussian_blur_3d(flow32, kernel_size=3, sigma=2.0)
                # flow43 = utils_basic.gaussian_blur_3d(flow43, kernel_size=3, sigma=2.0)
                # flow54 = utils_basic.gaussian_blur_3d(flow54, kernel_size=3, sigma=2.0)


                # flow01_vis.append(self.summ_writer.summ_3D_flow('', flow01, occ=rel0*occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                # flow01_vis.append(self.summ_writer.summ_3D_flow('', flow01, occ=occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                flow01_vis.append(self.summ_writer.summ_3D_flow('', flow01, occ=occ0, clip=clip, only_return=True))
                occ0_vis.append(self.summ_writer.summ_occ('', occ0, only_return=True))
                
                flow53 = flow54 + utils_samp.apply_flowAB_to_voxB(flow54, flow43)
                flow52 = flow53 + utils_samp.apply_flowAB_to_voxB(flow53, flow32)
                flow51 = flow52 + utils_samp.apply_flowAB_to_voxB(flow52, flow21)
                flow50 = flow51 + utils_samp.apply_flowAB_to_voxB(flow51, flow10)

                flow02 = flow01 + utils_samp.apply_flowAB_to_voxB(flow01, flow12)
                flow03 = flow02 + utils_samp.apply_flowAB_to_voxB(flow02, flow23)
                flow04 = flow03 + utils_samp.apply_flowAB_to_voxB(flow03, flow34)
                flow05 = flow04 + utils_samp.apply_flowAB_to_voxB(flow04, flow45)

                flow50_aligned = utils_samp.apply_flowAB_to_voxB(flow05, flow50)
                
                # flow05 = utils_basic.gaussian_blur_3d(flow05, kernel_size=3, sigma=2.0)                
                # flow05 = utils_basic.gaussian_blur_3d(flow05, kernel_size=3, sigma=2.0)                
                # flow05 = utils_basic.gaussian_blur_3d(flow05, kernel_size=3, sigma=2.0)                
                
                flow_cycle = flow05 + flow50_aligned
                consistency_mask = torch.exp(-torch.norm(flow_cycle, dim=1, keepdim=True))

                # flow05 = flow05 * consistency_mask

                # occ = self.vox_util.voxelize_xyz(self.xyz_camXAs[:,s], self.Z2, self.Y2, self.X2)
                # occ = self.crop_feat(occ)
                
                # flow05 = flow05 * occ0
                
                flow05 = flow05 * occ0
                # flow01_all.append(flow01)
                # flow02_all.append(flow02)
                # flow03_all.append(flow03)
                # flow04_all.append(flow04)
                flow05_all.append(flow05)

                # flow12_all.append(flow12)
                # flow23_all.append(flow23)
                # flow34_all.append(flow34)
                # flow45_all.append(flow45)
                
                # flow05_vis.append(self.summ_writer.summ_3D_flow('', flow05, occ=occ0*(1.0-occ_memXAI_median), clip=clip, only_return=True))
                flow05_vis.append(self.summ_writer.summ_3D_flow('', flow05, occ=occ0, clip=clip, only_return=True))

            self.summ_writer.summ_rgbs('3D_feats/flow01', flow01_vis)
            self.summ_writer.summ_rgbs('3D_feats/flow05', flow05_vis)
            self.summ_writer.summ_rgbs('3D_feats/occ0', occ0_vis)

            blue_vis_all = []
            # conn_vis = []
            # lrtlist_camXIs = []
            lrtlist_memXAI_all = []
            connlist_memXAI_all = []
            scorelist_all = []

            self.K = 32
            K = self.K
            crop_zyx = self.crop_guess

            for I in list(range(self.S-5)):
                print('getting box proposals for frame %d' % I)
                flow_mag = torch.norm(flow05_all[I], dim=1)
                boxes_image, boxlist_memXAI, scorelist_e, tidlist, connlist = utils_misc.get_boxes_from_flow_mag(
                    flow_mag, K)
                x, y, z, lx, ly, lz, rx, ry, rz = boxlist_memXAI.unbind(2)
                z = z + crop_zyx[0]
                y = y + crop_zyx[1]
                x = x + crop_zyx[2]
                boxlist_memXAI = torch.stack([x, y, z, lx, ly, lz, rx, ry, rz], dim=2)

                lrtlist_memXAI = utils_geom.convert_boxlist_to_lrtlist(boxlist_memXAI)
                lrtlist_memXAI_all.append(lrtlist_memXAI)
                connlist_memXAI_all.append(connlist)

                scorelist_e[scorelist_e > 0.0] = 1.0
                occ_memXAI = self.occ_memXAI_all[I]
                for n in list(range(K)):
                    mask_1 = connlist[:,n:n+1]
                    weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                    mask_3 = (F.conv3d(mask_1, weights, padding=1)).clamp(0, 1)

                    center_mask = mask_1.clone()
                    surround_mask = (mask_3-mask_1).clamp(0,1)

                    center_ = utils_basic.reduce_masked_mean(occ_memXAI, center_mask, dim=[2,3,4])
                    surround_ = utils_basic.reduce_masked_mean(occ_memXAI, surround_mask, dim=[2,3,4])
                    score_ = center_ - surround_
                    score_ = torch.clamp(torch.sigmoid(score_), min=1e-4)
                    score_[score_ < 0.55] = 0.0
                    scorelist_e[:,n] = score_
                scorelist_all.append(scorelist_e)

                # self.summ_writer.summ_rgb('proposals/anchor_frame', diff_memXAI_vis[self.anchor])
                # self.summ_writer.summ_rgb('proposals/get_boxes', boxes_image)
                blue_vis_all.append(boxes_image)

            # lrtlist_memXAI_all, connlist_memXAI_all, scorelist_all, blue_vis_all = propose_boxes_by_differencing(
            #     self.K, self.S, self.occ_memXAI_all, self.diff_memXAI_all, self.crop_guess,
            #     self.set_data_name, data_ind, super_iter)

            self.summ_writer.summ_rgbs('proposals/blue_boxes', blue_vis_all)

            camXs_T_camXAs_all = list(self.camXs_T_camXAs.unbind(1))
            lrtlist_camXAI_all = [self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_memXAI, self.Z2, self.Y2, self.X2)
                                  for lrtlist_memXAI in lrtlist_memXAI_all]
            lrtlist_camXI_all = [utils_geom.apply_4x4_to_lrtlist(camXI_T_camXA, lrtlist_camXAI)
                                 for (camXI_T_camXA, lrtlist_camXAI) in zip(camXs_T_camXAs_all, lrtlist_camXAI_all)]

            # quick eval:
            # note that since B=1, if i pack then i'll have tensors shaped S x N x 19
            super_lrtlist_ = __p(torch.stack(lrtlist_camXI_all, dim=1))
            super_scorelist_ = __p(torch.stack(scorelist_all, dim=1))
            full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
            full_scorelist_s_ = __p(self.full_scorelist_s)
            iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

            print('super_lrtlist_', super_lrtlist_.shape)
            print('super_scorelist_', super_scorelist_.shape)
            print('full_lrtlist_camXs_', full_lrtlist_camXs_.shape)
            print('full_scorelist_s_', full_scorelist_s_.shape)

            all_maps_3d = np.zeros([self.S, len(iou_thresholds)])
            all_maps_2d = np.zeros([self.S, len(iou_thresholds)])
            all_maps_pers = np.zeros([self.S, len(iou_thresholds)])
            all_maps_valid = np.zeros([self.S])
            for s in list(range(self.S-5)):
                lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                    super_lrtlist_[s:s+1], full_lrtlist_camXs_[s:s+1], super_scorelist_[s:s+1], full_scorelist_s_[s:s+1])

                if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:
                    all_maps_valid[s] = 1.0
                    maps_3d, maps_2d = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                    all_maps_3d[s] = maps_3d
                    all_maps_2d[s] = maps_2d
                    boxlist_e = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_e)
                    boxlist_g = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_g)
                    maps_pers = utils_eval.get_mAP_from_2d_boxlists(boxlist_e, scorelist_e, boxlist_g, iou_thresholds)
                    all_maps_pers[s] = maps_pers
                elif torch.sum(scorelist_g) > 0:
                    all_maps_valid[s] = 1.0
                    all_maps_3d[s] = 0.0
                    all_maps_2d[s] = 0.0
                    all_maps_pers[s] = 0.0
                
            for ind, overlap in enumerate(iou_thresholds):
                maps_3d = all_maps_3d[:,ind]
                maps_2d = all_maps_2d[:,ind]
                maps_pers = all_maps_pers[:,ind]

                map_3d_val = utils_py.reduce_masked_mean(maps_3d, all_maps_valid)
                map_2d_val = utils_py.reduce_masked_mean(maps_2d, all_maps_valid)
                map_pers_val = utils_py.reduce_masked_mean(maps_pers, all_maps_valid)
                
                self.summ_writer.summ_scalar('proposal_ap_3d/%.2f_iou' % overlap, map_3d_val)
                self.summ_writer.summ_scalar('proposal_ap_2d/%.2f_iou' % overlap, map_2d_val)
                self.summ_writer.summ_scalar('proposal_ap_pers/%.2f_iou' % overlap, map_pers_val)
            results['all_proposal_maps_3d'] = all_maps_3d
            results['all_proposal_maps_2d'] = all_maps_2d
            results['all_proposal_maps_pers'] = all_maps_pers

            box_vis_bev = []
            box_vis = []
            for I in list(range(self.S-5)):
                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,I],
                    torch.cat([self.full_lrtlist_camXs[:,I], lrtlist_camXI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.pix_T_cams[:,0], frame_id=I, only_return=True))
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.pad_feat(self.occ_memXAI_all[I]),
                    torch.cat([self.full_lrtlist_camXAs[:,I], lrtlist_camXAI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.vox_util, frame_id=I, only_return=True))
            self.summ_writer.summ_rgbs('proposals/all_boxes_bev', box_vis_bev)
            self.summ_writer.summ_rgbs('proposals/all_boxes', box_vis)

            # flow mode cannot go on to track
            return total_loss, results, False
        
        if hyp.do_use_cache:
            data_ind = feed['data_ind']
            med_cache_fn = 'cache/%s_%06d_s%d_med.npz' % (self.set_data_name, data_ind, self.S)
            # check if the thing exists
            if os.path.isfile(med_cache_fn):
                print('found median cache at %s; we will use this' % med_cache_fn)
                cache = np.load(med_cache_fn, allow_pickle=True)['save_dict'].item()
                have_medians = True
                self.occ_memXAI_median = torch.from_numpy(cache['occ_memXAI_median']).cuda()
                self.feat_memXAI_median = torch.from_numpy(cache['feat_memXAI_median']).cuda()
            else:
                print('could not find median cache at %s; we will write this' % med_cache_fn)
        
        if not have_medians:
            feat_memXAI_all_np = (torch.stack(self.feat_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            feat_memXAI_median_np = np.median(feat_memXAI_all_np, axis=0)
            self.feat_memXAI_median = torch.from_numpy(feat_memXAI_median_np).float().reshape(1, -1, Z2, Y2, X2).cuda()

            occ_memXAI_all_np = (torch.stack(self.occ_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            vis_memXAI_all_np = (torch.stack(vis_memXAI_all).detach().cpu().reshape(self.S, -1)).numpy()
            occ_memXAI_median_np_safe = np.median(occ_memXAI_all_np, axis=0)
            occ_memXAI_median_np = utils_py.reduce_masked_median(
                occ_memXAI_all_np.transpose(1, 0), vis_memXAI_all_np.transpose(1, 0), keep_batch=True)
            occ_memXAI_median_np[np.isnan(occ_memXAI_median_np)] = occ_memXAI_median_np_safe[np.isnan(occ_memXAI_median_np)]
            self.occ_memXAI_median = torch.from_numpy(occ_memXAI_median_np).float().reshape(1, -1, Z2, Y2, X2).cuda()

            # occ_memXAI_diff_np = np.mean(np.abs(occ_memXAI_all_np[1:] - occ_memXAI_all_np[:-1]), axis=0)
            # occ_memXAI_diff = torch.from_numpy(occ_memXAI_diff_np).float().reshape(1, 1, Z2, Y2, X2).cuda()

            if hyp.do_use_cache:
                # save this, so that we have it all next time
                save_dict = {}
                save_dict['occ_memXAI_median'] = self.occ_memXAI_median.detach().cpu().numpy()
                save_dict['feat_memXAI_median'] = self.feat_memXAI_median.detach().cpu().numpy()
                np.savez(med_cache_fn, save_dict=save_dict)
                print('saved medians to %s cache, for next time' % med_cache_fn)
                
        self.summ_writer.summ_feat('3D_feats/feat_memXAI_median', self.feat_memXAI_median, pca=True)
        self.summ_writer.summ_occ('3D_feats/occ_memXAI_median', self.occ_memXAI_median)

        
        # now, i should be able to walk through a second time, and collect great diff signals
        if use_feat_cache:
            self.diff_memXAI_all = []
            for I in list(range(self.S)):
                vis_memXAI = vis_memXAI_all[I]

                weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)
                # vis_memXAI = (F.conv3d(vis_memXAI, weights, padding=1)).clamp(0, 1)

                occ_memXAI = self.occ_memXAI_all[I]
                occrel_memXAI = occrel_memXAI_all[I]

                use_occrel = True
                if not use_occrel:
                    occrel_memXAI = torch.ones_like(occrel_memXAI)

                # diff_memXAI_all.append(torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(occ_memXAI * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))
                # diff_memXAI_all.append(occ_memXAI.round() * vis_memXAI * torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True))

                # diff = torch.norm(occ_memXAI - occ_memXAI_median, dim=1, keepdim=True)
                diff = torch.norm(occ_memXAI - self.occ_memXAI_median, dim=1, keepdim=True) * occrel_memXAI
                # diff = torch.nn.functional.relu(diff - occ_memXAI_diff)

                diff = occ_memXAI.round() * vis_memXAI * diff
                self.diff_memXAI_all.append(diff)

        
        self.K = 32

        super_lrtlist = []
        super_scorelist = []
        super_tidlist = []

        for super_iter in list(range(4)):
            print('-'*100)
            print('super_iter %d' % super_iter)

            diff_memXAI_vis = []
            for I in list(range(self.S)):
                diff_memXAI_vis.append(self.summ_writer.summ_oned('', self.diff_memXAI_all[I], bev=True, max_along_y=True, norm=False, only_return=True))
            self.summ_writer.summ_rgbs('3D_feats/diff_memXAI_all_%d' % super_iter, diff_memXAI_vis)
            
            lrtlist_memXAI_all, connlist_memXAI_all, scorelist_all, blue_vis_all = propose_boxes_by_differencing(
                self.K, self.S, self.occ_memXAI_all, self.diff_memXAI_all, self.crop_guess,
                self.set_data_name, data_ind, super_iter, use_box_cache=False)
            self.summ_writer.summ_rgbs('proposals/blue_boxes_%d' % super_iter, blue_vis_all)

            camXs_T_camXAs_all = list(self.camXs_T_camXAs.unbind(1))
            lrtlist_camXAI_all = [self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_memXAI, self.Z2, self.Y2, self.X2)
                                  for lrtlist_memXAI in lrtlist_memXAI_all]
            lrtlist_camXI_all = [utils_geom.apply_4x4_to_lrtlist(camXI_T_camXA, lrtlist_camXAI)
                                 for (camXI_T_camXA, lrtlist_camXAI) in zip(camXs_T_camXAs_all, lrtlist_camXAI_all)]

            if super_iter == 0:
                # quick eval:
                # note that since B=1, if i pack then i'll have tensors shaped S x N x 19
                super_lrtlist_ = __p(torch.stack(lrtlist_camXI_all, dim=1))
                super_scorelist_ = __p(torch.stack(scorelist_all, dim=1))
                full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
                full_scorelist_s_ = __p(self.full_scorelist_s)
                iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                all_maps_3d = np.zeros([self.S, len(iou_thresholds)])
                all_maps_2d = np.zeros([self.S, len(iou_thresholds)])
                all_maps_pers = np.zeros([self.S, len(iou_thresholds)])
                all_maps_valid = np.zeros([self.S])
                for s in list(range(self.S)):
                    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                        super_lrtlist_[s:s+1], full_lrtlist_camXs_[s:s+1], super_scorelist_[s:s+1], full_scorelist_s_[s:s+1])

                    if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:
                        all_maps_valid[s] = 1.0
                        maps_3d, maps_2d = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                        all_maps_3d[s] = maps_3d
                        all_maps_2d[s] = maps_2d
                        boxlist_e = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_e)
                        boxlist_g = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_g)
                        maps_pers = utils_eval.get_mAP_from_2d_boxlists(boxlist_e, scorelist_e, boxlist_g, iou_thresholds)
                        all_maps_pers[s] = maps_pers
                    elif torch.sum(scorelist_g) > 0:
                        all_maps_valid[s] = 1.0
                        all_maps_3d[s] = 0.0
                        all_maps_2d[s] = 0.0
                        all_maps_pers[s] = 0.0
                    
                for ind, overlap in enumerate(iou_thresholds):
                    maps_3d = all_maps_3d[:,ind]
                    maps_2d = all_maps_2d[:,ind]
                    maps_pers = all_maps_pers[:,ind]
                    
                    map_3d_val = utils_py.reduce_masked_mean(maps_3d, all_maps_valid)
                    map_2d_val = utils_py.reduce_masked_mean(maps_2d, all_maps_valid)
                    map_pers_val = utils_py.reduce_masked_mean(maps_pers, all_maps_valid)
                    
                    # if len(maps_3d):
                        # map_3d_val = np.mean(maps_3d)
                        # map_2d_val = np.mean(maps_2d)
                        # map_pers_val = np.mean(maps_pers)
                    # else:
                    #     map_3d_val = 0.0
                    #     map_2d_val = 0.0
                    #     map_pers_val = 0.0
                    self.summ_writer.summ_scalar('proposal_ap_3d/%.2f_iou' % overlap, map_3d_val)
                    self.summ_writer.summ_scalar('proposal_ap_2d/%.2f_iou' % overlap, map_2d_val)
                    self.summ_writer.summ_scalar('proposal_ap_pers/%.2f_iou' % overlap, map_pers_val)
                results['all_proposal_maps_3d'] = all_maps_3d
                results['all_proposal_maps_2d'] = all_maps_2d
                results['all_proposal_maps_pers'] = all_maps_pers
                
            box_vis_bev = []
            box_vis = []
            for I in list(range(self.S)):
                box_vis.append(self.summ_writer.summ_lrtlist(
                    '', self.rgb_camXs[:,I],
                    torch.cat([self.full_lrtlist_camXs[:,I], lrtlist_camXI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.pix_T_cams[:,0], frame_id=I, only_return=True))
                box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    '', self.pad_feat(self.occ_memXAI_all[I]),
                    torch.cat([self.full_lrtlist_camXAs[:,I], lrtlist_camXAI_all[I]], dim=1),
                    torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    self.vox_util, frame_id=I, only_return=True))
            self.summ_writer.summ_rgbs('proposals/all_boxes_bev_%d' % super_iter, box_vis_bev)
            self.summ_writer.summ_rgbs('proposals/all_boxes_%d' % super_iter, box_vis)

            # return here if you just want to see proposal perf
            return total_loss, results, False
        

            # at this stage
            # i need to assume i've done all the per-frame scoring i can do
            # next i need to figure out the objects

            best_size = torch.zeros_like(scorelist_all[0][:,0])
            best_score = torch.zeros_like(scorelist_all[0][:,0])
            best_lrt = lrtlist_camXAI_all[0][:,0]
            best_occ = torch.zeros_like(self.occ_memXAI_all[0])
            best_I = 0

            for b in list(range(self.B)):
                for I in list(range(self.S)):
                    scorelist = scorelist_all[I][b]
                    lrtlist_camXAI = lrtlist_camXAI_all[I][b]
                    occ_memXAI = self.occ_memXAI_all[I][b]
                    connlist_memXAI = connlist_memXAI_all[I][b]
                    # this is K x Z x Y x X
                    sizelist = torch.sum(connlist_memXAI, dim=[1,2,3])
                    # print('scorelist', scorelist.shape)
                    for k in list(range(self.K)):
                        # ind = torch.argmax(sizelist, dim=0)
                        score_here = scorelist[k]
                        lrt_here = lrtlist_camXAI[k]
                        size_here = sizelist[k]

                        # print('score_here, size_here', score_here, size_here)

                        if score_here >= 0.6 and size_here > best_size[b]:
                            # print('winner!')
                            best_size[b] = size_here
                            best_score[b] = score_here
                            best_lrt[b] = lrt_here
                            best_occ[b] = occ_memXAI
                            best_I = I
            print('biggest object we could find (iter %d)' % best_I, best_size.detach().cpu().numpy())
            
            # only proceed to track the object if it's at least 3 voxels large

            if best_size > 3:

                self.summ_writer.summ_lrtlist_bev(
                    'proposals/best_box_bev_%d' % super_iter, self.pad_feat(best_occ),
                    best_lrt.unsqueeze(1),
                    best_size.unsqueeze(1),
                    torch.ones_like(best_score.unsqueeze(1)).long(),
                    self.vox_util, frame_id=best_I)

                zoom = False
                if zoom:
                    # instead of immediately tracking, let's see if we can refine this box by a high-res bkg-subtraction 

                    Z_zoom, Y_zoom, X_zoom = hyp.Z_zoom, hyp.Y_zoom, hyp.X_zoom
                    zoom_centroid = utils_geom.get_clist_from_lrtlist(best_lrt.unsqueeze(1)).squeeze(1)
                    zoom_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=zoom_centroid, assert_cube=True)

                    # the centroid is in XA coords, so i should put everything into XA coords

                    vis_all = []
                    feat_all = []
                    occ_all = []
                    rel_all = []
                    # for s in list(range(best_I-2,bestI+2)):
                    for s in list(range(self.S)):
                        # note it is necessary to walk through the full seq here, to get a good median 

                        # print('zoom %d' % s)
                        rgb_memXII = zoom_vox_util.unproject_rgb_to_mem(
                            self.rgb_camXs[:,s], Z_zoom, Y_zoom, X_zoom, self.pix_T_cams[:,s])
                        rgb_memXAI = zoom_vox_util.apply_4x4_to_vox(self.camXAs_T_camXs[:,s], rgb_memXII)
                        occ_memXAI = zoom_vox_util.voxelize_xyz(self.xyz_camXAs[:,s], Z_zoom, Y_zoom, X_zoom)

                        occ_memXAIs, free_memXAIs, _, _ = zoom_vox_util.prep_occs_supervision(
                            self.camXAs_T_camXs[:,s:s+1],
                            self.xyz_camXs[:,I:I+1],
                            int(Z_zoom/2), int(Y_zoom/2), int(X_zoom/2),
                            agg=False)
                        occ_memXAI_g = self.crop_feat(occ_memXAIs.squeeze(1))
                        free_memXAI_g = self.crop_feat(free_memXAIs.squeeze(1))
                        vis_memXAI = (occ_memXAI_g + free_memXAI_g).clamp(0, 1)

                        feat_memXAI_input = torch.cat([
                            occ_memXAI, rgb_memXAI*occ_memXAI,
                        ], dim=1)
                        _, feat_memXAI, _ = self.featnet3D(feat_memXAI_input)
                        _, occ_memXAI_pred, _, _ = self.occnet(feat_memXAI)
                        occ_memXAI = F.sigmoid(occ_memXAI_pred)
                        _, occrel_memXAI = self.occrelnet(feat_memXAI)
                        # else:
                        #     vis_memXAI = torch.ones(

                        vis_all.append(vis_memXAI)
                        feat_all.append(feat_memXAI)
                        occ_all.append(occ_memXAI)
                        rel_all.append(occrel_memXAI)
                        # ind_all.append(s)

                    # print('feat_all[0]', feat_all[0].shape)
                    # print('vis_all[0]', vis_all[0].shape)
                    # print('occ_all[0]', occ_all[0].shape)
                    # print('rel_all[0]', rel_all[0].shape)
                    _, _, Z2_zoom, Y2_zoom, X2_zoom = feat_all[0].shape

                    self.summ_writer.summ_feats('3D_feats/zoom_feat_all_%d' % super_iter, feat_all, pca=True)
                    self.summ_writer.summ_occs('3D_feats/zoom_occ_all_%d' % super_iter, occ_all)

                    feat_all_np = (torch.stack(feat_all).detach().cpu().reshape(self.S, -1)).numpy()
                    feat_median_np = np.median(feat_all_np, axis=0)
                    feat_median = torch.from_numpy(feat_median_np).float().reshape(1, -1, int(Z2_zoom), int(Y2_zoom), int(X2_zoom)).cuda()

                    occ_all_np = (torch.stack(occ_all).detach().cpu().reshape(self.S, -1)).numpy()
                    vis_all_np = (torch.stack(vis_all).detach().cpu().reshape(self.S, -1)).numpy()
                    occ_median_np_safe = np.median(occ_all_np, axis=0)
                    occ_median_np = utils_py.reduce_masked_median(
                        occ_all_np.transpose(1, 0), vis_all_np.transpose(1, 0), keep_batch=True)
                    occ_median_np[np.isnan(occ_median_np)] = occ_median_np_safe[np.isnan(occ_median_np)]
                    occ_median = torch.from_numpy(occ_median_np).float().reshape(1, -1, int(Z2_zoom), int(Y2_zoom), int(X2_zoom)).cuda()

                    self.summ_writer.summ_feat('3D_feats/zoom_feat_median_%d' % super_iter, feat_median, pca=True)
                    self.summ_writer.summ_occ('3D_feats/zoom_occ_median_%d' % super_iter, occ_median)

                    # super_lrtlist is a list of B x N x 19 tensors
                    super_mask_all = []
                    for (super_lrt, super_score) in zip(super_lrtlist, super_scorelist):
                        # print('working on super_mask')
                        # print('super_lrt', super_lrt.shape)
                        # print('super_score', super_score.shape)

                        # super_lrt is B x N x 19 i think
                        # each lrt is lrt_camXAI
                        mask = zoom_vox_util.assemble_padded_obj_masklist(
                            super_lrt, super_score, int(Z_zoom/2), int(Y_zoom/2), int(X_zoom/2),
                            coeff=1.1, additive_coeff=2.0)
                        # this is B x S x 1 x Z2 x Y2 x X2
                        mask = mask.squeeze(2)
                        print('mask', mask.shape)

                        # mask = torch.sum(mask, dim=1).clamp(0, 1)
                        # # this is B x 1 x Z2 x Y2 x X2
                        mask = self.crop_feat(mask)
                        print('appending this mask to super_mask_all:', mask.shape)
                        super_mask_all.append(mask)
                    if len(super_mask_all):
                        # stack and squash the super dim
                        super_mask = torch.sum(torch.stack(super_mask_all, dim=1), dim=1, keepdim=True).clamp(0, 1)
                        print('super_mask', super_mask.shape)
                        # this is B x 1 x S x x Z2 x Y2 x X2
                        # unstack along the seq dim
                        super_mask_all = list(super_mask.unbind(2))
                    # else:
                    #     # super_mask = torch.ones_like(
                    #     # print('super_mask', super_mask

                    diff_some = []
                    occ_some = []
                    # ind_some = list(range(np.min([best_I-2, 0]), np.max([best_I+2, self.S])))
                    ind_some = [best_I]
                    camXs_T_camXAs_some = []

                    # for s in list(range(self.S)):
                    for s in ind_some:
                        vis = vis_all[s]

                        weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
                        vis = (F.conv3d(vis, weights, padding=1)).clamp(0, 1)
                        # vis = (F.conv3d(vis, weights, padding=1)).clamp(0, 1)
                        # vis = (F.conv3d(vis, weights, padding=1)).clamp(0, 1)

                        occ = occ_all[s]
                        occ_some.append(occ)
                        camXs_T_camXAs_some.append(self.camXs_T_camXAs[:,s])
                        rel = rel_all[s]

                        diff = torch.norm(occ - occ_median, dim=1, keepdim=True) * rel
                        diff = occ.round() * vis * diff

                        # suppress already-claimed objects
                        if len(super_mask_all):
                            diff = diff * (1.0 - super_mask_all[s])

                        diff_some.append(diff)
                        # ind_all.append(s)

                    print('diff_some[0]', diff_some[0].shape)
                    # # eliminate the claimed locations from the difference map
                    # diff_all = [diff*(1.0-mask) for (diff, mask) in zip(self.diff_memXAI_all, mask_memXAI_all)]

                    diff_vis = []
                    for s in list(range(len(diff_some))):
                        diff_vis.append(self.summ_writer.summ_oned('', diff_some[s], bev=True, max_along_y=True, norm=False, only_return=True))
                    self.summ_writer.summ_rgbs('3D_feats/zoom_diff_some_%d' % super_iter, diff_vis)

                    lrtlist_zoomXAI_some, connlist_zoomXAI_some, scorelist_some, blue_vis_some = propose_boxes_by_differencing(
                        self.K, len(diff_some), occ_some, diff_some, self.crop_guess,
                        self.set_data_name, data_ind, super_iter, use_box_cache=False, summ_writer=self.summ_writer)
                    self.summ_writer.summ_rgbs('proposals/zoom_blue_boxes_%d' % super_iter, blue_vis_some)

                    lrtlist_camXAI_some = [zoom_vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_zoomXAI, int(Z_zoom/2), int(Y_zoom/2), int(X_zoom/2))
                                          for lrtlist_zoomXAI in lrtlist_zoomXAI_some]
                    lrtlist_camXI_some = [utils_geom.apply_4x4_to_lrtlist(camXI_T_camXA, lrtlist_camXAI)
                                         for (camXI_T_camXA, lrtlist_camXAI) in zip(camXs_T_camXAs_some, lrtlist_camXAI_some)]
                    # # quick eval:
                    # # note that since B=1, if i pack then i'll have tensors shaped S x N x 19
                    # super_lrtlist_ = __p(torch.stack(lrtlist_camXI_all, dim=1))
                    # super_scorelist_ = __p(torch.stack(scorelist_all, dim=1))
                    # full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
                    # full_scorelist_s_ = __p(self.full_scorelist_s)
                    # iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    # all_maps_3d = np.zeros([self.S, len(iou_thresholds)])
                    # all_maps_2d = np.zeros([self.S, len(iou_thresholds)])
                    # for s in list(range(self.S)):
                    #     lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                    #         super_lrtlist_[s:s+1], full_lrtlist_camXs_[s:s+1], super_scorelist_[s:s+1], full_scorelist_s_[s:s+1])
                    #     maps_3d, maps_2d = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                    #     all_maps_3d[s] = maps_3d
                    #     all_maps_2d[s] = maps_2d
                    # for ind, overlap in enumerate(iou_thresholds):
                    #     maps_3d = all_maps_3d[:,ind]
                    #     maps_2d = all_maps_2d[:,ind]
                    #     if len(maps_3d):
                    #         map_3d_val = np.mean(maps_3d)
                    #         map_2d_val = np.mean(maps_2d)
                    #     else:
                    #         map_3d_val = 0.0
                    #         map_2d_val = 0.0
                    #     self.summ_writer.summ_scalar('zoom_proposal_ap_3d/%.2f_iou' % overlap, map_3d_val)
                    #     self.summ_writer.summ_scalar('zoom_proposal_ap_2d/%.2f_iou' % overlap, map_2d_val)
                    # results['all_zoom_proposal_maps_3d'] = all_maps_3d
                    # results['all_zoom_proposal_maps_2d'] = all_maps_2d

                    # box_vis_bev = []
                    # box_vis = []
                    # for I in list(range(self.S)):
                    #     box_vis.append(self.summ_writer.summ_lrtlist(
                    #         '', self.rgb_camXs[:,I],
                    #         torch.cat([self.full_lrtlist_camXs[:,I], lrtlist_camXI_all[I]], dim=1),
                    #         torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    #         torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    #         self.pix_T_cams[:,0], frame_id=I, only_return=True))
                    #     box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                    #         '', self.pad_feat(self.occ_memXAI_all[I]),
                    #         torch.cat([self.full_lrtlist_camXAs[:,I], lrtlist_camXAI_all[I]], dim=1),
                    #         torch.cat([self.full_scorelist_s[:,I], scorelist_all[I]], dim=1),
                    #         torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2*torch.ones_like(scorelist_all[I]).long()], dim=1),
                    #         self.vox_util, frame_id=I, only_return=True))
                    # self.summ_writer.summ_rgbs('proposals/all_zoom_boxes_bev_%d' % super_iter, box_vis_bev)
                    # self.summ_writer.summ_rgbs('proposals/all_zoom_boxes_%d' % super_iter, box_vis)




                    ## do not necessarily overwrite the initial obj; only do so if it wins
                    # in fact, size = size*2
                    # # to be safe, we'll mult by 1.5
                    # best_size = best_size * 1.5
                    # best_size = torch.zeros_like(scorelist_some[0][:,0])
                    # best_score = torch.zeros_like(scorelist_some[0][:,0])
                    # best_lrt = lrtlist_camXAI_some[0][:,0]
                    # best_occ = torch.zeros_like(self.occ_memXAI_all[0])
                    # best_I = 0

                    for b in list(range(self.B)):
                        for i, I in enumerate(ind_some):
                            scorelist = scorelist_some[i][b]
                            lrtlist_camXAI = lrtlist_camXAI_some[i][b]
                            occ_memXAI = self.occ_memXAI_all[I][b]
                            connlist_zoomXAI = connlist_zoomXAI_some[i][b]
                            # this is K x Z x Y x X
                            sizelist = torch.sum(connlist_zoomXAI, dim=[1,2,3])
                            # print('scorelist', scorelist.shape)
                            for k in list(range(self.K)):
                                # ind = torch.argmax(sizelist, dim=0)
                                score_here = scorelist[k]
                                lrt_here = lrtlist_camXAI[k]
                                size_here = sizelist[k]

                                if score_here > 0.5 and size_here > best_size[b]:
                                    best_size[b] = size_here
                                    best_score[b] = score_here
                                    best_lrt[b] = lrt_here
                                    best_occ[b] = occ_memXAI
                                    best_I = I
                    print('biggest zoom object we could find (iter %d)' % best_I, best_size.detach().cpu().numpy())

                    self.summ_writer.summ_lrtlist_bev(
                        'proposals/best_zoom_box_bev_%d' % super_iter, self.pad_feat(best_occ),
                        best_lrt.unsqueeze(1),
                        best_size.unsqueeze(1),
                        torch.ones_like(best_score.unsqueeze(1)).long(),
                        self.vox_util, frame_id=best_I)

                    # return total_loss, results, True


                # next thing i want to do is:
                # track that proposed object across the video
                # here i need code from model_nuscenes_zoom i think
                # then i will paste it into place

                lrt_camXAIs, scores, mask_memXAI_all, top_feat_memXAI_all, top_occ_memXAI_all = track_proposal(
                    self.B, self.S, 
                    best_lrt, best_I,
                    self.feat_memXAI_all,
                    self.occ_memXAI_all,
                    self.diff_memXAI_all,
                    self.feat_memXAI_median, 
                    self.occ_memXAI_median, 
                    self.pix_T_cams,
                    self.rgb_camXs,
                    self.xyz_camXAs,
                    self.camXAs_T_camXs,
                    self.featnet3D,
                    self.occrelnet,
                    self.vox_util,
                    self.crop_feat,
                    self.pad_feat,
                    self.crop_guess,
                    summ_writer=self.summ_writer)
                
                self.summ_writer.summ_oneds('track/top3d_mask_vis_%d' % super_iter, mask_memXAI_all, bev=True, norm=False)

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

                super_lrtlist.append(lrt_camXAIs)
                super_scorelist.append(scores)
                super_tidlist.append(torch.ones_like(scores).long() * super_iter)

                # eliminate the claimed locations from the difference map
                self.diff_memXAI_all = [diff*(1.0-mask) for (diff, mask) in zip(self.diff_memXAI_all, mask_memXAI_all)]


        # each lrt_camXAIs is B x S x 19
        super_lrtlist = torch.stack(super_lrtlist, dim=2)
        super_scorelist = torch.stack(super_scorelist, dim=2)
        super_tidlist = torch.stack(super_tidlist, dim=2)
        # super_lrtlist is B x S x N x 19
        box_vis_bev = []
        box_vis = []
        for I in list(range(self.S)):
            lrt_ = super_lrtlist[:,I]
            score_ = super_scorelist[:,I]
            tid_ = super_tidlist[:,I]
            box_vis_bev.append(self.summ_writer.summ_lrtlist_bev(
                '', self.pad_feat(self.occ_memXAI_all[I]),
                torch.cat([self.full_lrtlist_camXAs[:,I], lrt_], dim=1),
                torch.cat([self.full_scorelist_s[:,I], score_], dim=1),
                torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2+tid_], dim=1),
                self.vox_util, frame_id=I, only_return=True))
            lrt_ = utils_geom.apply_4x4_to_lrtlist(self.camXs_T_camXAs[:,I], lrt_)
            box_vis.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,I],
                torch.cat([self.full_lrtlist_camXs[:,I], lrt_], dim=1),
                torch.cat([self.full_scorelist_s[:,I], score_], dim=1),
                torch.cat([torch.ones_like(self.full_tidlist_s[:,I]).long(), 2+tid_], dim=1),
                self.pix_T_cams[:,I],
                frame_id=I, only_return=True))
        self.summ_writer.summ_rgbs('track/super_all_boxes_bev_%d' % super_iter, box_vis_bev)
        self.summ_writer.summ_rgbs('track/super_all_boxes_%d' % super_iter, box_vis)

        # note super_lrtlist is in camXAI coords
        super_lrtlist = __u(utils_geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camXAs), __p(super_lrtlist)))
        # now it is in camX coords
        # and i can evaluate it easily

        # note that since B=1, if i pack then i'll have tensors shaped S x N x 19
        super_lrtlist_ = __p(super_lrtlist)
        super_scorelist_ = __p(super_scorelist)
        full_lrtlist_camXs_ = __p(self.full_lrtlist_camXs)
        full_scorelist_s_ = __p(self.full_scorelist_s)

        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        all_maps_3d = np.zeros([self.S, len(iou_thresholds)])
        all_maps_2d = np.zeros([self.S, len(iou_thresholds)])
        all_maps_pers = np.zeros([self.S, len(iou_thresholds)])
        all_maps_valid = np.zeros([self.S])
        for s in list(range(self.S)):
            lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_lrts(
                super_lrtlist_[s:s+1], full_lrtlist_camXs_[s:s+1], super_scorelist_[s:s+1], full_scorelist_s_[s:s+1])

            if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:
                all_maps_valid[s] = 1.0
                maps_3d, maps_2d = utils_eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                all_maps_3d[s] = maps_3d
                all_maps_2d[s] = maps_2d
                boxlist_e = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_e)
                boxlist_g = utils_geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], lrtlist_g)
                maps_pers = utils_eval.get_mAP_from_2d_boxlists(boxlist_e, scorelist_e, boxlist_g, iou_thresholds)
                all_maps_pers[s] = maps_pers
            elif torch.sum(scorelist_g) > 0:
                all_maps_valid[s] = 1.0
                all_maps_3d[s] = 0.0
                all_maps_2d[s] = 0.0
                all_maps_pers[s] = 0.0
                
        for ind, overlap in enumerate(iou_thresholds):
            maps_3d = all_maps_3d[:,ind]
            maps_2d = all_maps_2d[:,ind]
            maps_pers = all_maps_pers[:,ind]

            map_3d_val = utils_py.reduce_masked_mean(maps_3d, all_maps_valid)
            map_2d_val = utils_py.reduce_masked_mean(maps_2d, all_maps_valid)
            map_pers_val = utils_py.reduce_masked_mean(maps_pers, all_maps_valid)
            
            self.summ_writer.summ_scalar('track_ap_3d/%.2f_iou' % overlap, map_3d_val)
            self.summ_writer.summ_scalar('track_ap_2d/%.2f_iou' % overlap, map_2d_val)
            self.summ_writer.summ_scalar('track_ap_pers/%.2f_iou' % overlap, map_pers_val)

        results['all_track_maps_3d'] = all_maps_3d
        results['all_track_maps_2d'] = all_maps_2d
        results['all_track_maps_pers'] = all_maps_pers
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False


    
            
        #     print('tracking on frame %d' % s)
        #     # remake the vox util and all the mem data
        #     self.scene_centroid = utils_geom.get_clist_from_lrtlist(lrt_camXAIs[:,s-1:s])[:,0]
        #     delta = self.scene_centroid - self.original_centroid
        #     self.vox_util = vox_util.Vox_util(self.Z, self.Y, self.X, 
        #         self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        #     self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        #     self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))

        #     self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
        #         __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        #     self.unp_memX0s = self.vox_util.apply_4x4s_to_voxs(self.camX0s_T_camXs, self.unp_memXs)
        #     self.summ_writer.summ_occ('track/reloc_occ_%d' % s, self.occ_memX0s[:,s])
        #     print('scene centroid:', self.scene_centroid.detach().cpu().numpy())
        # return 
        #     # inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z4, self.Y4, self.X, already_mem=False))
        #     inb = self.vox_util.get_inbounds(self.xyz_camX0s[:,s], self.Z4, self.Y4, self.X4, already_mem=False)
        #     num_inb = torch.sum(inb.float(), axis=1)
        #     # print('num_inb', num_inb, num_inb.shape)
        #     # num_inb = torch.sum(self.occ_memX0s.float(), axis=[2, 3, 4])
        #     inb_counts[:, s] = num_inb.cpu().numpy()

        #     feat_memI_input = torch.cat([
        #         self.occ_memX0s[:,s],
        #         self.unp_memX0s[:,s]*self.occ_memX0s[:,s],
        #     ], dim=1)
        #     _, feat_memI, valid_memI = self.featnet3D(feat_memI_input)

        #     self.summ_writer.summ_feat('3D_feats/feat_%d_input' % s, feat_memI_input, pca=True)
        #     self.summ_writer.summ_feat('3D_feats/feat_%d' % s, feat_memI, pca=True)


        # for s in range(self.S):
        #     self.summ_writer.summ_scalar('track/mean_iou_%02d' % s, torch.mean(ious[:,s]).cpu().item())

        # self.summ_writer.summ_scalar('track/mean_iou', torch.mean(ious).cpu().item())
        # self.summ_writer.summ_scalar('track/point_counts', np.mean(point_counts))
        # # self.summ_writer.summ_scalar('track/inb_counts', torch.mean(inb_counts).cpu().item())
        # self.summ_writer.summ_scalar('track/inb_counts', np.mean(inb_counts))

        # lrt_camX0s_e = lrt_camIs_e.clone()
        # lrt_camXs_e = utils_geom.apply_4x4s_to_lrts(self.camXs_T_camX0s, lrt_camX0s_e)

        # if self.include_vis:
        #     visX_e = []
        #     for s in list(range(self.S)):
        #         visX_e.append(self.summ_writer.summ_lrtlist(
        #             'track/box_camX%d_e' % s, self.rgb_camXs[:,s], lrt_camXs_e[:,s:s+1],
        #             self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
        #     self.summ_writer.summ_rgbs('track/box_camXs_e', visX_e)
        #     visX_g = []
        #     for s in list(range(self.S)):
        #         visX_g.append(self.summ_writer.summ_lrtlist(
        #             'track/box_camX%d_g' % s, self.rgb_camXs[:,s], self.lrt_camXs[:,s:s+1],
        #             self.score_s[:,s:s+1], self.tid_s[:,s:s+1], self.pix_T_cams[:,0], only_return=True))
        #     self.summ_writer.summ_rgbs('track/box_camXs_g', visX_g)


        # dists = torch.norm(obj_clist_camX0_e - self.obj_clist_camX0, dim=2)
        # # this is B x S
        # mean_dist = utils_basic.reduce_masked_mean(dists, self.score_s)
        # median_dist = utils_basic.reduce_masked_median(dists, self.score_s)
        # # this is []
        # self.summ_writer.summ_scalar('track/centroid_dist_mean', mean_dist.cpu().item())
        # self.summ_writer.summ_scalar('track/centroid_dist_median', median_dist.cpu().item())

        # return lrt_camXs_e
    
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
                return self.run_explain(feed)

        # # arriving at this line is bad
        # print('weird set_name:', set_name)

        # assert(False)
