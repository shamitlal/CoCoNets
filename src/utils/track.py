import torch
import numpy as np
import utils.geom
import utils.basic
# import utils.vox
import utils.samp
import utils.misc
import torch.nn.functional as F
import sklearn

np.set_printoptions(suppress=True, precision=6, threshold=2000)

def merge_rt_py(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def split_rt_py(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def apply_4x4_py(rt, xyz):
    # rt is 4 x 4
    # xyz is N x 3
    r, t = split_rt_py(rt)
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x 3 x N
    xyz = np.dot(r, xyz)
    # xyz is xyz1 x 3 x N
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x N x 3
    t = np.reshape(t, [1, 3])
    xyz = xyz + t
    return xyz

def rigid_transform_3d(xyz_cam0, xyz_cam1, do_ransac=True):
    # inputs are N x 3
    xyz_cam0 = xyz_cam0.detach().cpu().numpy()
    xyz_cam1 = xyz_cam1.detach().cpu().numpy()
    cam1_T_cam0 = rigid_transform_3d_py(xyz_cam0, xyz_cam1, do_ransac=do_ransac)
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).float().to('cuda')
    return cam1_T_cam0

def differentiable_rigid_transform_3d(xyz_cam0, xyz_cam1):
    # inputs are N x 3
    # xyz0 and xyz1 are each N x 3
    assert len(xyz_cam0) == len(xyz_cam1)

    N = xyz_cam0.shape[0] # total points
    assert(N > 8)

    # nPts = 8 # anything >3 is ok really
    # if N <= nPts:
    #     print('N = %d; returning an identity mat' % N)
    #     R = np.eye(3, dtype=np.float32)
    #     t = np.zeros(3, dtype=np.float32)
    #     cam1_T_cam0 = merge_rt_py(R, t)
    # else:
    
    cam1_T_cam0 = rigid_transform_3d_pt_helper(xyz_cam0, xyz_cam1)
    return cam1_T_cam0

    
    cam1_T_cam0 = rigid_transform_3d_pt(xyz_cam0, xyz_cam1, do_ransac=do_ransac)
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).float().to('cuda')
    return cam1_T_cam0

def rigid_transform_3d_py_helper(xyz0, xyz1):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    if N > 3:
        centroid_xyz0 = np.mean(xyz0, axis=0)
        centroid_xyz1 = np.mean(xyz1, axis=0)
        # print('centroid_xyz0', centroid_xyz0)
        # print('centroid_xyz1', centroid_xyz1)

        # center the points
        xyz0 = xyz0 - np.tile(centroid_xyz0, (N, 1))
        xyz1 = xyz1 - np.tile(centroid_xyz1, (N, 1))

        H = np.dot(xyz0.T, xyz1)

        U, S, Vt = np.linalg.svd(H)

        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = np.dot(Vt.T, U.T)

        t = np.dot(-R, centroid_xyz0.T) + centroid_xyz1.T
        t = np.reshape(t, [3])
    else:
        print('too few points; returning identity')
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
    rt = merge_rt_py(R, t)
    return rt

def rigid_transform_3d_pt_helper(xyz0, xyz1):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    assert(N > 3)
    centroid_xyz0 = torch.mean(xyz0, axis=0, keepdim=True)
    centroid_xyz1 = torch.mean(xyz1, axis=0, keepdim=True)

    # center the points
    xyz0 = xyz0 - centroid_xyz0.repeat(N, 1)
    xyz1 = xyz1 - centroid_xyz1.repeat(N, 1)

    H = torch.matmul(xyz0.transpose(0,1), xyz1)

    U, S, Vt = torch.svd(H)

    R = torch.matmul(Vt.transpose(0,1), U.transpose(0,1))

    # # special reflection case
    # if np.linalg.det(R) < 0:
    #    Vt[2,:] *= -1
    #    R = np.dot(Vt.transpose(0,1), U.transpose(0,1))

    t = torch.matmul(-R, centroid_xyz0.transpose(0,1)) + centroid_xyz1.transpose(0,1)
    t = t.reshape([3])

    rt = utils.geom.merge_rt_single(R, t)
    return rt

def rigid_transform_3d_py(xyz0, xyz1, do_ransac=True):
    # xyz0 and xyz1 are each N x 3
    assert len(xyz0) == len(xyz1)

    N = xyz0.shape[0] # total points

    nPts = 8 # anything >3 is ok really
    if N <= nPts:
        print('N = %d; returning an identity mat' % N)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        rt = merge_rt_py(R, t)
    elif not do_ransac:
        rt = rigid_transform_3d_py_helper(xyz0, xyz1)
    else:
        # print('N = %d' % N)
        # print('doing ransac')
        rts = []
        errs = []
        ransac_steps = 256
        for step in list(range(ransac_steps)):
            assert(N > nPts) 
            perm = np.random.permutation(N)
            cam1_T_cam0 = rigid_transform_3d_py_helper(xyz0[perm[:nPts]], xyz1[perm[:nPts]])
            # i got some errors in matmul when the arrays were too big, 
            # so let's just use 1k points for the error 
            perm = np.random.permutation(N)
            xyz1_prime = apply_4x4_py(cam1_T_cam0, xyz0[perm[:min([1000,N])]])
            xyz1_actual = xyz1[perm[:min([1000,N])]]
            err = np.mean(np.sum(np.abs(xyz1_prime-xyz1_actual), axis=1))
            rts.append(cam1_T_cam0)
            errs.append(err)
        ind = np.argmin(errs)
        rt = rts[ind]
    return rt

def compute_mem1_T_mem0_from_object_flow(flow_mem, mask_mem, occ_mem):
    B, C, Z, Y, X = list(flow_mem.shape)
    assert(C==3)
    mem1_T_mem0 = utils.geom.eye_4x4(B)

    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
    
    for b in list(range(B)):
        # i think there is a way to parallelize the where/gather but it is beyond me right now
        occ = occ_mem[b]
        mask = mask_mem[b]
        flow = flow_mem[b]
        xyz0 = xyz_mem0[b]
        # cam_T_obj = camR_T_obj[b]
        # mem_T_cam = mem_T_ref[b]

        flow = flow.reshape(3, -1).permute(1, 0)
        # flow is -1 x 3
        inds = torch.where((occ*mask).reshape(-1) > 0.5)
        # inds is ?
        flow = flow[inds]

        xyz0 = xyz0[inds]
        xyz1 = xyz0 + flow

        mem1_T_mem0_ = rigid_transform_3d(xyz0, xyz1)
        # this is 4 x 4 
        mem1_T_mem0[b] = mem1_T_mem0_

    return mem1_T_mem0

def compute_mem1_T_mem0_from_object_flow(flow_mem, mask_mem, occ_mem):
    B, C, Z, Y, X = list(flow_mem.shape)
    assert(C==3)
    mem1_T_mem0 = utils.geom.eye_4x4(B)

    xyz_mem0 = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
    
    for b in list(range(B)):
        # i think there is a way to parallelize the where/gather but it is beyond me right now
        occ = occ_mem[b]
        mask = mask_mem[b]
        flow = flow_mem[b]
        xyz0 = xyz_mem0[b]
        # cam_T_obj = camR_T_obj[b]
        # mem_T_cam = mem_T_ref[b]

        flow = flow.reshape(3, -1).permute(1, 0)
        # flow is -1 x 3
        inds = torch.where((occ*mask).reshape(-1) > 0.5)
        # inds is ?
        flow = flow[inds]

        xyz0 = xyz0[inds]
        xyz1 = xyz0 + flow

        mem1_T_mem0_ = rigid_transform_3d(xyz0, xyz1)
        # this is 4 x 4 
        mem1_T_mem0[b] = mem1_T_mem0_

    return mem1_T_mem0


def track_via_chained_flows(
        lrt_camIs_g,
        mask_mem0,
        model,
        occ_mems,
        occ_mems_half,
        unp_mems,
        summ_writer,
        include_image_summs=False,
        use_live_nets=False,
):
    B, S, _, Z, Y, X = list(occ_mems.shape)
    B, S, _, Z2, Y2, X2 = list(occ_mems_half.shape)
    
    flow_mem0 = torch.zeros(B, 3, Z2, Y2, X2, dtype=torch.float32, device=torch.device('cuda'))
    cam0_T_camI = utils.geom.eye_4x4(B)

    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    # this is B x S x 4 x 4

    cam0_T_obj = cams_T_obj0[:,0]
    obj_length = obj_lengths[:,0]

    occ_mem0 = occ_mems_half[:,0]

    input_mems = torch.cat([occ_mems, occ_mems*unp_mems], dim=2)

    mem_T_cam = utils.vox.get_mem_T_ref(B, Z2, Y2, X2)
    cam_T_mem = utils.vox.get_ref_T_mem(B, Z2, Y2, X2)

    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    lrt_camIs_e[:,0] = lrt_camIs_g[:,0] # init with gt box on frame0

    all_ious = []
    for s in list(range(1, S)):
        input_mem0 = input_mems[:,0]
        input_memI = input_mems[:,s]

        if use_live_nets:

            use_rigid_warp = True
            if use_rigid_warp:
                xyz_camI = model.xyz_camX0s[:,s]
                xyz_camI = utils.geom.apply_4x4(cam0_T_camI, xyz_camI)
                occ_memI = utils.vox.voxelize_xyz(xyz_camI, Z, Y, X)
                unp_memI = unp_mems[:,s]
                unp_memI = utils.vox.apply_4x4_to_vox(cam0_T_camI, unp_memI, already_mem=False, binary_feat=False)
                input_memI = torch.cat([occ_memI, occ_memI*unp_memI], dim=1)
                # input_memI = utils.vox.apply_4x4_to_vox(cam0_T_camI, input_memI, already_mem=False, binary_feat=False)
            else:
                input_memI = utils.samp.backwarp_using_3d_flow(input_memI, F.interpolate(flow_mem0, scale_factor=2, mode='trilinear'))
                
            featnet_output_mem0, _, _ = model.featnet(input_mem0, None, None)
            featnet_output_memI, _, _ = model.featnet(input_memI, None, None)
            _, residual_flow_mem0 = model.flownet(
                    featnet_output_mem0,
                    featnet_output_memI,
                    torch.zeros([B, 3, Z2, Y2, X2]).float().cuda(),
                    occ_mem0, 
                    False,
                    None)
        else:
            featnet_output_mem0 = model.feat_net.infer_pt(input_mem0)
            featnet_output_memI = model.feat_net.infer_pt(input_memI)
            featnet_output_memI = utils.samp.backwarp_using_3d_flow(featnet_output_memI, flow_mem0)
            residual_flow_mem0 = model.flow_net.infer_pt([featnet_output_mem0,
                                                          featnet_output_memI])

        # if use_live_nets:
        #     _, residual_flow_mem0 = model.flownet(
        #             featnet_output_mem0,
        #             featnet_output_memI,
        #             torch.zeros([B, 3, Z2, Y2, X2]).float().cuda(),
        #             occ_mem0, 
        #             False,
        #             summ_writer)
        # else:
        #     residual_flow_mem0 = model.flow_net.infer_pt([featnet_output_mem0,
        #                                                  featnet_output_memI])

        flow_mem0 = flow_mem0 + residual_flow_mem0

        if include_image_summs:
            summ_writer.summ_feats('3d_feats/featnet_inputs_%02d' % s, [input_mem0, input_memI], pca=True)
            summ_writer.summ_feats('3d_feats/featnet_outputs_warped_%02d' % s, [featnet_output_mem0, featnet_output_memI], pca=True)
            summ_writer.summ_3d_flow('flow/residual_flow_mem0_%02d' % s, residual_flow_mem0, clip=0.0)
            summ_writer.summ_3d_flow('flow/residual_masked_flow_mem0_%02d' % s, residual_flow_mem0*mask_mem0, clip=0.0)
            summ_writer.summ_3d_flow('flow/flow_mem0_%02d' % s, flow_mem0, clip=0.0)

        # compute the rigid motion of the object; we will use this for eval
        memI_T_mem0 = compute_mem1_T_mem0_from_object_flow(
            flow_mem0, mask_mem0, occ_mem0)
        mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
        cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

        # eval
        camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
        # this is B x 4 x 4

        lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
        ious = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1],
                                                             lrt_camIs_g[:,s:s+1])
        all_ious.append(ious)
        summ_writer.summ_scalar('box/mean_iou_%02d' % s, torch.mean(ious).cpu().item())

    # lrt_camIs_e is B x S x 19
    # this is B x S x 1 x 19
    return lrt_camIs_e, all_ious

def cross_corr_with_template(search_region, template):
    B, C, ZZ, ZY, ZX = list(template.shape)
    B2, C2, Z, Y, X = list(search_region.shape)
    assert(B==B2)
    assert(C==C2)
    corr = []

    Z_new = Z-ZZ+1
    Y_new = Y-ZY+1
    X_new = X-ZX+1
    corr = torch.zeros([B, 1, Z_new, Y_new, X_new]).float().cuda()

    # this loop over batch is ~2x faster than the grouped version
    for b in list(range(B)):
        search_region_b = search_region[b:b+1]
        template_b = template[b:b+1]
        corr[b] = F.conv3d(search_region_b, template_b).squeeze(0)

    # grouped version, for reference:
    # corr = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B) # fast valid conv
    
    # adjust the scale of responses, for stability early on
    corr = 0.001 * corr

    # since we did valid conv (which is smart), the corr map is offset from the search region
    # so we need to offset the xyz of the answer
    # _, _, Z_new, Y_new, X_new = list(corr.shape)
    Z_clipped = (Z - Z_new)/2.0
    Y_clipped = (Y - Y_new)/2.0
    X_clipped = (X - X_new)/2.0
    xyz_offset = np.array([X_clipped, Y_clipped, Z_clipped], np.float32).reshape([1, 3])
    xyz_offset = torch.from_numpy(xyz_offset).float().to('cuda')
    return corr, xyz_offset

def cross_corr_with_templates(search_region, templates):
    B, C, Z, Y, X = list(search_region.shape)
    B2, N, C2, ZZ, ZY, ZX = list(templates.shape)
    assert(B==B2)
    assert(C==C2)

    Z_new = Z-ZZ+1
    Y_new = Y-ZY+1
    X_new = X-ZX+1
    corr = torch.zeros([B, N, Z_new, Y_new, X_new]).float().cuda()

    # this loop over batch is ~2x faster than the grouped version
    for b in list(range(B)):
        search_region_b = search_region[b:b+1]
        for n in list(range(N)):
            template_b = templates[b:b+1,n]
            corr[b,n] = F.conv3d(search_region_b, template_b).squeeze(0)

    # grouped version, for reference:
    # corr = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B) # fast valid conv
    
    # adjust the scale of responses, for stability early on
    corr = 0.001 * corr

    # since we did valid conv (which is smart), the corr map is offset from the search region
    # so we need to offset the xyz of the answer
    # _, _, Z_new, Y_new, X_new = list(corr.shape)
    Z_clipped = (Z - Z_new)/2.0
    Y_clipped = (Y - Y_new)/2.0
    X_clipped = (X - X_new)/2.0
    xyz_offset = np.array([X_clipped, Y_clipped, Z_clipped], np.float32).reshape([1, 3])
    xyz_offset = torch.from_numpy(xyz_offset).float().to('cuda')
    return corr, xyz_offset

def track_via_inner_products(lrt_camIs_g, mask_mems, feat_mems, vox_util, mask_boxes=False, summ_writer=None):
    B, S, feat3d_dim, Z, Y, X = list(feat_mems.shape)

    feat_vecs = feat_mems.view(B, S, feat3d_dim, -1)
    # this is B x S x C x huge

    feat0_vec = feat_vecs[:,0]
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C

    obj_mask0_vec = mask_mems[:,0].reshape(B, -1).round()
    # this is B x huge

    orig_xyz = utils.basic.gridcloud3d(B, Z, Y, X)
    # this is B x huge x 3

    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    obj_length = obj_lengths[:,0]
    # this is B x S x 4 x 4

    # this is B x S x 4 x 4
    cam0_T_obj = cams_T_obj0[:,0]

    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    # we will fill this up

    mask_e_mems = torch.zeros_like(mask_mems)
    mask_e_mems_masked = torch.zeros_like(mask_mems)

    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)

    ious = torch.zeros([B, S]).float().cuda()
    point_counts = np.zeros([B, S])
    for s in list(range(S)):
        feat_vec = feat_vecs[:,s]
        feat_vec = feat_vec.permute(0, 2, 1)
        # B x huge x C

        memI_T_mem0 = utils.geom.eye_4x4(B)
        # we will fill this up

        # Use ground truth box to mask
        if s == 0:
            lrt = lrt_camIs_g[:,0].unsqueeze(1)
        # Use predicted box to mask
        else:
            lrt = lrt_camIs_e[:,s-1].unsqueeze(1)

        # Equal box length
        lrt[:,:,:3] = torch.ones_like(lrt[:,:,:3])*10

        # Remove rotation
        transform = lrt[:,:,3:].reshape(B, 1, 4, 4)
        transform[:,:,:3,:3] = torch.eye(3).unsqueeze(0).unsqueeze(0)
        transform = transform.reshape(B,-1)

        lrt[:,:,3:] = transform

        box_mask = vox_util.assemble_padded_obj_masklist(lrt, torch.ones(1,1).cuda(), Z, Y, X)

        # to simplify the impl, we will iterate over the batch dim
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

            corr_b = torch.matmul(feat_vec_b, obj_vec_b)
            # this is huge x med

            heat_b = corr_b.permute(1, 0).reshape(-1, 1, Z, Y, X)
            # this is med x 1 x Z4 x Y4 x X4

            
            # Mask by box to restrict area
            if mask_boxes:
                
                # Vanilla heatmap
                if summ_writer != None:
                    heat_map = heat_b.max(0)[0]
                    summ_writer.summ_feat("heatmap/vanilla", heat_map.unsqueeze(0), pca=False)
                    mask_e_mems[b,s] = heat_map
                
                box_mask = box_mask.squeeze(0).repeat(heat_b.shape[0],1,1,1,1)
                heat_b = heat_b*box_mask
               
                # Masked heatmap
                if summ_writer != None:
                    heat_map = heat_b.max(0)[0]
                    mask_e_mems_masked[b,s] = heat_map

                    summ_writer.summ_feat("heatmap/masked", heat_map.unsqueeze(0), pca=False)
            
            
            # for numerical stability, we sub the max, and mult by the resolution
            heat_b_ = heat_b.reshape(-1, Z*Y*X)
            heat_b_max = (torch.max(heat_b_, dim=1).values).reshape(-1, 1, 1, 1, 1)
            heat_b = heat_b - heat_b_max
            heat_b = heat_b * float(len(heat_b[0].reshape(-1)))

            xyzI = utils.basic.argmax3d(heat_b*float(Z*10), hard=False, stack=True)
            # this is med x 3
            memI_T_mem0[b] = rigid_transform_3d(xyz0, xyzI)

            # record #points, since ransac depends on this
            point_counts[b, s] = len(xyz0)
        # done stepping through batch

        mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
        cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)

        # eval
        camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
        # this is B x 4 x 4
        lrt_camIs_e[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
        ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)

    if summ_writer != None:
        summ_writer.summ_feats('heatmap/mask_e_memX0s', torch.unbind(mask_e_mems, dim=1), pca=False)
        summ_writer.summ_feats('heatmap/mask_e_memX0s_masked', torch.unbind(mask_e_mems_masked, dim=1), pca=False)

    return lrt_camIs_e, point_counts, ious

                             
def convert_corr_to_xyz(corr, xyz_offset, hard=True):
    # corr is B x 1 x Z x Y x X
    # xyz_offset is 1 x 3
    peak_z, peak_y, peak_x = utils.basic.argmax3d(corr, hard=hard)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_xyz_search

def convert_corrlist_to_xyzr(corrlist, radlist, xyz_offset, hard=True):
    # corrlist is a list of N different B x Z x Y x X tensors
    # radlist is N angles in radians
    # xyz_offset is 1 x 3
    corrcat = torch.stack(corrlist, dim=1)
    # this is B x N x Z x Y x X
    radcat = torch.from_numpy(np.array(radlist).astype(np.float32)).cuda()
    radcat = radcat.reshape(-1)
    # this is N
    peak_r, peak_z, peak_y, peak_x = utils.basic.argmax3dr(corrcat, radcat, hard=hard)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_r, peak_xyz_search

def convert_corrs_to_xyzr(corrcat, radcat, xyz_offset, hard=True, grid=None):
    # corrcat is B x N x Z x Y x X tensors
    # radcat is N
    # xyz_offset is 1 x 3
    # if grid is None we'll compute it during the argmax
    peak_r, peak_z, peak_y, peak_x = utils.basic.argmax3dr(corrcat, radcat, hard=hard, grid=grid)
    # these are B
    peak_xyz_corr = torch.stack([peak_x, peak_y, peak_z], dim=1)
    # this is B x 3, and in corr coords
    peak_xyz_search = xyz_offset + peak_xyz_corr
    # this is B x 3, and in search coords
    return peak_r, peak_xyz_search

def remask_via_inner_products(lrt_camIs_g, mask_mems, feat_mems, vox_util, mask_distance=False, summ_writer=None):
    B, S, feat3d_dim, Z, Y, X = list(feat_mems.shape)

    mask = mask_mems[:,0]
    distance_masks = torch.zeros_like(mask_mems)
    
    feat_vecs = feat_mems.view(B, S, feat3d_dim, -1)
    # this is B x S x C x huge

    feat0_vec = feat_vecs[:,0]
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C
    
    obj_mask0_vec = mask_mems[:,0].reshape(B, -1).round()
    # this is B x huge
    
    orig_xyz = utils.basic.gridcloud3d(B, Z, Y, X)
    # this is B x huge x 3
    
    obj_lengths, cams_T_obj0 = utils.geom.split_lrtlist(lrt_camIs_g)
    obj_length = obj_lengths[:,0]
    # this is B x S x 4 x 4
    
    # this is B x S x 4 x 4
    cam0_T_obj = cams_T_obj0[:,0]
    
    lrt_camIs_e = torch.zeros_like(lrt_camIs_g)
    # we will fill this up

    mem_T_cam = vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = vox_util.get_ref_T_mem(B, Z, Y, X)

    mask_e_mems = torch.zeros_like(mask_mems)
    mask_e_mems_thres = torch.zeros_like(mask_mems)
    mask_e_mems_hard  = torch.zeros_like(mask_mems)
    mask_e_mems_spatial = torch.zeros_like(mask_mems)

    ious = torch.zeros([B, S]).float().cuda()
    ious_hard = torch.zeros([B, S]).float().cuda()
    ious_spatial = torch.zeros([B, S]).float().cuda()

    point_counts = np.zeros([B, S])
    rough_centroids_mem = torch.zeros(B, S, 3).float().cuda()
    for s in list(range(S)):
        feat_vec = feat_vecs[:,s]
        feat_vec = feat_vec.permute(0, 2, 1)
        # B x huge x C

        memI_T_mem0 = utils.geom.eye_4x4(B)
        # we will fill this up
        # to simplify the impl, we will iterate over the batchmin
        
        for b in list(range(B)):
            # Expand mask
            # Code for growing a distance constraint mask
            if s == 0:
                 distance_mask = mask_mems[b,s]
                 distance_masks[b,s] = distance_mask
            else:
                 distance_mask = torch.nn.functional.conv3d(distance_masks[b,s-1].unsqueeze(0), torch.ones(1,1,3,3,3).cuda(), padding=1)
                 distance_mask = (distance_mask > 0).float()
                 distance_masks[b,s] = distance_mask

            distance_mask = distance_mask.reshape(X*Y*Z)

            feat_vec_b  = feat_vec[b]
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
            
            # Calculate similarities
            similarity_b = torch.exp(torch.matmul(feat_vec_b, obj_vec_b))
            
            # Remove entries which could never happen
            if mask_distance == True:
                similarity_b = torch.mul(distance_mask.repeat(similarity_b.shape[1],1).permute(1,0),similarity_b)
            
            # calculate attention
            similarity_b = similarity_b/torch.sum(similarity_b, dim=0, keepdim=True)    
            
            num_mask_channels = similarity_b.shape[1]
            
            # Calculate hard attention
            similarity_argmax = similarity_b.max(0)[1]
            hard_attention = torch.zeros_like(similarity_b)
            
            for i in range(num_mask_channels):
                hard_attention[similarity_argmax[i], i] = 1
    
            # Calculate positional average attention
            spatial_attention = hard_attention.permute(1,0)

            grid = utils.basic.gridcloud3d(1,Z,Y,X)
            
            pos_average = torch.zeros(3)
            
            spatial_attention_mask = torch.zeros(Z,Y,X)
            for i in range(num_mask_channels):
                weighted_grid = torch.mul(grid.squeeze(0), spatial_attention[i].unsqueeze(1))
                grid_average  = torch.sum(weighted_grid, dim=0)
                grid_average  = torch.round(grid_average)
                spatial_attention_mask[list(grid_average.long())] = 1
                #pos_average = pos_average + grid_average
                #import ipdb; ipdb.set_trace()

            #pos_average = pos_average/num_mask_channels
            #import ipdb; ipdb.set_trace()

            # this is huge x med normalized
            
            obj_mask0_vec_b = obj_mask0_vec[b]
            values = obj_mask0_vec_b[obj_inds_b].unsqueeze(1)
            #this is med x C

            # Propagated values are 1
            mask_e_mem = torch.matmul(similarity_b,values)
            mask_e_mem_hard = torch.matmul(hard_attention,values)
            # this is huge x 1
            
            # Threshold probabilities to be 1 close to the max
            mask_e_mem_t = (mask_e_mem > (mask_e_mem.mean()*0.3 + mask_e_mem.max()*0.7)).float()
            # this is huge x 1

            # Constrain to search region
            #mask_e_mem = torch.mul(distance_mask,mask_e_mem.reshape(-1))

            mask_e_mems[b,s] = mask_e_mem.reshape(1,Z,Y,X)
            mask_e_mems_thres[b,s] = mask_e_mem_t.reshape(1,Z,Y,X)
            mask_e_mems_hard[b,s]  = mask_e_mem_hard.reshape(1,Z,Y,X)
            mask_e_mems_spatial[b,s] = spatial_attention_mask.reshape(1,Z,Y,X)

            set_A = mask_mems[b,s].reshape(Z*Y*X).bool()
            set_B = mask_e_mem_t.reshape(Z*Y*X).bool()

            iou = sklearn.metrics.jaccard_score(set_A.bool().cpu().data.numpy(), set_B.bool().cpu().data.numpy(), average='binary')
            ious[b,s] = iou

            iou_hard = sklearn.metrics.jaccard_score(mask_mems[b,s].reshape(Z*Y*X).bool().bool().cpu().data.numpy(), mask_e_mem_hard.reshape(Z*Y*X).bool().cpu().data.numpy(), average='binary') 
            ious_hard[b,s] = iou_hard

            iou_spatial = sklearn.metrics.jaccard_score(mask_mems[b,s].reshape(Z*Y*X).bool().bool().cpu().data.numpy(), spatial_attention_mask.reshape(Z*Y*X).bool().cpu().data.numpy(), average='binary') 
            ious_spatial[b,s] = iou_spatial


    # Visualization Logs
    if summ_writer != None:
        summ_writer.summ_feats('track/mask_e_memX0s', torch.unbind(mask_e_mems, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_t', torch.unbind(mask_e_mems_thres, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_h', torch.unbind(mask_e_mems_hard, dim=1), pca=False)
        summ_writer.summ_feats('track/mask_e_memX0s_s', torch.unbind(mask_e_mems_spatial, dim=1), pca=False)

        for s in range(S):
            summ_writer.summ_scalar('track/mean_iou_hard_%02d' % s, torch.mean(ious_hard[:,s]).cpu().item())
            summ_writer.summ_scalar('track/mean_iou_spatial_%02d' % s, torch.mean(ious_spatial[:,s]).cpu().item())

    return ious
    
def track_one_step_via_inner_product(
        B, C,
        Z_, Y_, X_,
        lrt_camXAI,
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

    lrt_curr = lrt_camXAI.clone()
    _, camI_T_obj = utils.geom.split_lrt(lrt_curr)
    memI_T_mem0 = utils.basic.matmul4(mem_T_cam, camI_T_obj, obj_T_cam0, cam_T_mem)
    mem0_T_memI = memI_T_mem0.inverse()
    # mem0_T_memI = utils.basic.matmul4(mem_T_cam, cam0_T_obj, camI_T_obj.inverse(), cam_T_mem)

    feat0_map = feat0_vec.permute(0, 2, 1).reshape(B, -1, Z_, Y_, X_)
    featI_map = featI_vec.permute(0, 2, 1).reshape(B, -1, Z_, Y_, X_)

    xyzI_prior_full = utils.geom.apply_4x4(memI_T_mem0, orig_xyz)
    
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

        xyzI = utils.basic.argmax3D(heat_b, hard=True, stack=True)
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
            correspI = utils.samp.bilinear_sample3D(featI_map_b.unsqueeze(0), xyzI.unsqueeze(0) - crop_vec).squeeze(0)
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
            reverse_xyzI = utils.basic.argmax3D(reverse_heat_b, hard=False, stack=True)
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
        #         # utils.basic.print_stats('cycle dist %d' % cyc, dist)
        #         heat_b = zero_out_bad_peaks(xyzI, dist, heat_b, thresh=thresh)
        #     # note i recently switched to taking a hard argmax here
        #     xyzI = utils.basic.argmax3D(heat_b, hard=True, stack=True)

        # we need to get to cam coordinates to cancel the scene centroid delta
        xyzI_cam = vox_util.Mem2Ref(xyzI.unsqueeze(1), Z, Y, X)
        xyzI_cam += delta
        xyzI = vox_util.Ref2Mem(xyzI_cam, Z, Y, X).squeeze(1)

        memI_T_mem0[b] = utils.track.rigid_transform_3D(xyz0, xyzI)

        # record #points, since ransac depends on this
        # point_counts[b, s] = len(xyz0)
    # done stepping through batch

    mem0_T_memI = utils.geom.safe_inverse(memI_T_mem0)
    cam0_T_camI = utils.basic.matmul3(cam_T_mem, mem0_T_memI, mem_T_cam)
    # mem0_T_memIs_e[:,s] = mem0_T_memI

    # eval
    camI_T_obj = utils.basic.matmul4(cam_T_mem, memI_T_mem0, mem_T_cam, cam0_T_obj)
    # this is B x 4 x 4

    new_lrt_camXAI = utils.geom.merge_lrt(obj_length, camI_T_obj)
    score = torch.ones_like(lrt_camXAI[:,0])
    return new_lrt_camXAI, score, mem0_T_memI
    # lrt_camXAIs[:,s] = utils.geom.merge_lrt(obj_length, camI_T_obj)
    # # ious[:,s] = utils.geom.get_iou_from_corresponded_lrtlists(lrt_camIs_e[:,s:s+1], lrt_camIs_g[:,s:s+1]).squeeze(1)

def track_proposal(B, S, 
                   lrt_camXAI,
                   pix_T_cams,
                   rgb_camXs,
                   xyz_camXAs,
                   camXAs_T_camXs,
                   featnet3d,
                   scene_vox_util,
                   cropper,
                   padder,
                   crop_zyx,
                   super_iter=None,
                   summ_writer=None,
):
    # lrt is B x 19

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    original_centroid = utils.geom.get_clist_from_lrtlist(lrt_camXAI.unsqueeze(1)).squeeze(1)
    Z_zoom, Y_zoom, X_zoom = hyp.Z_zoom, hyp.Y_zoom, hyp.X_zoom
    orig_vox_util = vox_util.Vox_util(Z_zoom, Y_zoom, X_zoom, 'zoom', scene_centroid=original_centroid, assert_cube=True)

    rgb_memXII = orig_vox_util.unproject_rgb_to_mem(
        rgb_camXs[:,I], Z_zoom, Y_zoom, X_zoom, pix_T_cams[:,I])
    rgb_memXAI = orig_vox_util.apply_4x4_to_vox(camXAs_T_camXs[:,I], rgb_memXII)
    occ_memXAI = orig_vox_util.voxelize_xyz(xyz_camXAs[:,I], Z_zoom, Y_zoom, X_zoom)
    feat_memXAI_input = torch.cat([
        occ_memXAI, rgb_memXAI*occ_memXAI,
    ], dim=1)
    _, feat_memXAI, _ = featnet3d(feat_memXAI_input)

    B, C, Z_, Y_, X_ = list(feat_memXAI.shape)
    Z_pad, Y_pad, X_pad = crop_zyx
    Z = Z_ + Z_pad*2
    Y = Y_ + Y_pad*2
    X = X_ + X_pad*2

    occ_memXAI = orig_vox_util.voxelize_xyz(xyz_camXAs[:,I], Z, Y, X)

    lrt_camXAI_ = lrt_camXAI.unsqueeze(1)
    score_ = torch.ones_like(lrt_camXAI_[:,:,0])

    obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
        lrt_camXAI_, score_, Z, Y, X).squeeze(1)
    occ_obj_mask_memXAI = obj_mask_memXAI * occ_memXAI
    occ_obj_mask_memXAI = cropper(occ_obj_mask_memXAI)

    if torch.sum(occ_obj_mask_memXAI) < 8:
        print('using the full mask instead of just occ')
        # still too small!
        # use the full mask
        obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
            lrt_camXAI_, score_, Z, Y, X).squeeze(1)
        # pad by 1m on each side
        occ_obj_mask_memXAI = cropper(obj_mask_memXAI)

    # i need the feats of this object
    # let's use 0 to mean source/original, instead of AI
    feat0_vec = feat_memXAI.reshape(B, C, -1)
    # this is B x C x huge
    feat0_vec = feat0_vec.permute(0, 2, 1)
    # this is B x huge x C

    obj_mask0_vec = occ_obj_mask_memXAI.reshape(B, -1).round()
    # this is B x huge

    orig_xyz = utils.basic.gridcloud3D(B, Z, Y, X)
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

    obj_length_, camXAI_T_obj_ = utils.geom.split_lrtlist(lrt_camXAI_)
    obj_length = obj_length_[:,0]
    cam0_T_obj = camXAI_T_obj_[:,0]

    mem_T_cam = orig_vox_util.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = orig_vox_util.get_ref_T_mem(B, Z, Y, X)

    lrt_camXAIs = lrt_camXAI_.repeat(1, S, 1)
    scores = torch.ones_like(lrt_camXAIs[:,:,0])
    # now we need to write the non-I timesteps
    mem0_T_memIs_e = torch.zeros((B, S, 4, 4), dtype=torch.float32).cuda()

    clist = utils.geom.get_clist_from_lrtlist(lrt_camXAIs)
    havelist = torch.zeros_like(clist[:,:,0])
    havelist[:,I] = 1.0
    
    # first let's go forward
    for s in list(range(S)):
        
        prev_lrt = lrt_camXAIs[:,s-1]
        # if s==I+1 or s==1:
        #     # s-2 does not exist, so:
        #     prevprev_lrt = lrt_camXAIs[:,s-1]
        # else:
        #     prevprev_lrt = lrt_camXAIs[:,s-2]
        # print('using', s-1, s-2, 'to form the prior')

        clist = utils.geom.get_clist_from_lrtlist(lrt_camXAIs)
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
        #     clist = utils.geom.get_clist_from_lrtlist(lrt_camXAIs)

        #     # this is B x S x 3
            
        #     clist_a = clist[:,I+1:s]
        #     clist_b = clist[:,I:s-1]
        #     # print('clist_a', clist_a.detach().cpu().numpy())
        #     # print('clist_b', clist_b.detach().cpu().numpy())
        #     vel = torch.mean(clist_a-clist_b, dim=1)
        #     # print('mean vel', vel.detach().cpu().numpy())
            
        # # clist = utils.geom.get_clist_from_lrtlist(lrt_camXAIs)

        el, rt_prev = utils.geom.split_lrt(prev_lrt)
        r_prev, t_prev = utils.geom.split_rt(rt_prev)
        t_curr = t_prev + vel
        rt_curr = utils.geom.merge_rt(r_prev, t_curr)
        lrt_curr = utils.geom.merge_lrt(el, rt_curr)
        
        new_centroid = utils.geom.get_clist_from_lrtlist(lrt_camXAIs[:,s-1].unsqueeze(1)).squeeze(1)

        clist_camXAI = utils.geom.get_clist_from_lrtlist(lrt_curr.unsqueeze(1))
        clist_memXAI = scene_vox_util.Ref2Mem(clist_camXAI, Z_scene, Y_scene, X_scene)
        crop_vec = torch.from_numpy(np.reshape(np.array([X_pad, Y_pad, Z_pad]), (1, 1, 3))).float().cuda()
        inb = scene_vox_util.get_inbounds(clist_memXAI-crop_vec,
                                          Z_scene-Z_pad*2, Y_scene-Y_pad*2, X_scene-X_pad*2, already_mem=True, padding=1.0)
        if torch.sum(inb) == 0:
            # print('centroid predicted OOB; returning the prior')
            # return early; we won't find the object bc it's now oob
            mem0_T_memI = utils.geom.eye_4x4(B)
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

            obj_mask_memXAI = orig_vox_util.assemble_padded_obj_masklist(
                lrt_curr.unsqueeze(1), score_, Z, Y, X).squeeze(1)
            # pad by 1m on each side
            obj_mask_memXAI = cropper(obj_mask_memXAI)
            if torch.sum(obj_mask_memXAI) < 4:
                print('occrel is iffy at the pred location; returning the prior')
                mem0_T_memI = utils.geom.eye_4x4(B)
                # score = torch.zeros_like(lrt_camXAI[:,0])
                score = torch.ones_like(lrt_camXAI[:,0])*0.5
                # return lrt_camXAI, score, mem0_T_memI
                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = lrt_curr, score, mem0_T_memI
            else:
                # print('working on step %d' % s)
                featI_vec = feat_memXAI.view(B, C, -1)
                # this is B x C x huge
                featI_vec = featI_vec.permute(0, 2, 1)
                # this is B x huge x C

                lrt_camXAIs[:,s], scores[:,s], mem0_T_memIs_e[:,s] = track_one_step_via_inner_product(
                    B, C,
                    Z_, Y_, X_, 
                    prev_lrt,
                    featI_vec,
                    feat0_vec,
                    obj_mask0_vec,
                    obj_length,
                    cam0_T_obj,
                    orig_xyz,
                    new_vox_util,
                    cropper,
                    crop_zyx,
                    delta,
                    summ_writer=summ_writer,
                    use_window=False)
        # print('wrote ans for', s)
        havelist[:,s] = 1.0
        
    obj_clist_camXAI = utils.geom.get_clist_from_lrtlist(lrt_camXAIs)
    if summ_writer is not None:
        summ_writer.summ_traj_on_occ('track/traj_%d' % super_iter,
                                     obj_clist_camXAI,
                                     # padder(cropper(occ_memXAI)), # crop and pad, so i can see the empty area
                                     padder(occ_memXAI_all[I]), 
                                     scene_vox_util, 
                                     already_mem=False,
                                     sigma=2)
        
    return lrt_camXAIs, scores
        
    
