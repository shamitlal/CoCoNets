import io as sysio
import time
import hyperparams as hyp
import random
# import numba
# import hardPositiveMiner

import numpy as np
import utils.geom
import utils.box
import utils.ap
import torch

EPS = 1e-6

def make_border_green(vis):
    vis = np.copy(vis)
    vis[0,:,0] = 0
    vis[0,:,1] = 255
    vis[0,:,2] = 0
    
    vis[-1,:,0] = 0
    vis[-1,:,1] = 255
    vis[-1,:,2] = 0

    vis[:,0,0] = 0
    vis[:,0,1] = 255
    vis[:,0,2] = 0
    
    vis[:,-1,0] = 0
    vis[:,-1,1] = 255
    vis[:,-1,2] = 0
    return vis

def make_border_black(vis):
    vis = np.copy(vis)
    vis[0,:,:] = 0
    vis[-1,:,:] = 0
    vis[:,0,:] = 0
    vis[:,-1,:] = 0
    return vis

def compute_precision(xxx_todo_changeme, xxx_todo_changeme1, recalls=[1,3,5], pool_size=100):
    # inputs are lists
    # list elements are H x W x C
    
    (emb_e, vis_e) = xxx_todo_changeme
    (emb_g, vis_g) = xxx_todo_changeme1
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = np.zeros(len(recalls), np.float32)
    # print 'precision B = %d' % B

    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    vis_inds = perm[:10] # just vis 10 queries
    
    # print 'B = %d; pool_size = %d' % (B, pool_size)
    
    if B >= pool_size: # otherwise it's not going to be accurate
        emb_e = np.stack(emb_e, axis=0)
        emb_g = np.stack(emb_g, axis=0)
        # emb_e = np.concatenate(emb_e, axis=0)
        # emb_g = np.concatenate(emb_g, axis=0)
        vect_e = np.reshape(emb_e, [B, -1])
        vect_g = np.reshape(emb_g, [B, -1])
        scores = np.dot(vect_e, np.transpose(vect_g))
        ranks = np.flip(np.argsort(scores), axis=1)

        vis = []
        for i in vis_inds:
            minivis = []
            # first col: query
            # minivis.append(vis_e[i])
            minivis.append(make_border_black(vis_e[i]))
            
            # # second col: true answer
            # minivis.append(vis_g[i])
            
            # remaining cols: ranked answers
            for j in list(range(10)):
                v = vis_g[ranks[i, j]]
                if ranks[i, j]==i:
                    minivis.append(make_border_green(v))
                else:
                    minivis.append(v)
            # concat retrievals along width
            minivis = np.concatenate(minivis, axis=1)
            # print 'got this minivis:', 
            # print minivis.shape
            
            vis.append(minivis)
        # concat examples along height
        vis = np.concatenate(vis, axis=0)
        # print 'got this vis:', 
        # print vis.shape
            
        for recall_id, recall in enumerate(recalls):
            for emb_id in list(range(B)):
                if emb_id in ranks[emb_id, :recall]:
                    precision[recall_id] += 1
            # print("precision@", recall, float(precision[recall_id])/float(B))
        precision = precision/float(B)
    else:
        precision = np.nan*precision
        vis = np.zeros((H*10, W*11, 3), np.uint8)
    # print 'precision  %.2f' % np.mean(precision)
    
    return precision, vis

def compute_patch_based_vis(pool_e, pool_g, num_embeds,summ_writer):
    hpm = hardPositiveMiner.HardPositiveMiner()

    num_patches_per_emb = hyp.max.num_patches_per_emb
    scores = torch.zeros((num_embeds, num_embeds)).cuda()

    h_init_e = np.random.randint(2,14)
    d_init_e = np.random.randint(2,14)
    w_init_e = np.random.randint(2,14)
    _, unps_e = pool_e.fetch()
    _, unps_g = pool_e.fetch()
    featQuery_i , perm_i =  hpm.extractPatches_det(pool_e,d_init_e,h_init_e,w_init_e)
    featQuery_i = featQuery_i[:1]
    
    unp_e = unps_e[0]
    topkImg_i, topkScale, topkValue_i, topkW , topkH , topkD, topkR  = hpm.RetrievalResForExpectation(pool_g, featQuery_i)
    summ_writer.summ_evalmines("eval_mines",[[topkImg_i,topkD,topkH,topkW,topkR],[d_init_e,h_init_e,w_init_e],[unps_g,unp_e]])


def compute_precision_o_cuda(pool_e, pool_g, pool_size=100, summ_writer=None,steps_done=0):
    (emb_e, vis_e) = pool_e.fetch()
    (emb_g, vis_g) = pool_g.fetch()
    
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    if len(vis_e[0].shape)==4:
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) 

    perm = np.random.permutation(B)
    vis_inds = perm[:10] # just vis 10 queries    
    exp_done = False
    if (B >= pool_size) and summ_writer.save_this : # otherwise it's not going to be accurate
        emb_e = torch.stack(emb_e, axis=0)
        emb_g = torch.stack(emb_g, axis=0)
        vect_e = torch.nn.functional.normalize(torch.reshape(emb_e, [B, -1]),dim=1)
        vect_g = torch.nn.functional.normalize(torch.reshape(emb_g, [B, -1]),dim=1)
        compute_patch_based_vis(pool_e, pool_g, len(emb_e),summ_writer)

def drop_invalid_boxes(boxlist_e, boxlist_g, scorelist_e, scorelist_g):
    # print('before:')
    # print(boxlist_e.shape)
    # print(boxlist_g.shape)
    boxlist_e_, boxlist_g_, scorelist_e_, scorelist_g_ = [], [], [], []
    for i in list(range(len(boxlist_e))):
        box_e = boxlist_e[i]
        # print('box_e', box_e)
        score_e = scorelist_e[i]
        valid_e = np.where(box_e[:,3] > 0.0) # lx
        boxlist_e_.append(box_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    # print('boxlist_e_', boxlist_e_)
    for i in list(range(len(boxlist_g))):
        box_g = boxlist_g[i]
        score_g = scorelist_g[i]
        valid_g = np.where(score_g > 0.5)
        boxlist_g_.append(box_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    # print('boxlist_g_', boxlist_g_)
    boxlist_e, boxlist_g, scorelist_e, scorelist_g = np.array(boxlist_e_), np.array(boxlist_g_), np.array(scorelist_e_), np.array(scorelist_g_)
    return boxlist_e, boxlist_g, scorelist_e, scorelist_g

def drop_invalid_lrts(lrtlist_e, lrtlist_g, scorelist_e, scorelist_g):
    B, N, D = lrtlist_e.shape
    assert(B==1)
    # unlike drop_invalid_boxes, this is all in pt
    # print('before:')
    # print(lrtlist_e.shape)
    # print(lrtlist_g.shape)
    # print(scorelist_e.shape)
    # print(scorelist_g.shape)
    # lrtlists are shaped B x N x 19
    # scorelists are shaped B x N
    lrtlist_e_, scorelist_e_, lrtlist_g_, scorelist_g_ = [], [], [], []
    lenlist_e, _ = utils.geom.split_lrtlist(lrtlist_e)
    for i in list(range(len(lrtlist_e))):
        lrt_e = lrtlist_e[i]
        score_e = scorelist_e[i]
        len_e = lenlist_e[i]
        valid_e = torch.where(len_e[:, 0] > 0.01)
        lrtlist_e_.append(lrt_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    for i in list(range(len(lrtlist_g))):
        lrt_g = lrtlist_g[i]
        score_g = scorelist_g[i]
        valid_g = torch.where(score_g > 0.5)
        lrtlist_g_.append(lrt_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = torch.stack(lrtlist_e_), torch.stack(lrtlist_g_), torch.stack(scorelist_e_), torch.stack(scorelist_g_)
    # print('after')
    # print(lrtlist_e.shape)
    # print(lrtlist_g.shape)
    return lrtlist_e, lrtlist_g, scorelist_e, scorelist_g

def get_mAP(boxes_e, scores, boxes_g, iou_thresholds):
    # boxes are 1 x N x 9
    B, Ne, _ = list(boxes_e.shape)
    B, Ng, _ = list(boxes_g.shape)
    assert(B==1)
    boxes_e = np.reshape(boxes_e, (B*Ne, 9))
    boxes_g = np.reshape(boxes_g, (B*Ng, 9))
    corners_e = utils.geom.transform_boxes3D_to_corners_py(boxes_e)
    corners_g = utils.geom.transform_boxes3D_to_corners_py(boxes_g)
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            if(boxes_e[i,3]>0 and boxes_g[j,3]>0):
                iou_single, iou_2d_single = utils.box.box3d_iou(corners_e[i], corners_g[j])
                ious[i,j] = iou_single
    maps = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps = utils.ap.compute_ap(
            "box3D_"+str(iou_threshold), scores, ious, iou_threshold=iou_threshold)
        maps.append(map3d)
    maps = np.stack(maps, axis=0).astype(np.float32)
    if np.isnan(maps).any():
        print('got these nans in maps; setting to zero:', maps)
        maps[np.isnan(maps)] = 0.0
        # assert(False)
    
    # print("maps", maps)
    return maps

def get_mAP_from_lrtlist(lrtlist_e, scores, lrtlist_g, iou_thresholds):
    # lrtlist are 1 x N x 19
    B, Ne, _ = list(lrtlist_e.shape)
    B, Ng, _ = list(lrtlist_g.shape)
    assert(B==1)
    scores = scores.detach().cpu().numpy()
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious_3d = np.zeros((Ne, Ng), dtype=np.float32)
    ious_2d = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            iou_3d, iou_2d = utils.geom.get_iou_from_corresponded_lrtlists(lrtlist_e[:, i:i+1], lrtlist_g[:, j:j+1])
            ious_3d[i, j] = iou_3d[0, 0]
            ious_2d[i, j] = iou_2d[0, 0]
    maps_3d = []
    maps_2d = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps = utils.ap.compute_ap(
            "box3d_" + str(iou_threshold), scores, ious_3d, iou_threshold=iou_threshold)
        maps_3d.append(map3d)
        map2d, precision, recall, overlaps = utils.ap.compute_ap(
            "box2d_" + str(iou_threshold), scores, ious_2d, iou_threshold=iou_threshold)
        maps_2d.append(map2d)
    maps_3d = np.stack(maps_3d, axis=0).astype(np.float32)
    maps_2d = np.stack(maps_2d, axis=0).astype(np.float32)
    if np.isnan(maps_3d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_3d[np.isnan(maps_3d)] = 0.0
    if np.isnan(maps_2d).any():
        # print('got these nans in maps; setting to zero:', maps)
        maps_2d[np.isnan(maps_2d)] = 0.0

    # print("maps_3d", maps_3d)
    return maps_3d, maps_2d

def measure_semantic_retrieval_precision(feats, masks, debug=False):
    # feat_memXs is B x C x Z x Y x X
    # mask_memXs is B x 1 x Z x Y x X
    # mask_memXs is ones inside the object masks; zeros in the bkg
    B, C, Z, Y, X = list(feats.shape)
    assert(B>1)
    feats = feats[:2]
    masks = masks[:2]

    obj_feats = []
    bkg_feats = []
    for b in list(range(B)):
        feat = feats[b]
        mask = masks[b]
        # feat is C x Z x Y x X
        feat = feat.permute(1, 2, 3, 0).reshape(-1, C)
        mask = mask.permute(1, 2, 3, 0).reshape(-1)
        # feat is N x C
        # mask is N
        obj_inds = torch.where((mask).reshape(-1) > 0.5)
        bkg_inds = torch.where((mask).reshape(-1) < 0.5)
        # obj_inds is ?
        # bkg_inds is ?

        obj_feat = feat[obj_inds]
        bkg_feat = feat[bkg_inds]
        
        obj_feat = obj_feat.detach().cpu().numpy()
        bkg_feat = bkg_feat.detach().cpu().numpy()
        np.random.shuffle(obj_feat)
        np.random.shuffle(bkg_feat)

        def trim(feat, max_N=50):
            N, C = feat.shape
            if N > max_N:
                # print('trimming!')
                feat = feat[:max_N]
            return feat
        obj_feat = trim(obj_feat)
        bkg_feat = trim(bkg_feat)
        
        obj_feats.append(obj_feat)
        bkg_feats.append(bkg_feat)

    # let's just do this for b=0 for now

    source_obj_feat = obj_feats[0]
    source_bkg_feat = bkg_feats[0]
    other_obj_feats = np.concatenate(obj_feats[1:], axis=0)
    other_bkg_feats = np.concatenate(bkg_feats[1:], axis=0)
            # print('other_obj_feats', other_obj_feats.shape)
            # print('other_bkg_feats', other_bkg_feats.shape)
    
    # make it a fair game: even balance of obj vs bkg
    if other_bkg_feats.shape[0] > other_obj_feats.shape[0]:
        other_bkg_feats = other_bkg_feats[:other_obj_feats.shape[0]]
    else:
        other_obj_feats = other_obj_feats[:other_bkg_feats.shape[0]]

    other_feats = np.concatenate([other_obj_feats,
                                  other_bkg_feats], axis=0)
    labels = np.concatenate([np.ones_like(other_obj_feats[:,0]),
                             np.zeros_like(other_bkg_feats[:,0])], axis=0)
    # print('other_feats', other_feats.shape)

    precisions = []
    for n in list(range(len(source_obj_feat))):
        feat = source_obj_feat[n:n+1]
        dists = np.linalg.norm(other_feats - feat, axis=1)
        inds = np.argsort(dists)
        sorted_labels = labels[inds]
        sorted_dists = dists[inds]
        precision = np.mean(sorted_labels[:10])
        if not np.isnan(precision):
            precisions.append(precision)
        if np.isnan(precision) or debug:
            print('sorted dists and labels:')
            print(sorted_dists, sorted_dists.shape)
            print(sorted_labels, sorted_labels.shape)
            print('other_obj_feats', other_obj_feats.shape)
            print('other_bkg_feats', other_bkg_feats.shape)
            print('precision', precision)
            print('mean precision so far', np.mean(precisions))
            
    if len(precisions):
        mean_precision = np.mean(precisions)
    else:
        print('semantic retrieval bug!')
        print('some info:')
        print('source_obj_feat',source_obj_feat.shape)
        print('other_obj_feats', other_obj_feats.shape)
        print('other_bkg_feats', other_bkg_feats.shape)
        print('other_feats', other_feats.shape)
        assert(False)
        # input()
        # print(source_bkg_feat.shape)
        mean_precision = np.nan
        
    return mean_precision

def linmatch(labelpools, obj_inds, bkg_inds, codes_flat):
    num_embeddings = len(labelpools)
    # print('there seem to be %d embeddings')
    
    # now:
    # on every iter, take a balanced set of obj and bkg inds
    # for each index, see which codeword lights up
    # this is plinko for codewords
    # for each codeword, calculate if either "bkg" or "obj" dominates its inputs;
    # this means: max(mean(obj_yes), mean(1.0-obj_yes))

    np.random.shuffle(obj_inds)
    np.random.shuffle(bkg_inds)

    # print('before trim', len(obj_inds), len(bkg_inds))
    def trim(inds0, inds1):
        N0 = len(inds0)
        N1 = len(inds1)
        if N0 < N1:
            inds1 = inds1[:N0]
        else:
            inds0 = inds0[:N1]
        return inds0, inds1
    obj_inds, bkg_inds = trim(obj_inds, bkg_inds)

    # print('after trim', len(obj_inds), len(bkg_inds))

    print('making %d assignments' % (len(obj_inds) + len(bkg_inds)))
    # input()

    # print('ind_map', ind_map.shape)
    # print('expected:', self.H//8, self.W//8)
    if len(obj_inds) > 0:
        # at each index, i need to find the codeword used
        for ind in obj_inds:
            code = np.squeeze(codes_flat[ind])
            # print('obj: updating code', code)
            labelpools[code].update([1])
        for ind in bkg_inds:
            code = np.squeeze(codes_flat[ind])
            # print('bkg: updating code', code)
            labelpools[code].update([0])

    # print('updated the pools; got these:')

    accs = []
    pool_sizes = []
    for code in list(range(num_embeddings)):
        pool = labelpools[code].fetch()
        print('code', code, 'pool has %d items' % len(pool))
        pool_sizes.append(len(pool))
        # input()
        if len(pool) > 20:
            acc = np.mean(pool)
            acc = np.maximum(acc, 1.0-acc)
            accs.append(acc)
            print('acc = %.2f' % acc)
            # input()
    if len(accs) > 1:
        mean_acc = np.mean(accs)
        print('ok, have %d valid accs' % len(accs))
        print('overall acc: %.2f' % mean_acc)
    else:
        print('only have %d valid accs' % len(accs))
        mean_acc = np.nan
    num_codes_w_20 = len(accs)
    mean_pool_size = np.mean(pool_sizes)
    return mean_acc, mean_pool_size, num_codes_w_20

    # input()

    # an issue with that evaluation is that if most of the codewords devote themselves to
    # background, then they will dominate this accuracy measure, simply because they outnumber
    # the codes doing a mix.
    # maybe that's ok?

    # because of the way i am pooling, the data is evenly distributed between the classes


    # if this were empirically a problem,
    # then the more codewords i had, the higher this acc would be
    # but in my 2d experiments, this was not the case

