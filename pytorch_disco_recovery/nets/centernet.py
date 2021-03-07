import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic
import utils.geom
import utils.samp
import numpy as np

def _topk(objectness, K=10):
    B, C, Z, Y, X = list(objectness.shape)
    assert(C==1)
      
    scorelist, indlist = torch.topk(objectness.view(B, C, -1), K)

    indlist_z = indlist // (Y*X)
    indlist_y = (indlist % (Y*X)) // X
    indlist_x = (indlist % (Y*X)) % X

    scorelist = scorelist.reshape(B, K)
    indlist_z = indlist_z.reshape(B, K)
    indlist_y = indlist_y.reshape(B, K)
    indlist_x = indlist_x.reshape(B, K)

    xyzlist = torch.stack([indlist_x, indlist_y, indlist_z], dim=2).float()
    return scorelist, xyzlist

def _nms(heat, kernel=15):
    pad = (kernel - 1) // 2

    hmax = F.max_pool3d(
        heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def clamped_sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x whatever)
      gt_regr (batch x c x whatever)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  # loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  # num_pos  = pos_inds.float().sum()
  # pos_loss = pos_loss.sum()
  # neg_loss = neg_loss.sum()
  
  # print('num_pos', num_pos.detach().cpu().numpy())
  # print('pos_loss', pos_loss.detach().cpu().numpy())
  # print('neg_loss', neg_loss.detach().cpu().numpy())
  # print('neg_weights', neg_weights.detach().cpu().numpy())

  # if num_pos == 0:
  #   loss = loss - neg_loss
  # else:
  #   loss = loss - (pos_loss + neg_loss) / num_pos

  pos_loss = pos_loss.mean()
  neg_loss = neg_loss.mean()
  
  # loss = loss - (pos_loss + neg_loss)
  loss = - (pos_loss + neg_loss)
  # loss = - (torch.mean(pos_loss) + torch.mean(neg_loss))

  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class CenterNet(nn.Module):
    def __init__(self):
        super(CenterNet, self).__init__()

        print('CenterNet...')

        self.K = 8
        self.thresh = 0.0

        # for rotation, i have decided to be heading-unaware
        # so, 0 and 180 are treated as equivalent
        # self.num_rot_bins = 8
        self.num_rot_bins = 1
        bin_angles = np.linspace(0, np.pi, self.num_rot_bins, endpoint=False)
        bin_complements = bin_angles + np.pi
        all_bins = np.concatenate([bin_angles, bin_complements], axis=0)
        all_inds = np.concatenate([np.arange(self.num_rot_bins), np.arange(self.num_rot_bins)], axis=0)
        self.bin_angles = torch.from_numpy(bin_angles).float().cuda()
        self.all_bins = torch.from_numpy(all_bins).float().cuda()
        self.all_inds = torch.from_numpy(all_inds).long().cuda()

        # a thing missing here is: one set of bins, maybe rz, should not be fully determined
        # for now, let's remove rz. objects don't roll anyway

        obj_channels = 1
        size_channels = 3
        rot_channels = self.num_rot_bins*2
        offset_channels = 3
        self.total_channels = obj_channels + size_channels + rot_channels + offset_channels
        
        self.conv3d = nn.Conv3d(
            # in_channels=hyp.feat3d_dim,
            # in_channels=128+128+64+hyp.feat3d_dim,
            in_channels=128+128+64,
            out_channels=self.total_channels,
            kernel_size=1,
            stride=1,
            padding=0).cuda()

        self.focal = FocalLoss()

        self.mse = torch.nn.MSELoss(reduction='none')
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        
        
    def compute_center_loss(self, pred, occ, free, summ_writer):
        # occ is B x 1 x Z x Y x X
        pos = occ.clone()
        neg = free.clone()

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        # loss_vis = torch.mean(loss*mask_*valid, dim=3)
        loss_vis = torch.mean(loss*mask_, dim=3)
        summ_writer.summ_oned('center/prob_loss', loss_vis)

        # pos_loss = reduce_masked_mean(loss, pos*valid)
        # neg_loss = reduce_masked_mean(loss, neg*valid)
        pos_loss = utils.basic.reduce_masked_mean(loss, pos)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    def compute_rot_loss(self, rot_prob, rot_g, valid):
        # rot_prob is B x N x self.rot_bins
        # rot_g is B x N, with angles in radians
        # valid is B x N

        B, N = list(rot_g.shape)
        rot_prob = rot_prob.reshape(B*N, self.num_rot_bins)
        valid = valid.reshape(B*N)
        rot_g = rot_g.reshape(B*N, 1)

        # i need to assign rot_g into bins
        dist = utils.geom.angular_l1_dist(rot_g, self.all_bins.reshape(1, -1))
        # this is B*N x num_rot_bins
        min_inds = torch.argmin(dist, dim=1)
        # this is B*N and long
        # for safety, let's not parallelize the gather here
        labels = torch.zeros(B*N).long().cuda()
        for b in list(range(B*N)):
            labels[b] = self.all_inds[min_inds[b]]
        # print('labels', labels.detach().cpu().numpy())
        # print('rot_prob', rot_prob.shape)
        loss_vec = F.cross_entropy(rot_prob, labels, reduction='none')
        
        # rather than take a straight mean, we will balance across classes
        losses = []
        for cls in list(range(self.num_rot_bins)):
            mask = (labels==cls).float()
            cls_loss = utils.basic.reduce_masked_mean(loss_vec, mask*valid)
            if torch.sum(mask) >= 1:
                # print('adding loss for rot bin %d' % cls)
                losses.append(cls_loss)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss
        
    def forward(self, feat, crop, vox_util, center_g=None, lrtlist_cam_g=None, lrtlist_mem_g=None, scorelist_g=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)

        pred = self.conv3d(feat)
        center_e_ = pred[:,0:1]
        center_e = F.sigmoid(center_e_) # the values within are objectness
        # size_e = (pred[:,1:4] + 1.5).clamp(min=0.25) # the values within are in meters (cam coords)
        size_e = F.softplus(pred[:,1:4]) + 1.0
        rx_e = pred[:, 4+self.num_rot_bins*0:4+self.num_rot_bins*1]
        ry_e = pred[:, 4+self.num_rot_bins*1:4+self.num_rot_bins*2]
        # rz_e = pred[:, 4+self.num_rot_bins*2:4+self.num_rot_bins*3]
        # offset_e = pred[:, 4+self.num_rot_bins*3:]
        offset_e = pred[:, 4+self.num_rot_bins*2:]

        # print('center_e', center_e.shape)
        # print('size_e', size_e.shape)
        # print('rx_e', rx_e.shape)
        # print('ry_e', ry_e.shape)
        # print('offset_e', offset_e.shape)

        size_e_ = torch.abs(size_e.reshape(-1))
        utils.basic.print_stats('size_e_', size_e_)
        # penalize sizes for being larger than 3.0
        size_loss = F.relu(size_e_-3.0)
        utils.basic.print_stats('size_loss', size_loss)
        total_loss = utils.misc.add_loss('center/size_loss', total_loss, torch.mean(size_loss), 2.0, summ_writer)
        
        # # sizes_small = size_e_[size_e_ < 0.5]
        # # penalize sizes for being larger than 1.0
        # size_loss_large = torch.mean(torch.abs(sizes_large-4.0))
        # # penalize sizes for being smaller than 0.5
        # size_loss_small = torch.mean(torch.abs(sizes_small-0.5))
        # # size_loss = torch.mean(F.relu(torch.abs(size_e)-1.0))
        # size_loss = size_loss_large + size_loss_small
        # utils.basic.print_stats('size_loss', size_loss)
        # total_loss = utils.misc.add_loss('center/size_loss', total_loss, size_loss, 1.0, summ_writer)

        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(pred, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils.misc.add_loss('center/smooth_loss', total_loss, smooth_loss, hyp.center_smooth_coeff, summ_writer)
        
        Z_crop, Y_crop, X_crop = crop
        Z_mem = Z + Z_crop*2
        Y_mem = Y + Y_crop*2
        X_mem = X + X_crop*2
        feat_pad = F.pad(feat, (Z_crop, Z_crop, Y_crop, Y_crop, X_crop, X_crop), 'constant', 0)
        feat_pad = torch.abs(feat_pad[:,0:1])
        crop_vec = torch.from_numpy(np.reshape(np.array([X_crop, Y_crop, Z_crop]), (1, 1, 3))).float().cuda()

        if lrtlist_mem_g is not None:
            print('apparently we have gt!')
            # assume all the gt is here
            
            B2, N, D = list(lrtlist_cam_g.shape)
            assert(B==B2)
            assert(D==19)

            summ_writer.summ_lrtlist_bev(
                'center/boxes_g',
                utils.basic.normalize(feat_pad[0:1]), 
                lrtlist_mem_g[0:1],
                scorelist_g[0:1], # scores
                torch.ones(1,50).long().cuda(), # tids
                vox_util,
                already_mem=True)
            
            clist_g = utils.geom.get_clist_from_lrtlist(lrtlist_mem_g)
            clist_crop = clist_g - crop_vec
            # clist_g is B x N x 3

            print('clist_mem_g[0,0:2]', clist_g[0,0:2].detach().cpu().numpy())
            

            # now, extract the size and rotation estimates at the GT
            sizelist_e = utils.samp.bilinear_sample3d(size_e, clist_crop)
            rxlist_e = utils.samp.bilinear_sample3d(rx_e, clist_crop)
            rylist_e = utils.samp.bilinear_sample3d(ry_e, clist_crop)
            print('extracting data at gt')
            print('rxlist_e', rxlist_e.shape)
            print('rylist_e', rylist_e.shape)
            # rzlist_e = utils.samp.bilinear_sample3d(rz_e, clist_crop)
            offsetlist_e = utils.samp.bilinear_sample3d(offset_e, clist_crop)
            # this is B x 3 x N (since channels stays inside)
            sizelist_e = sizelist_e.permute(0, 2, 1)
            rxlist_e = rxlist_e.permute(0, 2, 1)
            rylist_e = rylist_e.permute(0, 2, 1)
            # rzlist_e = rzlist_e.permute(0, 2, 1)
            offsetlist_e = offsetlist_e.permute(0, 2, 1)
            # this is B x N x 3
            # these are now ready for regression or classification
            # for gt, we will extract data in cam coords
            sizelist_g, rtlist_cam_g = utils.geom.split_lrtlist(lrtlist_cam_g)
            
            print('sizelist_g[0,0:2]', sizelist_g[0,0:2].detach().cpu().numpy())
            print('corresp sizelist_e[0,0:2]', sizelist_e[0,0:2].detach().cpu().numpy())
            
            # sizelist_g is B x N x 3
            # rtlist_cam_g is B x N x 4 x 4
            rlist_, tlist_ = utils.geom.split_rt(rtlist_cam_g.reshape(B*N, 4, 4))
            rxlist_, rylist_, rzlist_ = utils.geom.rotm2eul(rlist_)
            rxlist_g = rxlist_.reshape(B, N)
            rylist_g = rylist_.reshape(B, N)
            # rzlist_g = rzlist_.reshape(B, N)

            rx_loss = self.compute_rot_loss(rxlist_e, rxlist_g, scorelist_g)
            ry_loss = self.compute_rot_loss(rylist_e, rylist_g, scorelist_g)
            # rz_loss = self.compute_rot_loss(rzlist_e, rzlist_g, scorelist_g, summ_writer)
            # rot_loss = rx_loss + ry_loss + rz_loss
            rot_loss = rx_loss + ry_loss
            # print('rot_loss', rot_loss.detach().cpu().numpy())
            total_loss = utils.misc.add_loss('center/rot_loss', total_loss, rot_loss, hyp.center_rot_coeff, summ_writer)

            # print('sizelist_e[0,0]', sizelist_e[0,0].detach().cpu().numpy())
            # print('sizelist_g[0,0]', sizelist_g[0,0].detach().cpu().numpy())

            # sizelist_diff = torch.sum(torch.abs(sizelist_e - sizelist_g), dim=2)
            sizelist_diff = torch.sum(self.smoothl1(sizelist_e, sizelist_g), dim=2)
            # this is B x N
            size_loss = utils.basic.reduce_masked_mean(sizelist_diff, scorelist_g)
            # print('size_loss', size_loss.detach().cpu().numpy())
            total_loss = utils.misc.add_loss('center/size_loss', total_loss, size_loss, hyp.center_size_coeff, summ_writer)

            # focal_loss = self.focal(center_e, center_g)
            # print('focal_loss', focal_loss.detach().cpu().numpy())
            # total_loss = utils.misc.add_loss('center/focal_loss%s' % suffix, total_loss, focal_loss, hyp.center_focal_coeff, summ_writer)

            prob_loss = self.compute_center_loss(center_e_, (center_g==1.0).float(), (center_g==0.0).float(), summ_writer)
            print('prob_loss', prob_loss.detach().cpu().numpy())
            total_loss = utils.misc.add_loss('center/prob_loss', total_loss, prob_loss, hyp.center_prob_coeff, summ_writer)
            if summ_writer is not None:
                summ_writer.summ_occ('center/center_g', center_g)

            offsetlist_g = clist_crop - torch.round(clist_crop) # get the decimal part
            offsetlist_diff = torch.sum(self.smoothl1(offsetlist_e, offsetlist_g), dim=2)
            offset_loss = utils.basic.reduce_masked_mean(offsetlist_diff, scorelist_g)
            print('offset_loss', offset_loss.detach().cpu().numpy())
            total_loss = utils.misc.add_loss('center/offset_loss', total_loss, offset_loss, hyp.center_offset_coeff, summ_writer)
            
        if summ_writer is not None:
            summ_writer.summ_occ('center/center_e', center_e)

        # now, let's convert the estimates into discrete boxes
        # this means: extract topk peaks from the centerness map,
        # and at those locations, extract the rotation and size estimates
        center_e_clean = center_e.clone()
        # center_e_clean[center_e_clean < 0.8] = 0.0
        center_e_clean = _nms(center_e_clean)
        if summ_writer is not None:
            summ_writer.summ_occ('center/center_e_clean', center_e_clean)
        scorelist_e, xyzlist_crop_e = _topk(center_e_clean, K=self.K)
        xyzlist_mem_e = xyzlist_crop_e + crop_vec
        # print('clist_mem_e[0,0:2]', xyzlist_mem_e[0,0:2].detach().cpu().numpy())
        xyzlist_cam_e = vox_util.Mem2Ref(xyzlist_mem_e, Z_mem, Y_mem, X_mem)
        sizelist_e = utils.samp.bilinear_sample3d(size_e, xyzlist_crop_e)
        rxlist_e = utils.samp.bilinear_sample3d(rx_e, xyzlist_crop_e)
        rylist_e = utils.samp.bilinear_sample3d(ry_e, xyzlist_crop_e)

        # print('extracting data at peaks')
        # print('rxlist_e', rxlist_e.shape)
        # print('rylist_e', rylist_e.shape)
        
        # rzlist_e = utils.samp.bilinear_sample3d(rz_e, xyzlist_crop_e)
        offsetlist_e = utils.samp.bilinear_sample3d(offset_e, xyzlist_crop_e)
        # these are B x 3 x N (since the channel dim stays inside)
        sizelist_e = sizelist_e.permute(0, 2, 1)
        # sizelist_e = sizelist_e.clamp(min=0.01)
        # print('peak sizelist_e[0,0:2]', sizelist_e[0,0:2].detach().cpu().numpy())
        rxlist_e = rxlist_e.permute(0, 2, 1)
        rylist_e = rylist_e.permute(0, 2, 1)
        # rzlist_e = rzlist_e.permute(0, 2, 1)
        offsetlist_e = offsetlist_e.permute(0, 2, 1) # TODO: clamp?
        xyzlist_cam_e = vox_util.Mem2Ref(xyzlist_mem_e + offsetlist_e, Z_mem, Y_mem, X_mem)

        # print('offsetlist_e[0,0:2]', offsetlist_e[0,0:2].detach().cpu().numpy())

        # fancy new idea:
        # at these locations, apply another loss, using the nearest gt
        # e.g., we would like offsets away from the object to point to the object
        if lrtlist_mem_g is not None:

            extra_size_loss = 0.0
            extra_offset_loss = 0.0
            extra_rot_loss = 0.0

            normalizer = 0.0
            for b in list(range(B)):
                for k in list(range(self.K)):
                    xyz_e = xyzlist_mem_e[b:b+1, k]
                    size_e = sizelist_e[b:b+1, k]
                    offset_e = offsetlist_e[b:b+1, k]
                    # these are 1 x 3
                    rx_e = rxlist_e[b:b+1, k]
                    ry_e = rylist_e[b:b+1, k]
                    # these are 1 x num_rot_bins
                    # print('xyz_e', xyz_e.shape)
                    # print('rx_e', rx_e.shape)
                    print('size_e', size_e.detach().cpu().numpy(), size_e.shape)
                    # rz = rzlist_e[b:b+1, k]
                    # these are 1 x 1
                    xyz_g = clist_g[b:b+1]
                    score_g = scorelist_g[b:b+1]
                    xyz_g[score_g < 1.0] = 100000
                    # this is 1 x N x 3
                    dist = utils.basic.sql2_on_axis(xyz_g - xyz_e.unsqueeze(1), 2)
                    # this is 1 x N
                    ind = torch.argmin(dist, dim=1).squeeze()
                    # print('ind', ind.detach().cpu().numpy(), ind.shape)
                    xyz_g = clist_g[b:b+1,ind]
                    size_g = sizelist_g[b:b+1,ind]
                    score_g = scorelist_g[b:b+1,ind]
                    mindist = dist[:,ind]
                    # only proceed if the nn is valid, and not too far away
                    if score_g.squeeze() == 1.0 and mindist.squeeze() < 8.0:
                        # offset_g = offsetlist_g[b:b+1,ind]
                        # for offset, we actually need to recompute
                        offset_g = xyz_g - xyz_e
                        rx_g = rxlist_g[b:b+1,ind]
                        ry_g = rylist_g[b:b+1,ind]
                        # print('xyz_e:', xyz_e.detach().cpu().numpy())
                        # print('nearest:', xyz_g.detach().cpu().numpy())

                        extra_rot_loss += self.compute_rot_loss(rx_e.unsqueeze(1), rx_g.unsqueeze(1), torch.ones_like(rx_g.unsqueeze(1)))
                        extra_rot_loss += self.compute_rot_loss(ry_e.unsqueeze(1), ry_g.unsqueeze(1), torch.ones_like(rx_g.unsqueeze(1)))

                        # print('rx_e', rx_e.shape)
                        # print('rx_g', rx_g.shape)
                        # print('ry_e', ry_e.shape)
                        # print('ry_g', ry_g.shape)
                        # print('offset_e', offset_e.shape)
                        # print('offset_g', offset_g.shape)
                        # print('size_e', size_e, size_e.shape)
                        # print('size_g', size_g, size_g.shape)

                        # all the tensors of interest are 1x3, or 1x16 for rots
                        # input()
                        extra_size_loss += torch.mean(torch.sum(self.smoothl1(size_e, size_g), dim=1))
                        # print('size_loss', size_loss.detach().cpu().numpy())
                        # total_loss = utils.misc.add_loss('center/size_loss', total_loss, size_loss, hyp.center_size_coeff, summ_writer)

                        extra_offset_loss += torch.mean(torch.sum(self.smoothl1(offset_e, offset_g), dim=1))
                        # total_loss = utils.misc.add_loss('center/offset_loss', total_loss, offset_loss, hyp.center_offset_coeff, summ_writer)

                        normalizer += 1
                    else:
                        print('discarding; mindist:', mindist.squeeze().detach().cpu().numpy())

            if normalizer > 0:
                total_loss = utils.misc.add_loss('center/extra_size_loss', total_loss, extra_size_loss/normalizer, hyp.center_size_coeff*hyp.center_peak_coeff, summ_writer)
                total_loss = utils.misc.add_loss('center/extra_offset_loss', total_loss, extra_offset_loss/normalizer, hyp.center_offset_coeff*hyp.center_peak_coeff, summ_writer)
                total_loss = utils.misc.add_loss('center/extra_rot_loss', total_loss, extra_rot_loss/normalizer, hyp.center_rot_coeff*hyp.center_peak_coeff, summ_writer)

        boxlist = scorelist_e.new_zeros((B, self.K, 9))
        scorelist = scorelist_e.new_zeros((B, self.K))
        for b in list(range(B)):
            boxlist_b = []
            scorelist_b = []
            for k in list(range(self.K)):
                score = scorelist_e[b:b+1, k]
                # print('score', score.shape)
                # print('score', score.squeeze().shape)
                # let's call it a real object
                if score.squeeze() > self.thresh:
                    xyz = xyzlist_cam_e[b:b+1, k]
                    size = sizelist_e[b:b+1, k]
                    # these are 1 x 3
                    rx = rxlist_e[b:b+1, k]
                    ry = rylist_e[b:b+1, k]
                    # rz = rzlist_e[b:b+1, k]
                    # these are 1 x num_rot_bins
                    # i need to convert this into an actual rot
                    rx = rx.squeeze()
                    rx_ind = torch.argmax(rx)
                    rx = self.bin_angles[rx_ind].reshape(1)

                    ry = ry.squeeze()
                    ry_ind = torch.argmax(ry)
                    ry = self.bin_angles[ry_ind].reshape(1)
                    
                    # rz = rz.squeeze()
                    # rz_ind = torch.argmax(rz)
                    # rz = self.bin_angles[rz_ind].reshape(1)
                    rz = torch.zeros_like(ry)
                    
                    rot = torch.cat([rx, ry, rz], dim=0).unsqueeze(0)
                    # this is 1 x 3
                    
                    box = torch.cat([xyz, size, rot], dim=1)
                    boxlist_b.append(box)
                    scorelist_b.append(score)
            if len(boxlist_b) > 0:
                boxlist_b = torch.stack(boxlist_b, dim=1) # 1 x ? x 3
                scorelist_b = torch.stack(scorelist_b, dim=1) # 1 x ? x 1
                boxlist_b = torch.cat((boxlist_b, torch.zeros([1, self.K, 9]).cuda()), dim=1)
                scorelist_b = torch.cat((scorelist_b, torch.zeros([1, self.K]).cuda()), dim=1)
                boxlist_b = boxlist_b[:, :self.K]
                scorelist_b = scorelist_b[:, :self.K]
            else:
                boxlist_b = torch.zeros([1, self.K, 9]).cuda()
                scorelist_b = torch.zeros([1, self.K]).cuda()
            boxlist[b:b+1] = boxlist_b
            scorelist[b:b+1] = scorelist_b
        lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(boxlist)
        lrtlist_mem = vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam, Z_mem, Y_mem, X_mem)

        if summ_writer is not None:
            summ_writer.summ_lrtlist_bev(
                'center/boxes_e',
                utils.basic.normalize(feat_pad[0:1]),
                lrtlist_mem[0:1],
                scorelist[0:1], # scores
                torch.ones(1,self.K).long().cuda(), # tids
                vox_util,
                already_mem=True)
        return total_loss, lrtlist_cam, scorelist
