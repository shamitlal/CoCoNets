import os
import torch
import numpy as np
from tqdm import tqdm

import utils_py
import utils_vox
import utils_geom
import utils_basic

from backend import inputs

import hyperparams as hyp

B = 1
__p = lambda x: utils_basic.pack_seqdim(x, B)
__u = lambda x: utils_basic.unpack_seqdim(x, B)

def write_file_list(filenames, set_name, shape):
    output_filename = hyp.trainset if set_name == 'train' else hyp.valset
    output_filename = os.path.join(hyp.dataset_location, output_filename + '_' + shape + '_extreme.txt')
    print("Writing to {}".format(output_filename))
    with open(output_filename, 'w') as f:
        for filename in filenames:
            f.write('%s\n' % filename)

def isCurved(feed):  
    results = {} 
    B = 1
    H, W, V, S, N = hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
    Z, Y, X = hyp.Z, hyp.Y, hyp.X
    ZZ, ZY, ZX = hyp.ZZ, hyp.ZY, hyp.ZX

    Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
    Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
    D = 9

    pix_T_cams = feed["pix_T_cams"]
    origin_T_camRs = feed["origin_T_camRs"]
    origin_T_camXs = feed["origin_T_camXs"]
    # xyz_camXs = feed["xyz_camXs"]

    camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
    camR0_T_camRs = utils_geom.get_camM_T_camXs(origin_T_camRs, ind=0)
    camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
    camRs_T_camR0 = __u(utils_geom.safe_inverse(__p(camR0_T_camRs)))

    box_traj_camR = feed["box_traj_camR"]

    lrtlist_camRs = utils_geom.convert_boxlist_to_lrtlist(box_traj_camR)
    lrtlist_camR0 = utils_geom.apply_4x4s_to_lrts(camR0_T_camRs, lrtlist_camRs)

    # xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
    # xyz_camR0s = __u(utils_geom.apply_4x4(__p(camR0_T_camRs), __p(xyz_camRs)))

    obj_lrtlist_camR0 = lrtlist_camR0

    clist_camR0 = utils_geom.get_clist_from_lrtlist(lrtlist_camR0)

    xyz0 = clist_camR0[:,0]
    xyz1 = clist_camR0[:,1]
    

    delta = xyz1-xyz0
    delta_norm = torch.norm(delta, dim=1)
    invalid_NY = delta_norm < 0.0001

    # if invalid_NY.sum() > 0:
    #     import pdb; pdb.set_trace()

    delta = delta.detach().cpu().numpy()
    dx = delta[:,0]
    dy = delta[:,1]
    dz = delta[:,2]

    yaw = np.arctan2(dz, dx)
    rot = np.stack([utils_py.eul2rotm(0,y,0) for y in yaw])
    rot = torch.from_numpy(rot).float().cuda()
    # rot is B x 3 x 3
    t = -xyz0
    # t is B x 3
    t0 = torch.zeros_like(t)

    rot0 = utils_geom.eye_3x3(B)
    zero_T_camR0 = utils_geom.merge_rt(rot0, t)
    noyaw_T_zero = utils_geom.merge_rt(rot, t0)

    mid_x = (hyp.XMAX + hyp.XMIN)/2.0
    mid_y = (hyp.YMAX + hyp.YMIN)/2.0
    mid_z = (hyp.ZMAX + hyp.ZMIN)/2.0
    mid_xyz = np.array([mid_x, mid_y, mid_z]).reshape(1, 3)
    tra = torch.from_numpy(mid_xyz).float().cuda().repeat(B, 1)
    center_T_noyaw = utils_geom.merge_rt(rot0, tra)

    noyaw_T_camR0 = utils_basic.matmul3(center_T_noyaw, noyaw_T_zero, zero_T_camR0)
    noyaw_T_camR0s = noyaw_T_camR0.unsqueeze(1).repeat(1, S, 1, 1)

    clist_camNY0s = utils_geom.apply_4x4(noyaw_T_camR0, clist_camR0) # B x S x 3
    curved = torch.abs(clist_camNY0s[:, 0, 2] - clist_camNY0s[:, -1, 2]) > 0.2 # B

    results['invalid_NY'] = invalid_NY.cpu().numpy()
    results['curved'] = curved.cpu().numpy()
    results['x_displacements'] = torch.abs(clist_camNY0s[:, 0, 2] - clist_camNY0s[:, -1, 2]).cpu().numpy()
    
    return results

def classify():
    all_inputs = inputs.get_inputs()
    for set_name, set_loader in all_inputs.items():
        curved_filenames = np.array([])
        straight_filenames = np.array([])
        x_displacements = np.array([])
        for feed in tqdm(set_loader, desc='Processing set {}'.format(set_name)):
        # for feed in set_loader:
            feed_cuda = {}
            for k in feed:
                try:
                    feed_cuda[k] = feed[k].cuda()
                except:
                    feed_cuda[k] = feed[k]
            results = isCurved(feed_cuda)
            invalid_NY = results['invalid_NY']
            # print('filename', feed['filename'])
            # input()
            # all_filenames = np.array(['/'.join(x.split('/')[7:]) for x in feed['filename']])
            all_filenames = np.array([x[len(hyp.dataset_location)+1:] for x in feed['filename']])
            # all_filenames = np.array(['/'.join(x.split('/')[7:]) for x in feed['filename']])
            # all_filenames = np.array([hyp.dataset_location]'/'.join(x.split('/')[7:]) for x in feed['filename']])
            # print('new filename', all_filenames[0])
            
            # If it's valid and 
            is_curved = np.logical_and(results['curved'], ~invalid_NY)
            is_straight = ~is_curved
            
            curved_files = all_filenames[is_curved] # Only valid curved
            straight_files = all_filenames[is_straight]
            if len(curved_files) != 0:
                curved_filenames = np.concatenate((curved_filenames, curved_files), axis=0)
            if len(straight_files) != 0:
                straight_filenames = np.concatenate((straight_filenames, straight_files), axis=0)
            x_displacements = np.concatenate((x_displacements, results['x_displacements'][~invalid_NY]), axis=0) # concatenate only valid NYs
        # Save Files
        np.save('x_displacements_{}'.format(set_name), x_displacements)
        write_file_list(straight_filenames.tolist(), set_name, 'straight')
        write_file_list(curved_filenames.tolist(), set_name, 'curved')

if __name__ == '__main__':
    classify()
