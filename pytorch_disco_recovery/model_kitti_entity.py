import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
# import imageio,scipy

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.matchnet import MatchNet
# from nets.entitynet import EntityNet

import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc
import utils.track

np.set_printoptions(precision=2)
np.random.seed(0)

class KITTI_ENTITY(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = KittiEntityModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        # if hyp.do_freeze_entity:
        #     self.model.entitynet.eval()
        #     self.set_requires_grad(self.model.entitynet, False)
            
class KittiEntityModel(nn.Module):
    def __init__(self):
        super(KittiEntityModel, self).__init__()
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_match:
            # self.deglist = [-4, -2, -1, 0, 1, 2, 4]
            self.deglist = [-4, -2, 0, 2, 4]
            # self.deglist = [-6, -3, 0, 3, 6]
            # self.deglist = [-6, 0, 6]
            # self.deglist = [0]
            self.radlist = [utils.geom.deg2rad(deg) for deg in self.deglist]
            self.trim = 5
            self.matchnet = MatchNet(self.radlist)
        # if hyp.do_entity:
        #     self.entitynet = EntityNet(
        #         num_scales=hyp.entity_num_scales,
        #         num_rots=hyp.entity_num_rots,
        #         max_deg=hyp.entity_max_deg,
        #         max_disp_z=hyp.entity_max_disp_z,
        #         max_disp_y=hyp.entity_max_disp_y,
        #         max_disp_x=hyp.entity_max_disp_x)

    def place_scene_at_dr(self, rgb_mem, xyz_cam, dr, Z, Y, X, vox_util):
        # this function voxelizes the scene with some rotation delta

        # dr is B x 3, containing drx, dry, drz (the rotation delta)
        # Z, Y, X are the resolution of the zoom
        # sz, sy, sx are the metric size of the zoom
        B, N, D = list(xyz_cam.shape)
        assert(D==3)

        # to help us create some mats:
        rot0 = utils.geom.eye_3x3(B)
        t0 = torch.zeros(B, 3).float().cuda()

        camr_T_cam = utils.geom.merge_rt(utils.geom.eul2rotm(dr[:,0], dr[:,1], dr[:,2]), t0)

        xyz_camr = utils.geom.apply_4x4(camr_T_cam, xyz_cam)
        occ_memr = vox_util.voxelize_xyz(xyz_camr, Z, Y, X)
        rgb_memr = vox_util.apply_4x4_to_vox(camr_T_cam, rgb_mem)
        return occ_memr, rgb_memr
            
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=16,
            just_gif=True)
        self.global_step = feed['global_step']

        self.B = feed['set_batch_size']
        self.S = feed['set_seqlen']
        self.set_name = feed['set_name']
        self.filename = feed['filename']
        self.data_ind = feed['data_ind']
        print('filename', self.filename)
        print('data_ind', self.data_ind)
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
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
        self.pix_T_cams = feed['pix_T_cams']

        # in this mode, we never use R coords, so we can drop the R/X notation
        self.origin_T_cams = feed['origin_T_camXs']
        self.xyz_velos = feed['xyz_veloXs']
        self.cams_T_velos = feed["cams_T_velos"]
        self.xyz_cams = __u(utils.geom.apply_4x4(__p(self.cams_T_velos), __p(self.xyz_velos)))

        scene_centroid_x = 0.0
        # scene_centroid_y = 1.0
        # scene_centroid_z = 18.0
        scene_centroid_y = 0.0
        scene_centroid_z = 0.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_cams = feed['rgb_camXs']
        self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_cams.unbind(1))

        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        assert(self.S==2)

        origin_T_cam0 = self.origin_T_cams[:, 0]
        origin_T_cam1 = self.origin_T_cams[:, 1]
        cam0_T_cam1 = utils.basic.matmul2(utils.geom.safe_inverse(origin_T_cam0), origin_T_cam1)

        # let's immediately discard the true motion and make some fake motion

        xyz0_cam0 = self.xyz_cams[:,0]
        xyz1_cam0 = utils.geom.apply_4x4(cam0_T_cam1, self.xyz_cams[:,1])

        # camX_T_cam0 = utils.geom.get_random_rt(
        xyz_cam_g, rx, ry, rz = utils.geom.get_random_rt(
            self.B,
            r_amount=4.0,
            t_amount=2.0,
            sometimes_zero=False,
            return_pieces=True)
        rot = utils.geom.eul2rotm(rx*0.1, ry, rz*0.1)
        camX_T_cam0 = utils.geom.merge_rt(rot, xyz_cam_g)
        
        cam0_T_camX = utils.geom.safe_inverse(camX_T_cam0)
        xyz1_camX = utils.geom.apply_4x4(camX_T_cam0, xyz1_cam0)

        occ0_mem0 = self.vox_util.voxelize_xyz(xyz0_cam0, self.Z, self.Y, self.X)
        occ1_memX = self.vox_util.voxelize_xyz(xyz1_camX, self.Z, self.Y, self.X)

        rgb0_mem0 = self.vox_util.unproject_rgb_to_mem(
            self.rgb_cams[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])
        rgb1_mem1 = self.vox_util.unproject_rgb_to_mem(
            self.rgb_cams[:,1], self.Z, self.Y, self.X, self.pix_T_cams[:,1])
        
        rgb1_memX = self.vox_util.apply_4x4_to_vox(
            utils.basic.matmul2(camX_T_cam0, cam0_T_cam1), rgb1_mem1)
        
        self.summ_writer.summ_occs('inputs/occ_mems', [occ0_mem0, occ1_memX])
        self.summ_writer.summ_unps('inputs/rgb_mems', [rgb0_mem0, rgb1_memX], [occ0_mem0, occ1_memX])

        if hyp.do_feat3d:
            feat_mem0_input = torch.cat([occ0_mem0, occ0_mem0*rgb0_mem0], dim=1)
            feat_memX_input = torch.cat([occ1_memX, occ1_memX*rgb1_memX], dim=1)
            feat_loss0, feat_halfmem0 = self.feat3dnet(feat_mem0_input, self.summ_writer)
            feat_loss1, feat_halfmemX = self.feat3dnet(feat_memX_input, self.summ_writer)
            total_loss += feat_loss0 + feat_loss1

        # if hyp.do_entity:
        #     assert(hyp.do_feat3d)
        #     entity_loss, cam0_T_cam1_e, _ = self.entitynet(
        #         feat_halfmem0,
        #         feat_halfmemX,
        #         cam0_T_camX,
        #         self.vox_util,
        #         self.summ_writer)
        #     total_loss += entity_loss

        if hyp.do_match:
            assert(hyp.do_feat3d)

            occ_rs = []
            rgb_rs = []
            feat_rs = []
            feat_rs_trimmed = []
            for ind, rad in enumerate(self.radlist):
                rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
                occ_r, rgb_r = self.place_scene_at_dr(
                    rgb0_mem0, self.xyz_cams[:,0], rad_,
                    self.Z, self.Y, self.X, self.vox_util)
                occ_rs.append(occ_r)
                rgb_rs.append(rgb_r)

                inp_r = torch.cat([occ_r, occ_r*rgb_r], dim=1)
                _, feat_r = self.feat3dnet(inp_r)
                feat_rs.append(feat_r)
                feat_r_trimmed = feat_r[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                # print('feat_r_trimmed', feat_r_trimmed.shape)
                feat_rs_trimmed.append(feat_r_trimmed)
                
            self.summ_writer.summ_occs('entity/occ_rs', occ_rs)
            self.summ_writer.summ_unps('entity/rgb_rs', rgb_rs, occ_rs)
            self.summ_writer.summ_feats('entity/feat_rs', feat_rs, pca=True)
            self.summ_writer.summ_feats('entity/feat_rs_trimmed', feat_rs_trimmed, pca=True)

            match_loss, camX_T_cam0_e, cam0_T_camX_e = self.matchnet(
                torch.stack(feat_rs_trimmed, dim=1), # templates
                feat_halfmemX, # search region
                self.vox_util,
                xyz_cam_g=xyz_cam_g,
                rad_g=ry,
                summ_writer=self.summ_writer)
            total_loss += match_loss

            occ1_mem0_e = self.vox_util.apply_4x4_to_vox(cam0_T_camX_e, occ1_memX)
            occ1_mem0_g = self.vox_util.apply_4x4_to_vox(cam0_T_camX, occ1_memX)

            self.summ_writer.summ_occs('entity/occ_mems_0', [occ0_mem0, occ1_memX])
            self.summ_writer.summ_occs('entity/occ_mems_e', [occ0_mem0, occ1_mem0_e.round()])
            self.summ_writer.summ_occs('entity/occ_mems_g', [occ0_mem0, occ1_mem0_g.round()])
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_sfm(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        occ_vis = []
        feat_vis = []
        feat_all = []
        stab_vis_e = []
        stab_vis_g = []
        diff_vis = []


        cam0s_T_camIs_e = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)
        cam0s_T_camIs_g = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)

        # xyz_e_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        # xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        # xyz_e = utils.geom.apply_4x4(__p(cam0s_T_camIs_e), xyz_e_).reshape(self.B, self.S, 3)
        # xyz_g = utils.geom.apply_4x4(__p(cam0s_T_camIs_g), xyz_g_).reshape(self.B, self.S, 3)
        
        xyz_e = torch.zeros((self.B, self.S, 3), dtype=torch.float32, device=torch.device('cuda'))
        xyz_g = torch.zeros((self.B, self.S, 3), dtype=torch.float32, device=torch.device('cuda'))

        for s in list(range(self.S-1)):
            origin_T_cam0 = self.origin_T_cams[:, s]
            origin_T_cam1 = self.origin_T_cams[:, s+1]
            cam0_T_origin = utils.geom.safe_inverse(origin_T_cam0)
            cam0_T_cam1_g = utils.basic.matmul2(cam0_T_origin, origin_T_cam1)
            cam0s_T_camIs_g[:,s+1] = utils.basic.matmul2(cam0s_T_camIs_g[:,s], cam0_T_cam1_g)
        xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        cam0s_T_camIs_g_ = __p(cam0s_T_camIs_g)
        xyz_g_ = utils.geom.apply_4x4(cam0s_T_camIs_g_, xyz_g_)
        xyz_g = xyz_g_.reshape(self.B, self.S, 3)

        assert(self.B==1)
        if '09_0000' in self.filename[0]:
            gt_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/ground_truth/09_full.txt'
            orb_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/orb_full_results/09_full.txt'
            # orb_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/orb_short_results/09_full.txt'
        elif '10_0000' in self.filename[0]:
            gt_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/ground_truth/10_full.txt'
            orb_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/orb_full_results/10_full.txt'
            # orb_fn = '/home/aharley/SfMLearner/kitti_eval/pose_data/orb_short_results/10_full.txt'
        else:
            print('weird filename:', self.filename)
            assert(False)
            
        # # with open(orb_fn) as f:
        # #     content = f.readlines()
        # # poses = [line.strip() for line in content]
        # # # xyz = np.array([[float(value) for value in gtruth_list[a][0:3]] for a,b in matches])

        def read_file_list(filename):
            """
            Reads a trajectory from a text file. 

            File format:
            The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
            and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 

            Input:
            filename -- File name

            Output:
            dict -- dictionary of (stamp,data) tuples

            """
            file = open(filename)
            data = file.read()
            lines = data.replace(","," ").replace("\t"," ").split("\n") 
            list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
            list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
            return dict(list)

        def associate(first_list, second_list,offset,max_difference):
            """
            Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
            to find the closest match for every input tuple.

            Input:
            first_list -- first dictionary of (stamp,data) tuples
            second_list -- second dictionary of (stamp,data) tuples
            offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
            max_difference -- search radius for candidate generation

            Output:
            matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

            """
            first_keys = list(first_list.keys())
            second_keys = list(second_list.keys())
            potential_matches = [(abs(a - (b + offset)), a, b) 
                                 for a in first_keys 
                                 for b in second_keys 
                                 if abs(a - (b + offset)) < max_difference]
            potential_matches.sort()
            matches = []
            for diff, a, b in potential_matches:
                if a in first_keys and b in second_keys:
                    first_keys.remove(a)
                    second_keys.remove(b)
                    matches.append((a, b))

            matches.sort()
            return matches

        gtruth_list = read_file_list(gt_fn)
        pred_list = read_file_list(orb_fn)
        matches = associate(gtruth_list, pred_list, 0, 0.01)
        
        gtruth_xyz = np.array([[float(value) for value in gtruth_list[a][0:3]] for a,b in matches])
        pred_xyz = np.array([[float(value) for value in pred_list[b][0:3]] for a,b in matches])

        offset = gtruth_xyz[0] - pred_xyz[0]
        pred_xyz += offset[None,:]

        # Optimize the scaling factor
        scale = np.sum(gtruth_xyz * pred_xyz)/np.sum(pred_xyz ** 2)
        alignment_error = pred_xyz * scale - gtruth_xyz
        pred_xyz = pred_xyz * scale
        
        gtruth_xyz = gtruth_xyz[:100]
        pred_xyz = pred_xyz[:100]

        print('gtruth_xyz', gtruth_xyz.shape)
        print('pred_xyz', pred_xyz.shape)
        
        xyz_g = torch.from_numpy(gtruth_xyz.astype(np.float32)).float().cuda().reshape(1, 100, 3)
        xyz_e = torch.from_numpy(pred_xyz.astype(np.float32)).float().cuda().reshape(1, 100, 3)
        
        # i just want EPE

        # rmse = rmse/100

        # print('rmse', rmse)

        # for s in list(range(self.S)):
        #     xyz_e_ = np.array(poses[s].split(' '))[1:4]
        #     xyz_e_ = xyz_e_.astype(np.float32)
        #     xyz_e_ = torch.from_numpy(xyz_e_).float().cuda().reshape(1, 1, 3)
        #     # xyz_e_ = utils.geom.apply_4x4(self.cams_T_velos[:,s], xyz_e_)
        #     xyz_e[:,s] = xyz_e_.reshape(1, 1, 3)
            
        #     print('xyz_e', xyz_e[:,s])
        #     print('xyz_g', xyz_g[:,s])
        #     input()

        # mean_epe = torch.mean(torch.norm(xyz_e[:,-1] - xyz_g[:,-1], dim=1))
        # self.summ_writer.summ_scalar('unscaled_entity/mean_epe', mean_epe)

        # xz_e = torch.stack([xyz_e[:,-1,0], xyz_e[:,-1,2]], dim=1)
        # xz_g = torch.stack([xyz_g[:,-1,0], xyz_g[:,-1,2]], dim=1)
        # mean_epe_bev = torch.mean(torch.norm(xz_e - xz_g, dim=1))
        # self.summ_writer.summ_scalar('unscaled_entity/mean_epe_bev', mean_epe_bev)

        self.vox_util_wide = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=torch.mean(xyz_g[:,:s+1], dim=1), assert_cube=False)
        wider = 8
        self.vox_util_wide.XMIN = self.vox_util_wide.XMIN - wider*8
        self.vox_util_wide.YMIN = self.vox_util_wide.YMIN #- wider*1
        self.vox_util_wide.ZMIN = self.vox_util_wide.ZMIN - wider*8
        self.vox_util_wide.XMAX = self.vox_util_wide.XMAX + wider*8
        self.vox_util_wide.YMAX = self.vox_util_wide.YMAX #+ wider*1
        self.vox_util_wide.ZMAX = self.vox_util_wide.ZMAX + wider*8
        occ_mem0 = self.vox_util_wide.voxelize_xyz(self.xyz_cams[:,0], self.Z*2, self.Y*1, self.X*2, assert_cube=False)
        self.summ_writer.summ_traj_on_occ(
            'entity/traj',
            xyz_e,
            occ_mem0,
            self.vox_util_wide,
            traj_g=xyz_g,
            already_mem=False,
            sigma=1)
        # total_loss += mean_epe
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    
    def run_orb(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        # cam0_T_cam1_list_e = []
        # cam0_T_cam1_list_g = []

        occ_vis = []
        feat_vis = []
        feat_all = []
        stab_vis_e = []
        stab_vis_g = []
        diff_vis = []


        cam0s_T_camIs_e = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)
        cam0s_T_camIs_g = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)


        assert(self.B==1)
        if '09_0000' in self.filename[0]:
            # orb_fn = '/projects/katefgroup/slam/ORB_KITTI_STEREO_100/09.txt'
            orb_fn = '/home/aharley/kitti_data/09_rt_e.txt'
        elif '10_0000' in self.filename[0]:
            # orb_fn = '/projects/katefgroup/slam/ORB_KITTI_STEREO_100/10.txt'
            orb_fn = '/home/aharley/kitti_data/10_rt_e.txt'
        else:
            print('weird filename:', self.filename)
            assert(False)
        with open(orb_fn) as f:
            content = f.readlines()
        content = content[1:]
        poses = [line.strip() for line in content]

        for s in list(range(self.S-1)):

            xyz_cams = self.xyz_cams[:,:s+1]
            cam0_T_cams_e = cam0s_T_camIs_e[:,:s+1]
            cam0_T_cams_g = cam0s_T_camIs_g[:,:s+1]
            xyz_cam0s_e = __u(utils.geom.apply_4x4(__p(cam0_T_cams_e), __p(xyz_cams)))
            xyz_cam0s_g = __u(utils.geom.apply_4x4(__p(cam0_T_cams_g), __p(xyz_cams)))
            xyz_cam0_e = xyz_cam0s_e.reshape(self.B, -1, 3)
            xyz_cam0_g = xyz_cam0s_g.reshape(self.B, -1, 3)
            xyz_camI_e = utils.geom.apply_4x4(utils.geom.safe_inverse(cam0_T_cams_e[:,s]), xyz_cam0_e)
            xyz_camI_g = utils.geom.apply_4x4(utils.geom.safe_inverse(cam0_T_cams_g[:,s]), xyz_cam0_g)

            xyz_e_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
            xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
            xyz_e = utils.geom.apply_4x4(__p(cam0s_T_camIs_e), xyz_e_).reshape(self.B, self.S, 3)
            xyz_g = utils.geom.apply_4x4(__p(cam0s_T_camIs_g), xyz_g_).reshape(self.B, self.S, 3)

            self.vox_util_wide = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=torch.mean(xyz_g[:,:s+1], dim=1), assert_cube=False)
            wider = 8
            self.vox_util_wide.XMIN = self.vox_util_wide.XMIN - wider*8
            self.vox_util_wide.YMIN = self.vox_util_wide.YMIN #- wider*1
            self.vox_util_wide.ZMIN = self.vox_util_wide.ZMIN - wider*8
            self.vox_util_wide.XMAX = self.vox_util_wide.XMAX + wider*8
            self.vox_util_wide.YMAX = self.vox_util_wide.YMAX #+ wider*1
            self.vox_util_wide.ZMAX = self.vox_util_wide.ZMAX + wider*8

            occ_mem0_e = self.vox_util_wide.voxelize_xyz(xyz_cam0_e, self.Z*2, self.Y*1, self.X*2, assert_cube=False)
            occ_mem0_g = self.vox_util_wide.voxelize_xyz(xyz_cam0_g, self.Z*2, self.Y*1, self.X*2, assert_cube=False)
            stab_vis_e.append(self.summ_writer.summ_traj_on_occ(
                '',
                xyz_e[:,:s+1],
                occ_mem0_e,
                self.vox_util_wide,
                traj_g=xyz_g[:,:s+1],
                already_mem=False,
                sigma=1,
                only_return=True))
            stab_vis_g.append(self.summ_writer.summ_traj_on_occ(
                '',
                xyz_e[:,:s+1],
                occ_mem0_g,
                self.vox_util_wide,
                traj_g=xyz_g[:,:s+1],
                already_mem=False,
                sigma=1,
                only_return=True))
            diff_vis.append(self.summ_writer.summ_oned('', torch.abs(occ_mem0_e - occ_mem0_g), bev=True, only_return=True, norm=True))

            if s==0:
                orb_curr_pose = np.eye(4)
                orb_next_pose = poses[s]
                orb_next_pose = np.array(orb_next_pose.split(' '))[1:]
            else:
                orb_curr_pose = poses[s-1]
                orb_next_pose = poses[s]
                orb_curr_pose = np.array(orb_curr_pose.split(' '))[1:]
                orb_next_pose = np.array(orb_next_pose.split(' '))[1:]
                
            # orb_curr_pose = poses[s]
            # orb_next_pose = poses[s+1]
            # print('orb_curr_pose', orb_curr_pose)
            # orb_curr_pose = np.array(orb_curr_pose.split(' '))
            # orb_next_pose = np.array(orb_next_pose.split(' '))
            
            print('orb_curr_pose', orb_curr_pose, orb_curr_pose.shape)
            
            # orb_curr_pose = np.reshape(orb_curr_pose, (1, 3, 4)).astype(np.float32)
            # orb_next_pose = np.reshape(orb_next_pose, (1, 3, 4)).astype(np.float32)
            orb_curr_pose = np.reshape(orb_curr_pose, (1, 4, 4)).astype(np.float32)
            orb_next_pose = np.reshape(orb_next_pose, (1, 4, 4)).astype(np.float32)
            print('orb_curr_pose', orb_curr_pose)
            orb_curr_pose = torch.from_numpy(orb_curr_pose).float().cuda()
            orb_next_pose = torch.from_numpy(orb_next_pose).float().cuda()

            # make these 4x4, by re-packing
            # orb_curr_pose = utils.geom.merge_rt(utils.geom.split_rt(orb_curr_pose))
            # orb_next_pose = utils.geom.merge_rt(utils.geom.split_rt(orb_next_pose))

            # make these 4x4, by re-packing
            
            # r, t = utils.geom.split_rt(orb_curr_pose)
            # origin_T_orb0 = utils.geom.merge_rt(r, t)
            # r, t = utils.geom.split_rt(orb_next_pose)
            # origin_T_orb1 = utils.geom.merge_rt(r, t)
            
            origin_T_orb0 = utils.geom.merge_rt(*utils.geom.split_rt(orb_curr_pose))
            origin_T_orb1 = utils.geom.merge_rt(*utils.geom.split_rt(orb_next_pose))
            orb0_T_origin = utils.geom.safe_inverse(origin_T_orb0)
            orb0_T_orb1 = utils.basic.matmul2(orb0_T_origin, origin_T_orb1)

            cam0_T_cam1_e = orb0_T_orb1.clone()
            
            
            print('seq step %d' % s)
            
            origin_T_cam0 = self.origin_T_cams[:, s]
            origin_T_cam1 = self.origin_T_cams[:, s+1]
            cam0_T_origin = utils.geom.safe_inverse(origin_T_cam0)
            cam0_T_cam1_g = utils.basic.matmul2(cam0_T_origin, origin_T_cam1)


            # cam0_T_cam1_list_e.append(cam0_T_cam1_e)
            # cam0_T_cam1_list_g.append(cam0_T_cam1_g)

            cam0s_T_camIs_e[:,s+1] = utils.basic.matmul2(cam0s_T_camIs_e[:,s], cam0_T_cam1_e)
            cam0s_T_camIs_g[:,s+1] = utils.basic.matmul2(cam0s_T_camIs_g[:,s], cam0_T_cam1_g)

        self.summ_writer.summ_rgbs('inputs/stab_vis_e', stab_vis_e)
        self.summ_writer.summ_rgbs('inputs/stab_vis_g', stab_vis_g)
        self.summ_writer.summ_rgbs('inputs/diff_vis', diff_vis)

        xyz_e_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        cam0s_T_camIs_e_ = __p(cam0s_T_camIs_e)
        cam0s_T_camIs_g_ = __p(cam0s_T_camIs_g)
        xyz_e_ = utils.geom.apply_4x4(cam0s_T_camIs_e_, xyz_e_)
        xyz_g_ = utils.geom.apply_4x4(cam0s_T_camIs_g_, xyz_g_)
        xyz_e = xyz_e_.reshape(self.B, self.S, 3)
        xyz_g = xyz_g_.reshape(self.B, self.S, 3)

        mean_epe = torch.mean(torch.norm(xyz_e[:,-1] - xyz_g[:,-1], dim=1))
        self.summ_writer.summ_scalar('unscaled_entity/mean_epe', mean_epe)

        xz_e = torch.stack([xyz_e[:,-1,0], xyz_e[:,-1,2]], dim=1)
        xz_g = torch.stack([xyz_g[:,-1,0], xyz_g[:,-1,2]], dim=1)
        mean_epe_bev = torch.mean(torch.norm(xz_e - xz_g, dim=1))
        self.summ_writer.summ_scalar('unscaled_entity/mean_epe_bev', mean_epe_bev)

        occ_mem0 = self.vox_util_wide.voxelize_xyz(self.xyz_cams[:,0], self.Z*2, self.Y*1, self.X*2, assert_cube=False)
        self.summ_writer.summ_traj_on_occ(
            'entity/traj',
            xyz_e,
            occ_mem0,
            self.vox_util_wide,
            traj_g=xyz_g,
            already_mem=False,
            sigma=1)

        total_loss += mean_epe
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
    
            
        
        
    def run_test(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        occ_vis = []
        feat_vis = []
        feat_all = []
        stab_vis_e = []
        stab_vis_g = []
        diff_vis = []

        cam0s_T_camIs_e = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)
        cam0s_T_camIs_g = utils.geom.eye_4x4(self.B*self.S).reshape(self.B, self.S, 4, 4)

        for s in list(range(self.S-1)):
            
            print('seq step %d' % s)
            
            origin_T_cam0 = self.origin_T_cams[:, s]
            origin_T_cam1 = self.origin_T_cams[:, s+1]
            cam0_T_origin = utils.geom.safe_inverse(origin_T_cam0)
            cam0_T_cam1_g = utils.basic.matmul2(cam0_T_origin, origin_T_cam1)

            if hyp.do_entity:
                # get 3d voxelized inputs
                occ_mem0 = self.vox_util.voxelize_xyz(self.xyz_cams[:,s], self.Z, self.Y, self.X)
                occ_mem1 = self.vox_util.voxelize_xyz(self.xyz_cams[:,s+1], self.Z, self.Y, self.X)
                unp_mem0 = self.vox_util.unproject_rgb_to_mem(self.rgb_cams[:,s], self.Z, self.Y, self.X, self.pix_T_cams[:,s])
                unp_mem1 = self.vox_util.unproject_rgb_to_mem(self.rgb_cams[:,s+1], self.Z, self.Y, self.X, self.pix_T_cams[:,s+1])

                occ_vis.append(self.summ_writer.summ_occ('', occ_mem0, only_return=True))
                
                feat_mem0_input = torch.cat([occ_mem0, unp_mem0*occ_mem0], dim=1)
                feat_mem1_input = torch.cat([occ_mem1, unp_mem1*occ_mem1], dim=1)
                _, feat_halfmem0 = self.feat3dnet(feat_mem0_input)
                _, feat_halfmem1 = self.feat3dnet(feat_mem1_input)

                entity_loss, cam0_T_cam1_e, _ = self.entitynet(
                    feat_halfmem0,
                    feat_halfmem1,
                    cam0_T_cam1_g,
                    self.vox_util)
                
            elif hyp.do_match:
                assert(hyp.do_feat3d)

                # what i want to do here is:
                # instead of using xyz_cams[:,s]
                # i want to use the total pointcloud accumulated so far,
                # in the coordinate frame of s (called 0 here)

                # i have cam0s_T_camIs_e
                # i think i can back-transform each pointcloud with this,
                # then forward-transform it to the s sys

                use_map = False
                use_second_stage = True
                # use_second_stage = False

                xyz_cams = self.xyz_cams[:,:s+1]
                cam0_T_cams_e = cam0s_T_camIs_e[:,:s+1]
                cam0_T_cams_g = cam0s_T_camIs_g[:,:s+1]
                xyz_cam0s_e = __u(utils.geom.apply_4x4(__p(cam0_T_cams_e), __p(xyz_cams)))
                xyz_cam0s_g = __u(utils.geom.apply_4x4(__p(cam0_T_cams_g), __p(xyz_cams)))
                xyz_cam0_e = xyz_cam0s_e.reshape(self.B, -1, 3)
                xyz_cam0_g = xyz_cam0s_g.reshape(self.B, -1, 3)
                xyz_camI_e = utils.geom.apply_4x4(utils.geom.safe_inverse(cam0_T_cams_e[:,s]), xyz_cam0_e)
                xyz_camI_g = utils.geom.apply_4x4(utils.geom.safe_inverse(cam0_T_cams_g[:,s]), xyz_cam0_g)

                xyz_e_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
                xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
                xyz_e = utils.geom.apply_4x4(__p(cam0s_T_camIs_e), xyz_e_).reshape(self.B, self.S, 3)
                xyz_g = utils.geom.apply_4x4(__p(cam0s_T_camIs_g), xyz_g_).reshape(self.B, self.S, 3)

                self.vox_util_wide = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=torch.mean(xyz_g[:,:s+1], dim=1), assert_cube=False)
                wider = 8
                self.vox_util_wide.XMIN = self.vox_util_wide.XMIN - wider*8
                self.vox_util_wide.YMIN = self.vox_util_wide.YMIN #- wider*1
                self.vox_util_wide.ZMIN = self.vox_util_wide.ZMIN - wider*8
                self.vox_util_wide.XMAX = self.vox_util_wide.XMAX + wider*8
                self.vox_util_wide.YMAX = self.vox_util_wide.YMAX #+ wider*1
                self.vox_util_wide.ZMAX = self.vox_util_wide.ZMAX + wider*8

                occ_mem0_e = self.vox_util_wide.voxelize_xyz(xyz_cam0_e, self.Z*2, self.Y*1, self.X*2, assert_cube=False)
                occ_mem0_g = self.vox_util_wide.voxelize_xyz(xyz_cam0_g, self.Z*2, self.Y*1, self.X*2, assert_cube=False)
                stab_vis_e.append(self.summ_writer.summ_traj_on_occ(
                    '',
                    xyz_e[:,:s+1],
                    occ_mem0_e,
                    self.vox_util_wide,
                    traj_g=xyz_g[:,:s+1],
                    already_mem=False,
                    sigma=1,
                    only_return=True))
                stab_vis_g.append(self.summ_writer.summ_traj_on_occ(
                    '',
                    xyz_e[:,:s+1],
                    occ_mem0_g,
                    self.vox_util_wide,
                    traj_g=xyz_g[:,:s+1],
                    already_mem=False,
                    sigma=1,
                    only_return=True))
                diff_vis.append(self.summ_writer.summ_oned('', torch.abs(occ_mem0_e - occ_mem0_g), bev=True, only_return=True, norm=True))

                if not use_map:
                    xyz_camI_e = self.xyz_cams[:,s].clone()
                    
                occ0_mem0 = self.vox_util.voxelize_xyz(xyz_camI_e, self.Z, self.Y, self.X)
                rgb0_mem0 = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_cams[:,s], self.Z, self.Y, self.X, self.pix_T_cams[:,s])
                
                occ1_mem1 = self.vox_util.voxelize_xyz(self.xyz_cams[:,s+1], self.Z, self.Y, self.X)
                rgb1_mem1 = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_cams[:,s+1], self.Z, self.Y, self.X, self.pix_T_cams[:,s+1])

                occ_vis.append(self.summ_writer.summ_occ('', occ0_mem0, only_return=True))
                
                occ_rs = []
                rgb_rs = []
                feat_rs = []
                feat_rs_trimmed = []
                for ind, rad in enumerate(self.radlist):
                    rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
                    occ_r, rgb_r = self.place_scene_at_dr(
                        rgb0_mem0, xyz_camI_e, rad_,
                        self.Z, self.Y, self.X, self.vox_util)
                    occ_rs.append(occ_r)
                    rgb_rs.append(rgb_r)

                    inp_r = torch.cat([occ_r, occ_r*rgb_r], dim=1)
                    _, feat_r = self.feat3dnet(inp_r)
                    feat_rs.append(feat_r)
                    feat_r_trimmed = feat_r[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                    feat_rs_trimmed.append(feat_r_trimmed)

                    if rad==0:
                        # feat_vis.append(self.summ_writer.summ_feat('', feat_r, pca=True, only_return=True))
                        feat_all.append(feat_r)
                
                feat_mem1_input = torch.cat([occ1_mem1, occ1_mem1*rgb1_mem1], dim=1)
                
                _, feat_halfmem1 = self.feat3dnet(feat_mem1_input)

                _, cam1_T_cam0_e, cam0_T_cam1_e = self.matchnet(
                    torch.stack(feat_rs_trimmed, dim=1), # templates
                    feat_halfmem1, # search region
                    self.vox_util)

                if use_second_stage:
                    # let's re-name our output, to make clear that it's just partway to the end
                    camP_T_cam0_e = cam1_T_cam0_e.clone()
                    cam0_T_camP_e = cam0_T_cam1_e.clone()

                    # xyz_camI_e is almost good;
                    # we just need to apply the transformation we got from the first step, to bring it a bit closer to the future
                    xyz_camI2_e = utils.geom.apply_4x4(cam1_T_cam0_e, xyz_camI_e)
                    rgb0_mem0 = self.vox_util.unproject_rgb_to_mem(
                        self.rgb_cams[:,s], self.Z, self.Y, self.X, self.pix_T_cams[:,s])
                    occ_rs = []
                    rgb_rs = []
                    feat_rs = []
                    feat_rs_trimmed = []
                    for ind, rad in enumerate(self.radlist):
                        rad_ = torch.from_numpy(np.array([0, rad, 0])).float().cuda().reshape(1, 3)
                        occ_r, rgb_r = self.place_scene_at_dr(
                            rgb0_mem0, xyz_camI2_e, rad_,
                            self.Z, self.Y, self.X, self.vox_util)
                        occ_rs.append(occ_r)
                        rgb_rs.append(rgb_r)

                        inp_r = torch.cat([occ_r, occ_r*rgb_r], dim=1)
                        _, feat_r = self.feat3dnet(inp_r)
                        feat_rs.append(feat_r)
                        feat_r_trimmed = feat_r[:,:,self.trim:-self.trim:,self.trim:-self.trim:,self.trim:-self.trim:]
                        feat_rs_trimmed.append(feat_r_trimmed)

                    feat_mem1_input = torch.cat([occ1_mem1, occ1_mem1*rgb1_mem1], dim=1)
                    _, feat_halfmem1 = self.feat3dnet(feat_mem1_input)
                    # this time, we are transforming between P and 1
                    _, cam1_T_camP_e, camP_T_cam1_e = self.matchnet(
                        torch.stack(feat_rs_trimmed, dim=1), # templates
                        feat_halfmem1, # search region
                        self.vox_util)
                    # print('cam1_T_camP_e', cam1_T_camP_e.detach().cpu().numpy())
                    cam1_T_cam0_e = utils.basic.matmul2(cam1_T_camP_e, camP_T_cam0_e)
                    cam0_T_cam1_e = utils.basic.matmul2(cam0_T_camP_e, camP_T_cam1_e)                

            # chain cam0_T_camN * camN_T_camM to get cam0_T_camM
            cam0s_T_camIs_e[:,s+1] = utils.basic.matmul2(cam0s_T_camIs_e[:,s], cam0_T_cam1_e)
            cam0s_T_camIs_g[:,s+1] = utils.basic.matmul2(cam0s_T_camIs_g[:,s], cam0_T_cam1_g)

        # diffs = [torch.abs(e - g) for (e,g) in zip(stab_vis_e, stab_vis_g)]
        self.summ_writer.summ_rgbs('inputs/occ_vis', occ_vis)
        # self.summ_writer.summ_rgbs('inputs/feat_vis', feat_vis)
        self.summ_writer.summ_rgbs('inputs/stab_vis_e', stab_vis_e)
        self.summ_writer.summ_rgbs('inputs/stab_vis_g', stab_vis_g)
        # self.summ_writer.summ_rgbs('inputs/stab_vis_d', diffs)
        self.summ_writer.summ_rgbs('inputs/diff_vis', diff_vis)
        self.summ_writer.summ_feats('inputs/feat_vis', feat_all, pca=True) # pca all together!

        xyz_e_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        xyz_g_ = torch.zeros((self.B*self.S, 1, 3), dtype=torch.float32, device=torch.device('cuda'))
        cam0s_T_camIs_e_ = __p(cam0s_T_camIs_e)
        cam0s_T_camIs_g_ = __p(cam0s_T_camIs_g)
        xyz_e_ = utils.geom.apply_4x4(cam0s_T_camIs_e_, xyz_e_)
        xyz_g_ = utils.geom.apply_4x4(cam0s_T_camIs_g_, xyz_g_)
        xyz_e = xyz_e_.reshape(self.B, self.S, 3)
        xyz_g = xyz_g_.reshape(self.B, self.S, 3)

        mean_epe = torch.mean(torch.norm(xyz_e[:,-1] - xyz_g[:,-1], dim=1))
        self.summ_writer.summ_scalar('unscaled_entity/mean_epe', mean_epe)
        
        xz_e = torch.stack([xyz_e[:,-1,0], xyz_e[:,-1,2]], dim=1)
        xz_g = torch.stack([xyz_g[:,-1,0], xyz_g[:,-1,2]], dim=1)
        mean_epe_bev = torch.mean(torch.norm(xz_e - xz_g, dim=1))
        self.summ_writer.summ_scalar('unscaled_entity/mean_epe_bev', mean_epe_bev)

        # wide_centroid = torch.mean(xyz_g, dim=1)
        # self.vox_util_wide = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=wide_centroid, assert_cube=True)
        # wider = 16
        # self.vox_util_wide.XMIN = self.vox_util_wide.XMIN - wider*4
        # self.vox_util_wide.YMIN = self.vox_util_wide.YMIN - wider*1
        # self.vox_util_wide.ZMIN = self.vox_util_wide.ZMIN - wider*4
        # self.vox_util_wide.XMAX = self.vox_util_wide.XMAX + wider*4
        # self.vox_util_wide.YMAX = self.vox_util_wide.YMAX + wider*1
        # self.vox_util_wide.ZMAX = self.vox_util_wide.ZMAX + wider*4
        
        occ_mem0 = self.vox_util_wide.voxelize_xyz(self.xyz_cams[:,0], self.Z*2, self.Y*1, self.X*2, assert_cube=False)
        self.summ_writer.summ_traj_on_occ(
            'entity/traj',
            xyz_e,
            occ_mem0,
            self.vox_util_wide,
            traj_g=xyz_g,
            already_mem=False,
            sigma=1)

        total_loss += mean_epe
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
    
    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)
        data_ok = False
        
        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            elif self.set_name=='test':
                return self.run_sfm(feed)
                # return self.run_orb(feed)
                # return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
