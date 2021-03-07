import torch
import hyperparams as hyp
import numpy as np
import utils_geom
import utils_samp
import utils_improc
import torch.nn.functional as F
from utils_basic import *
import utils_basic
import utils_py

class Search_util(object):
    def __init__(self, set_name, t_max=4.0, r_max=3.14, assert_cube=True):
        # on every step, we create this object
        
        self.set_name = set_name

        self.search_size_factor = 3.0

        if set_name == "train":
            t_max_x = t_max
            t_max_y = t_max/4.0
            t_max_z = t_max
            self.t_x = np.random.uniform(-t_max_x, t_max_x)
            self.t_y = np.random.uniform(-t_max_y, t_max_y)
            self.t_z = np.random.uniform(-t_max_z, t_max_z)

            # self.x_delta = np.random.uniform(-x_range*dx, x_range*dx)

            # y_range = 1.0
            # self.y_delta = np.random.uniform(-y_range*dy, y_range*dy)

            # z_range = 1.0
            # self.z_delta = np.random.uniform(-z_range*dz, z_range*dz)

            rot_max_x = r_max/4.0
            rot_max_y = r_max
            rot_max_z = r_max/4.0
            self.rot_x = np.random.uniform(-rot_max_x, rot_max_x)
            self.rot_y = np.random.uniform(-rot_max_y, rot_max_y)
            self.rot_z = np.random.uniform(-rot_max_z, rot_max_z)
            
        else:
            self.t_x = 0.0
            self.t_y = 0.0
            self.t_z = 0.0
            self.rot_x = 0.0
            self.rot_y = 0.0
            self.rot_z = 0.0


        r = utils_py.eul2rotm(self.rot_x, self.rot_y, self.rot_z).astype(np.float32)
        t = np.reshape(np.array([self.t_x, self.t_y, self.t_z], np.float32), [3, 1])
        # rt_py = utils_py.merge_rt(r, t)
        # print('rt:', rt_py)

        # r = torch.from_numpy

        r = torch.from_numpy(r).float().cuda().reshape(1, 3, 3)
        t = torch.from_numpy(t).float().cuda().reshape(1, 3)

        r0 = utils_geom.eye_3x3(1)
        t0 = torch.zeros_like(t)

        self.augrot_T_obj = utils_geom.merge_rt(r, t0)
        self.obj_T_augrot = utils_geom.safe_inverse(self.augrot_T_obj)

        self.search_T_augrot = utils_geom.merge_rt(r0, t)
        self.augrot_T_search = utils_geom.safe_inverse(self.search_T_augrot)
        
        # self.search_T_obj = torch.from_numpy(rt_py).float().cuda().reshape(1, 4, 4)
        # self.obj_T_aug = utils_geom.safe_inverse(self.search_T_obj)

        # VOX_SIZE_X = (self.XMAX-self.XMIN)/float(hyp.X)
        # VOX_SIZE_Y = (self.YMAX-self.YMIN)/float(hyp.Y)
        # VOX_SIZE_Z = (self.ZMAX-self.ZMIN)/float(hyp.Z)

        # print('VOX_SIZE_X', VOX_SIZE_X)
        # print('VOX_SIZE_Y', VOX_SIZE_Y)
        # print('VOX_SIZE_Z', VOX_SIZE_Z)

        # if assert_cube:
        #     # we assume cube voxels
        #     assert(np.isclose(VOX_SIZE_X, VOX_SIZE_Y))
        #     assert(np.isclose(VOX_SIZE_X, VOX_SIZE_Z))

    # def convert_obj_lrt_to_search_lrt(self, lrt):
    #     l, rt = utils_geom.split_lrt(lrt)
    #     l = l * self.search_size_factor
    #     search_rt = torch.matmul(self.search_T_given, rt)
    #     lrt = utils_geom.merge_lrt(l, search_rt)
    #     return lrt

    def convert_box_to_search_lrt(self, box):
        B, D = list(box.shape)
        assert(D==9)
        rt = self.convert_box_to_ref_T_search(box)
        l = box[:,3:6].reshape(B, 3)
        l = l * self.search_size_factor
        lrt = utils_geom.merge_lrt(l, rt)
        return lrt

    def convert_box_to_ref_T_search(self, box):
        # this borrows from utils_geom.convert_box_to_ref_T_obj
        B = list(box.shape)[0]

        # box is B x 9
        x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box, axis=1)
        rot0 = utils_geom.eye_3x3(B)
        tra = torch.stack([x, y, z], axis=1)
        center_T_ref = utils_geom.merge_rt(rot0, -tra)
        # center_T_ref is B x 4 x 4

        t0 = torch.zeros([B, 3])
        rot = utils_geom.eul2rotm(rx, ry, rz)
        rot = torch.transpose(rot, 1, 2) # other dir
        obj_T_center = utils_geom.merge_rt(rot, t0)

        # we want search_T_ref:
        # (1) we to translate to center, then
        # (2) rotate around the object's origin, then
        # (3) rotate some more, then
        # (4) add a displacement.
        search_T_ref = utils_basic.matmul4(
            self.search_T_augrot, self.augrot_T_obj, obj_T_center, center_T_ref)

        # return the inverse of this
        # (note this is a rigid transform, since there is no scaling)
        ref_T_search = utils_basic.safe_inverse(search_T_ref)
        return ref_T_search

    def get_sr_T_ref(self, lrt, Z, Y, X):
        # this borrows from utils_vox.get_zoom_T_ref
        # sr = search region (like a wide zoom)

        # actually it seems like get_zoom_T_ref suffices
        assert(False)
        
        # lrt is B x 19
        B, E = list(lrt.shape)
        assert(E==19)
        lens, ref_T_search = utils_geom.split_lrt(lrt)
        lx, ly, lz = lens.unbind(1)

        search_T_ref = utils_geom.safe_inverse(ref_T_search)
        # this is B x 4 x 4

        # translation
        center_T_search_r = utils_geom.eye_3x3(B)
        center_T_search_t = torch.stack([lx/2., ly/2., lz/2.], dim=1)
        center_T_search = utils_geom.merge_rt(center_T_search_r, center_T_search_t)

        # scaling
        Z_VOX_SIZE_X = (lx)/float(X)
        Z_VOX_SIZE_Y = (ly)/float(Y)
        Z_VOX_SIZE_Z = (lz)/float(Z)
        diag = torch.stack([1./Z_VOX_SIZE_X,
                            1./Z_VOX_SIZE_Y,
                            1./Z_VOX_SIZE_Z,
                            torch.ones([B], device=torch.device('cuda'))],
                           axis=1).view(B, 4)
        zoom_T_center = torch.diag_embed(diag)

        # compose these
        zoom_T_search = utils_basic.matmul2(zoom_T_center, center_T_search)

        zoom_T_ref = utils_basic.matmul2(zoom_T_obj, obj_T_ref)

        return zoom_T_ref
    
        
    # def Ref2Mem(self, xyz, Z, Y, X, bounds='default'):
    #     # xyz is B x N x 3, in ref coordinates
    #     # transforms velo coordinates into mem coordinates
    #     B, N, C = list(xyz.shape)
    #     mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, bounds=bounds)
    #     xyz = utils_geom.apply_4x4(mem_T_ref, xyz)
    #     return xyz

    # def Mem2Ref(self, xyz_mem, Z, Y, X, bounds='default'):
    #     # xyz is B x N x 3, in mem coordinates
    #     # transforms mem coordinates into ref coordinates
    #     B, N, C = list(xyz_mem.shape)
    #     ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, bounds=bounds)
    #     xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
    #     return xyz_ref

    # def get_ref_T_mem(self, B, Z, Y, X, bounds='default'):
    #     mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, bounds=bounds)
    #     # note safe_inverse is inapplicable here,
    #     # since the transform is nonrigid
    #     ref_T_mem = mem_T_ref.inverse()
    #     return ref_T_mem

    # def get_mem_T_ref(self, B, Z, Y, X, bounds='default'):
    #     # sometimes we want the mat itself
    #     # note this is not a rigid transform

    #     # translation
    #     center_T_ref = utils_geom.eye_4x4(B)
    #     center_T_ref[:,0,3] = -self.XMIN
    #     center_T_ref[:,1,3] = -self.YMIN
    #     center_T_ref[:,2,3] = -self.ZMIN

    #     VOX_SIZE_X = (self.XMAX-self.XMIN)/float(X)
    #     VOX_SIZE_Y = (self.YMAX-self.YMIN)/float(Y)
    #     VOX_SIZE_Z = (self.ZMAX-self.ZMIN)/float(Z)

    #     # scaling
    #     mem_T_center = utils_geom.eye_4x4(B)
    #     mem_T_center[:,0,0] = 1./VOX_SIZE_X
    #     mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    #     mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    #     mem_T_ref = utils_basic.matmul2(mem_T_center, center_T_ref)

    #     return mem_T_ref

    # def get_search_T_obj(self, B, Z, Y, X):
    #     # sometimes we want the mat itself
    #     # note this is not a rigid transform

    #     # translation
    #     center_T_ref = utils_geom.eye_4x4(B)
    #     center_T_ref[:,0,3] = -self.XMIN
    #     center_T_ref[:,1,3] = -self.YMIN
    #     center_T_ref[:,2,3] = -self.ZMIN

    #     VOX_SIZE_X = (self.XMAX-self.XMIN)/float(X)
    #     VOX_SIZE_Y = (self.YMAX-self.YMIN)/float(Y)
    #     VOX_SIZE_Z = (self.ZMAX-self.ZMIN)/float(Z)

    #     # scaling
    #     mem_T_center = utils_geom.eye_4x4(B)
    #     mem_T_center[:,0,0] = 1./VOX_SIZE_X
    #     mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    #     mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    #     mem_T_ref = utils_basic.matmul2(mem_T_center, center_T_ref)

    #     return mem_T_ref
    
    # def get_search_T_ref(self, lrt, Z, Y, X):
    #     # lrt is B x 19
    #     B, E = list(lrt.shape)
    #     assert(E==19)
    #     lens, ref_T_obj = utils_geom.split_lrt(lrt)
    #     lx, ly, lz = lens.unbind(1)

    #     # expand the search 
    #     lx = lx * self.search_size_factor
    #     ly = ly * self.search_size_factor
    #     lz = lz * self.search_size_factor
        
    #     obj_T_ref = utils_geom.safe_inverse(ref_T_obj)
    #     # this is B x 4 x 4

    #     # translation
    #     center_T_obj_r = utils_geom.eye_3x3(B)
    #     center_T_obj_t = torch.stack([
    #         lx/2. + self.x_delta,
    #         ly/2. + self.y_delta,
    #         lz/2. + self.z_delta], dim=1)
    #     center_T_obj = utils_geom.merge_rt(center_T_obj_r, center_T_obj_t)

    #     # scaling
    #     VOX_SIZE_X = (lx)/float(X)
    #     VOX_SIZE_Y = (ly)/float(Y)
    #     VOX_SIZE_Z = (lz)/float(Z)
    #     diag = torch.stack([1./VOX_SIZE_X,
    #                         1./VOX_SIZE_Y,
    #                         1./VOX_SIZE_Z,
    #                         torch.ones([B], device=torch.device('cuda'))],
    #                        axis=1).view(B, 4)
    #     search_T_center = torch.diag_embed(diag)

    #     # compose these
    #     search_T_obj = utils_basic.matmul2(search_T_center, center_T_obj)
    #     search_T_ref = utils_basic.matmul2(search_T_obj, obj_T_ref)
    #     return search_T_ref

    # def get_ref_T_search(self, lrt, Z, Y, X):
    #     # lrt is B x 19
    #     search_T_ref = self.get_search_T_ref(lrt, Z, Y, X)
    #     # note safe_inverse is inapplicable here,
    #     # since the transform is nonrigid
    #     ref_T_search = search_T_ref.inverse()
    #     return ref_T_search

    # def Ref2Search(self, xyz_ref, lrt_ref, Z, Y, X):
    #     # xyz_ref is B x N x 3, in ref coordinates
    #     # lrt_ref is B x 19, specifying the box in ref coordinates
    #     # this transforms ref coordinates into search coordinates
    #     B, N, _ = list(xyz_ref.shape)
    #     search_T_ref = self.get_search_T_ref(lrt_ref, Z, Y, X)
    #     xyz_search = utils_geom.apply_4x4(search_T_ref, xyz_ref)
    #     return xyz_search

    # def Search2Ref(self, xyz_search, lrt_ref, Z, Y, X):
    #     # xyz_search is B x N x 3, in search coordinates
    #     # lrt_ref is B x 9, specifying the box in ref coordinates
    #     B, N, _ = list(xyz_search.shape)
    #     ref_T_search = self.get_ref_T_search(lrt_ref, Z, Y, X)
    #     xyz_ref = utils_geom.apply_4x4(ref_T_search, xyz_search)
    #     return xyz_ref

    # def crop_search_from_mem(self, mem, lrt, Z2, Y2, X2):
    #     # mem is B x C x Z x Y x X
    #     # lrt is B x 19

    #     B, C, Z, Y, X = list(mem.shape)
    #     B2, E = list(lrt.shape)

    #     assert(E==19)
    #     assert(B==B2)

    #     xyz_mem = self.convert_lrt_to_sampling_coords(lrt, Z, Y, X, Z2, Y2, X2)

    #     search = utils_samp.sample3D(mem, xyz_mem, Z2, Y2, X2)
    #     search = torch.reshape(search, [B, C, Z2, Y2, X2])
    #     return search

    # def convert_lrt_to_sampling_coords(self, lrt, Z, Y, X, Z2, Y2, X2):
    #     # lrt is B x 19
    #     B, E = list(lrt.shape)
    #     assert(E==19)
    #     # Z, Y, X is the resolution we are sampling from
    #     # Z2, Y2, X2 is how many coords to sample

    #     # for each voxel in the search grid, 
    #     # i want to sample a voxel from the mem
    #     xyz_search = utils_basic.gridcloud3D(B, Z2, Y2, X2, norm=False)
    #     # these represent the search grid coordinates
    #     # we need to convert these to mem coordinates
    #     xyz_ref = self.Search2Ref(xyz_search, lrt, Z2, Y2, X2)
    #     xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X)
    #     return xyz_mem
    
