import numpy as np
import torch 
import open3d as o3d 
import matplotlib.pyplot as plt 
import ipdb 
import utils.geom
st = ipdb.set_trace


def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2


def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd



data = np.load('/Users/shamitlal/Desktop/temp/tsdf/raw/city_01_vehicles_150_episode_0039_2020-05-02_cam0_startframe000578_obj5.npz')
pix_T_cams = torch.tensor(data['pix_T_cams'][0:1])
rgb_cam0s = data['rgb_cam0s'][9:10]

xyz_cam0s = torch.tensor(data['xyz_cam0s'][9:10])
depth_cam0s,_ = utils.geom.create_depth_image(pix_T_cams, xyz_cam0s, rgb_cam0s.shape[1], rgb_cam0s.shape[2])
depth_cam0s[depth_cam0s>30] = 0
xyz_cam0s = utils.geom.depth2pointcloud(depth_cam0s, pix_T_cams)


xyz_cam1s = torch.tensor(data['xyz_cam3s'][9:10])
depth_cam1s,_ = utils.geom.create_depth_image(pix_T_cams, xyz_cam1s, rgb_cam0s.shape[1], rgb_cam0s.shape[2])
depth_cam1s[depth_cam1s>30] = 0
xyz_cam1s = utils.geom.depth2pointcloud(depth_cam1s, pix_T_cams)

origin_T_cam0s = torch.tensor(data['origin_T_cam0s'][9:10])
origin_T_cam1s = torch.tensor(data['origin_T_cam3s'][9:10])

xyz_origin0 = apply_4x4(origin_T_cam0s, xyz_cam0s)
xyz_origin1 = apply_4x4(origin_T_cam1s, xyz_cam1s)
xyz_combined = torch.cat([xyz_origin1, xyz_origin0], dim=1)

pcd0 = make_pcd(xyz_origin0[0])
pcd1 = make_pcd(xyz_origin1[0])
pcd2 = make_pcd(xyz_combined[0])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([mesh_frame, pcd0])
o3d.visualization.draw_geometries([mesh_frame, pcd1])
o3d.visualization.draw_geometries([mesh_frame, pcd2])

st()
aa=1

# rgb_cam0s = data['rgb_cam0s'][9:10]
# plt.imshow(rgb_cam0s[0])
# plt.show(block=True)
# rgb_cam1s = data['rgb_cam3s'][9:10]
# plt.imshow(rgb_cam1s[0])
# plt.show(block=True)


