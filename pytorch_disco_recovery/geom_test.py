import torch
import utils_geom
import utils_basic
import numpy as np

# XMIN = -16.0 # right (neg is left)
# XMAX = 16.0 # right
# YMIN = -8.0 # down (neg is up)
# YMAX = 8.0 # down
# ZMIN = -16.0 # forward
# ZMAX = 16.0 # forward

# XMIN = -64.0 # right (neg is left)
# XMAX = 64.0 # right
# YMIN = -32.0 # down (neg is up)
# YMAX = 32.0 # down
# ZMIN = -64.0 # forward
# ZMAX = 64.0 # forward

# XMIN = -0.5 # right (neg is left)
# XMAX = 127.5 # right
# YMIN = -0.5 # down (neg is up)
# YMAX = 63.5 # down
# ZMIN = -0.5 # forward
# ZMAX = 127.5 # forward


# XMIN = -0.5 # right (neg is left)
# XMAX = 127.5 # right
# YMIN = -0.5 # down (neg is up)
# YMAX = 63.5 # down
# ZMIN = -0.5 # forward
# ZMAX = 127.5 # forward

# SIZE = 32
# B = 1
# Z = (int(SIZE*4))
# Y = (int(SIZE*2))
# X = (int(SIZE*4))

XMIN = -2 # right (neg is left)
XMAX = 2 # right
YMIN = -1 # down (neg is up)
YMAX = 1 # down
ZMIN = -2 # forward
ZMAX = 2 # forward

B = 1
Z = 4
Y = 2
X = 4

print('X', X)
print('Y', Y)
print('Z', Z)

def get_mem_T_ref(B, Z, Y, X):
    # (sometimes we want the mat itself)
    # note this is not a rigid transform

    VOX_SIZE_X = (XMAX-XMIN)/float(X)
    VOX_SIZE_Y = (YMAX-YMIN)/float(Y)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(Z)
    print('VOX_SIZE_X', VOX_SIZE_X)
    print('VOX_SIZE_Y', VOX_SIZE_Y)
    print('VOX_SIZE_Z', VOX_SIZE_Z)
    
    # translation
    # (this makes the left edge of the leftmost voxel correspond to XMIN)
    center_T_ref = utils_geom.eye_4x4(B)
    center_T_ref[:,0,3] = -XMIN-VOX_SIZE_X/2.0
    center_T_ref[:,1,3] = -YMIN-VOX_SIZE_Y/2.0
    center_T_ref[:,2,3] = -ZMIN-VOX_SIZE_Z/2.0
    # print('center_T_ref', center_T_ref.cpu().numpy())

    # scaling
    # (this makes the right edge of the rightmost voxel correspond to XMAX)
    mem_T_center = utils_geom.eye_4x4(B)
    mem_T_center[:,0,0] = 1./VOX_SIZE_X
    mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    # print('mem_T_center', mem_T_center.cpu().numpy())
    mem_T_ref = utils_basic.matmul2(mem_T_center, center_T_ref)
    # print('mem_T_ref', mem_T_ref.cpu().numpy())

    return mem_T_ref

mem_T_ref = get_mem_T_ref(B, Z, Y, X)
ref_T_mem = mem_T_ref.inverse()
print('mem_T_ref', mem_T_ref)

print('MIN')
xyz_ref = np.array([XMIN,YMIN,ZMIN]).astype(np.float32)
xyz_ref = torch.from_numpy(xyz_ref).cuda()
xyz_ref = xyz_ref.reshape(1, 1, 3)
xyz_mem = utils_geom.apply_4x4(mem_T_ref, xyz_ref)
print('xyz_ref', xyz_ref.cpu().numpy())
print('xyz_mem', xyz_mem.cpu().numpy())

print('ONE REF')
xyz_ref = np.array([1.0,1.0,1.0]).astype(np.float32)
xyz_ref = torch.from_numpy(xyz_ref).cuda()
xyz_ref = xyz_ref.reshape(1, 1, 3)
xyz_mem = utils_geom.apply_4x4(mem_T_ref, xyz_ref)
print('xyz_ref', xyz_ref.cpu().numpy())
print('xyz_mem', xyz_mem.cpu().numpy())

print('ONE MEM')
xyz_mem = np.array([1.0,1.0,1.0]).astype(np.float32)
xyz_mem = torch.from_numpy(xyz_mem).cuda()
xyz_mem = xyz_mem.reshape(1, 1, 3)
xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
print('xyz_mem', xyz_mem.cpu().numpy())
print('xyz_ref', xyz_ref.cpu().numpy())

print('MID')
xyz_ref = np.array([0,0,0]).astype(np.float32)
xyz_ref = torch.from_numpy(xyz_ref).cuda()
xyz_ref = xyz_ref.reshape(1, 1, 3)
xyz_mem = utils_geom.apply_4x4(mem_T_ref, xyz_ref)
print('xyz_ref', xyz_ref.cpu().numpy())
print('xyz_mem', xyz_mem.cpu().numpy())



print('MAX')
xyz_ref = np.array([XMAX,YMAX,ZMAX]).astype(np.float32)
xyz_ref = torch.from_numpy(xyz_ref).cuda()
xyz_ref = xyz_ref.reshape(1, 1, 3)
xyz_mem = utils_geom.apply_4x4(mem_T_ref, xyz_ref)
print('xyz_ref', xyz_ref.cpu().numpy())
print('xyz_mem', xyz_mem.cpu().numpy())


