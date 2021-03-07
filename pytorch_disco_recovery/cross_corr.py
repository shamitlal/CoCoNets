import numpy as np 
import torch
import utils_basic
# import kornia
# import multiprocess_flag 
import hyperparams as hyp

'''
Using these references to generate rotations:
https://torchgeometry.readthedocs.io/en/latest/warp_affine.html
https://bitbucket.org/adamharley/discovery/src/d55d58e556ee9c66fead01cdf4deef42426321b5/cross_corr.py#lines-256
'''
def rotate_tensor_along_y_axis(tensor, gamma):
    B = tensor.shape[0]
    tensor = tensor.to("cpu")
    assert tensor.ndim == 6, "Tensors should have 6 dimensions."
    tensor = tensor.float()
    # B,S,C,D,H,W
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)
    tensor_ = __p(tensor)
    tensor_ = tensor_.permute(0, 1, 3, 2, 4) # Make it BS, C, H, D, W  (i.e. BS, C, y, z, x)
    BS, C, H, D, W = tensor_.shape
    
    # merge y dimension with channel dimension and rotate with gamma_
    tensor_y_reduced = tensor_.reshape(BS, C*H, D, W)
    # # gammas will be rotation angles along y axis.
    # gammas = torch.arange(10, 360, 10)

    # define the rotation center
    center = torch.ones(1, 2)
    center[..., 0] = tensor_y_reduced.shape[3] / 2  # x
    center[..., 1] = tensor_y_reduced.shape[2] / 2  # z
    
    # define the scale factor
    scale = torch.ones(1)
    
    gamma_ = torch.ones(1) * gamma
    
    # compute the transformation matrix
    M = kornia.get_rotation_matrix2d(center, gamma_, scale)
    M = M.repeat(BS, 1, 1)
    # apply the transformation to original image
    # st()
    tensor_y_reduced_warped = kornia.warp_affine(tensor_y_reduced, M, dsize=(D, W))
    tensor_y_reduced_warped = tensor_y_reduced_warped.reshape(BS, C, H , D, W)
    tensor_y_reduced_warped = tensor_y_reduced_warped.permute(0, 1, 3, 2, 4)
    tensor_y_reduced_warped = __u(tensor_y_reduced_warped)
    return tensor_y_reduced_warped.cuda()
    
'''
Rotate tensor_e along y axis and find rotation which best aligns it with tensor_g
'''
def orient_and_calculate_scores(tensor_g, tensor_e):
    assert tensor_g.shape  == tensor_e.shape, "Both tensors shape should match exactly"
    tensor_g = torch.from_numpy(tensor_g).cuda()
    tensor_g = tensor_g.permute(0, 4, 1, 2, 3).unsqueeze(1)
    tensor_e = torch.from_numpy(tensor_e).cuda()
    tensor_e = tensor_e.permute(0, 4, 1, 2, 3).unsqueeze(1)
    B, S, C, D, H, W = tensor_g.shape
    vec_g = tensor_g.reshape(B*S, -1)
    # st()
    # gammas will be rotation angles along z axis.
    gammas = torch.arange(0, 360, 10)
    tensor_g_vec = tensor_g.reshape(-1)
    scores = torch.zeros((tensor_g.shape[0], tensor_e.shape[0])).cuda() - 1000000
    for gamma in gammas:
        rotated_tensor_e = rotate_tensor_along_y_axis(tensor_e, gamma)
        vec_e = rotated_tensor_e.reshape(B*S, -1)

        vec_e = utils_basic.l2_normalize(vec_e)
        vec_g = utils_basic.l2_normalize(vec_g)

        scores_gamma = torch.matmul(vec_g, vec_e.t())
        scores = torch.max(scores, scores_gamma)
    return scores

def orient_and_calculate_scores_cuda(tensor_g, tensor_e):
    assert tensor_g.shape  == tensor_e.shape, "Both tensors shape should match exactly"
    tensor_g = tensor_g.permute(0, 4, 1, 2, 3).unsqueeze(1)
    tensor_e = tensor_e.permute(0, 4, 1, 2, 3).unsqueeze(1)
    B, S, C, D, H, W = tensor_g.shape
    vec_g = tensor_g.reshape(B*S, -1)
    # st()
    # gammas will be rotation angles along z axis.
    gammas = torch.arange(0, 360, 10)
    tensor_g_vec = tensor_g.reshape(-1)
    scores = torch.zeros((tensor_g.shape[0], tensor_e.shape[0])).cuda() - 1000000
    for gamma in gammas:
        rotated_tensor_e = rotate_tensor_along_y_axis(tensor_e, gamma)
        vec_e = rotated_tensor_e.reshape(B*S, -1)
        vec_e = utils_basic.l2_normalize(vec_e)
        vec_g = utils_basic.l2_normalize(vec_g)
        scores_gamma = torch.matmul(vec_g, vec_e.t())
        scores = torch.max(scores, scores_gamma)
    return scores


class meshgrid_based_rotation:
    """
    This helper precomputed a fixed grid of indices for rotation.
    This is suitable when you always want to rotate a fixed list of angles for all
    the voxels in the batch and you don't want to compute the rotation matrix
    everytimes you want to transform
    """
    def __init__(self, D, H, W, angleIncrement=10.0):
        self.D = D
        self.H = H
        self.W = W
        self.centerD = self.D//2
        self.centerW = self.W//2
        self.EPS = 1e-5
        self.device = torch.device("cuda")
        self.angleIncrement = angleIncrement
        # with torch.no_grad():
        self.anglesDeg = -1*torch.arange(0, 360, angleIncrement).to(self.device)
        self.anglesRad = kornia.deg2rad(self.anglesDeg).to(self.device)
        self.precomputeMeshGrids()
    
    '''
    Get rotation matrix which rotates embedding along center
    https://math.stackexchange.com/questions/2093314/rotation-matrix-of-rotation-around-a-point-other-than-the-origin
    '''
    def precomputeMeshGrids(self):
        # Meshgrid along D and W
        
        dInd = torch.arange(self.D).to(self.device)
        wInd = torch.arange(self.W).to(self.device)

        dMesh, wMesh = torch.meshgrid(dInd, wInd)

        cosThetas = torch.cos(self.anglesRad)
        sinThetas = torch.sin(self.anglesRad)
        numAngles = self.anglesRad.shape[0]
        self.numAngles = numAngles
        
        dMesh = dMesh.unsqueeze(0).repeat(numAngles,1,1).to(torch.float)
        wMesh = wMesh.unsqueeze(0).repeat(numAngles,1,1).to(torch.float)

        cosThetas = cosThetas.view(-1, 1, 1)
        sinThetas = sinThetas.view(-1, 1, 1)

        # We will be rotating along the center.
        self.dRot = cosThetas*dMesh - sinThetas*wMesh - cosThetas*self.centerD + sinThetas*self.centerW + self.centerD #+ self.EPS # [36, 5, 5]
        self.wRot   = sinThetas*dMesh + cosThetas*wMesh - sinThetas*self.centerD - cosThetas*self.centerW + self.centerW #+ self.EPS # [36, 5, 5]
        
        self.dRot = torch.clamp(self.dRot, 0+self.EPS, self.D-1-self.EPS)
        self.wRot = torch.clamp(self.wRot, 0+self.EPS, self.W-1-self.EPS)
    
    def rotateTensor(self, tensor, interpolation="bilinear"):
        assert tensor.ndim == 5, "Tensor should have 5 dimensions (B,C,D,H,W)"

        B,C,D,H,W = tensor.shape
        tensor = tensor.permute(0, 1, 3, 2, 4) # torch.Size([2, 32, 16, 16, 16])
        tensor = tensor.reshape(B, C*H, D, W)
        rotated =  self.rotate2D(tensor, interpolation) # torch.Size([2, 512, 36, 16, 16])
        rotated = rotated.reshape(B, C, H, self.numAngles, D, W) 
        rotated = rotated.permute(0, 3, 1, 4, 2, 5) # B, numAngles, C, D, H, W
        return rotated
    
    def rotateTensorToPose(self,tensor,pose,interpolation="bilinear"):
        assert tensor.ndim == 5, "Tensor should have 5 dimensions (B,C,D,H,W)"
        B,C,D,H,W = tensor.shape
        tensor = tensor.permute(0, 1, 3, 2, 4) # torch.Size([2, 32, 16, 16, 16])
        tensor = tensor.reshape(B, C*H, D, W)
        rotated =  self.rotate2D_pose(tensor,pose, interpolation=interpolation).squeeze(1) # torch.Size([2, 512, 36, 16, 16])
        rotated = rotated.reshape(B, C, H,1, D, W) 
        rotated = rotated.permute(0, 3, 1, 4, 2, 5).squeeze(1)
        return rotated

    def rotate2D(self, tensor, interpolation="bilinear"):
        if interpolation == "nearestNeighbor":
            out = self.nearestNeighborInterpolation(tensor)
        else:
            out = self.bilinearInterpolation(tensor)
            out[:,:,0,:,:] = tensor # 0 degree rotation is original tensor.
            out[:, :, :, self.centerD, self.centerW] = tensor.unsqueeze(2)[:, :, :, self.centerD, self.centerW] # set the value of center pixel
        return out

    def rotate2D_pose(self, tensor,pose, interpolation="bilinear"):
        if interpolation == "nearestNeighbor":
            assert False
            out = self.nearestNeighborInterpolation(tensor)
        else:
            out = self.bilinearInterpolation_toPose(tensor,pose)
            # out[:,:,0,:,:] = tensor # 0 degree rotation is original tensor.
            # out[:, :, :, self.centerD, self.centerW] = tensor.unsqueeze(2)[:, :, :, self.centerD, self.centerW] # set the value of center pixel
        return out

    
    def bilinearInterpolation(self, tensor):
        dfloor, dceil, wfloor, wceil = self.getFloorAndCeil() # torch.Size([36, 60, 60])
        fq12 = tensor[:,:,dceil, wfloor] # torch.Size([2, 3, 36, 60, 60])
        fq22 = tensor[:,:,dceil, wceil]
        fq11 = tensor[:,:,dfloor,wfloor]
        fq21 = tensor[:,:,dfloor,wceil]
        # y1, y2, x1, x2 = dfloor.unsqueeze(0).unsqueeze(0), dceil.unsqueeze(0).unsqueeze(0), wfloor.unsqueeze(0).unsqueeze(0), wceil.unsqueeze(0).unsqueeze(0)
        # y = self.dRot.unsqueeze(0).unsqueeze(0)
        # x = self.wRot.unsqueeze(0).unsqueeze(0)
        y1, y2, x1, x2 = dfloor.unsqueeze(0).unsqueeze(0).to(torch.float32), dceil.unsqueeze(0).unsqueeze(0).to(torch.float32), wfloor.unsqueeze(0).unsqueeze(0).to(torch.float32), wceil.unsqueeze(0).unsqueeze(0).to(torch.float32)
        y = self.dRot.unsqueeze(0).unsqueeze(0).to(torch.float32)
        x = self.wRot.unsqueeze(0).unsqueeze(0).to(torch.float32)
        one = (x2-x )*(y2-y)
        two = (x-x1)*(y2-y)
        three = (x2-x)*(y-y1)
        four = (x-x1)*(y-y1)


        one[torch.where(one == 0.0)] = 0.25
        two[torch.where(two == 0.0)] = 0.25
        three[torch.where(three == 0.0)] = 0.25
        four[torch.where(four == 0.0)] = 0.25

        # st()
        # self.EPS = 0.0
        numerator = fq11*one + fq21*two + fq12*three + fq22*four
        # numerator = torch.clamp(numerator,min=self.EPS)
        denominator = (x2-x1)*(y2-y1)
        
        denominator[torch.where(denominator == 0.0)] = 1.0
        # st()
        out = numerator/denominator
        return out
    

    def bilinearInterpolation_toPose(self, tensor, pose):
        rotated_tensors = []
        for index,tensor_i in enumerate(tensor):
            tensor_i = tensor_i.unsqueeze(0)
            pose_i = pose[index:index+1]
            dRot,wRot,dfloor, dceil, wfloor, wceil = self.getFloorAndCeil_pose(pose_i)

            fq12 = tensor_i[:,:,dceil, wfloor] 
            fq22 = tensor_i[:,:,dceil, wceil]
            fq11 = tensor_i[:,:,dfloor,wfloor]
            fq21 = tensor_i[:,:,dfloor,wceil]

            y1, y2, x1, x2 = dfloor.unsqueeze(0).unsqueeze(0).to(torch.float32), dceil.unsqueeze(0).unsqueeze(0).to(torch.float32), wfloor.unsqueeze(0).unsqueeze(0).to(torch.float32), wceil.unsqueeze(0).unsqueeze(0).to(torch.float32)
            y = dRot.unsqueeze(0).unsqueeze(0).to(torch.float32)
            x = wRot.unsqueeze(0).unsqueeze(0).to(torch.float32)
            # st()
            one = (x2-x )*(y2-y)
            two = (x-x1)*(y2-y)
            three = (x2-x)*(y-y1)
            four = (x-x1)*(y-y1)
            # st()
            one[one == 0.0] = 0.25
            two[two == 0.0] = 0.25
            three[three == 0.0] = 0.25
            four[four == 0.0] = 0.25


            numerator = fq11*one + fq21*two + fq12*three + fq22*four
            denominator = (x2-x1)*(y2-y1)
            
            denominator[torch.where(denominator == 0.0)] = 1.0
            out = numerator/denominator
            rotated_tensors.append(out)
        rotated_tensors = torch.cat(rotated_tensors, dim=0)
        return rotated_tensors

    def nearestNeighborInterpolation(self, tensor):
        dfloor, dceil, wfloor, wceil = self.getFloorAndCeil()
        out = tensor[:, :, dfloor, wfloor]
        return out

    def getFloorAndCeil(self):
        dfloor = torch.floor(self.dRot).long()
        dceil = torch.ceil(self.dRot).long()
        wfloor = torch.floor(self.wRot).long()
        wceil = torch.ceil(self.wRot).long()
        return dfloor, dceil, wfloor, wceil

    def get_inverse(self,pose):
        inverted_poses = (self.numAngles - pose)%self.numAngles
        return inverted_poses

    def getFloorAndCeil_pose(self,pose):
        inverted_pose = self.get_inverse(pose)
        dRot_temp = self.dRot.clone()
        wRot_temp = self.wRot.clone()

        dRot = dRot_temp[inverted_pose]
        wRot = wRot_temp[inverted_pose]
        dfloor = torch.floor(dRot).long()
        dceil = torch.ceil(dRot).long()
        wfloor = torch.floor(wRot).long()
        wceil = torch.ceil(wRot).long()
        return dRot, wRot, dfloor, dceil, wfloor, wceil
    
    
    def getFloorAndCeil2(self):
        dfloor = torch.floor(self.dRot)
        dfloor = torch.clamp(dfloor, 0, self.D-1).long()

        dceil = torch.ceil(self.dRot)
        dceil = torch.clamp(dceil, 0, self.D-1).long()

        wfloor = torch.floor(self.wRot)
        wfloor = torch.clamp(wfloor, 0, self.W-1).long()
        
        wceil = torch.ceil(self.wRot)
        wceil = torch.clamp(wceil, 0, self.W-1).long()
        return dfloor, dceil, wfloor, wceil
