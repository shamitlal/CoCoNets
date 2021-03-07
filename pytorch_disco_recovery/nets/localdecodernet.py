import torch
import numpy as np
import torch.nn as nn
import utils.misc
import torch.nn.functional as F
import pickle
from torch import distributions as dist
from utils.basic import *
# import utils_implicit
import ipdb
st = ipdb.set_trace
# st()
import sys
sys.path.append("..")

# import archs.encoder3D
import hyperparams as hyp
# from utils_basic import *
# import utils_improc
# import utils_misc


def normalize_gridcloud(xyz, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    
    z = 2.0*(z / float(Z-1)) - 1.0
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xyz = torch.stack([x,y,z], dim=-1)
    
    if clamp_extreme:
        xyz = torch.clamp(xyz, min=-2.0, max=2.0)
    return xyz
    
class LocalDecoderParent(nn.Module):
    def __init__(self):
        super(LocalDecoderParent, self).__init__()
        if hyp.do_implicit_occ or hyp.do_tsdf_implicit_occ:
            self.localdecoder_occ = LocalDecoder(dim=3, c_dim=hyp.feat3d_dim,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1).cuda()

        if hyp.do_localdecoder_render:
            if hyp.hypervoxel:
                self.localdecoder_feats = LocalDecoderHyperVoxel(dim=hyp.feat3d_dim, c_dim=3,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=3).cuda()
                
            else:
                self.localdecoder_feats = LocalDecoder(dim=3, c_dim=hyp.feat3d_dim,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=3).cuda()
        else:
            if hyp.do_concat:
                self.localdecoder_feats = LocalDecoder_Concat(dim=3, c_dim=hyp.feat3d_dim,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=hyp.feat3d_dim).cuda()

            elif hyp.hypervoxel:
                self.localdecoder_feats = LocalDecoderHyperVoxel(dim=hyp.feat3d_dim, c_dim=3,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=hyp.feat3d_dim).cuda()
                
            else:
                self.localdecoder_feats = LocalDecoder(dim=3, c_dim=hyp.feat3d_dim,
                                hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=hyp.feat3d_dim).cuda()

    def forward(self, embed_g, embed_e, xyz_mem_g, xyz_mem0_g=None, summ_writer=None, rgb_g=None):
        '''
        ipdb> pointcl
        ipdb> pcl_mem.shape
        torch.Size([1, 100000, 3])
        ipdb> feats.shape 
        torch.Size([1, 3, 72, 72, 72])
        '''
        if hyp.do_localdecoder_render:
            assert rgb_g is not None
            total_loss = 0.0
            feats_source_emb3d, _ = self.localdecoder_feats(None, xyz_mem_g, embed_e, summ_writer=summ_writer)
            # st()
            rgb_e = torch.nn.functional.tanh(feats_source_emb3d)*0.5
            loss_rgb = torch.mean(l1_on_axis(rgb_e-rgb_g, 2, keepdim=True))
            total_loss = utils.misc.add_loss(f'imp_rendernet/implicit_loss_render', total_loss, loss_rgb, 1, summ_writer)
            return total_loss,rgb_e

        elif embed_g is None and (hyp.summ_pca_points_3d or hyp.summ_pca_points_2d) :
            feats_target_emb3d, _ = self.localdecoder_feats(None, xyz_mem_g, embed_e, summ_writer=summ_writer)
            return feats_target_emb3d
        
        else:
            feats_target_emb3d, _ = self.localdecoder_feats(None, xyz_mem_g, embed_g, summ_writer=summ_writer)
            if hyp.implicit_camX:
                feats_source_emb3d, _ = self.localdecoder_feats(None, xyz_mem0_g, embed_e, summ_writer=summ_writer)
            else:
                feats_source_emb3d, _ = self.localdecoder_feats(None, xyz_mem_g, embed_e, summ_writer=summ_writer)
            dim = feats_target_emb3d.shape[1]
            ml_3d_sample_points = np.random.permutation(dim)[:hyp.implicit_ml_num_sample_points]
            feats_target_sampled_emb3d = feats_target_emb3d[:, ml_3d_sample_points].permute(0,2,1)
            feats_source_sampled_emb3d = feats_source_emb3d[:, ml_3d_sample_points].permute(0,2,1)
            
            neg_ml_3d_sample_points = np.random.permutation(dim)[:hyp.implicit_ml_num_sample_points]
            feats_neg_sampled_emb3d = feats_target_emb3d[:, neg_ml_3d_sample_points].permute(0,2,1)

            return feats_target_sampled_emb3d, feats_source_sampled_emb3d, feats_neg_sampled_emb3d
        

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx





# Resnet Blocks
class ResnetBlockFC_Hyper(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        # self.fc_0 = nn.Linear(size_in, size_h)
        # self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = True
            # self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        # nn.init.zeros_(self.fc_1.weight)

    def forward(self,hyperwts,x,batch_index):
        kls,bias = hyperwts
        net = F.linear(self.actvn(x),kls[0][batch_index],bias[0][batch_index])
        dx = F.linear(self.actvn(net),kls[1][batch_index],bias[1][batch_index])

        if self.shortcut is not None:
            x_s = F.linear(x,kls[2][batch_index])
        else:
            x_s = x

        return x_s + dx





class LocalDecoder_Hyper(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super(LocalDecoder_Hyper, self).__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        weight_info = pickle.load(open("hypernet.p","rb"))

        decoder_kernel_names,decoder_kernel_shape = weight_info['decoder_kernel']
        decoder_bias_names,decoder_bias_shape = weight_info['decoder_bias']
        # st()
        # if c_dim != 0:
        #     self.fc_c = nn.ModuleList([
        #         nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        #     ])


        # self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC_Hyper(hidden_size) for i in range(n_blocks)
        ])

        # self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    
    # def sample_grid_feature(self, p, c):
    #     # st()
    #     # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
    #     # st()
    #     _,  _,  Z, Y, X = c.shape
    #     vgrid = normalize_gridcloud(p, Z , Y, X)
    #     vgrid = vgrid[:, :, None, None].float()
    #     # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
    #     # acutally trilinear interpolation if mode = 'bilinear'
    #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
    #     return c
    # @classmethod
    def sample_grid_feature(self, p, c):
        # st()
        # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        # st()
        _,  _,  Z, Y, X = c.shape
        vgrid = normalize_gridcloud(p, Z , Y, X)
        vgrid = vgrid[:, :, None, None].float()
        # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    # @classmethod
    def get_sampled_features(self, pcl_mem, c_plane):
        c = self.sample_grid_feature(pcl_mem, c_plane)
        c = c.transpose(1, 2)
        return c


    def forward(self, hyper_wts, pcl, pcl_mem, c_plane, summ_writer=None):
        if self.c_dim != 0:
            c = 0
            c += self.sample_grid_feature(pcl_mem, c_plane)
            c = c.transpose(1, 2)
        implicit_feats = c
        hyper_kls, hyper_bias = hyper_wts

        fc_p_kls = hyper_kls[:1]
        fc_p_bias = hyper_bias[:1]

        fc_c_kls = hyper_kls[1:6]
        fc_c_bias = hyper_bias[1:6]

        blocks_kls = hyper_kls[6:16]
        blocks_bias = hyper_bias[6:16]

        fc_out_kls = hyper_kls[16:]
        fc_out_bias = hyper_bias[16:]

        pcl_norm = (pcl_mem.float()/(hyp.Z_train//2) - 0.5)

        out_list = []

        for index_batch in range(c.shape[0]):
            net = F.linear(pcl_norm[index_batch:index_batch+1], fc_p_kls[0][index_batch], bias=fc_p_bias[0][index_batch])

            for i in range(self.n_blocks):
                if self.c_dim != 0:
                    net = net + F.linear(c[index_batch:index_batch+1], fc_c_kls[i][index_batch], bias=fc_c_bias[i][index_batch])

                net = self.blocks[i]([blocks_kls[2*i:2*(i+1)],blocks_bias[2*i:2*(i+1)]],net,index_batch)

            out = F.linear(self.actvn(net),fc_out_kls[0][index_batch],bias=fc_out_bias[0][index_batch])

            out_list.append(out)

        out_stack = torch.cat(out_list,dim=0)
        out_stack = out_stack.squeeze(-1)
        return out_stack, implicit_feats






class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    # sample_mode = "bilinear"
    def __init__(self, dim=3, c_dim=128,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1):
        super(LocalDecoder, self).__init__()
        print('Implicit Local Decoder...')
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.fc_p = nn.Linear(dim, hidden_size)
        # st()
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)
        
        if leaky:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        else:
            self.actvn = F.relu
        if hyp.nearest_neighbour:
            self.sample_mode = "nearest"
        else:
            self.sample_mode = "bilinear"
        self.padding = padding
    
    def normalize_3d_coordinate(self, p, padding=0.1):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        '''
        p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
        p_nor = p_nor + 0.5 # range (0, 1)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor

    # @classmethod
    def sample_grid_feature(self, p, c):
        # st()
        # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        # st()
        _,  _,  Z, Y, X = c.shape
        vgrid = normalize_gridcloud(p, Z , Y, X)
        vgrid = vgrid[:, :, None, None].float()
        # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    # @classmethod
    def get_sampled_features(self, pcl_mem, c_plane):
        c = self.sample_grid_feature(pcl_mem, c_plane)
        c = c.transpose(1, 2)
        return c

    def forward(self, pcl, pcl_mem, c_plane,summ_writer=None):
        if self.c_dim != 0:
            c = 0
            c += self.sample_grid_feature(pcl_mem, c_plane)
            c = c.transpose(1, 2)

        implicit_feats = c

        if hyp.use_delta_mem_coords:
            pcl_norm = (pcl_mem - pcl_mem.int()) - 0.5
        else:
            pcl_norm = (pcl_mem.float()/(hyp.Z_train//2) - 0.5)        
        # st()
        
        net = self.fc_p(pcl_norm.cuda())

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out, implicit_feats





class LocalDecoder_Concat(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    def __init__(self, dim=3, c_dim=128,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1):
        super(LocalDecoder_Concat, self).__init__()
        print('Implicit Local Decoder...')
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.fc_p = nn.Linear(dim, hidden_size)
        # st()
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(160) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(160, out_dim)
        
        if leaky:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        else:
            self.actvn = F.relu

        self.padding = padding
        if hyp.nearest_neighbour:
            self.sample_mode = "nearest"
        else:
            self.sample_mode = "bilinear"    
    def normalize_3d_coordinate(self, p, padding=0.1):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        '''
        p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
        p_nor = p_nor + 0.5 # range (0, 1)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor

    # @classmethod
    def sample_grid_feature(self, p, c):
        # st()
        # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        # st()
        _,  _,  Z, Y, X = c.shape
        vgrid = normalize_gridcloud(p, Z , Y, X)
        vgrid = vgrid[:, :, None, None].float()
        # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
        # acutally trilinear interpolation if mode = 'bilinear'
        c_bilinear = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)
        c_nearest = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode='nearest').squeeze(-1).squeeze(-1)
        c_total = torch.cat([c_bilinear,c_nearest],dim=1)
        return c_total

    # @classmethod
    def get_sampled_features(self, pcl_mem, c_plane):
        c = self.sample_grid_feature(pcl_mem, c_plane)
        c = c.transpose(1, 2)
        return c

    def forward(self, pcl, pcl_mem, c_plane,summ_writer=None):
        if self.c_dim != 0:
            c = 0
            c += self.sample_grid_feature(pcl_mem, c_plane)
            c = c.transpose(1, 2)

        implicit_feats = c
        
        if hyp.use_delta_mem_coords:
            pcl_norm = (pcl_mem - pcl_mem.int()) - 0.5
        else:
            pcl_norm = (pcl_mem.float()/(hyp.Z_train//2) - 0.5)        
        
        net = self.fc_p(pcl_norm.cuda())
        net = torch.cat([c,net],dim=-1)

        for i in range(self.n_blocks):
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out, implicit_feats














class LocalDecoderHyperVoxel(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1):
        super(LocalDecoderHyperVoxel, self).__init__()
        print('Implicit Hyper voxel Local Decoder...')
        self.c_dim = c_dim
        self.use_bias = True

        # architecture for prediction network:
        self.arch = torch.tensor([[c_dim, hidden_size], [hidden_size, hidden_size], [hidden_size, out_dim]]).cuda()

        total_params = self.get_total_params()
        
        self.hypernet = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),            
            nn.Linear(hidden_size, total_params)
        ).cuda()

        self.hypernet.apply(self.weights_init)
        
        # self.kernel_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances[index])*lambda_val[index]/((variance**2-weight_variances[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.layer_size)])
        # self.bias_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, i[0]), mean=0, std=(variance*bias_variances[index])*lambda_val[index]/((variance**2-bias_variances[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.layer_size)])
        if hyp.nearest_neighbour:
            self.sample_mode = "nearest"
        else:
            self.sample_mode = "bilinear"

        self.padding = padding
    def weights_init(self,m):
        classname = m.__class__.__name__
        # st()
        if classname == "Linear":
            torch.nn.init.normal_(m.weight, 0.0, 0.21)
            torch.nn.init.zeros_(m.bias)        
    
    def get_total_params(self):
        total_params = torch.sum(torch.prod(self.arch, dim=1))
        if self.use_bias:
            total_params += torch.sum(self.arch[:,-1])
        return total_params.item()


    def sample_grid_feature(self, p, c):
        # st()
        # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        # st()
        _,  _,  Z, Y, X = c.shape
        vgrid = normalize_gridcloud(p, Z , Y, X)
        vgrid = vgrid[:, :, None, None].float()
        # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def get_sampled_features(self, pcl_mem, c_plane):
        c = self.sample_grid_feature(pcl_mem, c_plane)
        c = c.transpose(1, 2)
        return c

    def normalize_3d_coordinate(self, p, padding=0.1):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        '''
        p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
        p_nor = p_nor + 0.5 # range (0, 1)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor

    # def sample_grid_feature(self, p, c):
    #     # st()
    #     # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
    #     # st()
    #     _,  _,  Z, Y, X = c.shape
    #     vgrid = normalize_gridcloud(p, Z , Y, X)
    #     vgrid = vgrid[:, :, None, None].float()
    #     # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
    #     # acutally trilinear interpolation if mode = 'bilinear'
    #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
    #     return c

    def forward(self, pcl, pcl_mem, c_plane, summ_writer=None):
        B, N, _ = pcl_mem.shape
        c = self.sample_grid_feature(pcl_mem, c_plane)
        c = c.transpose(1, 2)
        implicit_feats = c
        hyperwts = self.hypernet(c)
        # if hyp.vis_feat_weights:
        #     summ_writer.summ_histogram("hypernet_wts", hyperwts.clone().cpu().data.numpy())        
            # for ind,weight_name in enumerate(self.decoder_kernel_names):
            # st()
            # hyperwts
            # summ_writer.summ_histogram("decoder_weight_"+weight_name, decoder_kernels[ind].clone().cpu().data.numpy())        
        pointer = 0
        # st()

        if hyp.use_delta_mem_coords:
            pcl_norm = (pcl_mem - pcl_mem.int()) - 0.5
        else:
            pcl_norm = (pcl_mem.float()/(hyp.Z_train//2) - 0.5)


        # st()
        out = pcl_norm.reshape(B*N, -1).unsqueeze(1)


        for layer_num, layer in enumerate(self.arch):
            if layer_num > 0:
                out =  F.leaky_relu(out)
            
            wts_to_use = torch.prod(layer)
            wts = hyperwts[:, :, pointer: pointer + wts_to_use]
            pointer += wts_to_use

            wts = wts.reshape(B*N, layer[0], layer[1])
            out = out @ wts

            if self.use_bias:
                bias = hyperwts[:, :, pointer: pointer + layer[-1].item()]
                pointer += layer[-1].item()
                bias = bias.reshape(B*N, -1).unsqueeze(1)
                # bias = bias.permute(1,0,2)
                out = out + bias
        # st()
        out = out.squeeze(1)
        out = out.reshape(B,N,-1)
        out = out.squeeze(-1)

        return out, implicit_feats




class LocalDecoderAddBaseline(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1):
        super(LocalDecoderAddBaseline, self).__init__()
        print('Implicit Hyper voxel Local Decoder...')
        self.c_dim = c_dim
        self.use_bias = True

        # architecture for prediction network:
        self.fc_p = nn.Linear(dim, hidden_size)

        self.fc_c_1 = nn.Linear(c_dim, hidden_size)

        self.fc_c_2 = nn.Linear(c_dim, hidden_size)

        self.fc_cs = [self.fc_c_1, self.fc_c_2]
        

        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        ).cuda()

        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        ).cuda()

        self.blocks = [self.block1, self.block2]

        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.sample_mode = sample_mode
        self.padding = padding
    
    def get_total_params(self):
        total_params = torch.sum(torch.prod(self.arch, dim=1))
        if self.use_bias:
            total_params += torch.sum(self.arch[:,-1])
        return total_params.item()


    def normalize_3d_coordinate(self, p, padding=0.1):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        '''
        p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
        p_nor = p_nor + 0.5 # range (0, 1)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor

    def sample_grid_feature(self, p, c):
        # st()
        # p_nor = self.normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        # st()
        _,  _,  Z, Y, X = c.shape
        vgrid = normalize_gridcloud(p, Z , Y, X)
        vgrid = vgrid[:, :, None, None].float()
        # vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)\
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, pcl, pcl_mem, c_plane,summ_writer=None):
        if self.c_dim != 0:
            c = 0
            c += self.sample_grid_feature(pcl_mem, c_plane)
            c = c.transpose(1, 2)

        if hyp.use_delta_mem_coords:
            pcl_norm = (pcl_mem - pcl_mem.int()) - 0.5
        else:
            pcl_norm = (pcl_mem.float()/(hyp.Z_train//2) - 0.5)
        
        # st()

        net = self.fc_p(pcl_norm.cuda())

        for i in range(2):
            if self.c_dim != 0:
                net = net + self.fc_cs[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(net)
        out = out.squeeze(-1)
        return out
