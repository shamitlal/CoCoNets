import torch
import torch.nn as nn
# import spconv
import time
import numpy as np
import inspect
import torch.nn.functional as F
 
def generate_sparse(feat, mask):
    B, C, D, H, W = list(feat.shape)
    coords = torch.nonzero(mask[:,0].int())
    # should be [N, 4]
    b, d, h, w = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
    coords_flatten = w + h*W + d*H*W + b*D*H*W
    # should be [N]
    feat_flatten = torch.reshape(feat.permute(0,2,3,4,1), (-1, C)).float()
    feat_flatten = feat_flatten[coords_flatten]
    coords = coords.int()
    sparse_feat = spconv.SparseConvTensor(feat_flatten, coords.to('cpu'), [D, H, W], B)
    # sparse_feat = spconv.SparseConvTensor(feat_flatten, coords, [D, H, W], B)
    return sparse_feat

class SparseNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(SparseNet3D, self).__init__()
        conv3d = []
        up_bn = [] #batch norm layer for deconvolution
        conv3d_transpose = []

        # self.conv3d = torch.nn.Conv3d(4, 32, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        # self.layers = []
        self.down_in_dims = [in_channel, chans, 2*chans]
        self.down_out_dims = [chans, 2*chans, 4*chans]
        self.down_ksizes = [4, 4, 4]
        self.down_strides = [2, 2, 2]
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('down dims: ', self.down_out_dims)

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            # print('3D CONV', end=' ')
             
            conv3d += [
                spconv.SparseConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                ]

        # self.conv3d = nn.ModuleList(conv3d)
        self.conv3d = spconv.SparseSequential(*conv3d)

        self.up_in_dims = [4*chans, 6*chans]
        self.up_bn_dims = [6*chans, 3*chans]
        self.up_out_dims = [4*chans, 2*chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
             
            conv3d_transpose += [
                spconv.SparseConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                ]
            up_bn.append(nn.BatchNorm1d(num_features=bn_dim))

        self.pre_final_feature = spconv.SparseSequential(
                spconv.ToDense(),
        )
        
        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Sequential(
                nn.Conv3d(in_channels=2*chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        )
        self.conv3d_transpose = spconv.SparseSequential(*conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)


    def forward(self, feat, mask):
        mask = mask > 0.5
        feat = feat
        sparse_feat = generate_sparse(feat, mask)
        sparse_feat = self.conv3d(sparse_feat)
        sparse_feat = self.conv3d_transpose(sparse_feat)
        dense_feat = self.pre_final_feature(sparse_feat)
        mask = 1.0 - (dense_feat==0).all(dim=1, keepdim=True).float()
        dense_feat = self.final_feature(dense_feat)
        return dense_feat, mask
        

        '''
        skipcons = []
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv3d_transpose_layer, bn_layer) in enumerate(zip(self.conv3d_transpose, self.up_bn)):
            feat = conv3d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) #skip connection by concatenation
            feat = bn_layer(feat)

        feat = self.final_feature(feat)
        '''
        # return feat



class SparseResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(SparseResNet3D, self).__init__()

        # in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        # self.down_sampler = nn.Sequential(
        #     nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
        #     nn.BatchNorm3d(num_features=out_dim),
        #     nn.LeakyReLU(),
        # )
        # self.maxpooling = nn.MaxPool3d(2, stride=2)

        # self.prep_layer = nn.Sequential(
        #     nn.Conv3d(in_channels=in_channel, out_channels=chans, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm3d(num_features=out_chans),
        #     nn.LeakyReLU(),
        # )

        self.squeeze_block = self.generate_squeeze(in_channel, chans)
        
        ksize, stride, padding = 3, 1, 1
        self.res_block1 = self.generate_block(chans, chans, ksize, stride, padding)
        self.lrelu_block1 = nn.LeakyReLU()
        self.res_block2 = self.generate_block(chans, chans, ksize, stride, padding)
        self.lrelu_block2 = nn.LeakyReLU()
        self.res_block3 = self.generate_block(chans, chans, ksize, stride, padding)
        self.lrelu_block3 = nn.LeakyReLU()
        self.res_block4 = self.generate_block(chans, chans, ksize, stride, padding)
        self.lrelu_block4 = nn.LeakyReLU()
        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)

    def generate_block(self, in_dim, out_dim, ksize, stride, padding):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, indice_key='subm0'),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            spconv.ToDense(),
            )
        return block

    def generate_squeeze(self, in_dim, out_dim):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.ToDense(),
            )
        return block

    def forward(self, feat, mask):
        # mask = self.maxpooling(mask)
        mask = mask > 0.5
        print('feat', feat.shape)
        # feat = self.down_sampler(feat)

        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat = self.squeeze_block(feat_sparse)
        
        feat_before = feat.clone()
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block1(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block1(feat)
        print('feat', feat.shape)

        feat_before = feat.clone()
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block2(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block2(feat)
        print('feat', feat.shape)

        feat_before = feat.clone()
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block3(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block3(feat)
        print('feat', feat.shape)

        feat_before = feat.clone()
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block4(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block4(feat)
        print('feat', feat.shape)

        feat = self.final_feature(feat)
        print('feat', feat.shape)
        return feat, mask.float()

class Simple3d(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Simple3d, self).__init__()
        
        self.super_block = self.generate_super(in_dim, mid_dim, out_dim)

        # self.super_block1 = self.generate_super(in_dim, mid_dim, mid_dim)
        # self.super_block2 = self.generate_super(mid_dim*2, mid_dim*2, mid_dim*2)
        # ksize, stride, padding = 3, 1, 1
        # self.res_block1 = self.generate_block(chans, chans, ksize, stride, padding)
        # self.lrelu_block1 = nn.LeakyReLU()
        # self.res_block2 = self.generate_block(chans, chans, ksize, stride, padding)
        # self.lrelu_block2 = nn.LeakyReLU()
        # self.res_block3 = self.generate_block(chans, chans, ksize, stride, padding)
        # self.lrelu_block3 = nn.LeakyReLU()
        # self.res_block4 = self.generate_block(chans, chans, ksize, stride, padding)
        # self.lrelu_block4 = nn.LeakyReLU()
        # # final 1x1x1 conv to get our desired pred_dim
        # self.final = self.generate_last(chans, pred_dim)

        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=mid_dim, out_channels=mid_dim*2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(num_features=mid_dim*2),
            nn.LeakyReLU(),
        )
        self.max_pooler = nn.MaxPool3d(2, stride=2)
        self.final = nn.Sequential(
            nn.Conv3d(in_channels=mid_dim*3, out_channels=out_dim, kernel_size=1, stride=1, padding=0),
        )

    def generate_super(self, in_dim, mid_dim, out_dim):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=mid_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=0, indice_key="subm0"),
            nn.BatchNorm1d(num_features=mid_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=0, indice_key="subm0"),
            nn.BatchNorm1d(num_features=mid_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=0, indice_key="subm0"),
            nn.BatchNorm1d(num_features=mid_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=0, indice_key="subm0"),
            nn.BatchNorm1d(num_features=mid_dim),
            nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1, stride=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=mid_dim),
            # nn.LeakyReLU(),
            # spconv.SubMConv3d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=out_dim),
            # nn.LeakyReLU(),
            # spconv.ToDense(),
            spconv.SubMConv3d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            spconv.ToDense(),
            )
        return block

    def generate_squeeze(self, in_dim, out_dim):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            # spconv.ToDense(),
            )
        return block

    def generate_block(self, in_dim, out_dim, ksize, stride, padding):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, indice_key='subm0'),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            # spconv.ToDense(),
            )
        return block

    def generate_last(self, in_dim, out_dim):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.ToDense(),
            )
        return block

    def forward(self, feat, mask):
        # mask = self.maxpooling(mask)
        # mask = mask > 0.5
        # print('feat', feat.shape)
        # feat = self.down_sampler(feat)

        # feat = generate_sparse(feat, mask) # sparse feat
        # feat = self.squeeze_block(feat)
        # feat = self.super_block(feat)

        feat = generate_sparse(feat, mask) # sparse feat
        feat = self.super_block(feat)

        # print('feat', feat.shape)
        # feat = generate_sparse(feat, mask > 0.5)
        # feat = self.super_block1(feat)
        # print('feat', feat.shape)

        # low_feat = feat.clone()
        
        # feat = self.down_sampler(feat)
        # mask = self.max_pooler(mask)
        # print('feat', feat.shape)

        # feat = generate_sparse(feat, mask > 0.5) 
        # feat = self.super_block2(feat)
        # print('feat', feat.shape)


        # feat = F.interpolate(feat, scale_factor=2, mode='trilinear')
        # feat = torch.cat([feat, low_feat], dim=1)
        # feat = self.final(feat)
        # print('feat', feat.shape)
        
        
        # feat_before = feat.clone()
        # feat_after = self.res_block1(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block1(feat)
        # print('feat', feat.shape)

        # feat_before = feat.clone()
        # feat_after = self.res_block2(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block2(feat)
        # print('feat', feat.shape)

        # feat_before = feat.clone()
        # feat_after = self.res_block3(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block3(feat)
        # print('feat', feat.shape)

        # feat_before = feat.clone()
        # feat_after = self.res_block4(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block4(feat)
        # print('feat', feat.shape)

        # feat = self.final(feat)
        # print('feat', feat.shape)
        return feat, mask.float()


if __name__ == "__main__":
    shape = [2, 4, 128,128, 32] 
    sparse_featnet = SparseResNet3D(in_channel=4, pred_dim=32).cuda()
    print(sparse_featnet.named_parameters)
    inputs = torch.rand(shape) # N, C, D, H, W
    mask = torch.max(inputs, 1, keepdim=True)[0] 
    mask[:,:,0:64]*=0.0
    time1 = time.time()
    out = sparse_featnet(inputs.cuda(), mask.cuda())
    print("time for sparse:", time.time()-time1)
    print(out.size())


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper
class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        # self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(
                spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm0"))
            # middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        # middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(
                spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm1"))
            # middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        # middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, feat, mask):
        mask = mask > 0.5
        feat = generate_sparse(feat, mask)
        feat = self.middle_conv(feat)
        feat = feat.dense()
        N, C, D, H, W = feat.shape
        return feat
    





class Custom3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Custom3D, self).__init__()

        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )
        self.maxpooling = nn.MaxPool3d(2, stride=2)

        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, dilation=3)
        self.lrelu_block1 = nn.LeakyReLU()
        self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, dilation=3)
        self.lrelu_block2 = nn.LeakyReLU()
        self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, dilation=3)
        self.lrelu_block3 = nn.LeakyReLU()
        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)

    def generate_block(self, in_dim, out_dim, ksize, stride, dilation=1):
        block = spconv.SparseSequential(
                spconv.SubMConv3d(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=1,
                        stride=1,
                        indice_key="subm0"),
                nn.BatchNorm1d(num_features=out_dim),
                nn.LeakyReLU(),
                spconv.SubMConv3d(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=3,
                        stride=1,
                        dilation=dilation,
                        padding=dilation,
                        indice_key='subm0'),
                nn.BatchNorm1d(num_features=out_dim),
                nn.LeakyReLU(),
                spconv.SubMConv3d(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=1,
                        stride=1,
                        indice_key="subm0"),
                nn.BatchNorm1d(num_features=out_dim),
                nn.LeakyReLU(),
                # spconv.ToDense(),
        )
        return block

    def forward(self, feat, mask):
        mask = self.maxpooling(mask)
        mask = mask > 0.5
        feat = self.down_sampler(feat)
            
        # feat_before = feat.clone()
        # # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # feat_after = self.res_block1(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block1(feat)

        # feat_before = feat.clone()
        # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # feat_after = self.res_block2(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block2(feat)

        # feat_before = feat.clone()
        # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # feat_after = self.res_block3(feat_sparse) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block3(feat)
        
        # feat = feat.dense()

        # # v2: sparse all at once; no ToDense()
        # # unsupported operand type(s) for +: 'SparseConvTensor' and 'SparseConvTensor
        
        # feat = generate_sparse(feat, mask) # sparse feat
            
        # feat_before = feat
        # feat_after = self.res_block1(feat) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block1(feat)

        # feat_before = feat
        # feat_after = self.res_block2(feat) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block2(feat)

        # feat_before = feat
        # feat_after = self.res_block3(feat) # dense feat
        # feat = feat_before + feat_after
        # feat = self.lrelu_block3(feat)
        
        # feat = feat.dense()

        # v3: no skips; put a lrelu at the end of the resblock
        feat = generate_sparse(feat, mask)
        feat = self.res_block1(feat)
        feat = self.res_block2(feat)
        feat = self.res_block3(feat)
        feat = feat.dense()
        feat = self.final_feature(feat)
        return feat, mask.float()


class SASparseResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(SASparseResNet3D, self).__init__()

        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
            # nn.BatchNorm3d(num_features=out_dim),
            # nn.LeakyReLU(),
        )
        self.down_bn_lrelu = nn.Sequential(
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )
        
        self.maxpooling = nn.MaxPool3d(2, stride=2)
        
        self.down_weights = torch.ones([1, 1, 4, 4, 4]).float().cuda()

        in_dim, out_dim, ksize, stride, padding, dilation = chans, chans, 3, 1, 1, 1
        self.weights = torch.ones([1, 1, 3, 3, 3]).float().cuda()
        self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        self.lrelu_block1 = nn.LeakyReLU()
        self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        self.lrelu_block2 = nn.LeakyReLU()
        self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        self.lrelu_block3 = nn.LeakyReLU()
        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)


    def generate_block(self, in_dim, out_dim, ksize, stride, padding, dilation):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                indice_key='subm0',
            ),
            # nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            # nn.BatchNorm1d(num_features=out_dim),
            spconv.ToDense(),
        )
        return block

    # def run_layer(x, m, layer):
    #     x = x * m
    #     x = self.conv(x)
    #     weights = torch.ones_like(self.conv.weight[0:1,0:1])
    #     mc = F.conv3d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
    #     mc = torch.clamp(mc, min=1e-5)
    #     mc = 1. / mc * torch.sum(weights)
    #     x = x * mc
    #     if self.if_bias:
    #         x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
    #     m = self.pool(m)
    #     return x, m

    def forward(self, feat, mask):
        mc = F.conv3d(mask.float(), self.down_weights, bias=None, stride=2, padding=1, dilation=1)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc * torch.sum(self.down_weights)
        
        feat = self.down_sampler(feat)
        feat = feat * mc
        feat = self.down_bn_lrelu(feat)
        
        mask = self.maxpooling(mask)
        mask = mask > 0.5
        mc = F.conv3d(mask.float(), self.weights, bias=None, stride=1, padding=1, dilation=1)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc * torch.sum(self.weights)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block1(feat_sparse) # dense feat
        feat_after = feat_after * mc
        feat = feat_before + feat_after
        feat = self.lrelu_block1(feat)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block2(feat_sparse) # dense feat
        feat_after = feat_after * mc
        feat = feat_before + feat_after
        feat = self.lrelu_block2(feat)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block3(feat_sparse) # dense feat
        feat_after = feat_after * mc
        feat = feat_before + feat_after
        feat = self.lrelu_block3(feat)

        feat = self.final_feature(feat)
        return feat, mask.float()


class CarefulSparseResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(CarefulSparseResNet3D, self).__init__()

        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
        )
        self.down_bn_lrelu = nn.Sequential(
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )
        self.maxpooling = nn.MaxPool3d(2, stride=2)
        self.down_weights = torch.ones([1, 1, 4, 4, 4]).float().cuda()

        # in_dim, out_dim, ksize, stride, padding, dilation = chans, chans, 3, 1, 1, 1
        # self.weights = torch.ones([1, 1, 3, 3, 3]).float().cuda()
        # self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        # self.lrelu_block1 = nn.LeakyReLU()
        # self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        # self.lrelu_block2 = nn.LeakyReLU()
        # self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding, dilation)
        # self.lrelu_block3 = nn.LeakyReLU()

        in_dim, out_dim = chans, chans
        stride, padding, dilation = 1, 1, 1
        self.weights = torch.ones([1, 1, 3, 3, 3]).float().cuda()
        self.block1_conv1 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        self.block1_conv2 = self.generate_block(in_dim, out_dim, 3, stride, padding, dilation)
        self.block1_conv3 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        
        self.block2_conv1 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        self.block2_conv2 = self.generate_block(in_dim, out_dim, 3, stride, padding, dilation)
        self.block2_conv3 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        
        self.block3_conv1 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        self.block3_conv2 = self.generate_block(in_dim, out_dim, 3, stride, padding, dilation)
        self.block3_conv3 = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)

        # final 1x1x1 conv to get our desired pred_dim
        # self.final_feature = self.generate_block(in_dim, out_dim, 1, stride, padding, dilation)
        self.final_block = self.generate_bn_block(in_dim, out_dim, 1, stride, padding, dilation)
        
        self.lrelu = nn.LeakyReLU()
        self.bn1_1 = nn.BatchNorm3d(num_features=out_dim)
        self.bn1_2 = nn.BatchNorm3d(num_features=out_dim)
        self.bn1_3 = nn.BatchNorm3d(num_features=out_dim)
        self.bn2_1 = nn.BatchNorm3d(num_features=out_dim)
        self.bn2_2 = nn.BatchNorm3d(num_features=out_dim)
        self.bn2_3 = nn.BatchNorm3d(num_features=out_dim)
        self.bn3_1 = nn.BatchNorm3d(num_features=out_dim)
        self.bn3_2 = nn.BatchNorm3d(num_features=out_dim)
        self.bn3_3 = nn.BatchNorm3d(num_features=out_dim)

    def generate_block(self, in_dim, out_dim, ksize, stride, padding, dilation):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                indice_key='subm0',
            ),
            spconv.ToDense(),
        )
        return block

    def generate_bn_block(self, in_dim, out_dim, ksize, stride, padding, dilation):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                indice_key='subm0',
            ),
            nn.BatchNorm1d(num_features=out_dim),
            spconv.ToDense(),
        )
        return block

    # def run_layer(x, m, layer):
    #     x = x * m
    #     x = self.conv(x)
    #     weights = torch.ones_like(self.conv.weight[0:1,0:1])
    #     mc = F.conv3d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
    #     mc = torch.clamp(mc, min=1e-5)
    #     mc = 1. / mc * torch.sum(weights)
    #     x = x * mc
    #     if self.if_bias:
    #         x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
    #     m = self.pool(m)
    #     return x, m

    def forward(self, feat, mask):
        mc = F.conv3d(mask.float(), self.down_weights, bias=None, stride=2, padding=1, dilation=1)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc * torch.sum(self.down_weights)
        
        feat = self.down_sampler(feat)
        feat = feat * mc
        feat = self.down_bn_lrelu(feat)
        
        mask = self.maxpooling(mask)
        mask = mask > 0.5
        mc = F.conv3d(mask.float(), self.weights, bias=None, stride=1, padding=1, dilation=1)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc * torch.sum(self.weights)

        # def generate_block(self, in_dim, out_dim, ksize, stride, padding, do_subm=True):
        #     block = spconv.SparseSequential(
        #         spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
        #         nn.BatchNorm1d(num_features=out_dim),
        #         nn.LeakyReLU(),
        #         spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, indice_key='subm0'),
        #         nn.BatchNorm1d(num_features=out_dim),
        #         nn.LeakyReLU(),
        #         spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
        #         nn.BatchNorm1d(num_features=out_dim),
        #         spconv.ToDense(),
        #         )
        #     return block

        feat_before = feat.clone()
        feat_after = generate_sparse(feat, mask)
        feat_after = self.block1_conv1(feat_after)
        feat_after = self.bn1_1(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block1_conv2(feat_after)
        feat_after = feat_after * mc
        feat_after = self.bn1_2(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block1_conv3(feat_after)
        feat_after = self.bn1_3(feat_after)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)


        feat_before = feat.clone()
        feat_after = generate_sparse(feat, mask)
        feat_after = self.block2_conv1(feat_after)
        feat_after = self.bn2_1(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block2_conv2(feat_after)
        feat_after = feat_after * mc
        feat_after = self.bn2_2(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block2_conv3(feat_after)
        feat_after = self.bn2_3(feat_after)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat_before = feat.clone()
        feat_after = generate_sparse(feat, mask)
        feat_after = self.block3_conv1(feat_after)
        feat_after = self.bn3_1(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block3_conv2(feat_after)
        feat_after = feat_after * mc
        feat_after = self.bn3_2(feat_after)
        feat_after = self.lrelu(feat_after)
        feat_after = generate_sparse(feat_after, mask)
        feat_after = self.block3_conv3(feat_after)
        feat_after = self.bn3_3(feat_after)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)


        # feat_before = feat
        # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # feat_after = self.res_block2(feat_sparse) # dense feat
        # feat_after = feat_after * mc
        # feat = feat_before + feat_after
        # feat = self.lrelu_block2(feat)

        # feat_before = feat
        # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # feat_after = self.res_block3(feat_sparse) # dense feat
        # feat_after = feat_after * mc
        # feat = feat_before + feat_after
        # feat = self.lrelu_block3(feat)

        feat = generate_sparse(feat, mask)
        feat = self.final_block(feat)
        
        return feat, mask.float()
