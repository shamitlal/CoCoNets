import torch
import torch.nn as nn
import time
# import hyperparams as hyp
# from utils_basic import *
import torch.nn.functional as F
import archs.pixelshuffle3d
# import spconv

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

class Altnet3d(nn.Module):
    def __init__(self, in_chans, mid_chans=64, out_chans=1):
        super(Altnet3d, self).__init__()
        conv3d = []
        conv3d_transpose = []
        up_bn = []

        self.down_in_dims = [in_chans, mid_chans, mid_chans, mid_chans, mid_chans]
        self.down_out_dims = [mid_chans, mid_chans, mid_chans, mid_chans, mid_chans]
        self.down_ksizes = [3, 3, 3, 3, 3]
        self.down_strides = [2, 2, 2, 2, 2]
        padding = 1

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            conv3d.append(nn.Sequential(
                # nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=0),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_dim),
            ))
        self.conv3d = nn.ModuleList(conv3d)

        self.up_in_dims = [mid_chans, mid_chans*2, mid_chans*2, mid_chans*2]
        self.up_out_dims = [mid_chans, mid_chans, mid_chans, mid_chans]
        self.up_bn_dims = [mid_chans*2, mid_chans*2, mid_chans*2, mid_chans*2]
        self.up_ksizes = [4, 4, 4, 4]
        self.up_strides = [2, 2, 2, 2]
        padding = 1 # Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
            conv3d_transpose.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm3d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired out_chans
        self.final_feature = nn.Conv3d(in_channels=mid_chans*2, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.conv3d_transpose = nn.ModuleList(conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        # print('in feat', feat.shape)
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            # print('down feat', feat.shape)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv3d_transpose_layer, bn_layer) in enumerate(zip(self.conv3d_transpose, self.up_bn)):
            feat = conv3d_transpose_layer(feat)
            # print('up feat', feat.shape)
            feat = torch.cat([feat, skipcons.pop()], dim=1) # skip connection by concatenation
            # print('cat feat', feat.shape)
            feat = bn_layer(feat)

        feat = self.final_feature(feat)
        # print('final feat', feat.shape)

        return feat
    
class Skipnet3d(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Skipnet3d, self).__init__()
        conv3d = []
        up_bn = [] #batch norm layer for deconvolution
        conv3d_transpose = []

        # self.conv3d = torch.nn.Conv3d(4, 32, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        # self.layers = []
        self.down_in_dims = [in_dim, mid_dim, 2*mid_dim]#, 4*mid_dim]
        self.down_out_dims = [mid_dim, 2*mid_dim, 4*mid_dim, 8*mid_dim]
        self.down_ksizes = [4, 4, 4, 4]
        self.down_strides = [2, 2, 2, 2]
            
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('down dims: ', self.down_out_dims)

        for i, (in_chan, out_chan, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            # print('3d CONV', end=' ')
             
            conv3d.append(nn.Sequential(
                nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_chan),
            ))

        self.conv3d = nn.ModuleList(conv3d)

        # self.up_in_dims = [8*chans, 8*chans]
        # self.up_out_dims = [4*chans, 4*chans]
        # self.up_bn_dims = [8*chans, 6*chans]
        # self.up_ksizes = [4, 4]
        # self.up_strides = [2, 2]
        
        self.up_in_dims = [4*mid_dim, 6*mid_dim]
        self.up_out_dims = [4*mid_dim, 4*mid_dim]
        self.up_bn_dims = [6*mid_dim, 5*mid_dim]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        # print('up dims: ', self.up_out_dims)

        for i, (in_chan, bn_chan, out_chan, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
             
            conv3d_transpose.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm3d(num_features=bn_chan))

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=self.up_bn_dims[-1], out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.conv3d_transpose = nn.ModuleList(conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv3d_transpose_layer, bn_layer) in enumerate(zip(self.conv3d_transpose, self.up_bn)):
            # print('feat before up', feat.shape)
            feat = conv3d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) #skip connection by concatenation
            # print('feat before bn', feat.shape)
            # print('feat before bn', feat.shape)
            feat = bn_layer(feat)

        feat = self.final_feature(feat)
        # print('final feat', feat.shape)
        # feat = F.interpolate(feat, scale_factor=2)
        
        return feat

class Res3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, ksize=3, padding=1):
        super(Res3dBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=ksize, stride=1, padding=padding),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=padding),
            nn.BatchNorm3d(out_planes)
        )
        assert(padding==1 or padding==0)
        self.padding = padding
        self.ksize = ksize

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        # print('res', res.shape)
        skip = self.skip_con(x)
        if self.padding==0 and self.ksize==3:
            # the data has shrunk a bit
            skip = skip[:,:,2:-2,2:-2,2:-2]
        # print('skip', skip.shape)
        # # print('trim', skip.shape)
        return F.relu(res + skip, True)

class Conv3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)

class Deconv3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Deconv3dBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.deconv(x)

class Up3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, scale=2, relu=True):
        super(Up3dBlock, self).__init__()
        # self.res_branch = nn.Sequential(
        #     nn.Conv3d(in_planes, out_planes*8, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm3d(out_planes),
        #     nn.ReLU(True),
        #     nn.Conv3d(out_planes, out_planes*8, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm3d(out_planes)
        # )
        
        channel_factor = int(scale**3)
        if relu:
            self.conv = nn.Sequential(
                # nn.Conv3d(in_planes, out_planes*channel_factor, kernel_size=3, stride=1, padding=0),
                # nn.Conv3d(in_planes, out_planes*channel_factor, kernel_size=2, stride=1, padding=0),
                nn.Conv3d(in_planes, out_planes*channel_factor, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_planes, out_planes*channel_factor, kernel_size=1, stride=1, padding=0),
            )
            
        self.unpack = archs.pixelshuffle3d.PixelShuffle3d(scale)
        # archs.nn.PixelShuffle(9)
        # pixelshuffle3d
        
        # if in_planes == out_planes:
        #     self.skip_con = nn.Sequential()
        # else:
        #     self.skip_con = nn.Sequential(
        #         nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm3d(out_planes)
        #     )

    def forward(self, x):
        # res = self.res_branch(x)
        # # print('res', res.shape)
        # skip = self.skip_con(x)
        # # print('skip', skip.shape)
        # skip = skip[:,:,2:-2,2:-2,2:-2]
        # # print('trim', skip.shape)
        # out = F.relu(res + skip, True)
        # # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        out = self.conv(x)
        # print('intermed out', out.shape)
        return self.unpack(out)
    # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
    

class Pool3dBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3dBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Encoder3d(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()

        chans = 32
        self.encoder_layer0 = Res3dBlock(in_dim, chans)
        self.encoder_layer1 = Pool3dBlock(2)
        self.encoder_layer2 = Res3dBlock(chans, chans)
        self.encoder_layer3 = Res3dBlock(chans, chans)
        self.encoder_layer4 = Res3dBlock(chans, chans)
        self.encoder_layer5 = Pool3dBlock(2)
        self.encoder_layer6 = Res3dBlock(chans, chans)
        self.encoder_layer7 = Res3dBlock(chans, chans)
        self.encoder_layer8 = Res3dBlock(chans, chans)
        self.final_layer = nn.Conv3d(in_channels=chans, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # skip_x1 = self.skip_res1(x)
        x = self.encoder_layer0(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = self.encoder_layer6(x)
        x = self.encoder_layer7(x)
        x = self.encoder_layer8(x)
        x = self.final_layer(x)
        return x
    
    
class EncoderDecoder3d(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()

        chans = 128
        chans2 = int(chans/2)
        chans4 = int(chans/4)

        def generate_sparse_block(in_dim, out_dim, ksize, stride, padding):
            block = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                spconv.SparseConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                spconv.SparseConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                spconv.SparseConv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                spconv.ToDense(),
                )
            return block
        def generate_dense_block(in_dim, out_dim, ksize, stride, padding):
            block = nn.Sequential(
                nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_dim),
                nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_dim),
                # nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                # nn.LeakyReLU(),
                # nn.BatchNorm3d(num_features=out_dim),
                # nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                # nn.LeakyReLU(),
                # nn.BatchNorm3d(num_features=out_dim),
                )
            return block

        # self.sparse_encoder = generate_sparse_block(in_dim, chans4, 3, 2, 0)
        # self.sparse_encoder = generate_sparse_block(in_dim, chans4, 4, 2, 0)
        # self.sparse_encoder = generate_sparse_block(in_dim, chans4, 3, 2, 1)
        self.dense_encoder = generate_dense_block(in_dim, chans2, 3, 2, 1)
        
        # self.encoder_layer1 = generate_sparse_block(in_dim, chans4, 3, 1, 0)
        # self.encoder_layer2 = self.generate_sparse_block(chans4, chans4, 3, 1, 0)
        # self.encoder_layer3 = self.generate_sparse_block(chans4, chans4, 3, 1, 0)
        
        # self.encoder_layer3 = self.generate_block(in_dim, chans4, 3, 1, 0)
        # self.encoder_layer2 = Res3dBlock(chans4, chans4)
        # self.encoder_layer3 = Res3dBlock(chans4, chans4)
        
        
        # self.encoder_layer0 = Pool3dBlock(2)
        # self.encoder_layer1 = Conv3dBlock(in_dim, chans4, stride=2)
        # self.encoder_layer2 = Conv3dBlock(chans4, chans2, stride=2)
        # self.encoder_layer3 = Conv3dBlock(chans2, chans2, stride=2)
        # self.encoder_layer3 = Conv3dBlock(chans4, chans2)
        # self.encoder_layer4 = Pool3dBlock(2)
        # self.encoder_layer3 = Res3dBlock(chans2, chans2)
        
        # self.encoder_layer4 = Res3dBlock(chans4, chans2, padding=0)
        # self.encoder_layer5 = Res3dBlock(chans2, chans2)
        # self.encoder_layer6 = Res3dBlock(chans2, chans2)
        # self.encoder_layer7 = Res3dBlock(chans2, chans)

        self.encoder_layer4 = Res3dBlock(chans2, chans, padding=0)
        self.encoder_layer5 = Res3dBlock(chans, chans, padding=0)
        self.encoder_layer6 = Res3dBlock(chans, chans, padding=0)
        self.encoder_layer7 = Res3dBlock(chans, chans, padding=0)
        # self.encoder_layer8 = Res3dBlock(chans, chans)
        # self.encoder_layer9 = Res3dBlock(chans, chans)
        
        # self.encoder_layer7 = Res3dBlock(chans, chans, padding=0)
        # self.encoder_layer8 = Res3dBlock(chans, chans, padding=0)
        # self.encoder_layer9 = Res3dBlock(chans, chans, padding=0)



        
        # self.encoder_layer9 = Res3dBlock(chans, chans)
        
        # # self.encoder_layer8 = Pool3dBlock(2)
        # self.encoder_layer9 = Res3dBlock(chans2, chans)
        # self.encoder_layer10 = Res3dBlock(chans, chans)
        # self.encoder_layer11 = Res3dBlock(chans, chans)
        
        # self.decoder_layer = Up3dBlock(chans, chans2, scale=4)
        # self.decoder_layer = Up3dBlock(chans, chans2, scale=2)
        
        # self.up_layer = Up3dBlock(chans, chans2, scale=2, relu=False)
        # self.up_layer = Deconv3dBlock(chans, chans2)
        self.up_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.encoder_layer8 = Res3dBlock(chans, chans2, padding=0)
        self.final_layer = nn.Conv3d(in_channels=chans2, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

        # self.up_layer = Up3dBlock(chans, out_dim, scale=16)
        
    def forward(self, x):
        # print('x in', x.shape)
        # x = self.encoder_layer0(x)
        # print('x 00', x.shape)

        # mask = mask > 0.5
        # feat_before = feat.clone()
        # # feat_sparse = generate_sparse(feat, mask) # sparse feat
        # x = generate_sparse(x, mask)
        # x, _ = self.encoder_layer1(x, mask)
        # print('x 01', x.shape)

        # x = generate_sparse(x, mask)
        # x = self.sparse_encoder(x)
        # print('x sp', x.shape)

        x = self.dense_encoder(x)
        # print('x de', x.shape)
        
        
        # x = self.encoder_layer1(x)
        # print('x 01', x.shape)
        # x = self.encoder_layer2(x)
        # print('x 02', x.shape)
        # x = self.encoder_layer3(x)
        # print('x 03', x.shape)

        x = self.encoder_layer4(x)
        # print('x 04', x.shape)

        low_feat = x.clone()
        
        
        x = self.encoder_layer5(x)
        # print('x 05', x.shape)
        x = self.encoder_layer6(x)
        # print('x 06', x.shape)
        
        # x = self.encoder_layer8(x)
        # print('x 08', x.shape)
        # x = self.encoder_layer9(x)
        # print('x 09', x.shape)
        
        # x = self.encoder_layer10(x)
        # print('x 10', x.shape)
        # x = self.encoder_layer11(x)
        # print('x 11', x.shape)

        
        # x = self.up_layer(x)
        # print('x up', x.shape)

        x = self.encoder_layer7(x)
        # print('x 07', x.shape)
        
        mid_feat = x.clone()

        x = self.up_layer(x)
        # print('x up', x.shape)

        x = self.encoder_layer8(x)
        # print('x 07', x.shape)
        
        high_feat = x.clone()
        
        x = self.final_layer(x)
        # print('x', x.shape)
        
        return x, (low_feat, mid_feat, high_feat)
    
class Up3d(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, chans=64, scale=8):
        super().__init__()
        # self.conv_layer = Res3dBlock(in_dim, chans, padding=0)
        # self.up_layer = Res3dBlock(chans, out_dim, padding=0)
        # self.conv_layer2 = Res3dBlock(chans, out_dim)
        # self.conv_layer2 = Res3dBlock(chans, out_dim)
        # self.up_layer = Up3dBlock(chans, out_dim, scale=scale)

        # v1 
        # self.conv_layer1 = Res3dBlock(in_dim, chans, ksize=1, padding=0)
        # self.up_layer = Up3dBlock(chans, chans, scale=scale, relu=False)
        # self.conv_layer2 = Res3dBlock(chans, chans, ksize=3, padding=1)

        # v2
        self.conv_layer1 = Res3dBlock(in_dim, chans, ksize=1, padding=0)
        self.conv_layer2 = Res3dBlock(chans, chans, ksize=1, padding=0)
        self.up_layer = Up3dBlock(chans, out_dim, scale=scale, relu=False)

        # self.final_conv = nn.Conv3d(in_channels=chans, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        # print('x in', x.shape)

        # v1
        # x = self.conv_layer1(x)
        # x = self.up_layer(x)
        # x = self.conv_layer2(x)
        # x = self.final_conv(x)

        # v2
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.up_layer(x)
        
        # print('x up', x.shape)
        return x
    
    



    
