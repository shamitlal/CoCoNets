import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
# import hyperparams as hyp
# from utils_basic import *

class Pad(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, amount):
        super(Pad, self).__init__()
        self.pad = nn.ConstantPad2d(amount, 0)
    def forward(self, input):
        x, m = input
        x, m = self.pad(x), self.pad(m)
        return x, m

class BatchNorm(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, out_channels):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, input):
        x, m = input
        x = self.batch_norm(x)
        return x, m

class LeakyRelu(nn.Module):
    # standard leaky relu, but allow packed (x,m) inputs
    def __init__(self):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    def forward(self, input):
        x, m = input
        x = self.leaky_relu(x)
        return x, m

class Relu(nn.Module):
    # standard relu, but allow packed (x,m) inputs
    def __init__(self):
        super(Relu, self).__init__()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x, m = input
        x = self.relu(x)
        return x, m

class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SparseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.pool.require_grad = False
    def forward(self, input):
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x = self.conv(x)
        weights = torch.ones_like(self.conv.weight)
        mc = F.conv2d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
        m = self.pool(m)
        return x, m

class Conv(nn.Module):
    # standard conv, but allow packed (x,m) inputs
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
    def forward(self, input):
        x, m = input
        x = self.conv(x)
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
        return x, m

class SimpleCNN(nn.Module):
    def __init__(self, in_channel, pred_dim, num_layers=5):
        super(SimpleCNN, self).__init__()

        chan = 64
        stride = 1
        self.layers = []
        for layer_num in list(range(num_layers)):
            if layer_num==0:
                in_dim = in_channel
                kernel_size = 7
                pad = 3
            else:
                in_dim = chan
                kernel_size = 5
                pad = 2
            out_dim = chan
            dilation = 1
            print('in, out, stride, dilation: %d, %d, %d, %d' % (in_dim, out_dim, stride, dilation))
            self.layers.append(
                self.generate_conv_block(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=False))
        # final 1x1x1 conv to get our desired pred_dim
        self.final_layer = nn.Conv2d(in_channels=chan, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        
    def generate_conv_block(self, in_dim, out_dim, kernel_size=3, stride=1, dilation=2, bias=True):
        block = nn.Sequential(
            # Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=1),
            # BatchNorm(out_dim),
            # LeakyRelu(),
            Pad(dilation), # pre-pad, to help the sparse conv
            SparseConv(in_channels=in_dim,
                       out_channels=out_dim,
                       kernel_size=3,
                       stride=stride,
                       padding=0,
                       dilation=dilation,
                       bias=bias),
            BatchNorm(out_dim),
            # LeakyRelu(),
            Relu(),
            # Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # BatchNorm(out_dim),
        ).cuda()
        return block
    
    def forward(self, feat, mask):

        # def run_layer(feat_and_mask, layer):
        #     feat_and_mask = layer(feat_and_mask)
        #     feat, mask = feat_and_mask
        #     return feat, mask
        
        for layer in self.layers:
            feat, mask = layer((feat, mask))

        feat = self.final_layer(feat)

        # # up one
        # feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
        # mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        
        
        return feat, mask


class SingleresCNN(nn.Module):
    def __init__(self, in_channel, pred_dim, num_layers=5):
        super(SingleresCNN, self).__init__()

        # dims = [16, 32, 64, 128, 256, 512]
        dims = [16, 32, 64, 64, 128, 128, 256, 256, 512, 512]
        stride = 1

        self.layers = []
        
        for layer_num in list(range(num_layers)):
            if layer_num==0:
                in_dim = in_channel
            else:
                in_dim = dims[layer_num-1]
                
            if np.mod(layer_num+1, 2)==0:
                dilation = 2
            else:
                dilation = 1
                
            out_dim = dims[layer_num]

            print('in, out, stride, dilation: %d, %d, %d, %d' % (in_dim, out_dim, stride, dilation))
            
            self.layers.append(self.generate_conv_block(in_dim, out_dim, stride, dilation=dilation))
        
        self.layers = nn.ModuleList(self.layers)
        self.lrelu = nn.LeakyReLU()
        # final 1x1x1 conv to get our desired pred_dim
        self.final_layer = nn.Conv2d(in_channels=dims[num_layers-1], out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        
    def generate_conv_block(self, in_dim, out_dim, stride=1, dilation=2):
        block = nn.Sequential(
            # Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=1),
            # BatchNorm(out_dim),
            # LeakyRelu(),
            Pad(dilation), # pre-pad, to help the sparse conv
            SparseConv(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=0, dilation=dilation),
            BatchNorm(out_dim),
            LeakyRelu(),
            # Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # BatchNorm(out_dim),
        ).cuda()
        return block
    
    def forward(self, feat, mask):

        for layer in self.layers:
            feat, mask = layer((feat, mask))
        feat = self.final_layer(feat)
        
        return feat, mask

class CustomCNN(nn.Module):
    def __init__(self, in_channel, pred_dim, num_sparse_layers=3):
        super(CustomCNN, self).__init__()

        dims = [16, 32, 64, 64, 128, 128, 256, 256, 512, 512]

        self.sparse_layers = []
        
        for layer_num in list(range(num_sparse_layers)):
            if layer_num==0:
                in_dim = in_channel
            else:
                in_dim = dims[layer_num-1]
                
            if layer_num==0 or layer_num==5:
                stride = 2
            else:
                stride = 1
                
            if np.mod(layer_num+1, 2)==0:
                dilation = 2
            else:
                dilation = 1
                
            out_dim = dims[layer_num]

            print('in, out, stride, dilation: %d, %d, %d, %d' % (in_dim, out_dim, stride, dilation))
            
            self.sparse_layers.append(self.generate_conv_block(in_dim, out_dim, stride, dilation=dilation))


        # now skipcon net

        conv2d = []
        conv2d_transpose = []
        up_bn = []

        in_chans = dims[num_sparse_layers-1]+2
        mid_chans = 32
        out_chans = pred_dim
        self.down_in_dims = [in_chans, mid_chans, 2*mid_chans]
        self.down_out_dims = [mid_chans, 2*mid_chans, 4*mid_chans]
        self.down_ksizes = [3, 3, 3]
        self.down_strides = [2, 2]#, 2]
        padding = 1

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            conv2d.append(nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=out_dim),
            ))
        self.conv2d = nn.ModuleList(conv2d)

        # self.up_in_dims = [4*mid_chans, 6*mid_chans]
        # self.up_bn_dims = [6*mid_chans, 3*mid_chans]
        # self.up_out_dims = [4*mid_chans]#, 2*mid_chans]

        self.up_in_dims = [2*mid_chans]
        self.up_bn_dims = [3*mid_chans]
        self.up_out_dims = [2*mid_chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 # Note: this only holds for ksize=4 and stride=2!


        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
            conv2d_transpose.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm2d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired out_chans
        # self.final_feature = nn.Conv2d(in_channels=3*mid_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.final_feature = nn.Conv2d(in_channels=3*mid_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.conv2d_transpose = nn.ModuleList(conv2d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def generate_conv_block(self, in_dim, out_dim, stride=1, dilation=2):
        block = nn.Sequential(
            # Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=1),
            # BatchNorm(out_dim),
            # LeakyRelu(),
            Pad(dilation), # pre-pad, to help the sparse conv
            SparseConv(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=0, dilation=dilation),
            BatchNorm(out_dim),
            LeakyRelu(),
            # Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # BatchNorm(out_dim),
        ).cuda()
        return block
    
    def forward(self, feat, mask, halfgrid):
        # first a few sparse layers, to densify
        for sparse_layer in self.sparse_layers:
            feat, mask = sparse_layer((feat, mask))

        feat = torch.cat([feat, halfgrid], dim=1)
        skipcons = []
        for conv2d_layer in self.conv2d:
            feat = conv2d_layer(feat)
            skipcons.append(feat)
        skipcons.pop() # we don't want the innermost layer as skipcon
        for i, (conv2d_transpose_layer, bn_layer) in enumerate(zip(self.conv2d_transpose, self.up_bn)):
            feat = conv2d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) # skip connection by concatenation
            feat = bn_layer(feat)
        feat = self.final_feature(feat)

        # up one
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        
        return feat, mask
    
