import torch
import torch.nn as nn
import time
import torch.nn.functional as F
# import hyperparams as hyp
# from utils_basic import *

class Pad(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, amount):
        super(Pad, self).__init__()
        self.pad = nn.ConstantPad3d(amount, 0)
    def forward(self, input):
        x, m = input
        x, m = self.pad(x), self.pad(m)
        return x, m

class BatchNorm(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, out_channels):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
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

class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SparseConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.pool.require_grad = False
    def forward(self, input):
        x, m = input
        ## v0
        # mc = m.repeat(1, C, 1, 1, 1)
        # _, C, _, _, _ = x.shape
        ## v1
        mc = m.expand_as(x)
        x = x * mc
        x = self.conv(x)
        weights = torch.ones_like(self.conv.weight)
        mc = F.conv3d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
        m = self.pool(m)
        return x, m

class Conv(nn.Module):
    # standard conv, but allow packed (x,m) inputs
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
    def forward(self, input):
        x, m = input
        x = self.conv(x)
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
        return x, m
    
class Bottle3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Bottle3D, self).__init__()
        conv3d = []

        self.out_chans = [chans, 2*chans, 4*chans, 8*chans]

        n_layers = len(self.out_chans)
        
        for i in list(range(n_layers)):
            if i==0:
                in_dim = in_channel
            else:
                in_dim = self.out_chans[i-1]
            out_dim = self.out_chans[i]
            conv3d.append(nn.Sequential(
                SparseConv(in_dim, out_dim, kernel_size=4, stride=2, padding=0),
                LeakyRelu(),
                BatchNorm(out_dim),
            ))
        self.conv3d = nn.ModuleList(conv3d)

        hidden_dim = 1024
        self.linear_layers = nn.Sequential(
            nn.Linear(self.out_chans[-1]*2*2*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, pred_dim),
        )
        
    def forward(self, feat, mask):
        B, C, Z, Y, X = list(feat.shape)
        # print(mask.shape)
        # print(feat.shape)
        for conv3d_layer in self.conv3d:
            feat, mask = conv3d_layer((feat, mask))
            # print(feat.shape)
        # discard the mask
        feat = feat.reshape(B, -1)
        # print(feat.shape)
        feat = self.linear_layers(feat)
        return feat

