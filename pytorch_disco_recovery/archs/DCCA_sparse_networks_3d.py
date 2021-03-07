import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt

# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
#     elif norm_type == 'none':
#         norm_layer = None
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer

# def get_scheduler(optimizer, opt):
#     if opt.lr_policy == 'lambda':
#         lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    net = net
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'pretrained':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)

#     for root_child in net.children():
#         for children in root_child.children():
#             if children in root_child.need_initialization:
#                 init_weights(children, init_type, gain=init_gain)
#     return net

# def define_DCCASparseNet(rgb_enc=True, depth_enc=True, depth_dec=True, norm='batch', use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=[]):
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)
#     net = DCCASparsenetGenerator(rgb_enc=rgb_enc, depth_enc=depth_enc, depth_dec=depth_dec)
#     return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class SAConv(nn.Module):
    # Convolution layer for sparse data
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(SAConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pool.require_grad = False

    def forward(self, input):
        x, m = input
        x = x * m
        x = self.conv(x)
        weights = torch.ones_like(self.conv.weight[0:1,0:1])
        mc = F.conv3d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc * torch.sum(weights) # replaced "9" with this sum
        x = x * mc # added by adam; this is req to keep magnitudes invariant to sparsity
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
        m = self.pool(m)

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

class Relu(nn.Module):
    # standard relu, but allow packed (x,m) inputs
    def __init__(self):
        super(Relu, self).__init__()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x, m = input
        x = self.relu(x)
        return x, m
    
class SAConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(SAConvBlock, self).__init__()
        self.sparse_conv = SAConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x, m = input
        x, m = self.sparse_conv((x, m))
        assert (m.size(1)==1)
        # x = self.relu(x)
        
        return x, m

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# def make_layers_from_size(sizes):
#     layers = []
#     for size in sizes:
#         layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1],momentum = 0.1), nn.ReLU(inplace=True)]
#     return nn.Sequential(*layers)

def make_blocks_from_names(names, in_dim, out_dim, dilation=1):
    layers = []
    if names[0] == "block1" or names[0] == "block2":
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    elif names[0] == "simpleblock":
        # layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2)]
        # layers += [SAConvBlock(in_dim, out_dim, 3, stride=1, padding=1)]
        # layers += [SAConvBlock(in_dim, out_dim, 3, stride=1, padding=1), BatchNorm(out_dim), Relu()]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2), BatchNorm(out_dim), Relu()]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2, bias=False), Relu()]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2, bias=False), BatchNorm(out_dim), Relu()]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2, bias=False), BatchNorm(out_dim), Relu()]
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=1, padding=0, bias=False), BatchNorm(out_dim), Relu()]
        # layers += [SAConvBlock(in_dim, out_dim, 5, stride=1, padding=2, dilation=dilation, bias=True), Relu()]
    elif names[0] == "convblock":
        layers += [nn.Conv3d(in_dim, out_dim, 5, stride=1, padding=2, bias=False), nn.BatchNorm3d(out_dim), nn.ReLU(True)]
    else:
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    return nn.Sequential(*layers)

class SimpleCNN(nn.Module):
    # def __init__(self, in_dim, out_dim, num_layers, chans=64):
    def __init__(self, in_dim, out_dim, chans=64):
        super(SimpleCNN, self).__init__()
        batchNorm_momentum = 0.1
        self.need_initialization = []

        self.num_layers = 0

        # i can't seem to make the layer declaration (and appropriate weight usage) work in a loop
        # so, i will declare each layer in order,
        # but still fill up a dict, so that forward() is easy
        
        self.layers = {}
        
        self.layer0 = make_blocks_from_names(["simpleblock"], in_dim, chans)
        self.layers['layer0'] = self.layer0
        self.num_layers += 1
        
        self.layer1 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer1'] = self.layer1
        self.num_layers += 1
        
        self.layer2 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer2'] = self.layer2
        self.num_layers += 1
        
        self.layer3 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer3'] = self.layer3
        self.num_layers += 1
        
        self.layer4 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer4'] = self.layer4
        self.num_layers += 1
        
        self.layer5 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer5'] = self.layer5
        self.num_layers += 1
        
        self.layer6 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer6'] = self.layer6
        self.num_layers += 1
        
        self.layer7 = make_blocks_from_names(["simpleblock"], chans, chans)
        self.layers['layer7'] = self.layer7
        self.num_layers += 1

        # self.layer8 = make_blocks_from_names(["simpleblock"], chans, chans)
        # self.layers['layer8'] = self.layer8
        # self.num_layers += 1

        # self.layer9 = make_blocks_from_names(["simpleblock"], chans, chans)
        # self.layers['layer9'] = self.layer9
        # self.num_layers += 1

        # regular conv layers, since it's dense enough by now

        # self.layer8 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer8'] = self.layer8

        # self.layer9 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer9'] = self.layer9

        # self.layer10 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer10'] = self.layer10

        # self.layer11 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer11'] = self.layer11
        
        # self.layer12 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer12'] = self.layer12
        
        # self.layer13 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer13'] = self.layer13
        
        # self.layer14 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer14'] = self.layer14
        
        # self.layer15 = make_blocks_from_names(["convblock"], chans, chans)
        # self.layers['layer15'] = self.layer15
        
        # self.need_initialization.append(self.layer8)
        # self.need_initialization.append(self.layer9)
        # self.need_initialization.append(self.layer10)
        # self.need_initialization.append(self.layer11)
        # self.need_initialization.append(self.layer12)
        # self.need_initialization.append(self.layer13)
        # self.need_initialization.append(self.layer14)
        # self.need_initialization.append(self.layer15)
        
        self.final_layer = nn.Conv3d(chans, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.need_initialization.append(self.final_layer)

    def forward(self, feat, mask):
        # for li in list(range(self.num_layers)):
        #     feat, mask = self.layers['layer%d' % li]((feat, mask))
        
        # print('feat', feat.shape)
        for li in list(range(self.num_layers)):
            feat, mask = self.layers['layer%d' % li]((feat, mask))
            # print('feat', feat.shape)
        # feat = self.layers['layer6'](feat)
        # feat = self.layers['layer7'](feat)
        # feat = self.layers['layer8'](feat)
        # feat = self.layers['layer9'](feat)
        # feat = self.layers['layer10'](feat)
        # feat = self.layers['layer11'](feat)
        # feat = self.layers['layer12'](feat)
        # feat = self.layers['layer13'](feat)
        # feat = self.layers['layer14'](feat)
        # feat = self.layers['layer15'](feat)
        feat = self.final_layer(feat)
        # print('feat', feat.shape)
        return feat, mask

def make_block(name, in_dim, out_dim, stride=1):
    layers = []
    if name == "conv_block":
        # no real skipcon here
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=stride, padding=1, bias=True), BatchNorm(out_dim), Relu()]
    else:
        print('name:', name)
        assert(False) # not ready
    # elif names[0] == "convblock":
    #     layers += [nn.Conv3d(in_dim, out_dim, 5, stride=1, padding=2, bias=False), nn.BatchNorm3d(out_dim), nn.ReLU(True)]
    # else:
    #     layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
    #     layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    #     layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    return nn.Sequential(*layers)

class Encoder3D(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()

        chans = 64
        self.encoder_layer0 = make_block('conv_block', in_dim, chans, stride=1) 
        self.encoder_layer1 = make_block('conv_block', chans, chans, stride=1) 
        self.encoder_layer2 = make_block('conv_block', chans, chans, stride=2)
        self.encoder_layer3 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer4 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer5 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer6 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer7 = make_block('conv_block', chans, chans, stride=2)
        self.encoder_layer8 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer9 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer10 = make_block('conv_block', chans, chans, stride=1)
        self.encoder_layer11 = make_block('conv_block', chans, chans, stride=1)
        self.final_layer = nn.Conv3d(in_channels=chans, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, m):
        x, m = self.encoder_layer0((x, m))
        x, m = self.encoder_layer1((x, m))
        x, m = self.encoder_layer2((x, m))
        x, m = self.encoder_layer3((x, m))
        x, m = self.encoder_layer4((x, m))
        x, m = self.encoder_layer5((x, m))
        x, m = self.encoder_layer6((x, m))
        x, m = self.encoder_layer7((x, m))
        x, m = self.encoder_layer8((x, m))
        x, m = self.encoder_layer9((x, m))
        x, m = self.encoder_layer10((x, m))
        x, m = self.encoder_layer11((x, m))
        x = self.final_layer(x)
        return x, m
