from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-k', '--kernel-size', type=int, default=1)
parser.add_argument('--patch', type=int, default=4)
parser.add_argument('--patch_dilation', type=int, default=1)
parser.add_argument('-c', '--channel', type=int, default=2)
parser.add_argument('-d', '--depth', type=int, default=6)
parser.add_argument('--height', type=int, default=6)
parser.add_argument('-w', '--width', type=int, default=6)
parser.add_argument('-s', '--stride', type=int, default=1)
parser.add_argument('-p', '--pad', type=int, default=0)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

from spatial_correlation_sampler import SpatialCorrelationSampler
# args.kernel_size = 3
# args.patch = [2,3,5]
#args.height = 7
#args.depth = 6
#args.width = 9
input1 = torch.ones(args.batch_size,
                     args.channel,
                     args.depth,
                     args.height,
                     args.width).double()
input2 = torch.ones(args.batch_size,
                     args.channel,
                     args.depth,
                     args.height,
                     args.width).double()
input1.requires_grad = True
input2.requires_grad = True


print("creating model...")
correlation_sampler = SpatialCorrelationSampler(
    args.kernel_size,
    args.patch,
    args.stride,
    args.pad,
    args.patch_dilation)
torch.set_printoptions(threshold=1000000)
print("running test...")
device = torch.device("cuda")
cuda_values = correlation_sampler(input1.to(device), input2.to(device))
print("forward:")
N, PD, PH, PW, D, H, W = list(cuda_values.shape)
print(N, PD, PH, PW, D, H, W)
# print(cuda_values.view(N, PD*PH*PW, D, H, W))
print(cuda_values.view(N, PD*PH*PW, D*H*W).sum(2))

def get_grads(variables):
    return [var.grad.clone() for var in variables]
cuda_values.sum().backward()
grad_cuda = get_grads([input1, input2])
print("backward:")
print(list(grad_cuda[0].shape))
print(grad_cuda[0])
print(grad_cuda[1])

