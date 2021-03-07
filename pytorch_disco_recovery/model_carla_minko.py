import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.matchnet import MatchNet
from nets.rendernet import RenderNet
from nets.occnet import OccNet
from nets.rgbnet import RgbNet
from nets.sigen3dnet import Sigen3dNet
from backend import saverloader, inputs

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc
import utils.track

import MinkowskiEngine as ME

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10

# the idea here is to minko to higher resolutions in a smart way,
# following the advice of PointRend and even NERF

class CompletionNet(nn.Module):

    # ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    # DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    # ENC_CHANNELS = [8, 16, 32, 64, 128, 256, 512, 1024]
    # DEC_CHANNELS = [8, 16, 32, 64, 128, 256, 512, 1024]
    # ENC_CHANNELS = [8, 16, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    # DEC_CHANNELS = [8, 16, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    ENC_CHANNELS = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    DEC_CHANNELS = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

    def __init__(self, resolution):
        nn.Module.__init__(self)

        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(
                1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s64s128 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[6], enc_ch[7], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[7]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[7], enc_ch[7], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[7]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s128s64 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                enc_ch[7],
                dec_ch[6],
                kernel_size=4,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[6], dec_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[6]),
            ME.MinkowskiELU(),
        )

        self.dec_s64_cls = ME.MinkowskiConvolution(
            dec_ch[6], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, has_bias=True, dimension=3)

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, has_bias=True, dimension=3)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(
                out.coords_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, pos_key, neg_key):
        out_cls, positives, negatives = [], [], []

        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)
        enc_s128 = self.enc_block_s64s128(enc_s64)

        ##################################################
        # Decoder 128 -> 32
        ##################################################
        dec_s64 = self.dec_block_s128s64(enc_s128)

        # Add encoder features
        dec_s64 = dec_s64 + enc_s64
        dec_s64_cls = self.dec_s64_cls(dec_s64)
        keep_s64 = (dec_s64_cls.F > 0).cpu().squeeze()

        positive = self.get_target(dec_s64, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s64, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s64_cls)

        if self.training:
            keep_s64 += positive
            keep_s64 += negative

        # Remove voxels s64
        dec_s64 = self.pruning(dec_s64, keep_s64.cpu())

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).cpu().squeeze()

        positive = self.get_target(dec_s32, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s32, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s32_cls)

        if self.training:
            keep_s32 += positive
            keep_s32 += negative

        # Remove voxels s32
        dec_s32 = self.pruning(dec_s32, keep_s32.cpu())

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).cpu().squeeze()

        positive = self.get_target(dec_s16, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s16, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s16_cls)

        if self.training:
            keep_s16 += positive
            keep_s16 += negative

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16.cpu())

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        positive = self.get_target(dec_s8, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s8, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).cpu().squeeze()

        if self.training:
            keep_s8 += positive
            keep_s8 += negative

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8.cpu())

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        positive = self.get_target(dec_s4, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s4, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).cpu().squeeze()

        if self.training:
            keep_s4 += positive
            keep_s4 += negative

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4.cpu())

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        positive = self.get_target(dec_s2, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s2, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).cpu().squeeze()

        if self.training:
            keep_s2 += positive
            keep_s2 += negative

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2.cpu())

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        positive = self.get_target(dec_s1, pos_key)
        positives.append(positive)
        negative = self.get_target(dec_s1, neg_key)
        negatives.append(negative)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).cpu().squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += positive
        #     keep_s1 += negative

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1.cpu())

        return out_cls, positives, negatives, dec_s1
    
class GenerativeNet(nn.Module):

    CHANNELS = [1024, 512, 256, 128, 64, 32, 16, 16]

    def __init__(self, resolution, in_nchannel=889):
        nn.Module.__init__(self)

        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_nchannel,
                ch[0],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolutionTranspose(
                ch[0],
                ch[1],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, has_bias=True, dimension=3)

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[1],
                ch[2],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, has_bias=True, dimension=3)

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[2],
                ch[3],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, has_bias=True, dimension=3)

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[3],
                ch[4],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, has_bias=True, dimension=3)

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[4],
                ch[5],
                kernel_size=2,
                stride=2,
                generate_new_coords=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block5_cls = ME.MinkowskiConvolution(
            ch[5], 1, kernel_size=1, has_bias=True, dimension=3)

        # # Block 6
        # self.block6 = nn.Sequential(
        #     ME.MinkowskiConvolutionTranspose(
        #         ch[5],
        #         ch[6],
        #         kernel_size=2,
        #         stride=2,
        #         generate_new_coords=True,
        #         dimension=3),
        #     ME.MinkowskiBatchNorm(ch[6]),
        #     ME.MinkowskiELU(),
        #     ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
        #     ME.MinkowskiBatchNorm(ch[6]),
        #     ME.MinkowskiELU(),
        # )

        # self.block6_cls = ME.MinkowskiConvolution(
        #     ch[6], 1, kernel_size=1, has_bias=True, dimension=3)

        # # Block 7
        # self.block7 = nn.Sequential(
        #     ME.MinkowskiConvolutionTranspose(
        #         ch[6],
        #         ch[7],
        #         kernel_size=2,
        #         stride=2,
        #         generate_new_coords=True,
        #         dimension=3),
        #     ME.MinkowskiBatchNorm(ch[7]),
        #     ME.MinkowskiELU(),
        #     ME.MinkowskiConvolution(ch[7], ch[7], kernel_size=3, dimension=3),
        #     ME.MinkowskiBatchNorm(ch[7]),
        #     ME.MinkowskiELU(),
        # )

        # self.block7_cls = ME.MinkowskiConvolution(
        #     ch[7], 1, kernel_size=1, has_bias=True, dimension=3)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(
                out.coords_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z, target_key):
        out_cls, targets = [], []
        
        print('z', z.shape)
        # this is B x 889; the 889 is apparently a one-hot encoding the instance of a class

        # Block1
        out1 = self.block1(z)
        print('out1', out1.shape)
        # this is B*64 x 512
        
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).cpu().squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        out1 = self.pruning(out1, keep1.cpu())
        print('out1 pruned', out1.shape)
        # this is something like B*N x 512
        # where N<64, is the number of voxels not pruned

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).cpu().squeeze()

        if self.training:
            keep2 += target

        print('out2', out2.shape)
        # this is B*N*8
        # and, indeed, all steps from here onward  
        # produce 8x their input points
        # the bookkeeping of this must be a nightmare
        # however, i suppose it's the case that
        # each voxel on the batch dim can be treated independently
        # > no, wait.
        # > that can't be true
        # > well, this is the case if kernel size 1, but it's not
        # so, these "keep" tensors must be helping out a lot
        # 
        
        # Remove voxels 16
        out2 = self.pruning(out2, keep2.cpu())
        print('out2 pruned', out2.shape)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).cpu().squeeze()

        if self.training:
            keep3 += target

        print('out3', out3.shape)
        # Remove voxels 8
        out3 = self.pruning(out3, keep3.cpu())
        print('out3 pruned', out3.shape)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).cpu().squeeze()

        if self.training:
            keep4 += target

        print('out4', out4.shape)
        # Remove voxels 4
        out4 = self.pruning(out4, keep4.cpu())
        print('out4 pruned', out4.shape)

        # Block 5
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).cpu().squeeze()

        if self.training:
            keep5 += target

        print('out5', out5.shape)
        # Remove voxels 2
        out5 = self.pruning(out5, keep5.cpu())
        print('out5 pruned', out5.shape)

        # # Block 5
        # out6 = self.block6(out5)
        # out6_cls = self.block6_cls(out6)
        # target = self.get_target(out6, target_key)
        # targets.append(target)
        # out_cls.append(out6_cls)
        # keep6 = (out6_cls.F > 0).cpu().squeeze()

        # # Last layer does not require keep
        # # if self.training:
        # #   keep6 += target

        # print('out6', out6.shape)
        # # Remove voxels 1
        # out6 = self.pruning(out6, keep6.cpu())
        # print('out6 pruned', out6.shape)



        # # Block 6
        # out7 = self.block7(out6)
        # out7_cls = self.block7_cls(out7)
        # target = self.get_target(out7, target_key)
        # targets.append(target)
        # out_cls.append(out7_cls)
        # keep7 = (out7_cls.F > 0).cpu().squeeze()

        # # Last layer does not require keep
        # # if self.training:
        # #   keep7 += target

        # print('out7', out7.shape)
        # # Remove voxels 1
        # out7 = self.pruning(out7, keep7.cpu())
        # print('out7 pruned', out7.shape)

        # return out_cls, targets, out7


        return out_cls, targets, out5
        

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                has_bias=False,
                dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=out_feat,
                kernel_size=3,
                stride=2,
                dimension=D))
            # ME.MinkowskiGlobalPooling(),
            # ME.MinkowskiLinear(128, out_feat))

    def forward(self, x):
        return self.net(x)

class CARLA_MINKO(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaMinkoModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        if hyp.do_freeze_sigen3d:
            self.model.sigen3dnet.eval()
            self.set_requires_grad(self.model.sigen3dnet, False)
            
    # take over go() from base
    def go(self):
        self.start_time = time.time()
        self.initialize_model()
        print("------ Done creating models ------")

        set_nums = []
        set_names = []
        set_batch_sizes = []
        set_data_formats = []
        set_seqlens = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []

        for set_name in hyp.set_names:
            if hyp.sets_to_run[set_name]:
                set_nums.append(hyp.set_nums[set_name])
                set_data_formats.append(hyp.data_formats[set_name])
                set_seqlens.append(hyp.seqlens[set_name])
                set_names.append(set_name)
                set_batch_sizes.append(hyp.batch_sizes[set_name])
                set_inputs.append(self.all_inputs[set_name])
                set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                set_log_freqs.append(hyp.log_freqs[set_name])
                set_do_backprops.append(hyp.sets_to_backprop[set_name])
                set_dicts.append({})
                set_loaders.append(iter(set_inputs[-1]))

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': hyp.lr},
        ])
        
        model_state_dict = self.model.state_dict()
        for k in model_state_dict.keys():
            print('key', k)
        
        self.start_iter = saverloader.load_weights(self.model, None)
            
        print("------ Done loading weights ------")

        for step in list(range(self.start_iter+1, hyp.max_iters+1)):
            # reset set_loader after each epoch
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0:
                    set_loaders[i] = iter(set_input)
            for (set_num,
                 set_data_format,
                 set_seqlen,
                 set_name,
                 set_batch_size,
                 set_input,
                 set_writer,
                 set_log_freq,
                 set_do_backprop,
                 set_dict,
                 set_loader
            ) in zip(
                set_nums,
                set_data_formats,
                set_seqlens,
                set_names,
                set_batch_sizes,
                set_inputs,
                set_writers,
                set_log_freqs,
                set_do_backprops,
                set_dicts,
                set_loaders
            ):   
                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0
                
                output_dict = dict()


                if log_this or set_do_backprop or hyp.do_test:
                    # print('%s: set_num %d; set_data_format %s; set_seqlen %s; log_this %d; set_do_backprop %d; ' % (
                    #     set_name, set_num, set_data_format, set_seqlen, log_this, set_do_backprop))
                    # print('log_this = %s' % log_this)
                    # print('set_do_backprop = %s' % set_do_backprop)

                    read_start_time = time.time()
                    feed, data_ind = next(set_loader)
                    data_ind = data_ind.detach().cpu().numpy()
                    # print('data_ind', data_ind)
                    feed_cuda = {}
                    for k in feed:
                        try:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True)
                        except:
                            # some things are not tensors (e.g., filename)
                            feed_cuda[k] = feed[k]

                    read_time = time.time() - read_start_time

                    feed_cuda['writer'] = set_writer
                    feed_cuda['global_step'] = step
                    feed_cuda['set_num'] = set_num
                    feed_cuda['set_log_freq'] = set_log_freq
                    feed_cuda['set_data_format'] = set_data_format
                    feed_cuda['set_seqlen'] = set_seqlen
                    feed_cuda['set_name'] = set_name
                    feed_cuda['set_batch_size'] = set_batch_size
                    feed_cuda['data_ind'] = data_ind

                    iter_start_time = time.time()

                    if set_do_backprop:
                        self.model.train()
                        loss, results, returned_early = self.model(feed_cuda)
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            loss, results, returned_early = self.model(feed_cuda)
                    loss_py = loss.cpu().item()

                    if (not returned_early) and (set_do_backprop) and (hyp.lr > 0):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (
                        hyp.name,
                        step,
                        hyp.max_iters,
                        total_time,
                        read_time,
                        iter_time,
                        loss_py,
                        set_name))
                    
            if np.mod(step, hyp.snap_freq) == 0 and hyp.lr > 0:
                saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

        for writer in set_writers: #close writers to flush cache into file
            writer.close()

class CarlaMinkoModel(nn.Module):
    def __init__(self):
        super(CarlaMinkoModel, self).__init__()
        
        # self.crop_2x = (18*2,18*2,18*2)
        # self.crop = (18,18,18)
        # self.crop_low = (2,2,2)
        # self.crop_mid = (8,8,8)

        # self.crop = (0,0,0)
        self.crop = (4,4,4)
        # self.crop = (5,5,5)
        # self.crop = (6,6,6)
        # self.crop = (7,7,7)
        # self.crop = (8,8,8)
        # self.crop = (9,9,9)
        # self.crop = (10,10,10)
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)

        if hyp.do_render:
            self.rendernet = RenderNet()

        if hyp.do_occ:
            self.occnet = OccNet()

        if hyp.do_sigen3d:
            self.sigen3dnet1 = Sigen3dNet().cuda()
            self.sigen3dnet2 = Sigen3dNet().cuda()
            
        if hyp.do_rgb:
            self.rgbnet = RgbNet()

        # self.example_net = ExampleNetwork(in_feat=3, out_feat=5, D=3).cuda()
        self.generator = GenerativeNet(64, 10).cuda()
        self.completor = CompletionNet(hyp.Z).cuda()
        self.crit = nn.BCEWithLogitsLoss()
        
    def zero_border(self, feat, crop):
        feat = self.crop_feat(feat, crop)
        feat = self.pad_feat(feat, crop)
        return feat
    
    def crop_feat(self, feat_pad, crop):
        Z_pad, Y_pad, X_pad = crop
        feat = feat_pad[:,:,
                        Z_pad:-Z_pad,
                        Y_pad:-Y_pad,
                        X_pad:-X_pad].clone()
        return feat
    
    def pad_feat(self, feat, crop):
        Z_pad, Y_pad, X_pad = crop
        feat_pad = F.pad(feat, (Z_pad, Z_pad, Y_pad, Y_pad, X_pad, X_pad), 'constant', 0)
        return feat_pad

    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']
        self.data_ind = feed['data_ind']
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V = hyp.H, hyp.W, hyp.V
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
            
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams"]
        set_data_format = feed['set_data_format']
        

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        # scene_centroid_x = np.random.uniform(-8.0, 8.0)
        # scene_centroid_y = np.random.uniform(0.0, 2.0)
        # scene_centroid_z = np.random.uniform(8.0, 26.0)
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_camXs = feed['rgb_camXs']
        self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))

        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camX0s_ = utils.geom.apply_4x4(__p(self.camX0s_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camX0s_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        
        # self.summ_writer.summ_oned('inputs/valid_camX0_before', self.valid_camXs[:,0], norm=False)

        # weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
        # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        # self.valid_camXs = __u((F.conv2d(__p(self.valid_camXs), weights, padding=1)).clamp(0, 1))
        
        self.summ_writer.summ_oned('inputs/depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        # self.summ_writer.summ_oned('inputs/valid_camX0_after', self.valid_camXs[:,0], norm=False)
        

        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        self.rgb_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))

        self.summ_writer.summ_rgb('inputs/rgb_camX0', self.rgb_camXs[:,0])

        rgb_memX0 = self.vox_util.unproject_rgb_to_mem(
            self.rgb_camXs[:,0], self.Z1, self.Y1, self.X1, self.pix_T_cams[:,0])
        occ_memX0 = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,0], self.Z1, self.Y1, self.X1)
        self.summ_writer.summ_occ('inputs/occ_memX0', occ_memX0)
        
        xyz_camX0 = self.xyz_camX0s[:,0]
        xyz_memX0 = self.vox_util.Ref2Mem(xyz_camX0, self.Z1, self.Y1, self.X1)

        xyz_camX0_all = self.xyz_camX0s.reshape(self.B, -1, 3)
        xyz_memX0_all = self.vox_util.Ref2Mem(xyz_camX0_all, self.Z1, self.Y1, self.X1)
        
        occ_memX0_all = self.vox_util.voxelize_xyz(xyz_camX0_all, self.Z1, self.Y1, self.X1)
        self.summ_writer.summ_occ('inputs/occ_memX0_all', occ_memX0_all)

        occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
            self.camX0s_T_camXs,
            self.xyz_camXs,
            self.Z, self.Y, self.X,
            agg=True)
        # sub2ind3d(depth, height, width, d, h, w)
        # meshgrid3d
        xyz_grid = utils.basic.meshgrid3d(self.B, self.Z, self.Y, self.X, stack=True)[0]
        # this is B x Z x Y x X x 3
        xyz_pos = xyz_grid[occ_memX0_sup[0,0] > 0]
        xyz_neg = xyz_grid[free_memX0_sup[0,0] > 0]

        perm = np.random.permutation(len(xyz_pos))
        if len(xyz_pos) > 5000:
            xyz_pos = xyz_pos[perm[:5000]]
        
        # print('xyz_pos', xyz_pos.shape)
        # print('xyz_neg', xyz_neg.shape)
        perm = np.random.permutation(len(xyz_neg))
        # if len(xyz_neg) > len(xyz_pos):
        #     xyz_neg = xyz_neg[perm[:len(xyz_pos)]]
        #     print('xyz_neg trim', xyz_neg.shape)
        if len(xyz_neg) > 10000:
            xyz_neg = xyz_neg[perm[:10000]]
            print('xyz_neg trim', xyz_neg.shape)
        # these are N x 3

        device = torch.device('cuda')
        
        debug_with_generator = False
        debug_custom = False
        
        # debug_with_generator = True
        debug_custom = True

        if debug_with_generator:
            self.generator.train()
            
            init_coords = torch.zeros((self.B, 4), dtype=torch.int)
            init_coords[:, 0] = torch.arange(self.B)

            in_feat = torch.zeros((self.B, 10))
            in_feat[torch.arange(self.B), self.data_ind] = 1

            sin = ME.SparseTensor(
                feats=in_feat,
                coords=init_coords,
                allow_duplicate_coords=True,  # for classification, it doesn't matter
                tensor_stride=self.Z1,
            ).to(device)

            out_coord = torch.floor(xyz_memX0[0]).detach().cpu().numpy()
            inds = ME.utils.sparse_quantize(out_coord, return_index=True)
            out_coord = out_coord[inds]
            out_coord = torch.from_numpy(out_coord).float().cuda()
            print('out_coord', out_coord.shape)

            # Generate target sparse tensor
            cm = sin.coords_man
            target_key = cm.create_coords_key(
                ME.utils.batched_coordinates([out_coord]),
                force_creation=True,
                allow_duplicate_coords=True)

            out_cls, targets, sout = self.generator(sin, target_key)
            print('sout', sout.shape)

            dense_output, min_coord, tensor_stride = sout.dense()

            batch_coords, batch_feats = sout.decomposed_coordinates_and_features
            print('batch_coords', batch_coords[0].shape)

            xyz_mem = batch_coords[0].unsqueeze(0)
            xyz_mem = xyz_mem.float().cuda()
            
            occ = self.vox_util.get_occupancy(xyz_mem, self.Z1, self.Y1, self.X1)
            # input()
            # print('dense_output', dense_output.shape)

            self.summ_writer.summ_occ('minko/dense_output_occ', occ)
            self.summ_writer.summ_oned('minko/dense_output_oned', occ, bev=True, norm=True)

            num_layers, loss = len(out_cls), 0
            for out_cl, target in zip(out_cls, targets):
                curr_loss = self.crit(out_cl.F.squeeze(),
                                      target.type(out_cl.F.dtype).to(device))
                loss += curr_loss / num_layers
            total_loss = utils.misc.add_loss('minko/loss', total_loss, loss, 1.0, self.summ_writer)
        elif debug_custom:
            # self.generator.train()
            self.completor.train()
            
            in_coord = torch.floor(xyz_memX0[0]).detach().cpu().numpy()
            inds, inv_inds = ME.utils.sparse_quantize(in_coord, return_index=True, return_inverse=True)
            in_coord = in_coord[inds]
            in_coord = torch.from_numpy(in_coord).float().cuda()
            in_feat = torch.ones_like(in_coord[:,0:1])
            print('in_coord', in_coord.shape)
            print('in_feat', in_feat.shape)

            sin = ME.SparseTensor(
                feats=in_feat,
                # coords=ME.utils.batch_sparse_collate([in_coord]),
                coords=ME.utils.batched_coordinates([in_coord]),
                # coords=in_coord,
                # allow_duplicate_coords=True,  # for classification, it doesn't matter
                # tensor_stride=self.Z1,
            ).to(device)
            print('-'*10)

            pos_coord = torch.floor(xyz_pos).detach().cpu().numpy()
            inds = ME.utils.sparse_quantize(pos_coord, return_index=True)
            pos_coord = pos_coord[inds]
            pos_coord = torch.from_numpy(pos_coord).float().cuda()
            print('pos_coord', pos_coord.shape)

            neg_coord = torch.floor(xyz_neg).detach().cpu().numpy()
            inds = ME.utils.sparse_quantize(neg_coord, return_index=True)
            neg_coord = neg_coord[inds]
            neg_coord = torch.from_numpy(neg_coord).float().cuda()
            print('neg_coord', neg_coord.shape)

            # Generate target sparse tensor
            coords_manager = sin.coords_man
            pos_key = coords_manager.create_coords_key(
                ME.utils.batched_coordinates([pos_coord]),
                force_creation=True,
                allow_duplicate_coords=True)
            neg_key = coords_manager.create_coords_key(
                ME.utils.batched_coordinates([neg_coord]),
                force_creation=True,
                allow_duplicate_coords=True)
            
            # ins, outs = coords_manager.get_coords_map(1,1)
            # print('ins', ins[:10], ins.shape)
            # print('outs', outs[:10], outs.shape)

            # out_cls, targets, sout = self.generator(sin, target_key)
            out_cls, positives, negatives, sout = self.completor(sin, pos_key, neg_key)
            # print('out_cls[-1]', out_cls[-1].shape)
            print('sout', sout.shape)

            # dense_output, min_coord, tensor_stride = sout.dense()
            # print('min_coord', min_coord)

            sparse_output, min_coord, tensor_stride = sout.sparse()
            print('min_coord', min_coord)

            # batch_coords, batch_feats = sout.decomposed_coordinates_and_features
            # print('batch_coords', batch_coords[0].shape)

            xyz_mem = sout.coordinates_at(0)
            # we already pruned!!

            
            # xyz_feat = sout.features_at(0)

            # print('xyz_mem', xyz_mem.shape)
            # print('xyz_feat', xyz_feat.shape)
            # xyz_sig = F.sigmoid(xyz_feat)
            # utils.basic.print_stats('xyz_feat', xyz_feat)
            # utils.basic.print_stats('xyz_sig', xyz_sig)
            # xyz_mem = xyz_mem[xyz_sig > 0.5]
            # input()
            
            xyz_mem = xyz_mem.unsqueeze(0)
            xyz_mem = xyz_mem + min_coord.reshape(1, 1, 3)

            # i would like to add min_coord, but i do not want to risk OOM by densifying
            
            # xyz_mem = batch_coords[0].unsqueeze(0)
            # xyz_mem = ins[xyz_mem]
            # xyz_mem[xyz_mem==ins] = 
            # xyz_mem = torch.from_numpy(xyz_mem).float().cuda()
            xyz_mem = xyz_mem.float().cuda()
            occ = self.vox_util.get_occupancy(xyz_mem, self.Z1, self.Y1, self.X1)
            # input()
            # print('dense_output', dense_output.shape)

            self.summ_writer.summ_occ('minko/dense_output_occ', occ)
            self.summ_writer.summ_oned('minko/dense_output_oned', occ, bev=True, norm=True)

            num_layers, loss = len(out_cls), 0
            for out_cl, positive, negative in zip(out_cls, positives, negatives):
                # print('out_cl', out_cl.shape, out_cl.F.squeeze()[:10])
                # print('target', target.shape, target[:10])

                out_cl = out_cl.F.squeeze()
                positive = positive.type(out_cl.dtype).to(device)
                negative = negative.type(out_cl.dtype).to(device)

                out_pos = out_cl[positive > 0]
                positive = positive[positive > 0]

                out_neg = out_cl[negative > 0]
                negative = 1.0 - negative[negative > 0]

                pos_loss = self.crit(out_pos, positive)
                neg_loss = self.crit(out_neg, negative)
                
                # curr_loss = self.crit(out_cl.F.squeeze(),
                #                       target.type(out_cl.F.dtype).to(device))
                # pos_loss = self.crit(out_cl.F.squeeze(),
                #                      positive.type(out_cl.F.dtype).to(device))
                # neg_loss = self.crit(out_cl.F.squeeze(),
                #                      negative.type(out_cl.F.dtype).to(device))
                loss += (pos_loss + neg_loss) / num_layers

            # out_cl, target = out_cls[-1], targets[-1]
            # curr_loss = self.crit(out_cl.F.squeeze(),
            #                       target.type(out_cl.F.dtype).to(device))
            # loss = curr_loss

            # xyz_mem = batch_coords[0].unsqueeze(0).detach()
            # xyz_mem = xyz_mem.float().cuda()
            # occ_memX0_sup, free_memX0_sup, _, _ = self.vox_util.prep_occs_supervision(
            #     self.camX0s_T_camXs,
            #     self.xyz_camXs,
            #     self.Z, self.Y, self.X,
            #     agg=True)
            # samp_occ = utils.samp.bilinear_sample3d(occ_memX0_sup, xyz_mem)
            # samp_free = utils.samp.bilinear_sample3d(free_memX0_sup, xyz_mem)

            # def compute_loss(pred, occ, free):
            #     pos = occ.clone()
            #     neg = free.clone()

            #     # occ is B x 1 x Z x Y x X
            #     label = pos*2.0 - 1.0
            #     a = -label * pred
            #     b = F.relu(a)
            #     loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

            #     mask_ = (pos+neg>0.0).float()

            #     pos_loss = utils.basic.reduce_masked_mean(loss, pos)
            #     neg_loss = utils.basic.reduce_masked_mean(loss, neg)

            #     balanced_loss = pos_loss + neg_loss
            #     return balanced_loss
            # loss = compute_loss(batch_feats[0].unsqueeze(0),
            #                     samp_occ,
            #                     samp_free)
            total_loss = utils.misc.add_loss('minko/loss', total_loss, loss, 1.0, self.summ_writer)

            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)
        # data_ok = False
        
        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            elif self.set_name=='val':
                return self.run_test(feed)
            elif self.set_name=='test':
                return self.run_test(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)
                
