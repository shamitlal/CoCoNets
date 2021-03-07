import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.v2v2d 
import utils_geom
import utils_vox
import utils_misc
import utils_basic
import utils_improc

class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

EPS = 1e-4
class ColNet2D(nn.Module):
    def __init__(self, in_dim=1):
        super(ColNet2D, self).__init__()
        
        # feat_dim = 32
        # self.encoder = archs.v2v2d.V2VModel(1, feat_dim).cuda()
        # self.down = Interpolate(0.5, 'bilinear')
        # self.crit = torch.nn.CrossEntropyLoss()
        
        self.huber = torch.nn.SmoothL1Loss(reduction='none')

    def correlation(self, feat_0, feat_1):
        B, F, N = list(feat_0.shape)

        # Becomes B, N, F
        feat_0 = feat_0.permute(0,2,1)

        # Becomes B, N, N
        innerproduct = torch.matmul(feat_0, feat_1)
        return innerproduct

    def attention(self, corr, temperature=1):
        B, N, N = list(corr.shape)
        
        # Softmax over origin image
        softmax_axis = 2

        similarity = torch.softmax(corr / temperature, softmax_axis)
        return similarity

    def propagate(self, labels, attention):
        B, C, N = list(labels.shape)
        B, N, N = list(attention.shape)

        # Becomes B, N, C
        labels_t = labels.permute(0,2,1)
        
        prediction = torch.matmul(attention, labels_t)
        prediction = prediction.permute(0,2,1)
        return prediction

    def forward(self,
                rgb0, rgb1, 
                feat0, feat1,
                summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        
        B, C, H, W = list(rgb0.shape)
        BF, F, HF, WF = list(feat0.shape)
        assert(B==BF)
        assert(H==HF)
        assert(W==WF)

        # lab0 = utils_improc.rgb_to_lab(rgb0+0.5) / 100.0
        # lab1 = utils_improc.rgb_to_lab(rgb1+0.5) / 100.0
        lab0 = utils_improc.rgb2lab(rgb0)
        lab1 = utils_improc.rgb2lab(rgb1)
        
        # print('rgb0', rgb0.shape)
        # print('lab0', lab0.shape)
        # print('feat0', feat0.shape)
        # utils_basic.print_stats('lab0', lab0)

        feat0_ = feat0.reshape(B, F, H*W)
        feat1_ = feat1.reshape(B, F, H*W)

        feat0_ = feat0_.permute(0,2,1)
        # this is B x H*W x F
        corr = torch.matmul(feat0_, feat1_) / 0.07
        # this is B x H*W x H*W
        softmax_ = torch.softmax(corr, 2)
        # this is (still) B x H*W x H*W
        # this tells us, for each pixel in rgb0, which colors to grab from rgb1

        use_rgb = False # the MAST paper recommends lab space for the loss
        if use_rgb:
            rgb1_ = rgb1.reshape(B, C, H*W, 1).permute(0, 3, 2, 1)
            # this is B x 1 x H*W x C
            softmax_ = softmax_.unsqueeze(3)
            # this is B x H*W x H*W x 1
            rgb0_e_ = torch.sum(softmax_ * rgb1_, dim=2)
            # this is B x H*W x C
            rgb0_e = rgb0_e_.permute(0, 2, 1).reshape(B, C, H, W)
            l1_loss_im = utils_basic.l1_on_axis(rgb0_e-rgb0, 1, keepdim=True)
            huber_loss_im = torch.sum(self.huber(rgb0_e, rgb0), dim=1, keepdim=True)
        else:
            lab1_ = lab1.reshape(B, C, H*W, 1).permute(0, 3, 2, 1)
            # this is B x 1 x H*W x C
            softmax_ = softmax_.unsqueeze(3)
            # this is B x H*W x H*W x 1
            lab0_e_ = torch.sum(softmax_ * lab1_, dim=2)
            # this is B x H*W x C
            lab0_e = lab0_e_.permute(0, 2, 1).reshape(B, C, H, W)
            rgb0_e = utils_improc.lab2rgb(lab0_e) # just for vis
            l1_loss_im = utils_basic.l1_on_axis(lab0_e-lab0, 1, keepdim=True)
            huber_loss_im = torch.sum(self.huber(lab0_e, lab0), dim=1, keepdim=True)
        
        summ_writer.summ_rgbs('col/rgb0', [rgb0_e.clamp(-0.5, 0.5), rgb0])
        summ_writer.summ_rgb('col/rgb0_e', rgb0_e.clamp(-0.5, 0.5))
        summ_writer.summ_rgb('col/rgb0_g', rgb0.clamp(-0.5, 0.5))
        summ_writer.summ_oned('col/l1_loss', l1_loss_im)
        summ_writer.summ_oned('col/huber_loss', huber_loss_im)
        
        l1_loss = torch.mean(l1_loss_im)
        huber_loss = torch.mean(huber_loss_im)
        total_loss = utils_misc.add_loss('col/l1_loss', total_loss, l1_loss, hyp.col2D_l1_coeff, summ_writer)
        total_loss = utils_misc.add_loss('col/huber_loss', total_loss, huber_loss, hyp.col2D_huber_coeff, summ_writer)
        
        # label0_g_one_hot_ = I[label0_g_.squeeze(1)].permute(0,2,1)
        
        # # frame1_labels_e_ = self.propagate(label0_g_one_hot_.float(), att) 

        # # Calculate loss
        # color_loss = self.crit(frame1_labels_e_, label1_g_.squeeze(1))
        # total_loss = utils_misc.add_loss('colorization/color_loss', total_loss, color_loss, 1, summ_writer)

        # frame1_labels_e_idxs = frame1_labels_e_.max(1)[1]
        
        # rgb_e = self.normalized_clusters[frame1_labels_e_idxs]
        # rgb_e_channels = rgb_e[:,:,1:]

        # # Reshape output back to image dimensions
        # frame1_channels_e = rgb_e_channels.reshape(B, 2, HF, WF)

        # return total_loss, frame1_channels_e

        return total_loss
        
