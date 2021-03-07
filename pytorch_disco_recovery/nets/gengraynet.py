import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3D as encoder3D
from utils_basic import *
import utils_geom
import utils_vox
import utils_misc
import utils_basic

EPS = 1e-4
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    
class GengrayNet(nn.Module):
    def __init__(self):#, input_dim=2, num_layers=15):
        super().__init__()

        fm = 64
        self.net = nn.Sequential(
            MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.Conv2d(fm, 256, 1))#.cuda()
        print(self.net)
        
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # # embedding layer to embed input
        # self.embedding = nn.Embedding(input_dim, self.emb_dim)
        
        # # we will build the PixelCNN layer by layer
        # self.layers = nn.ModuleList()
        # # make initial block with Mask-A convolution; 
        # # make others with Mask-B convolutions
        # for i in range(num_layers):
        #     mask_type = 'A' if i == 0 else 'B'
        #     kernel = 5 if i == 0 else 3
        #     residual = False if i == 0 else True
        #     self.layers.append(
        #         GatedMaskedConv3d(mask_type, self.emb_dim, kernel, residual)
        #     )
        # # final layer
        # self.output_conv = nn.Sequential(
        #     nn.Conv3d(self.emb_dim, 512, 1),
        #     nn.ReLU(True),
        #     nn.Conv3d(512, input_dim, 1)
        # )

        # self.apply(weights_init)
    
    # def run_net(self, feat):
    #     B, C, Z, Y, X = list(feat.shape)
    #     # the input feat should be type int64/long
        
    #     feat = self.embedding(feat.view(-1)).view(B, Z, Y, X, self.emb_dim)
    #     feat = feat.permute(0, 4, 1, 2, 3)  # (B, self.emb_dim, Z, Y, X)

    #     t_x, t_y, t_z = (feat, feat, feat)
    #     for i, layer in enumerate(self.layers):
    #         t_x, t_y, t_z = layer(t_x, t_y, t_z)

    #     feat = self.output_conv(t_x)
        
    #     return feat

    def generate_sample(self, B, H, W):
        # param = next(self.parameters())
        # feat = torch.zeros((B, 1, Z, Y, X), dtype=torch.int64, device=param.device)
        
        # for i in list(range(Z)):
        #     for j in list(range(Y)):
        #         for k in list(range(X)):
        #             logits = self.run_net(feat)
        #             probs = F.softmax(logits[:,:,i,j,k],1)
        #             feat.data[:,:,i,j,k].copy_(probs.multinomial(1).squeeze().data)

        sample = torch.Tensor(B, 1, H, W).cuda()
        for i in range(H):
            for j in range(W):
                out = self.net(torch.autograd.Variable(sample, volatile=True))
                probs = F.softmax(out[:, :, i, j]).data
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
        return sample
    
    def forward(self, image_input, summ_writer=None):
        B, C, H, W = list(image_input.shape)
        total_loss = torch.tensor(0.0).cuda()
        
        summ_writer.summ_oned('gengray/image_input', image_input)
        # feat = feat.long() # discrete input

        # logit = self.run_net(feat)

        # print(image_input.shape)
        # target = torch.autograd.Variable((image_input.data[:,0] * 255).long()).cuda()
        # target = (image_input.data[:,0] * 255).long().cuda()
        target = ((image_input.data[:,0] + 0.5) * 255).long().cuda()
        # print(target.shape)

        image_output_logits = self.net(image_input)
        image_output = torch.argmax(image_output_logits, dim=1, keepdim=True)
        summ_writer.summ_oned('gengray/image_output', image_output.float()/255.0 - 0.5)
        
        ce_loss_image = F.cross_entropy(image_output_logits, target, reduction='none')

        summ_writer.summ_oned('gengray/ce_loss', ce_loss_image.unsqueeze(1))
        
        
        # # smooth loss
        # dz, dy, dx = gradient3D(logit, absolute=True)
        # smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        # summ_writer.summ_oned('gengray/smooth_loss', torch.mean(smooth_vox, dim=3))
        # # smooth_loss = utils_basic.reduce_masked_mean(smooth_vox, valid)
        # smooth_loss = torch.mean(smooth_vox)
        # total_loss = utils_misc.add_loss('gengray/smooth_loss', total_loss,
        #                                  smooth_loss, hyp.genocc_smooth_coeff, summ_writer)
        # summ_writer.summ_feat('gengray/feat_output', logit, pca=False)
        # occ_e = torch.argmax(logit, dim=1, keepdim=True)
        

        # loss_pos = self.criterion(logit, (occ_g[:,0]).long())
        # loss_neg = self.criterion(logit, (1-free_g[:,0]).long())

        # summ_writer.summ_oned('gengray/ce_loss',
        #                       torch.mean((loss_pos+loss_neg), dim=1, keepdim=True) * \
        #                       torch.clamp(occ_g+(1-free_g), 0, 1),
        #                       bev=True)
        
        ce_loss = torch.mean(ce_loss_image)
        total_loss = utils_misc.add_loss('gengray/ce_loss', total_loss, ce_loss, hyp.gengray_coeff, summ_writer)

        # sample = self.generate_sample(1, int(Z/2), int(Y/2), int(X/2))
        # occ_sample = torch.argmax(sample, dim=1, keepdim=True)
        # summ_writer.summ_occ('gengray/occ_sample', occ_sample)
        
        return total_loss


    
