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
import utils_py

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
    
class Gen2dvqNet(nn.Module):
    def __init__(self):#, input_dim=2, num_layers=15):
        super().__init__()

        self.dict_dim = 512
        self.emb_dim = 64
        self.embed_dict = nn.Embedding(self.dict_dim, self.emb_dim).cuda()

        fm = 64
        self.net = nn.Sequential(
            MaskedConv2d('A', self.emb_dim,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.Conv2d(fm, self.dict_dim, 1))
        print(self.net)

    def embed(self, discrete_image):
        B, H, W = list(discrete_image.shape)
        # utils_py.print_stats('discrete_image', discrete_image.cpu().detach().numpy())
        emb = self.embed_dict(discrete_image.view(-1)).view(B, H, W, self.emb_dim)
        emb = emb.permute(0, 3, 1, 2) # (B, self.emb_dim, H, W)
        # print('emb', emb.shape)
        return emb
    
    def generate_sample(self, B, H, W):
        sample = torch.zeros(B, H, W).long().cuda()
        for i in range(H):
            for j in range(W):
                emb = self.embed(sample)
                out = self.net(emb)
                probs = F.softmax(out[:, :, i, j]).data
                sample[:, i, j] = torch.multinomial(probs, 1).squeeze().data
        return sample
    
    def forward(self, image_input, summ_writer=None):
        B, H, W = list(image_input.shape)
        total_loss = torch.tensor(0.0).cuda()
        summ_writer.summ_oned('gen2dvq/image_input', image_input.unsqueeze(1)/512.0, norm=False)

        emb = self.embed(image_input)
        image_output_logits = self.net(emb)
        # print('logits', image_output_logits.shape)
        
        image_output = torch.argmax(image_output_logits, dim=1, keepdim=True)
        # print('output', image_output.shape)
        summ_writer.summ_oned('gen2dvq/image_output', image_output.float()/512.0, norm=False)
        ce_loss_image = F.cross_entropy(image_output_logits, image_input, reduction='none')

        summ_writer.summ_oned('gen2dvq/ce_loss', ce_loss_image.unsqueeze(1))
        
        ce_loss = torch.mean(ce_loss_image)
        total_loss = utils_misc.add_loss('gen2dvq/ce_loss', total_loss, ce_loss, hyp.gen2dvq_coeff, summ_writer)
        
        return total_loss


    
    
