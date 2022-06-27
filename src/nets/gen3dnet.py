import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3D as encoder3D
import archs.focal_loss
from utils_basic import *
import utils_geom
import utils_vox
import utils_misc
import utils_basic
import utils_py

EPS = 1e-4
class MaskedConv3d(nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kZ, kY, kX = self.weight.size()
        self.mask.fill_(1)
        # all slices after the current one
        self.mask[:, :, kZ // 2 + 1:] = 0
        # current slice, everything below the center zero
        self.mask[:, :, kZ // 2, kY // 2 + 1:] = 0
        # current slice, middle row, everything after the center zero
        self.mask[:, :, kZ // 2, kY // 2, kX // 2 + (mask_type == 'B'):] = 0
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)
    
class Gen3dNet(nn.Module):
    def __init__(self):#, input_dim=2, num_layers=15):
        super().__init__()

        self.dict_dim = hyp.vq3d_num_embeddings
        self.emb_dim = 64 # note this is not tied to the dim of the dict on the other side
        self.embed_dict = nn.Embedding(self.dict_dim, self.emb_dim).cuda()

        # self.focal = archs.focal_loss.FocalLoss()

        fm = 64
        self.net = nn.Sequential(
            MaskedConv3d('A', self.emb_dim,  fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            MaskedConv3d('B', fm, fm, 5, 1, 2, bias=False), nn.BatchNorm3d(fm), nn.ReLU(True),
            nn.Conv3d(fm, self.dict_dim, 1))
        print(self.net)

    def embed(self, discrete_image):
        B, Z, Y, X = list(discrete_image.shape)
        # utils_py.print_stats('discrete_image', discrete_image.cpu().detach().numpy())
        emb = self.embed_dict(discrete_image.view(-1)).view(B, Z, Y, X, self.emb_dim)
        emb = emb.permute(0, 4, 1, 2, 3) # (B, self.emb_dim, Z, Y, X)
        return emb
    
    def generate_sample(self, B, Z, Y, X, stop_early=False):
        sample = torch.zeros(B, Z, Y, X).long().cuda()
        
        if stop_early:
            # Z_new = int(Z//2)
            Z_new = int(Z//4)
        else:
            Z_new = int(Z)

        print('plan is to fill up a sample shaped', sample.shape)
            
        for i in range(Z_new):
            print('working on z=%d' % i)
            for j in range(Y):
                for k in range(X):
                    emb = self.embed(sample)
                    out = self.net(emb)
                    probs = F.softmax(out[:, :, i, j, k]).data
                    sample[:, i, j, k] = torch.multinomial(probs, 1).squeeze().data
        return sample
    
    def forward(self, vox_input, summ_writer=None):
        B, Z, Y, X = list(vox_input.shape)
        total_loss = torch.tensor(0.0).cuda()
        summ_writer.summ_oned('gen3d/vox_input', vox_input.unsqueeze(1)/512.0, bev=True, norm=False)

        emb = self.embed(vox_input)
        vox_output_logits = self.net(emb)
        # print('logits', vox_output_logits.shape)
        
        vox_output = torch.argmax(vox_output_logits, dim=1)
        # print('output', vox_output.shape)
        summ_writer.summ_oned('gen3d/vox_output', vox_output.unsqueeze(1).float()/512.0, bev=True, norm=False)
        ce_loss_vox = F.cross_entropy(vox_output_logits, vox_input, reduction='none')
        # ce_loss_vox = self.focal(vox_output_logits, vox_input, reduction='none')
        summ_writer.summ_oned('gen3d/ce_loss', ce_loss_vox.unsqueeze(1), bev=True)
        
        ce_loss = torch.mean(ce_loss_vox)
        total_loss = utils_misc.add_loss('gen3d/ce_loss', total_loss, ce_loss, hyp.gen3d_coeff, summ_writer)
        
        return total_loss, vox_output


    
    
