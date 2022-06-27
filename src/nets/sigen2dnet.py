import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
# import archs.encoder3D as encoder3D
import archs.sparse_invar_encoder2D
import archs.DCCA_sparse_networks
from utils_basic import *
import utils_geom
import utils_improc
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
    
class Sigen2dNet(nn.Module):
    def __init__(self):#, input_dim=2, num_layers=15):
        super().__init__()

        self.dict_dim = 512
        self.emb_dim = 64
        self.embed_dict = nn.Embedding(self.dict_dim, self.emb_dim).cuda()

        self.num_choices = 64

        self.use_grid = True
        if self.use_grid:
            # since we are using a grid as additional input, we add 2 to the emb_dim
            self.net = archs.DCCA_sparse_networks.SimpleCNN(self.emb_dim+2, self.dict_dim, chans=64)
        else:
            self.net = archs.DCCA_sparse_networks.SimpleCNN(self.emb_dim, self.dict_dim, chans=64)
        
        print(self.net)

    def embed(self, discrete_image):
        B, H, W = list(discrete_image.shape)
        # utils_py.print_stats('discrete_image', discrete_image.cpu().detach().numpy())
        emb = self.embed_dict(discrete_image.view(-1)).view(B, H, W, self.emb_dim)
        emb = emb.permute(0, 3, 1, 2) # (B, self.emb_dim, H, W)
        if self.use_grid:
            grid = utils_basic.meshgrid2D(B, H, W, stack=True, norm=True).permute(0, 3, 1, 2)
            emb = torch.cat([emb, grid], dim=1)
        return emb
    
    def generate_confidence(self, sparse_image):
        B, C, H, W = list(sparse_image.shape)
        assert(C==1)
        
        num_layers = 7
        
        weights = torch.ones(torch.Size([1, 1, 3, 3])).cuda()
        for layer in list(range(num_layers)):
            sparse_image = F.conv2d(
                sparse_image,
                weights,
                bias=None,
                stride=1,
                padding=1,
                dilation=1)
        return sparse_image
        
    # def generate_sample(self, B, H, W):
    #     # if sample==None:
    #     sample = torch.zeros(B, H, W).long().cuda()
    #     sample_mask = torch.zeros(B, 1, H, W).float().cuda()
    #     for i in range(H):
    #         for j in range(W):
    #             emb = self.embed(sample)
    #             out, _ = self.net(emb, sample_mask)
    #             probs = F.softmax(out[:, :, i, j]).data
    #             sample[:, i, j] = torch.multinomial(probs, 1).squeeze().data
    #             sample_mask[:, :, i, j] = 1.0
    #     return sample
    
    def generate_uncond_sample(self, sample, sample_mask, summ_writer=None, mod=''):
        # assume the topleft corner is done already
        B, H, W = list(sample.shape)
        B2, C, H2, W2 = list(sample_mask.shape)
        sample = sample.clone()
        sample_mask = sample_mask.clone()
        assert(B==1)
        assert(C==1)
        assert(B==B2)
        assert(H==H2)
        assert(W==W2)
        count = 0
        
        if summ_writer is not None:
            sample_masks = []
            sample_masks.append(sample_mask.clone())
            
        for i in range(H):
            for j in range(W):
                if not (i==0 and j==0):
                    emb = self.embed(sample)
                    out, _ = self.net(emb, sample_mask)
                    probs = F.softmax(out[:, :, i, j]).data
                    sample[:, i, j] = torch.multinomial(probs, 1).squeeze().data
                    sample_mask[:, :, i, j] = 1.0

                    if np.mod(count, 80)==0.0 and (summ_writer is not None):
                        sample_masks.append(sample_mask.clone())
                    count += 1
                    
        if summ_writer is not None:
            sample_masks.append(sample_mask.clone())
            print('saving %d masks as a gif' % len(sample_masks))
            summ_writer.summ_oneds('sigen2d/%scond_masks' % mod, sample_masks, norm=False)
                    
        return sample
    
    def generate_cond_sample(self, sample, sample_mask, summ_writer=None, mod=''):
        B, H, W = list(sample.shape)
        B2, C, H2, W2 = list(sample_mask.shape)
        sample = sample.clone()
        sample_mask = sample_mask.clone()
        assert(B==1)
        assert(C==1)
        assert(B==B2)
        assert(H==H2)
        assert(W==W2)

        count = 0

        # for i in range(H):
        
        if summ_writer is not None:
            sample_masks = []
            sample_masks.append(sample_mask.clone())
        
        while torch.sum(sample_mask) < (H*W):
            emb = self.embed(sample)
            out, _ = self.net(emb, sample_mask)

            # find the conv confidence in each pixel
            conf = self.generate_confidence(sample_mask)
            # do not choose a pixel we've already decided on
            conf = conf * (1.0 - sample_mask)
            # choose the most confident of the rest
            y, x = utils_basic.hard_argmax2D(conf)
            y = y[0]
            x = x[0]
            
            assert(sample[:,y,x]==0.0) # otw what are we doing
            probs = F.softmax(out[:, :, y, x]).data
            sample[:, y, x] = torch.multinomial(probs, 1).squeeze().data
            sample_mask[:, :, y, x] = 1.0

            if np.mod(count, 80)==0.0 and (summ_writer is not None):
                sample_masks.append(sample_mask.clone())
                # print('up to %d masks' % len(sample_masks))
            count += 1
                    
        # summ_writer.summ_oned('sigen2d/conf_iter_%06d' % count, conf, norm=True)
        if summ_writer is not None:
            sample_masks.append(sample_mask.clone())
            print('saving %d masks as a gif' % len(sample_masks))
            summ_writer.summ_oneds('sigen2d/%scond_masks' % mod, sample_masks, norm=False)
                
        return sample
    
    def forward(self, image_input, summ_writer=None, is_train=True):
        B, H, W = list(image_input.shape)
        total_loss = torch.tensor(0.0).cuda()
        summ_writer.summ_oned('sigen2d/image_input', image_input.unsqueeze(1)/512.0, norm=False)

        emb = self.embed(image_input)

        y = torch.randint(low=0, high=H, size=[B, self.num_choices, 1])
        x = torch.randint(low=0, high=W, size=[B, self.num_choices, 1])
        choice_mask = utils_improc.xy2mask(torch.cat([x, y], dim=2), H, W, norm=False)
        summ_writer.summ_oned('sigen2d/choice_mask', choice_mask, norm=False)

        # cover up the 3x3 region surrounding each choice
        xy = torch.cat([torch.cat([x-1, y-1], dim=2),
                        torch.cat([x+0, y-1], dim=2),
                        torch.cat([x+1, y-1], dim=2),
                        torch.cat([x-1, y], dim=2),
                        torch.cat([x+0, y], dim=2),
                        torch.cat([x+1, y], dim=2),
                        torch.cat([x-1, y+1], dim=2),
                        torch.cat([x+0, y+1], dim=2),
                        torch.cat([x+1, y+1], dim=2)], dim=1)
        input_mask = 1.0 - utils_improc.xy2mask(xy, H, W, norm=False)
        
        # if is_train:
        #     input_mask = (torch.rand((B, 1, H, W)).cuda() > 0.5).float()
        # else:
        #     input_mask = torch.ones((B, 1, H, W)).cuda().float()
        # input_mask = input_mask * (1.0 - choice_mask)
        # input_mask = 1.0 - choice_mask
        
        emb = emb * input_mask
        summ_writer.summ_oned('sigen2d/input_mask', input_mask, norm=False)

        image_output_logits, _ = self.net(emb, input_mask)
        
        image_output = torch.argmax(image_output_logits, dim=1, keepdim=True)
        summ_writer.summ_feat('sigen2d/emb', emb, pca=True)
        summ_writer.summ_oned('sigen2d/image_output', image_output.float()/512.0, norm=False)
        
        ce_loss_image = F.cross_entropy(image_output_logits, image_input, reduction='none').unsqueeze(1)
        # summ_writer.summ_oned('sigen2d/ce_loss', ce_loss_image)
        # ce_loss = torch.mean(ce_loss_image)

        # only apply loss at the choice pixels
        ce_loss = utils_basic.reduce_masked_mean(ce_loss_image, choice_mask)
        total_loss = utils_misc.add_loss('sigen2d/ce_loss', total_loss, ce_loss, hyp.sigen2d_coeff, summ_writer)
        
        return total_loss


    
    
