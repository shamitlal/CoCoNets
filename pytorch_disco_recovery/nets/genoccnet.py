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
def weights_init(m):
    classname = m.__class__.__name__
    # print('trying to init %s' % classname)
    if classname.find('Conv') != -1 and hasattr(m,'weight'):
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
            # print("ok looks good")
        except AttributeError:
            print("Skipping initialization of", classname)

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)

class GatedMaskedConv3d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        kernel_shp = (kernel // 2 + 1, kernel, kernel)
        padding_shp = (kernel // 2, kernel // 2, kernel // 2)
        self.z_stack = nn.Conv3d(dim, dim*2, kernel_shp, 1, padding_shp)
        print('initializing GatedMaskedConv3d z_stack with kernel', kernel_shp)

        self.z_to_x = nn.Conv3d(dim*2, dim*2, 1)

        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.y_stack = nn.Conv2d(dim, dim*2, kernel_shp, 1, padding_shp)
        print('initializing GatedMaskedConv3d y_stack with kernel', kernel_shp)

        self.y_to_x = nn.Conv3d(dim*2, dim*2, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.x_stack = nn.Conv2d(dim, dim*2, kernel_shp, 1, padding_shp)
        print('initializing GatedMaskedConv3d x_stack with kernel', kernel_shp)

        self.x_resid = nn.Conv3d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.z_stack.weight.data[:,:,-1].zero_() # Mask final Z
        self.y_stack.weight.data[:,:,-1].zero_()  # Mask final Y
        self.x_stack.weight.data[:,:,:,-1].zero_()  # Mask final X

    def condZ(self, t_z):
        B,CI,Z,Y,X = t_z.shape
        t_z = self.z_stack(t_z)[:,:,:Z] # [B,CO,Z,Y,X]
        return t_z

    def condY(self, t_y):
        B,CI,Z,Y,X = t_y.shape
        t_y = t_y.permute(0,2,1,3,4).reshape(B*Z,CI,Y,X)
        t_y = self.y_stack(t_y)[:,:,:Y] # [B*Z,CO,Y,X]
        CO = t_y.shape[1]
        t_y = t_y.view(B,Z,CO,Y,X).permute(0,2,1,3,4) # [B,CO,Z,Y,X]
        return t_y

    def condX(self, t_x):
        B,CI,Z,Y,X = t_x.shape
        t_x = t_x.permute(0,2,1,3,4).reshape(B*Z,CI,Y,X)
        t_x = self.x_stack(t_x)[:,:,:,:X] # [B*Z,CO,Y,X]
        CO = t_x.shape[1]
        t_x = t_x.view(B,Z,CO,Y,X).permute(0,2,1,3,4) # [B,CO,Z,Y,X]
        return t_x

    def forward(self, t_x, t_y, t_z, h=0.0):
        if self.mask_type == 'A':
            self.make_causal()

        # h = self.class_cond_embedding(h)
        # h = h[:,:,None,None,None]
        h = 0.0

        t_z = self.condZ(t_z)
        t_z2x = self.z_to_x(t_z)
        out_z = self.gate(t_z + h)

        t_y = self.condY(t_y)
        t_y2x = self.y_to_x(t_y)
        out_y = self.gate(t_y + h)

        t_x_prev = t_x
        t_x = self.condX(t_x)
        out_x = self.gate(t_z2x + t_y2x + t_x + h)

        if self.residual:
            out_x = self.x_resid(out_x) + t_x_prev
        else:
            out_x = self.x_resid(out_x)

        return out_x, out_y, out_z
    
class GenoccNet(nn.Module):
    def __init__(self, input_dim=2, num_layers=15):
        super().__init__()
        
        self.emb_dim = 32

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, self.emb_dim)
        
        # we will build the PixelCNN layer by layer
        self.layers = nn.ModuleList()
        # make initial block with Mask-A convolution; 
        # make others with Mask-B convolutions
        for i in range(num_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 5 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(
                GatedMaskedConv3d(mask_type, self.emb_dim, kernel, residual)
            )
        # final layer
        self.output_conv = nn.Sequential(
            nn.Conv3d(self.emb_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv3d(512, input_dim, 1)
        )

        self.apply(weights_init)
    
    def run_net(self, feat):
        B, C, Z, Y, X = list(feat.shape)
        # the input feat should be type int64/long
        
        feat = self.embedding(feat.view(-1)).view(B, Z, Y, X, self.emb_dim)
        feat = feat.permute(0, 4, 1, 2, 3)  # (B, self.emb_dim, Z, Y, X)

        t_x, t_y, t_z = (feat, feat, feat)
        for i, layer in enumerate(self.layers):
            t_x, t_y, t_z = layer(t_x, t_y, t_z)

        feat = self.output_conv(t_x)
        
        return feat

    def generate_sample(self, feat, occ, free):
        B, C, Z, Y, X = list(feat.shape)
        assert(B==1)
        assert(C==1)
        for i in list(range(Z)):
            for j in list(range(Y)):
                for k in list(range(X)):
                    # print(occ.shape)
                    # print(occ[0,0,i,j,k])
                    # print(free[0,0,i,j,k])
                    if occ[0,0,i,j,k]==0.0 and free[0,0,i,j,k]==0.0:
                        logits = self.run_net(feat)
                        probs = F.softmax(logits[:,:,i,j,k],1)
                        feat.data[:,:,i,j,k].copy_(probs.multinomial(1).squeeze().data)
        return feat
    
    def generate_uncond_sample(self, B, Z, Y, X):
        param = next(self.parameters())
        feat = torch.zeros((B, 1, Z, Y, X), dtype=torch.int64, device=param.device)
        
        for i in list(range(Z)):
            for j in list(range(Y)):
                for k in list(range(X)):
                    logits = self.run_net(feat)
                    probs = F.softmax(logits[:,:,i,j,k],1)
                    feat.data[:,:,i,j,k].copy_(probs.multinomial(1).squeeze().data)
        return feat
    
    def forward(self, feat, occ_g, free_g, summ_writer=None):
        B, C, Z, Y, X = list(feat.shape)
        total_loss = torch.tensor(0.0).cuda()
        
        summ_writer.summ_feat('genoccnet/feat_input', feat, pca=False)
        feat = feat.long() # discrete input

        logit = self.run_net(feat)

        # smooth loss
        dz, dy, dx = gradient3D(logit, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        summ_writer.summ_oned('genoccnet/smooth_loss', torch.mean(smooth_vox, dim=3))
        # smooth_loss = utils_basic.reduce_masked_mean(smooth_vox, valid)
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('genoccnet/smooth_loss', total_loss,
                                         smooth_loss, hyp.genocc_smooth_coeff, summ_writer)
        
        summ_writer.summ_feat('genoccnet/feat_output', logit, pca=False)
        occ_e = torch.argmax(logit, dim=1, keepdim=True)
        
        summ_writer.summ_occ('genoccnet/occ_e', occ_e)
        summ_writer.summ_occ('genoccnet/occ_g', occ_g)

        loss_pos = self.criterion(logit, (occ_g[:,0]).long())
        loss_neg = self.criterion(logit, (1-free_g[:,0]).long())

        summ_writer.summ_oned('genoccnet/loss',
                              torch.mean((loss_pos+loss_neg), dim=1, keepdim=True) * \
                              torch.clamp(occ_g+(1-free_g), 0, 1),
                              bev=True)
        
        loss_pos = utils_basic.reduce_masked_mean(loss_pos.unsqueeze(1), occ_g)
        loss_neg = utils_basic.reduce_masked_mean(loss_neg.unsqueeze(1), free_g)
        loss_bal = loss_pos + loss_neg
        total_loss = utils_misc.add_loss('genoccnet/loss_bal', total_loss, loss_bal, hyp.genocc_coeff, summ_writer)

        # sample = self.generate_sample(1, int(Z/2), int(Y/2), int(X/2))
        # occ_sample = torch.argmax(sample, dim=1, keepdim=True)
        # summ_writer.summ_occ('genoccnet/occ_sample', occ_sample)
        
        return logit, total_loss


