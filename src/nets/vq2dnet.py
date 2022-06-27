import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc
import utils_basic
import utils_py

class IndPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.inds = []
            
    def fetch(self):
        return self.inds
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, inds):
        for ind in inds:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.inds.pop(0)
            # add to the back
            self.inds.append(ind)
        return self.inds

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCZYX -> BZYXC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BZYXC -> BCZYX
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized, perplexity, encodings, encoding_indices
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # print('inputs', inputs.shape)
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # print('flat_input', flat_input.shape)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings, encoding_indices

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = F.relu(x)
        
        x = self._conv_4(x)
        
        x = self._residual_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._conv_trans_3(x)
        return x
        
class VqrgbNet(nn.Module):
    def __init__(self):
        super(VqrgbNet, self).__init__()

        print('VqrgbNet...')

        # self.net = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
        # print('net', net)

        # we'll keep track of the last 10k used inds,
        # as an interpretable measure of "perplexity"
        self.ind_pool = IndPool(10000)
        
        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 3
        embedding_dim = 64
        commitment_cost = 0.25

        self._encoder = Encoder(
            3,
            num_hiddens,
            num_residual_layers, 
            num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, 
            out_channels=embedding_dim,
            kernel_size=1, 
            stride=1)
        self._vq_vae = VectorQuantizer(
            hyp.vqrgb_num_embeddings, 
            embedding_dim,
            commitment_cost)
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens, 
            num_residual_layers, 
            num_residual_hiddens)

    def forward(self, rgb_g, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, H, W = list(rgb_g.shape)
        
        z = self._encoder(rgb_g)
        z = self._pre_vq_conv(z)
        # print('encoded z', z.shape)
        
        latent_loss, quantized, perplexity, _, inds = self._vq_vae(z)
        rgb_e = self._decoder(quantized)

        recon_loss = F.mse_loss(rgb_g, rgb_e)
        total_loss = utils_misc.add_loss('vqrgbnet/recon', total_loss, recon_loss, hyp.vqrgb_recon_coeff, summ_writer)

        utils_misc.add_loss('vqrgbnet/perplexity', 0.0, perplexity, 0.0, summ_writer)
        
        # utils_py.print_stats('rgb_e', rgb_e.detach().cpu().numpy())
        # utils_py.print_stats('rgb_g', rgb_g.detach().cpu().numpy())
        
        if summ_writer is not None:
            summ_writer.summ_rgb('vqrgbnet/rgb_e', rgb_e.clamp(-0.5, 0.5))
            summ_writer.summ_rgb('vqrgbnet/rgb_g', rgb_g)

        total_loss = utils_misc.add_loss('vqrgbnet/latent', total_loss, latent_loss, hyp.vqrgb_latent_coeff, summ_writer)

        # count the number of unique inds being used 
        unique_inds_here = np.unique(inds.detach().cpu().numpy())
        self.ind_pool.update(unique_inds_here)
        all_used_inds = self.ind_pool.fetch()
        unique_used_inds = np.unique(all_used_inds)
        utils_misc.add_loss('vqrgbnet/num_used_inds', 0.0, len(unique_used_inds), 0.0, summ_writer)
        
        ind_image = inds.reshape(B, int(H/8), int(W/8))
        
        return total_loss, rgb_e, ind_image

    def convert_inds_to_embeds(self, ind_map):
        # ind_map is B x H x W
        B, H, W = ind_map.shape

        encoding_indices = ind_map.reshape(B*H*W, 1)
        encodings = torch.zeros(encoding_indices.shape[0], hyp.vqrgb_num_embeddings, device=ind_map.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._vq_vae._embedding.weight).view(B, H, W, -1)
        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized

    
