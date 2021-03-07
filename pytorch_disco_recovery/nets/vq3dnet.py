import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
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
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
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
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
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
        # convert inputs from BCHWD -> BHWDC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
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
        # convert quantized from BHWDC -> BCHWD
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings, encoding_indices

class Vq3dNet(nn.Module):
    def __init__(self):
        super(Vq3dNet, self).__init__()

        print('Vq3dNet...')

        # we'll keep track of the last 10k used inds,
        # as an interpretable measure of "perplexity"
        self.ind_pool = IndPool(10000)
        
        num_embeddings = hyp.vq3d_num_embeddings
        commitment_cost = 0.25

        # self._vq_vae = VectorQuantizer(
        self._vq_vae = VectorQuantizerEMA(
            num_embeddings,
            128, # this is the dim of the featurespace we are quantizing
            commitment_cost).cuda()

    def forward(self, feat, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat.shape)
        
        latent_loss, quantized_vox, perplexity, _, inds = self._vq_vae(feat)
        total_loss = utils_misc.add_loss('vq3dnet/latent', total_loss, latent_loss, hyp.vq3d_latent_coeff, summ_writer)

        # count the number of unique inds being used 
        unique_inds_here = np.unique(inds.detach().cpu().numpy())
        self.ind_pool.update(unique_inds_here)
        all_used_inds = self.ind_pool.fetch()
        unique_used_inds = np.unique(all_used_inds)
        
        ind_vox = inds.reshape(B, Z, Y, X)
        
        if summ_writer is not None:
            summ_writer.summ_scalar('unscaled_vq3dnet/perplexity', perplexity.cpu().item())
            summ_writer.summ_scalar('unscaled_vq3dnet/num_used_inds', float(len(unique_used_inds)))
            summ_writer.summ_feat('vq3dnet/quantized', quantized_vox, pca=True)
            
        return total_loss, quantized_vox, ind_vox

    def convert_inds_to_embeds(self, ind_vox):
        # ind_vox is B x Z, Y, X
        B, Z, Y, X = ind_vox.shape
        encoding_indices = ind_vox.reshape(B*Z*Y*X, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self._vq_vae._num_embeddings, device=ind_vox.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._vq_vae._embedding.weight).view(B, Z, Y, X, -1)
        # convert quantized from BZYXC -> BCZYX
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        return quantized
        

    
