from tqdm import tqdm,trange

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv3d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel, kernel)
        padding_shp = (kernel // 2, kernel // 2, kernel // 2)
        self.z_stack = nn.Conv3d(dim, dim*2, kernel_shp, 1, padding_shp)

        self.z_to_x = nn.Conv3d(dim*2, dim*2, 1)

        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.y_stack = nn.Conv2d(dim, dim*2, kernel_shp, 1, padding_shp)

        self.y_to_x = nn.Conv3d(dim*2, dim*2, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.x_stack = nn.Conv2d(dim, dim*2, kernel_shp, 1, padding_shp)

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

    def forward(self, t_x, t_y, t_z, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h = h[:,:,None,None,None]

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


class GatedVoxelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv3d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv3d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv3d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, t, label):
        shp = t.size() + (-1, )
        t = self.embedding(t.view(-1)).view(shp)  # (B, Z, Y, X, C)
        t = t.permute(0, 4, 1, 2, 3)  # (B, C, Z, Y, X)

        t_x,t_y,t_z = (t,t,t)
        for i, layer in enumerate(self.layers):
            t_x,t_y,t_z = layer(t_x,t_y,t_z,label)

        return self.output_conv(t_x)

    def generate(self, label, shape, batch_size=64):
        param = next(self.parameters())
        Z,Y,X = shape
        t = torch.zeros((batch_size, *shape),dtype=torch.int64, device=param.device)
        
        with tqdm(total=Z*Y*X) as pbar:
            for i in range(Z):
                for j in range(Y):
                    for k in range(X):
                        logits = self.forward(t,label)
                        probs = F.softmax(logits[:,:,i,j,k],-1)
                        t.data[:,i,j,k].copy_(probs.multinomial(1).squeeze().data)
                        pbar.update(1)
        return t
