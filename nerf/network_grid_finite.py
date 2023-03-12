import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x



class ResLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.nn = nn.Linear(dim_in, dim_out, bias=bias)
    def forward(self, x):
        if self.dim_in == self.dim_out:
            return self.nn(x) + x
        return self.nn(x)

class MLP_swish(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l > 0:
                net.append(nn.SiLU(inplace=True))
                # net.append(nn.LayerNorm(self.dim_hidden))
            net.append(ResLinear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
            # net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.Sequential(*net)
        # self.net = nn.ModuleList(net)
    
    def forward(self, x):
        x = self.net(x)
        # x = self.net[0](x)
        # for l in range(1, self.num_layers, 2):
        #     if l != self.num_layers - 2:
        #         x = F.silu(x, inplace=True)
        #     x = self.net[l](x)
        #     if l != self.num_layers - 2:
        #         x = self.net[l + 1](x) + x # resblock
        #     else:
        #         x = self.net[l + 1](x)
        # print(x.shape)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=96,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        num_levels = 16
        level_dim = 2
        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep', num_levels=16, level_dim=2)
        self.max_level = num_levels * level_dim
        self.cur_level = num_levels * level_dim
        # self.cur_level = num_levels // 2 * level_dim

        self.sigma_net = MLP_swish(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else F.softplus

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    # add a density blob to the scene center
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        # g = 5.0 * torch.exp(- d / (2 * 0.2 ** 2))
        g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))

        return g


    def normal(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal[torch.isnan(normal)] = 0

        return normal


    def common_forward(self, x):

        # sigma
        enc = self.encoder(x, bound=self.bound)
        # print('enc', enc.shape) # [..., 32]
        enc[:, self.cur_level:] = 0

        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def forward(self, x, d, l, l_p, l_a, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None
        
        else:
            # query normal

            sigma, albedo = self.common_forward(x)
            normal = self.normal(x)

            ww = safe_normalize(l - x)
            lambertian = ((normal * ww).sum(-1, keepdim=True)).clamp(min=0)

            if shading == 'textureless':
                # color = lambertian
                # print(lambertian.shape, albedo.shape)
                color = lambertian.repeat(1, 3)
                # color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            elif shading == 'lambertian': # 'lambertian'
                color = albedo * lambertian
            else:
                # mixed shading from dreamfusion
                ambient = albedo
                diffuse = albedo * lambertian

                ratio = 0 # ours
                color = ambient * ratio * 0.5 + diffuse * (1 - ratio * 0.5)

        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params