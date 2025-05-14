# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from models.encoders.pointCNNencoder import PointCNNEncoder
from timm.layers import to_2tuple
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from modules.voxelization import Voxelization
import modules.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        voxel_size = (voxel_size, voxel_size, voxel_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#not needed for us
# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         # print('token drop drop_ids:', drop_ids, drop_ids.shape)
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         # print('token drop labels:', labels, labels.shape)
#         embeddings = self.embedding_table(labels)
#         return embeddings

#latent shape embedder
class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with the architecture shown in the image (bottom to top):
    1. Layer Norm -> Multi-Head Self-Attention
    2. Layer Norm -> Multi-Head Cross-Attention (with conditioning input)
    3. Layer Norm -> Pointwise Feedforward
    Each with a residual connection
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        
        # Layer Norm -> Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # Layer Norm -> Multi-Head Cross-Attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # For cross-attention, we need a custom implementation that can use conditioning as context
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Layer Norm -> Pointwise Feedforward
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.feedforward = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def cross_attention(self, q, c): # c is [B, 1, HiddenSize]
        batch_size = q.shape[0]
        num_patches_q = q.shape[1] # Sequence length of q

        # Project queries from input sequence
        # q input shape: [B, NumPatches, HiddenSize]
        q_proj = self.q_proj(q).reshape(batch_size, num_patches_q, self.num_heads, self.head_dim).transpose(1, 2)
        # q_proj shape: [B, NumHeads, NumPatches, HeadDim]

        # Project keys and values from conditioning c
        # c input shape: [B, 1, HiddenSize]
        k_proj = self.k_proj(c).reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(c).reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # k_proj, v_proj shape: [B, NumHeads, 1, HeadDim]

        # Compute scaled dot-product attention
        attn_weights = (q_proj @ k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B,H,N,D_h] @ [B,H,D_h,1] -> [B,H,N,1]
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        out = attn_weights @ v_proj # [B,H,N,1] @ [B,H,1,D_h] -> [B,H,N,D_h]
        out = out.transpose(1, 2).reshape(batch_size, num_patches_q, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out

    def forward(self, x, c):
        # Layer Norm -> Multi-Head Self-Attention
        input_x = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = x + input_x  # Residual connection
        
        # Layer Norm -> Multi-Head Cross-Attention
        input_x = x
        x = self.norm2(x)
        # Reshape conditioning to match batch size
        if c.dim() == 2:
            c = c.unsqueeze(1)  # [B, D] -> [B, 1, D]
        x = self.cross_attention(x, c)
        x = x + input_x  # Residual connection
        
        # Layer Norm -> Pointwise Feedforward
        input_x = x
        x = self.norm3(x)
        x = self.feedforward(x)
        x = x + input_x  # Residual connection
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT - simple Layer Norm followed by Linear.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, c):
        # Simple Layer Norm and Linear
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False,
        latent_dim=256  # Added parameter for latent dimension
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.input_size = input_size
        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)

        self.x_embedder = PatchEmbed_Voxel(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = PointCNNEncoder(zdim=latent_dim, input_dim=3)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.shape_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU() # or another activation
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.input_size//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize shape projection layer
        nn.init.xavier_uniform_(self.shape_proj[0].weight)
        if self.shape_proj[0].bias is not None:
            nn.init.constant_(self.shape_proj[0].bias, 0)

    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = z = self.input_size // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        points = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return points

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C, P) tensor of spatial inputs (point clouds or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """

        # Voxelization
        features, coords = x, x
        x_voxelized, voxel_coords = self.voxelization(features, coords)

        # Get shape embedding from the input pointcloud (before adding noise)
        # PointCNNEncoder expects [B, N, 3] format
        B, C, P = x.shape
        point_cloud = x.transpose(1, 2)  # [B, P, C]
        shape_embedding, _ = self.y_embedder(point_cloud)
        shape_embedding = self.shape_proj(shape_embedding) # Project to hidden_size
        t = self.t_embedder(t)
        c = t + shape_embedding

        # Embed voxelized noisy pointcloud
        x = self.x_embedder(x_voxelized) 
        x = x + self.pos_embed 

        for block in self.blocks:
            x = block(x, c)                      
        x = self.final_layer(x, c)                
        x = self.unpatchify_voxels(x)                   

        # Devoxelization
        x = F.trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)

        return x

    #we wont need classifier-free guidance
    def forward_with_cfg(self, x, t, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t)
        
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if not k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_XL_4(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)
    
    return model

def DiT_XL_8(pretrained=False, **kwargs):

    model = DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if not k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_L_2(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_B_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def DiT_B_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=32, num_heads=12, **kwargs)

def DiT_S_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(pretrained=False, **kwargs):

    model = DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
    if pretrained:
        checkpoint = torch.load('/path/to/DiT2D_pretrained_weights/DiT-S-4.pth', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

def DiT_S_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_S_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)

def DiT_S_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=32, num_heads=6, **kwargs)

DiT3D_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}