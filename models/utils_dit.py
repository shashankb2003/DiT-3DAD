import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Create 1D sin/cos positional embedding from a grid of positions.
    Args:
        embed_dim: Output dimension for each position
        pos: 1D positions to be embedded
    Returns:
        Positional embedding of shape (len(pos), embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    3D sine-cosine positional embedding.
    Args:
        embed_dim: Output dimension
        grid_size: Size of the 3D grid
        cls_token: Whether to include a class token
    Returns:
        Positional embedding of shape (grid_size^3, embed_dim) or (1+grid_size^3, embed_dim)
    """
    grid_h = grid_w = grid_d = grid_size
    
    # Use 1/3 of dimensions for each spatial position
    d_per_dimension = embed_dim // 3
    
    # Create grid of positions
    grid_h_pos = np.arange(grid_h, dtype=np.float32)
    grid_w_pos = np.arange(grid_w, dtype=np.float32)
    grid_d_pos = np.arange(grid_d, dtype=np.float32)
    
    # Create meshgrid for all positions
    pos_h, pos_w, pos_d = np.meshgrid(
        grid_h_pos, grid_w_pos, grid_d_pos, indexing='ij'
    )
    
    # Flatten positions
    pos_h = pos_h.reshape(-1)
    pos_w = pos_w.reshape(-1)
    pos_d = pos_d.reshape(-1)
    
    # Create embeddings for each dimension
    embed_h = get_1d_sincos_pos_embed_from_grid(d_per_dimension, pos_h)
    embed_w = get_1d_sincos_pos_embed_from_grid(d_per_dimension, pos_w)
    embed_d = get_1d_sincos_pos_embed_from_grid(d_per_dimension, pos_d)
    
    # Concatenate all embeddings
    pos_embed = np.concatenate([embed_h, embed_w, embed_d], axis=1)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def unpatchify_voxels(x, patch_size, out_chans, input_size=None):
    """
    Convert patched feature maps to 3D voxel grid.
    Args:
        x: Input tensor of shape (B, N, C) where N is the number of patches
        patch_size: Size of each patch
        out_chans: Number of output channels
        input_size: Original input size (before patching). If None, will be inferred from x.
    Returns:
        Voxel grid of shape (B, C, H, W, D)
    """
    p = patch_size
    c = out_chans
    
    if input_size is not None:
        # DiT-3D approach: Calculate grid dimensions from input size and patch size
        h = w = d = input_size // patch_size
        # Verify the calculation is correct
        assert h * w * d == x.shape[1], f"Grid dimensions {h}x{w}x{d} don't match number of patches {x.shape[1]}"
    else:
        # Fallback: Infer grid dimensions from number of patches
        h = w = d = int(x.shape[1]**(1/3))
        # Check if the inference is accurate
        if h**3 != x.shape[1]:
            print(f"Warning: Cube root of {x.shape[1]} is not an integer. Grid might not be cubic.")
    
    x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, c))
    x = torch.einsum('nhwdpqrc->nchpwdqr', x)
    voxels = x.reshape(shape=(x.shape[0], c, h * p, w * p, d * p))
    return voxels


def modulate(x, shift, scale):
    """
    Modulate the input with shift and scale.
    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter
    Returns:
        Modulated tensor
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
