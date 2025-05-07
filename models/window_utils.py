import torch
import torch.nn.functional as F
import numpy as np


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    
    Args:
        x (tensor): input tokens with [B, X, Y, Z, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Xp, Yp, Zp): padded height and width before partition
    """
    B, X, Y, Z, C = x.shape

    pad_x = (window_size - X % window_size) % window_size
    pad_y = (window_size - Y % window_size) % window_size
    pad_z = (window_size - Z % window_size) % window_size
    
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        x = F.pad(x, (0, 0, 0, pad_z, 0, pad_y, 0, pad_x))

    Xp, Yp, Zp = X + pad_x, Y + pad_y, Z + pad_z

    x = x.view(B, Xp // window_size, window_size, Yp // window_size, window_size, Zp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    
    return windows, (Xp, Yp, Zp)


def window_unpartition(windows, window_size, pad_xyz, xyz):
    """
    Window unpartition into original sequences and removing padding.
    
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_xyz (Tuple): padded dimensions (Xp, Yp, Zp).
        xyz (Tuple): original dimensions (X, Y, Z) before padding.

    Returns:
        x: unpartitioned sequences with [B, X, Y, Z, C].
    """
    Xp, Yp, Zp = pad_xyz
    X, Y, Z = xyz
    B = windows.shape[0] // (Xp * Yp * Zp // window_size // window_size // window_size)
    
    x = windows.view(B, Xp // window_size, Yp // window_size, Zp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Xp, Yp, Zp, -1)

    if Xp > X or Yp > Y or Zp > Z:
        x = x[:, :X, :Y, :Z, :].contiguous()
        
    return x
