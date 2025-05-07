import torch
import torch.nn as nn

# Import the CUDA-accelerated modules
from modules.functional import avg_voxelize


class Voxelization(nn.Module):
    """
    Voxelizes a point cloud to a 3D grid using CUDA-accelerated implementation.
    
    Args:
        resolution: Resolution of the voxel grid
        normalize: Whether to normalize the point cloud
        eps: Small epsilon for numerical stability
    """
    def __init__(self, resolution=32, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        """
        Forward pass.
        
        Args:
            features: Point features (B, N, C)
            coords: Point coordinates (B, N, 3)
            
        Returns:
            voxel_features: Voxelized features (B, C, R, R, R)
            voxel_coords: Voxel coordinates for each point (B, 3, N)
        """
        # Transpose features from (B, N, C) to (B, C, N) for CUDA implementation
        features = features.transpose(1, 2).contiguous()
        
        # Transpose coords from (B, N, 3) to (B, 3, N) for CUDA implementation
        coords = coords.transpose(1, 2).contiguous()
        
        # Detach coordinates to avoid backpropagation through coordinates
        coords = coords.detach()
        
        # Normalize coordinates
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        
        # Scale to voxel resolution
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        
        # Convert to integer coordinates
        vox_coords = torch.round(norm_coords).to(torch.int32)
        
        # Use CUDA-accelerated voxelization
        voxel_features = avg_voxelize(features, vox_coords, self.r)
        
        # Note: We return norm_coords in (B, 3, N) format for CUDA devoxelization
        return voxel_features, norm_coords


def trilinear_devoxelize(voxel_features, voxel_coords, resolution, training=True):
    """
    Devoxelizes a voxel grid back to a point cloud using trilinear interpolation.
    Uses CUDA-accelerated implementation for faster processing.
    
    Args:
        voxel_features: Voxelized features (B, C, R, R, R)
        voxel_coords: Voxel coordinates for each point (B, 3, N)
        resolution: Resolution of the voxel grid
        training: Whether in training mode
        
    Returns:
        point_features: Devoxelized point features (B, C, N)
    """
    # Import the CUDA-accelerated devoxelization function
    from modules.functional import trilinear_devoxelize as cuda_trilinear_devoxelize
    
    # Ensure voxel_features is contiguous for CUDA operations
    voxel_features = voxel_features.contiguous()
    
    # Ensure voxel_coords is contiguous for CUDA operations
    voxel_coords = voxel_coords.contiguous()
    
    # Use CUDA-accelerated devoxelization
    point_features = cuda_trilinear_devoxelize(voxel_features, voxel_coords, resolution, training)
    
    return point_features
