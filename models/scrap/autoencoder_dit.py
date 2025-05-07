import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import math

from .encoders import PointNetEncoder
from .diffusion import DiffusionPoint, VarianceSchedule
from .utils_dit import get_3d_sincos_pos_embed, unpatchify_voxels, modulate
from .voxelization import Voxelization, trilinear_devoxelize
from .window_utils import window_partition, window_unpartition
from .encoders import PointNetEncoder

# Define the DiT components we need
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
        """
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


class PatchEmbed_Voxel(nn.Module):
    """
    3D voxel to patch embedding.
    """
    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.grid_size = voxel_size // patch_size
        self.num_patches = self.grid_size ** 3
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=bias
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (B, C, H, W, D)
        Returns:
            Patch tokens: (B, N, C)
        """
        B, C, H, W, D = x.shape
        assert H == W == D == self.voxel_size, f"Input voxel size ({H}x{W}x{D}) doesn't match model voxel size ({self.voxel_size}x{self.voxel_size}x{self.voxel_size})"
        
        x = self.proj(x)  # (B, E, H//P, W//P, D//P)
        x = x.flatten(2)  # (B, E, (H//P)*(W//P)*(D//P))
        x = x.transpose(1, 2)  # (B, (H//P)*(W//P)*(D//P), E)
        
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, window_size=0, input_size=None, use_rel_pos=False, rel_pos_zero_init=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.window_size = window_size
        self.input_size = input_size
        
        # Attention module with support for relative position embeddings
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size)
        )
            
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size, 
            hidden_features=mlp_hidden_dim, 
            act_layer=nn.GELU, 
            drop=0
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Save shortcut for residual connection
        shortcut = x
        
        # Apply normalization
        x = self.norm1(x)
        
        # Reshape to 5D format for attention
        B = x.shape[0]
        x = x.reshape(B, self.input_size[0], self.input_size[1], self.input_size[2], -1)
        
        # Window partition if needed
        if self.window_size > 0:
            X, Y, Z = x.shape[1], x.shape[2], x.shape[3]
            x, pad_xyz = window_partition(x, self.window_size)
        
        # Reshape to 2D for modulation
        x = x.reshape(B, self.input_size[0] * self.input_size[1] * self.input_size[2], -1)
        x = modulate(x, shift_msa, scale_msa)
        x = x.reshape(B, self.input_size[0], self.input_size[1], self.input_size[2], -1)
        
        # Apply attention
        x = self.attn(x)
        
        # Reverse window partition if needed
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_xyz, (X, Y, Z))
        
        # Reshape back to 2D for residual connection
        x = x.reshape(B, self.input_size[0] * self.input_size[1] * self.input_size[2], -1)
        
        # Apply residual connection and gating
        x = shortcut + gate_msa.unsqueeze(1) * x
        
        # Apply MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        '''x=x+shortcut #residual connection
        shortcut=x
        x=self.norm2(x)
        
        # Reshape conditioning to match batch size
        if c.dim() == 2:
            c = c.unsqueeze(1)  # [B, D] -> [B, 1, D]
        x=self.cross_attention(x,c)
        
        x=x+shortcut
        # Layer Norm -> Pointwise Feedforward
        shortcut=x
        x = self.norm3(x)
        x = self.feedforward(x)
        x = x + shortcut 
        '''
        return x


class Attention(nn.Module):
    """
    Multi-head Attention block with optional relative position embeddings.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False, rel_pos_zero_init=True, input_size=None):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple or None): Input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and input_size is not None:
            # Initialize relative positional embeddings
            self.rel_pos_x = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_y = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_z = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_x, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_y, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_z, std=0.02)

    def forward(self, x):
        B, X, Y, Z, C = x.shape
        
        # Reshape to get qkv
        qkv = self.qkv(x).reshape(B, X*Y*Z, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, X * Y * Z, -1).unbind(0)
        
        # Compute attention
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative positional embeddings if enabled
        if self.use_rel_pos:
            # This would require implementing add_decomposed_rel_pos function
            # For now, we'll skip this as it's not essential for the core functionality
            pass

        # Apply softmax and compute weighted sum
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, X, Y, Z, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, X, Y, Z, -1)
        
        # Project back to output dimension
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., eta=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(hidden_features), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(out_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        x = self.gamma1 * self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gamma2 * self.fc2(x)
        x = self.drop2(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTNoisePredictor(Module):
    """
    DiT-based noise predictor for diffusion models.
    """
    def __init__(
        self,
        point_dim=3,
        hidden_size=512,
        depth=8,
        num_heads=8,
        patch_size=4,
        input_size=32,
        mlp_ratio=4.0,
        window_size=0,
        window_block_indexes=(),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        latent_dim=256
    ):
        super().__init__()
        self.point_dim = point_dim
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Voxelization layer
        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)
        
        # Patch embedding
        self.x_embedder = PatchEmbed_Voxel(input_size, patch_size, point_dim, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Context embedding (replaces class embedding)
        self.y_embedder = PointNetEncoder(zdim=latent_dim, input_dim=3)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(input_size // patch_size, input_size // patch_size, input_size // patch_size),
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init
            ) for i in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, point_dim)
        self.shape_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU() # or another activation
        )
        # Initialize weights
        self._initialize_weights()
    
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

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # Initialize shape projection layer
        nn.init.xavier_uniform_(self.shape_proj[0].weight)
        if self.shape_proj[0].bias is not None:
            nn.init.constant_(self.shape_proj[0].bias, 0)

    def forward(self, x, beta):
        """
        Forward pass.
        Args:
            x: Point clouds at timestep t, (B, N, d)
            beta: Time. (B, )
        Returns:
            Predicted noise: (B, N, d)
        """
        batch_size = x.size(0)
        
        # Voxelization
        features, coords = x, x
        x_vox, voxel_coords = self.voxelization(features, coords)
        # Note: x_vox is (B, C, R, R, R) and voxel_coords is (B, 3, N)
        
        # Patch embedding
        x_emb = self.x_embedder(x_vox)
        x_emb = x_emb + self.pos_embed
        
        # Time embedding
        t_emb = self.t_embedder(beta)
        
        point_cloud = x.transpose(1, 2)  # [B, P, C]
        shape_embedding, _ = self.y_embedder(point_cloud)
        shape_embedding = self.shape_proj(shape_embedding) #
        
        # Combined conditioning
        combined_cond = t_emb + shape_embedding
        
        # Apply transformer blocks
        for block in self.blocks:
            x_emb = block(x_emb, combined_cond)
        
        # Final layer
        x_emb = self.final_layer(x_emb, combined_cond)
        
        # Unpatchify
        x_unpatch = unpatchify_voxels(x_emb, self.patch_size, self.point_dim, self.input_size)
        
        # Devoxelize - using CUDA-accelerated implementation
        # voxel_coords is already in (B, 3, N) format as required
        x_out = trilinear_devoxelize(x_unpatch, voxel_coords, self.input_size, self.training)
        
        # Transpose output from (B, C, N) to (B, N, C) to match expected output format
        x_out = x_out.transpose(1, 2).contiguous()
        
        return x_out


class DiTAutoEncoder(Module):
    """
    Autoencoder with DiT-based diffusion model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Keep the original encoder
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=3)
        
        # Parse window block indexes if provided
        window_block_indexes = []
        if hasattr(args, 'dit_window_block_indexes') and args.dit_window_block_indexes:
            window_block_indexes = [int(i) for i in args.dit_window_block_indexes.split(',')]
        
        # Replace the diffusion model's noise predictor
        self.diffusion = DiffusionPoint(
            net=DiTNoisePredictor(
                point_dim=3,
                hidden_size=args.dit_hidden_size,
                depth=args.dit_depth,
                num_heads=args.dit_num_heads,
                patch_size=args.dit_patch_size,
                input_size=args.dit_input_size,
                mlp_ratio=args.dit_mlp_ratio,
                window_size=args.dit_window_size if hasattr(args, 'dit_window_size') else 0,
                window_block_indexes=window_block_indexes,
                use_rel_pos=args.dit_use_rel_pos if hasattr(args, 'dit_use_rel_pos') else False,
                rel_pos_zero_init=True,
                latent_dim=args.latent_dim
            ),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
    
    def encode(self, x):
        """
        Encode point clouds.
        Args:
            x: Point clouds to be encoded, (B, N, d).
        Returns:
            code: (B, C)
        """
        code, _ = self.encoder(x)
        return code
    
    def decode(self, num_points, batch_size=1, flexibility=0.0, ret_traj=False):
        """
        Generate point clouds.
        """
        return self.diffusion.sample(num_points,code, point_dim=3, flexibility=flexibility, ret_traj=ret_traj)
    
    def get_loss(self, x, x_raw=None):
        """
        Get loss for training.
        """
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code, x_raw=x_raw)
        return loss



