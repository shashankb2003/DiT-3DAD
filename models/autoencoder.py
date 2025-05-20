import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *
from .dit_ada_window import *

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=3)
        self.diffusion = DiffusionPoint(
            net=DiT(
                hidden_size=args.dit_hidden_size,
                depth=args.dit_depth,
                num_heads=args.dit_num_heads,
                patch_size=args.dit_patch_size,
                input_size=args.dit_input_size,
                mlp_ratio=args.dit_mlp_ratio,
                window_size=args.dit_window_size if hasattr(args, 'dit_window_size') else 0,
                window_block_indexes=args.dit_window_block_indexes,
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
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        Returns:
            code: (B, C)
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, point_dim=3, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, x_raw=None):
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code, x_raw=x_raw)
        return loss

class AutoEncoderTNet(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoderTNet(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        Returns:
            code: (B, C)
        """
        code, _, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        code, trans, trans_feat = self.encoder(x)
        loss = self.diffusion.get_loss(x, code)
        loss_mat = feature_transform_reguliarzer(trans_feat)
        return loss + loss_mat * 0.001
    
class DenoisingAutoEncoder(AutoEncoder):
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        Returns:
            code: (B, C)
        """
        x += torch.normal(mean=0, std=0.05, size=x.size()).to(x.device)
        code, _ = self.encoder(x)
        return code