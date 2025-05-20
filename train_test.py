import argparse
import os
import time
from pathlib import Path

from utils.config import cmd_from_config
from utils.dataset import all_shapenetad_cates


def main(args):
    # Get local_rank from environment variable
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    # Only run on rank 0 to avoid multiple processes running the same training
    if local_rank in [-1, 0]:  # -1 for non-distributed, 0 for distributed
        exp_name = Path(args.config).stem
        time_fix = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        cfg_cmd = cmd_from_config(args.config)

        if 'ShapeNetAD' in cfg_cmd:
            cates = all_shapenetad_cates
            dataset = 'shapenet-ad'
        else:
            raise NotImplementedError
        
        for cate in cates:
            # Use torchrun for distributed training
            cmd = f"torchrun --nproc_per_node={args.nproc_per_node} train_ae.py --category {cate} --log_root logs_{dataset}/{exp_name}_{time_fix}_{args.tag}/ --model_type {args.model_type}" + cfg_cmd
            os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--model_type', type=str, default='default', choices=['default', 'dit'], 
                      help='Model type: default (original R3D-AD) or dit (with DiT transformer)')
    # Add distributed training arguments
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of GPUs to use for training')
    args = parser.parse_args()
    main(args)
