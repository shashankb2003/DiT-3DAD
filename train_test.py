import argparse
import os
import time
from pathlib import Path

from utils.config import cmd_from_config
from utils.dataset import all_shapenetad_cates


def main(args):

    exp_name = Path(args.config).stem
    time_fix = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    cfg_cmd = cmd_from_config(args.config)

    if 'ShapeNetAD' in cfg_cmd:
        cates = all_shapenetad_cates
        dataset = 'shapenet-ad'
    else:
        raise NotImplementedError
    
    for cate in cates:
        cmd = f"python train_ae.py --category {cate} --log_root logs_{dataset}/{exp_name}_{time_fix}_{args.tag}/ --model_type {args.model_type}" + cfg_cmd
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--model_type', type=str, default='default', choices=['default', 'dit'], 
                      help='Model type: default (original R3D-AD) or dit (with DiT transformer)')
    args = parser.parse_args()
    main(args)