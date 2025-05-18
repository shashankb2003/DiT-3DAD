import os
import sys
import argparse
import torch
import torch.utils.tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import ROC_AP


# Arguments
parser = argparse.ArgumentParser()
# Distributed training arguments
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
# Model arguments
parser.add_argument('--model', type=str, default='AutoEncoder')
parser.add_argument('--model_type', type=str, default='default', choices=['default', 'dit'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# DiT model arguments
parser.add_argument('--dit_hidden_size', type=int, default=1152)
parser.add_argument('--dit_depth', type=int, default=8)
parser.add_argument('--dit_num_heads', type=int, default=8)
parser.add_argument('--dit_patch_size', type=int, default=4)
parser.add_argument('--dit_input_size', type=int, default=32)
parser.add_argument('--dit_mlp_ratio', type=float, default=4.0)
parser.add_argument('--dit_window_size', type=int, default=0, help='Window size for window attention (0 means global attention)')
parser.add_argument('--dit_window_block_indexes', type=tuple, default='0,3,6,9', help='Comma-separated list of block indexes to use window attention (e.g., "0,4,8")')
parser.add_argument('--dit_use_rel_pos', type=eval, default=False, choices=[True, False], help='Whether to use relative position embeddings in attention') 

# Datasets and loaders
parser.add_argument('--dataset', type=str, default='ShapeNetAD')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet-ad')
parser.add_argument('--category', type=str, default='ashtray0')
parser.add_argument('--scale_mode', type=str, default=None)
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_aug', type=int, default=2048)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=128)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
parser.add_argument('--rel', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix=args.category + '_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

def main():
    # Initialize distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = f'cuda:{args.local_rank}'
        dist.init_process_group(backend='nccl')
    
    # Datasets and loaders
    train_transforms = []
    val_transforms = []
    if args.rotate:
        train_transforms.append(RandomRotate(180, ['pointcloud']))
    logger.info('Train Transforms: %s' % repr(train_transforms))
    logger.info('Val Transforms: %s' % repr(val_transforms))
    logger.info('Loading datasets...')
    
    train_dset = getattr(sys.modules[__name__], args.dataset)(
        path=args.dataset_path,
        cates=[args.category],
        split='train',
        scale_mode=args.scale_mode,
        num_points=args.num_points,
        num_aug = args.num_aug,
        transforms=train_transforms,
    )
    val_dset = getattr(sys.modules[__name__], args.dataset)(
        path=args.dataset_path,
        cates=[args.category],
        split='test',
        scale_mode=args.scale_mode,
        num_points=args.num_points,
        transforms=val_transforms,
    )
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset)
    
    train_loader = DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dset, 
        batch_size=args.val_batch_size, 
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    train_iter = get_data_iterator(train_loader)

    # Model
    logger.info('Building model...')
    if args.resume is not None:
        logger.info('Resuming from checkpoint...')
        ckpt = torch.load(args.resume)
        model = getattr(sys.modules[__name__], args.model)(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        logger.info('Using default model')
        model = getattr(sys.modules[__name__], args.model)(args).to(args.device)
    
    # Wrap model with DDP
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])
    logger.info(repr(model))

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    memory_bank = []
    
    # Train, validate 
    def train(it):
        train_sampler.set_epoch(it)
        # Load data
        batch = next(train_iter)
        x = batch['pointcloud'].to(args.device)
        x = x.transpose(1, 2)
        if it == 1 and args.local_rank in [-1, 0]:  # Only log on main process
            memory_bank.append(x)
            writer.add_mesh('train/pc', x, global_step=it)
        # Reset grad and model state
        optimizer.zero_grad()
        model.train()

        # Forward
        if args.rel:
            x_raw = batch['pointcloud_raw'].to(args.device)
            x_raw = x_raw.transpose(1, 2)
            loss = model.module.get_loss(x, x_raw) if args.local_rank != -1 else model.get_loss(x, x_raw)
        else:
            loss = model.module.get_loss(x) if args.local_rank != -1 else model.get_loss(x)

        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if args.local_rank in [-1, 0]:  # Only log on main process
            logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
            writer.add_scalar('train/loss', loss, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()

    def validate_loss(it):
        val_sampler.set_epoch(it)
        all_ref = []
        all_recons = []
        all_label = []
        all_mask = []
        for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
            if args.num_val_batches > 0 and i >= args.num_val_batches:
                break
            ref = batch['pointcloud'].to(args.device)
            shift = batch['shift'].to(args.device)
            scale = batch['scale'].to(args.device)
            with torch.no_grad():
                model.eval()
                if args.local_rank != -1:
                    code = model.module.encode(ref)
                    recons = model.module.decode(code, ref.size(1), flexibility=args.flexibility)
                else:
                    code = model.encode(ref)
                    recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
                if args.rel:
                    recons += ref
            
            all_ref.append(ref * scale + shift)
            all_recons.append(recons * scale + shift)
            all_label.append(batch['label'].to(args.device))
            all_mask.append(batch['mask'].to(args.device))

        all_ref = torch.cat(all_ref, dim=0)
        all_recons = torch.cat(all_recons, dim=0)
        all_label = torch.cat(all_label, dim=0)
        all_mask = torch.cat(all_mask, dim=0)

        if args.local_rank != -1:
            # Gather results from all processes
            all_ref = [torch.zeros_like(all_ref) for _ in range(dist.get_world_size())]
            all_recons = [torch.zeros_like(all_recons) for _ in range(dist.get_world_size())]
            all_label = [torch.zeros_like(all_label) for _ in range(dist.get_world_size())]
            all_mask = [torch.zeros_like(all_mask) for _ in range(dist.get_world_size())]
            
            dist.all_gather(all_ref, all_ref)
            dist.all_gather(all_recons, all_recons)
            dist.all_gather(all_label, all_label)
            dist.all_gather(all_mask, all_mask)
            
            all_ref = torch.cat(all_ref, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
            all_label = torch.cat(all_label, dim=0)
            all_mask = torch.cat(all_mask, dim=0)

        metrics = ROC_AP(all_ref, all_recons, all_label, all_mask)
        roc_i, roc_p, ap_i, ap_p = metrics['ROC_i'].item(), metrics['ROC_p'].item(), metrics['AP_i'].item(), metrics['AP_p'].item()
        
        if args.local_rank in [-1, 0]:  # Only log on main process
            logger.info('[Val] Iter %04d | ROC_i_cdist %.6f | ROC_p_cdist %.6f | AP_i_cdist %.6f | AP_p_cdist %.6f' % (it, roc_i, roc_p, ap_i, ap_p))
            roc_i_nn, roc_p_nn, ap_i_nn, ap_p_nn = metrics['ROC_i_nn'].item(), metrics['ROC_p_nn'].item(), metrics['AP_i_nn'].item(), metrics['AP_p_nn'].item()
            logger.info('[Val] Iter %04d | ROC_i_nn %.6f | ROC_p_nn %.6f | AP_i_nn %.6f | AP_p_nn %.6f' % (it, roc_i_nn, roc_p_nn, ap_i_nn, ap_p_nn))
            writer.add_scalar('val/ROC_i', roc_i_nn, it)
            writer.add_scalar('val/ROC_p', roc_p_nn, it)
            writer.add_scalar('val/AP_i', ap_i_nn, it)
            writer.add_scalar('val/AP_p', ap_p_nn, it)
            writer.flush()

            np.save(os.path.join(log_dir, 'ref.npy'), all_ref.cpu().numpy())
            np.save(os.path.join(log_dir, 'out.npy'), all_recons.cpu().numpy())
            np.save(os.path.join(log_dir, 'mask.npy'), all_mask.cpu().numpy())

        return roc_i

    def validate_inspect(it):
        for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
            x = batch['pointcloud'].to(args.device)
            model.eval()
            if args.local_rank != -1:
                code = model.module.encode(x)
                recons = model.module.decode(code, x.size(1), flexibility=args.flexibility).detach()
            else:
                code = model.encode(x)
                recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()
            if args.rel:
                recons += x

            if i >= args.num_inspect_batches:
                break

        if args.local_rank in [-1, 0]:  # Only log on main process
            writer.add_mesh('val/pc_in', x[:args.num_inspect_pointclouds], global_step=it)
            writer.add_mesh('val/pc_out', recons[:args.num_inspect_pointclouds], global_step=it)
            writer.flush()

    # Main loop
    logger.info('Start training...')
    try:
        it = 1
        while it <= args.max_iters:
            train(it)
            if it % args.val_freq == 0:
                with torch.no_grad():
                    score = validate_loss(it)
                    validate_inspect(it)
            # save checkpoint only at the final iteration
            if it == args.max_iters and args.local_rank in [-1, 0]:  # Only save on main process
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                model_to_save = model.module if args.local_rank != -1 else model
                ckpt_mgr.save(model_to_save, args, score, opt_states, it)
            it += 1

    except KeyboardInterrupt:
        logger.info('Terminating...')
    
    if args.local_rank != -1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()