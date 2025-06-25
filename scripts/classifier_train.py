"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
from torch.autograd import Variable
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.acneloader import AcneDataset
import blobfile as bf
import torch as th
os.environ['OMP_NUM_THREADS'] = '8'

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from visdom import Visdom
import numpy as np
viz = Visdom(port=8850)
loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='acc', title='accuracy'))

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict



def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=1000
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )


    logger.log("creating data loader...")

    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)
        data = iter(datal)
        print(f"Training on {len(ds)} data points")

    elif args.dataset == 'acne':  # Add this entire elif block
        ds = AcneDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)
        data = iter(datal)
        print(f'Acne dataset loaded with {len(ds)} images')

    elif args.dataset == 'chexpert':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
        print('dataset is chexpert')



    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    print_detailed_config_and_model(args, model, diffusion, opt)
    logger.log("training classifier model...")


    def forward_backward_log(data_loader, step, prefix="train"):
        if args.dataset=='brats':
            batch, extra, labels,_ , _ = next(data_loader)
            print('IS BRATS')

        elif args.dataset=='acne':  # Add this elif block
            batch, extra, labels, _, _ = next(data_loader)
            print('IS ACNE')

        elif  args.dataset=='chexpert':
            batch, extra = next(data_loader)
            labels = extra["y"].to(dist_util.dev())
            print('IS CHEXPERT')

        print('labels', labels)
        batch = batch.to(dist_util.dev())
        labels= labels.to(dist_util.dev())
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
          
            sub_batch = Variable(sub_batch, requires_grad=True)
            logits = model(sub_batch, timesteps=sub_t)
         
            loss = F.cross_entropy(logits, sub_labels, reduction="none")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@2"] = compute_top_k(
                logits, sub_labels, k=2, reduction="none"
            )
            print('acc', losses[f"{prefix}_acc@1"])
            log_loss_dict(diffusion, sub_t, losses)

            loss = loss.mean()
            accuracy = losses[f"{prefix}_acc@1"].mean()
            if prefix=="train":
                viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([loss]).unsqueeze(0).cpu(),
                     win=loss_window, name='loss_cls',
                     update='append')
                viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([accuracy]).unsqueeze(0).cpu(),
                     win=acc_window, name='train_acc', update='append')

            else:

                output_idx = logits[0].argmax()
                print('outputidx', output_idx)
                output_max = logits[0, output_idx]
                print('outmax', output_max, output_max.shape)
                output_max.backward()
                saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
                print('saliency', saliency.shape)
                viz.heatmap(visualize(saliency[0, ...]))
                viz.image(visualize(sub_batch[0, 0,...]))
                viz.image(visualize(sub_batch[0, 1, ...]))
                th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return losses

    correct=0; total=0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        print('step', step + resume_step)
        try:
            losses = forward_backward_log(data, step + resume_step)
        except:
            data = iter(datal)
            losses = forward_backward_log(data, step + resume_step)

        correct+=losses["train_acc@1"].sum()
        total+=args.batch_size
        acctrain=correct/total

        mp_trainer.optimize(opt)
          
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"modelbratsclass{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"optbratsclass{step:06d}.pt"))

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=8,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=1,
        eval_interval=1000,
        save_interval=5000,
        dataset='acne'
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def print_detailed_config_and_model(args, model, diffusion, opt):
    """Print detailed configuration and model information"""
    
    # Get model parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    
    # Extract classifier configuration from args
    print("Classifier Configuration:")
    print(f"   spatial_dims: 2")
    print(f"   in_channels: {3 if args.dataset == 'acne' else 4 if args.dataset == 'brats' else 1}")
    print(f"   out_channels: {4 if args.dataset == 'acne' else 2}")
    print(f"   model_channels: {args.classifier_width}")
    print(f"   classifier_depth: {args.classifier_depth}")
    print(f"   attention_resolutions: {args.classifier_attention_resolutions}")
    print(f"   use_scale_shift_norm: {args.classifier_use_scale_shift_norm}")
    print(f"   resblock_updown: {args.classifier_resblock_updown}")
    print(f"   pool: {args.classifier_pool}")
    print(f"   with_conditioning: True")
    print(f"   dataset: {args.dataset}")

    print(f"\nClassifier Training Configuration:")
    print(f"   iterations: {args.iterations}")
    print(f"   learning_rate: {args.lr}")
    print(f"   weight_decay: {args.weight_decay}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   microbatch: {args.microbatch}")
    print(f"   noised: {args.noised}")
    print(f"   schedule_sampler: {args.schedule_sampler}")
    print(f"   log_interval: {args.log_interval}")
    print(f"   save_interval: {args.save_interval}")
    print(f"   anneal_lr: {args.anneal_lr}")
    print(f"   use_fp16: {args.classifier_use_fp16}")

    print(f"\nCreating classifier model...")
    
    print(f"\nClassifier Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model_size_mb:.1f} MB")
    print(f"   Image size: {args.image_size}")
    print(f"   Classes: {4 if args.dataset == 'acne' else 2} ({'acne severity levels' if args.dataset == 'acne' else 'binary classification'})")
    
    # Print architecture breakdown
    print(f"   Architecture breakdown:")
    for name, module in model.named_modules():
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            if hasattr(module, 'kernel_size'):
                print(f"      {name}: {module.in_channels} -> {module.out_channels} (kernel: {module.kernel_size})")

    print(f"\nOptimizer Configuration:")
    print(f"   Type: {opt.__class__.__name__}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Parameters: {total_params:,}")
    
    # Print diffusion configuration
    print(f"\nDiffusion Configuration:")
    print(f"   num_timesteps: {diffusion.num_timesteps}")
    print(f"   noise_schedule: linear")
    print(f"   model_mean_type: {diffusion.model_mean_type}")
    print(f"   model_var_type: {diffusion.model_var_type}")
    print(f"   loss_type: {diffusion.loss_type}")

    print("="*80)

if __name__ == "__main__":
    main()
