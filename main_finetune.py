# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import *
import util.misc as misc

import models_vit
import models_vit_mask

from engine_finetune import train_one_epoch, evaluate
from torch.utils.data import Subset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--dump_freq', default=1000, type=int,
                        help='every nth epoch to save checkpoints')
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=None, type=list,  # changed
                        help='images input size')
    parser.add_argument('--patch_size', default=None, type=int,  # changed
                        help='images patch size')
    parser.add_argument('--patch_size1', default=1, type=int,  # changed
                        help='images patch size')
    parser.add_argument('--patch_size2', default=None, type=int,  # changed
                        help='images patch size')
    parser.add_argument('--patch_num', default=10, type=int,  # changed - added
                        help='number of patches per one row in the image')
    parser.add_argument('--in_chans', default=6, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--alt', action='store_true',
                        help='using [n, c, l, 1] format instead') # changed - added
    parser.set_defaults(alt=False) # changed - added
    parser.add_argument('--mask_finetune', default=0, type=float)
    parser.add_argument('--finetune_interpolate_patches', action='store_true',
                        help='different number of patches in finetune)') # changed - added
    parser.set_defaults(finetune_interpolate_patches=False) # changed - added


    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data', default='UCIHAR', type=str, 
                        help='data to use')
    parser.add_argument('--data_path', default='', type=str, # changed
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int, # changed
                        help='number of the classification types')
    parser.add_argument('--RISE_hz', default=10, type=int, # changed
                        help='data frequency for RISE data')
    parser.add_argument('--normalization', action='store_true',
                        help='train and test data set normalization') # changed - added
    parser.set_defaults(normalization=False) # changed - added
    parser.add_argument('--normalization_chan', action='store_true',
                        help='train and test data set normalization per channel') 
    parser.set_defaults(normalization_chan=False) # changed - added

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--transform', action='store_true',
                        help='transform dataset to image (244*244)') # changed - added
    parser.set_defaults(transform=False) # changed - added
    parser.add_argument('--padding_transform', action='store_true',
                        help='transform dataset to image (244*244) with zero padding', default=False) # changed - added
    parser.add_argument('--hz_adjustment', action='store_true') 
    parser.set_defaults(hz_adjustment=False) 
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--SCV", default=False, action='store_true', help="Do subject-level 5-fold CV")
    parser.add_argument('--fold', default=0, type=int, 
                        help='cross-validation folds')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # evaluation
    parser.add_argument('--confusion_matrix_plot', action='store_true')
    parser.set_defaults(confusion_matrix_plot=False)
    parser.add_argument('--plot_save_name', type=str, default=None)
    parser.add_argument('--plot_title', type=str, default=None)
    parser.add_argument('--RISE_collapse_labels', action='store_true')
    parser.set_defaults(RISE_collapse_labels=False)


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
  

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()

    cudnn.benchmark = False # changed from True

    print("started data loading")
    if args.SCV == True:
        full_dataset = UCIHAR(data_path=args.data_path, is_all_data=True, normalization=args.normalization,
                            normalization_chan=args.normalization_chan,
                            pre_mix_up=False, mix_up=False, alt=args.alt, nb_classes=args.nb_classes,
                            transform=args.transform, padding_transform=args.padding_transform)
        subject_ids = np.array(torch.load(os.path.join(args.data_path, "CV_sub_ids.pt")))
        unique_subjects = np.unique(subject_ids)
        # retreive dataset_train
        train_subj_idx = torch.load(os.path.join(args.data_path, f"SCV/train_subj_idx_fold{args.fold}.pt"))
        test_subj_idx = torch.load(os.path.join(args.data_path, f"SCV/val_subj_idx_fold{args.fold}.pt"))
        train_subjects = unique_subjects[train_subj_idx]
        test_subjects = unique_subjects[test_subj_idx]
        train_indices = [i for i, sid in enumerate(subject_ids) if sid in train_subjects]
        val_indices = [i for i, sid in enumerate(subject_ids) if sid in test_subjects]
        print(f"\n--- Fold {args.fold + 1}/5 ---, train fold subjects: {train_subj_idx}")

        dataset_train = Subset(full_dataset, train_indices)
        dataset_val = Subset(full_dataset, val_indices)
    else:
        if args.data == "UCIHAR":
            dataset_train = UCIHAR(data_path=args.data_path, is_test=False, normalization=args.normalization,
                                normalization_chan=args.normalization_chan,
                                pre_mix_up=False, mix_up=False, alt=args.alt, nb_classes=args.nb_classes,
                                transform=args.transform, padding_transform=args.padding_transform)
            dataset_val = UCIHAR(data_path=args.data_path, is_test=True, normalization=args.normalization,
                                normalization_chan=args.normalization_chan,
                                pre_mix_up=False, mix_up=False, alt=args.alt,  nb_classes=args.nb_classes,
                                transform=args.transform, padding_transform=args.padding_transform)
        if args.data == "RISE":
            dataset_train = RISE(data_path=args.data_path, is_test=False, normalization=args.normalization,
                                normalization_chan=args.normalization_chan, RISE_hz = args.RISE_hz,
                                mix_up=False, alt=args.alt,transform=args.transform,
                                hz_adjustment = args.hz_adjustment, padding_transform=args.padding_transform)
            dataset_val = RISE(data_path=args.data_path, is_test=True, normalization=args.normalization,
                                normalization_chan=args.normalization_chan, RISE_hz = args.RISE_hz,
                                mix_up=False, alt=args.alt, transform=args.transform,
                                hz_adjustment = args.hz_adjustment, padding_transform=args.padding_transform)
    print("finished data loading")

    # input_dim, input_size, in_chans, patch_size sanity check
    input_dim = [dataset_train[0][0].shape[1], dataset_train[0][0].shape[2]]

    if args.input_size == None:     # changed - added 
        args.input_size = input_dim
    
    if args.input_size != input_dim: # changed - added 
        print("input_size incorrect")

    if args.patch_size2 == None:
        args.patch_size2 = int(args.input_size[1]/args.patch_num)

    args.patch_size = [args.patch_size1, args.patch_size2]

    if int(args.input_size[1]/args.patch_size2) != args.patch_num:
        args.patch_num = int(args.input_size[1]/args.patch_size2)

    if args.in_chans != dataset_train[0][0].shape[0]:
        args.in_chans = dataset_train[0][0].shape[0]

    print(f"input_size: {args.input_size}, patch_size:{args.patch_size}, patch_num: {args.patch_num}")


    sampler_train = torch.utils.data.RandomSampler(dataset_train) #changed 
    sampler_test = torch.utils.data.SequentialSampler(dataset_val) #changed 
    
    if args.log_dir is not None and not args.eval:  #changed - global_rank == 0 and ommitted
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    print(f"Using in_chans={args.in_chans}") 

    if args.mask_finetune > 0:
        model = models_vit_mask.__dict__[args.model](
            img_size=args.input_size, patch_size=args.patch_size,  # changed - added to reflect input_size change
            num_classes=args.nb_classes, in_chans = args.in_chans, #alt=args.alt,
            drop_path_rate=args.drop_path, mask_finetune=args.mask_finetune,
            global_pool=args.global_pool
        )
    
    if args.mask_finetune == 0:
         model = models_vit.__dict__[args.model](
            img_size=args.input_size, patch_size=args.patch_size,  # changed - added to reflect input_size change
            num_classes=args.nb_classes, in_chans = args.in_chans, #alt=args.alt,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool
        )       

    print(model)




    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        if args.finetune_interpolate_patches:
            interpolate_patch_shape = checkpoint_model['patch_embed.proj.weight'].shape
            interpolate_pos_embed(model, checkpoint_model, orig_size=(14,14), new_size=(int(args.input_size[0]/interpolate_patch_shape[-2]),int(args.input_size[1]/interpolate_patch_shape[-1]))) 
        else:
            interpolate_pos_embed(model, checkpoint_model, orig_size =(6,10), new_size=(6,10))
            
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False) # changed from strict=False
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed: #changed - hashed out
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler) 

    if args.eval:
        test_stats = evaluate(data_loader=data_loader_val, model=model, device=device,
        confusion_matrix_plot=args.confusion_matrix_plot, RISE_collapse_labels=args.RISE_collapse_labels, plot_save_name=args.plot_save_name, plot_title=args.plot_title)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer,  epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args, device=device,
        )
        if args.output_dir and (epoch % args.dump_freq == 0 or epoch + 1 == args.epochs): # changed - added and~ for less frequent dump
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch) # changed - changed to acc3
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
