# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('nMSE', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if not args.alt:
        metric_logger.add_meter('loss_x_gyr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added
        metric_logger.add_meter('loss_y_gyr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added
        metric_logger.add_meter('loss_z_gyr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added
        metric_logger.add_meter('loss_x_acc', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added
        metric_logger.add_meter('loss_y_acc', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added
        metric_logger.add_meter('loss_z_acc', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')) #changed : added


    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        if args.alt: # loss = [loss, loss_mse, loss_nmse]
            with torch.cuda.amp.autocast():
                loss_all, _, _ = model(samples, mask_ratio=args.mask_ratio, masking_scheme=args.masking_scheme,
                                        var_mask_ratio=args.var_mask_ratio, time_mask_ratio=args.time_mask_ratio)
            loss = loss_all[0]
            loss_value = loss_all[0].item()
            loss_nmse = loss_all[1].item()

        else:   # loss_all = [[loss, loss_mse, loss_nmse], loss_x_gyr, loss_y_gyr, loss_z_gyr, loss_x_acc, loss_y_acc, loss_z_acc]
            with torch.cuda.amp.autocast():
                loss_all, _, _ = model(samples, mask_ratio=args.mask_ratio, masking_scheme=args.masking_scheme,
                                        var_mask_ratio=args.var_mask_ratio, time_mask_ratio=args.time_mask_ratio) #changed
            loss = loss_all[0][0] #changed: changed from loss.item() since now we have list of losses
            loss_value = loss_all[0][0].item() #changed: changed from loss.item() since now we have list of losses
            loss_nmse = loss_all[0][1].item()
            loss_x_gyr_value = loss_all[1].item() #changed: added for per channel loss
            loss_y_gyr_value = loss_all[2].item() #changed: added for per channel loss
            loss_z_gyr_value = loss_all[3].item() #changed: added for per channel loss
            loss_x_acc_value = loss_all[4].item() #changed: added for per channel loss
            loss_y_acc_value = loss_all[5].item() #changed: added for per channel loss
            loss_z_acc_value = loss_all[6].item() #changed: added for per channel loss


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
            

        metric_logger.update(loss=loss_value)
        metric_logger.update(nMSE=loss_nmse)        

        if args.alt == False:
            metric_logger.update(loss_x_gyr=loss_x_gyr_value) #changed : added
            metric_logger.update(loss_y_gyr=loss_y_gyr_value) #changed : added
            metric_logger.update(loss_z_gyr=loss_z_gyr_value) #changed : added
            metric_logger.update(loss_x_acc=loss_x_acc_value) #changed : added
            metric_logger.update(loss_y_acc=loss_y_acc_value) #changed : added
            metric_logger.update(loss_z_acc=loss_z_acc_value) #changed : added

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)  
        
        if args.alt == False:
            loss_x_gyr_value_reduce = misc.all_reduce_mean(loss_x_gyr_value) #changed : added
            loss_y_gyr_value_reduce = misc.all_reduce_mean(loss_y_gyr_value) #changed : added
            loss_z_gyr_value_reduce = misc.all_reduce_mean(loss_z_gyr_value) #changed : added
            loss_x_acc_value_reduce = misc.all_reduce_mean(loss_x_acc_value) #changed : added
            loss_y_acc_value_reduce = misc.all_reduce_mean(loss_y_acc_value) #changed : added
            loss_z_acc_value_reduce = misc.all_reduce_mean(loss_z_acc_value) #changed : added


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)

            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('nMSE', loss_nmse, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


            if not args.alt:
                log_writer.add_scalar('train_loss/gyr_x', loss_x_gyr_value_reduce, epoch_1000x)  #changed : added
                log_writer.add_scalar('train_loss/gyr_y', loss_y_gyr_value_reduce, epoch_1000x)  #changed : added
                log_writer.add_scalar('train_loss/gyr_z', loss_z_gyr_value_reduce, epoch_1000x)  #changed : added
                log_writer.add_scalar('train_loss/acc_x', loss_x_acc_value_reduce, epoch_1000x)  #changed : added
                log_writer.add_scalar('train_loss/acc_y', loss_y_acc_value_reduce, epoch_1000x)  #changed : added
                log_writer.add_scalar('train_loss/acc_z', loss_z_acc_value_reduce, epoch_1000x)  #changed : added



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}