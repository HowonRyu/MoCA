import torch
import numpy as np

import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
from functools import partial
from timm.utils import accuracy


from models_mae import MaskedAutoencoderViT
import models_vit
import models_vit_mask
from timm.models.layers import trunc_normal_

from util.pos_embed import *
from util.datasets import *


def load_dataset(data_path, data, test_data=True, RISE_hz=30, alt=True, normalization=False, transform=False):
    ids = None
    times = None

    test_train_label = ["train", "test"]

    if data == "RISE":
        dataset = RISE(data_path=data_path, is_test=test_data, normalization=normalization,
                        normalization_chan=False, RISE_hz=RISE_hz,
                            mix_up=False, alt=alt, transform=transform)
        labels = ["sedentary", "standing", "stepping", "cycling", "primary_lying", "secondary_lying", "seated_transport"]
        activity_labels = {0: "SEDENTARY", 1: "STANDING", 2: "STEPPING", 3: "CYCLING", 4: "PRIMARY_LYING", 5: "SECONDARY_LYING", 6: "SEATED_TRANSPORT"}
        ids = torch.load(os.path.join(data_path, f"id_{test_train_label[test_data]}.pt"))
        #times = torch.load(os.path.join(data_path, f"time_{test_train_label[test_data]}.pt"))
    elif data == "UCIHAR":
        dataset = UCIHAR(data_path=data_path, is_test=test_data, normalization=normalization,
                        normalization_chan=False, mix_up=False, alt=alt, transform=transform)
        
        labels = ['Transition', 'Walking', 'Walking-upstairs', 'Walking-downstairs', 'Sitting', 'Standing', 'Laying']
        activity_labels = {0: "Transition", 1: "Walking", 2: "Walking-upstairs", 3: "Walking-downstairs",
                            4: "Sitting", 5: "Standing", 6: "Laying"}

    return dataset, labels, activity_labels, ids, times




## reconstruction
def load_model_pretrain(checkpoint_path, input_size=[3,200], patch_size=[1,20], finetune_interpolate_patches=False, alt=True, in_chans=1):
    #define new model
    model_mae = MaskedAutoencoderViT(img_size=input_size, patch_size=patch_size, in_chans=in_chans,     # vit_base
                embed_dim=768, depth=12, num_heads=12, 
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), alt=alt)
    #bring in checkpoint model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    print("decoder_pos_embed shape BEFORE loading into model:", checkpoint_model['decoder_pos_embed'].shape)
    print("model expected shape:", model_mae.decoder_pos_embed.shape)

    # interpolate position embedding
    if finetune_interpolate_patches:
        interpolate_patch_len = checkpoint_model['patch_embed.proj.weight'].shape[-1]
        interpolate_pos_embed(model_mae, checkpoint_model, orig_size=(6,10), new_size=(input_size[0],int(input_size[1]/interpolate_patch_len))) 
    else:
        interpolate_pos_embed(model_mae, checkpoint_model, orig_size =(6,10), new_size=(6,10))

    model_mae.load_state_dict(checkpoint_model, strict=False)

    return model_mae




def mae_reconstruction_pass(x, model, alt, mask_ratio=0.75, norm_pix_loss=False, import_mask=None, seed=None):
    """
    input
        x: torch of dim (BS, 1, H', W) = (BS, 1, C, L) 
    output
        im_masked: (BS, H', W, 1) = (BS, C, L, 1)
        im_paste: (BS, H', W, 1) = (BS, C, L, 1)
        x_prime: (BS, H', W, 1) = (BS, C, L, 1)
        y: (BS, H', W, 1) = (BS, C, L, 1)
        loss: [MSE, nMSE]
    """

    model.alt = alt
    in_chans = x.shape[1]
    # if norm_pix_loss enabled, normalize x patch-wise
    if norm_pix_loss:
        x_patch = model.patchify(x)  # patchify to get patch-wise mean and variance
        mean = x_patch.mean(dim=-1, keepdim=True)
        var = x_patch.var(dim=-1, keepdim=True)
        x_patch_norm = (x_patch - mean) / (var + 1.e-6)**.5
        x = model.unpatchify(x_patch_norm)


    # load reconstructed y and masks - deconstructed self.forward() to designate masking
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio, import_mask=import_mask, seed=seed) 
    #print("y shape:", y.shape, ", mask (1-remove or 0-keep) shape:", mask.shape)

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()


    # make mask into the same dimension as x so we can apply operation in the next step
    #print("mask shape before:", mask.shape)
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]* model.patch_embed.patch_size[1] * in_chans)  # (N, H*W, p0*p1*3)  #changed
    
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    #print("mask shape after:", mask.shape)


    # revert x back to (1, h, w, c)
    x_prime = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x_prime * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x_prime * (1 - mask) + y * mask

    return im_masked, im_paste, x_prime, y, loss


def plot_reconstruction(model, x, alt, input_length, mask_ratio=0.75, time_x = None, norm_pix_loss=False, title = None, marg = 0.2, 
                        activity_label=None, labels=None, import_mask=None, seed=None, figsize=(18,13), linestyle=""):

    masked, paste, x_prime, y, loss = mae_reconstruction_pass(x=x, model=model, mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss, alt=alt,
                                                               import_mask=import_mask, seed=seed)

    print(f"test loss= {loss[0]:.3f}(MSE), {loss[1]:.3f}(nMSE)")

    if alt:
        masked = masked.permute(0,3,2,1).reshape(1,1,-1,3)
        paste = paste.permute(0,3,2,1).reshape(1,1,-1,3)
        x_prime = x_prime.permute(0,3,2,1).reshape(1,1,-1,3)
        y = y.permute(0,3,2,1).reshape(1,1,-1,3)

    print(masked.shape, paste.shape, x_prime.shape, y.shape)
    #print("time:", time_x.shape) #TODO: add time as x-axis

    # y limits
    y_squeeze = y.squeeze(dim=0).squeeze(dim=0)
    all_data = torch.cat((y, x_prime), dim=0)
    ymin_lims = torch.amin(all_data, dim=(0, 1, 2))
    ymax_lims = torch.amax(all_data, dim=(0, 1, 2))
    ms = 2
    linestyle_unmasked = ''
    modalities = ["Acc"]
    axes = ['X', "Y", "Z"]
    fig, axs = plt.subplots(len(modalities)*2, len(axes), figsize=figsize)

    for i, modal in enumerate(modalities):
        for j, ax in enumerate(axes):
            # xaxis
            axs[2*i,j].plot(x_prime[0, 0, :, 3*i+j], label="original (masked)", color="black", marker='o', linestyle=linestyle, markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked[0, 0, :, 3*i+j] != 0
            indices_x = np.where(masked[0, 0, :, 3*i+j] != 0)[0]
            axs[2*i,j].plot(indices_x, masked[0,0,unmasked_x,3*i+j], label="original (unmasked)", color="orange", marker='o', linestyle=linestyle_unmasked, markersize=ms)
            axs[2*i,j].legend()
            axs[2*i,j].set_title(f"{ax}_{modal}")
            axs[2*i,j].set_ylim([ymin_lims[3*i+j]-marg, ymax_lims[3*i+j]+marg])

            axs[2*i+1,j].plot(y[0,0,:,3*i+j], label="reconstructed", color="blue", marker='o', linestyle=linestyle, markersize=ms)
            #axs[i,j].plot(paste[0,0,:,i], label="reconstruction + visible")
            axs[2*i+1,j].legend()
            axs[2*i+1,j].set_title(f"{ax}_{modal}")
            axs[2*i+1,j].set_ylim([ymin_lims[3*i+j]-marg, ymax_lims[3*i+j]+marg])
            for k in range(0, masked.shape[2], input_length):
                axs[2*i, j].axvline(x=k, color='gray', linestyle='--', linewidth=0.5)
                axs[2*i+1, j].axvline(x=k, color='gray', linestyle='--', linewidth=0.5)
    
    

    plt.suptitle(f"Reconstruction {title}, \n Activity={[activity_label[al.item()] for al in labels]}")
    plt.subplots_adjust(hspace=0.1, top=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()


def mae_reconstruction_loop_UCIHAR(model, x, mask_ratio, alt, norm_pix_loss=False, import_mask=False):

    masked_all = torch.Tensor()
    paste_all = torch.Tensor()
    x_prime_all = torch.Tensor()
    y_all = torch.Tensor()
    x_squeeze_all = torch.Tensor()
    loss_MSE = list()
    loss_nMSE = list()

    for i in range(len(x)):
        if i % 100 == 0:
            print(f"{i}/{len(x)}")
        x = x[i:i+1,:,:,:]
        
        if alt:
            x = x.permute(0, 3, 2, 1) 

        x = x.permute(0,3,1,2)
            
        masked, paste, x_prime, y, loss = mae_reconstruction_pass(x=x, model=model, mask_ratio=mask_ratio, alt=alt, 
                                                                  norm_pix_loss=norm_pix_loss, import_mask=import_mask)

        x_squeeze = x.squeeze(dim=0)

        masked_all = torch.cat((masked_all, masked), dim=0)
        paste_all = torch.cat((paste_all, paste), dim=0)
        x_prime_all = torch.cat((x_prime_all, x_prime), dim=0)
        y_all = torch.cat((y_all, y), dim=0)

        # loss
        if alt:
            loss_MSE.append(loss[0].detach().numpy())
        else:
            loss_MSE.append(loss[0][0].detach().numpy())
            loss_nMSE.append(loss[0][1].detach().numpy())


        x_squeeze_all = torch.cat((x_squeeze_all, x_squeeze), dim=0)

    if alt:
        masked_all = masked_all.permute(0,3,2,1)
        paste_all = paste_all.permute(0,3,2,1)
        x_prime_all = x_prime_all.permute(0,3,2,1)
        y_all = y_all.permute(0,3,2,1)
        #x_squeeze_all = x_squeeze_all.permute(2, 1, 0)

    return masked_all, paste_all, x_prime_all, y_all, x_squeeze_all, loss_MSE, loss_nMSE


def mae_reconstruction_loop(model, dataset, batch_size, mask_ratio, alt, norm_pix_loss=False, import_mask=False, device='cuda'):
    """
    input
        dataset: Dataset object
    output
        masked_all: (len(dataset), H, W, C) = (N, 1, L, C)
        paste_all: (len(dataset), H, W, C) = (N, 1, L, C)
        x_prime_all: (len(dataset), H, W, C) = (N, 1, L, C)
        y_all: (len(dataset), H, W, C) = (N, 1, L, C)
        x_squeeze_all: (len(dataset), H, W, C) = (N, 1, L, C)
        loss_MSE: (N/BS,)
        loss_nMSE: (N/BS,)
    """
    masked_all = torch.Tensor()
    paste_all = torch.Tensor()
    x_prime_all = torch.Tensor()
    y_all = torch.Tensor()
    x_squeeze_all = torch.Tensor()
    loss_MSE = list()
    loss_nMSE = list()

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=1, drop_last=False)
    
    for batch in iter(data_loader):

        x = batch[0].to(device, non_blocking=True)
        masked, paste, x_prime, y, loss = mae_reconstruction_pass(x=x, model=model, mask_ratio=mask_ratio, alt=alt, 
                                                                  norm_pix_loss=norm_pix_loss, import_mask=import_mask)

        x_squeeze = x.squeeze(dim=0)
        masked_all = torch.cat((masked_all, masked), dim=0)
        paste_all = torch.cat((paste_all, paste), dim=0)
        x_prime_all = torch.cat((x_prime_all, x_prime), dim=0)
        y_all = torch.cat((y_all, y), dim=0)
  
        # loss
        if alt:
            loss_MSE.append(loss[0].detach().numpy())
            loss_nMSE.append(loss[1].detach().numpy())
        else:
            loss_MSE.append(loss[0][0].detach().numpy())
            loss_nMSE.append(loss[0][1].detach().numpy())


        x_squeeze_all = torch.cat((x_squeeze_all, x_squeeze), dim=0)

    if alt:
        masked_all = masked_all.permute(0,3,2,1)
        paste_all = paste_all.permute(0,3,2,1)
        x_prime_all = x_prime_all.permute(0,3,2,1)
        y_all = y_all.permute(0,3,2,1)
        #x_squeeze_all = x_squeeze_all.permute(2, 1, 0)

    return masked_all, paste_all, x_prime_all, y_all, x_squeeze_all, loss_MSE, loss_nMSE



def plot_all_reconstruction(masked_all, x_prime_all, y_all, X_test_subseq_indicator, label_y, sample_sub, loss_MSE, loss_nMSE, test_only=False, 
                            ms=0.5, figsize=(30, 50), plot_save=False, plot_save_name=None):
    """
    only use for UCI-HAR data
    """

    fig, axs = plt.subplots(12, 1, figsize=figsize)

    plot_labels = ['Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']


    image_length = masked_all.shape[2]
    cutoff = 17600
    n_images_in_trjt = int(cutoff/image_length)

    label_y_per_sub = label_y[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt]


    test_indicator = list(X_test_subseq_indicator[:,0,0,0])
    test_loss_MSE = [a for a, b in  zip(loss_MSE, test_indicator) if b == 1]
    training_loss_MSE = [a for a, b in  zip(loss_MSE, test_indicator) if b == 0]

    test_loss_nMSE = [a for a, b in  zip(loss_nMSE, test_indicator) if b == 1]
    training_loss_nMSE = [a for a, b in  zip(loss_nMSE, test_indicator) if b == 0]


    test_loss_MSE = np.round(np.mean(test_loss_MSE),4)
    training_loss_MSE = np.round(np.mean(training_loss_MSE),4)

    test_loss_nMSE = np.round(np.mean(test_loss_nMSE),4)
    training_loss_nMSE = np.round(np.mean(training_loss_nMSE),4)

    print(f"training MSE: {training_loss_MSE}, training nMSE: {training_loss_nMSE}, test MSE: {test_loss_MSE}, test nMSE: {test_loss_nMSE}")

    if test_only:    
        title =  f"Test images, subject{sample_sub}"
    else:
        title =  f"Train + Test images, subject{sample_sub}"


    for i in range(6):
        # reshape data
        masked_plot = masked_all[sample_sub*n_images_in_trjt : (sample_sub+1)*n_images_in_trjt, 0, :, i].reshape(-1)
        #paste_plot = paste_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)
        x_prime_plot = x_prime_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)
        y_plot = y_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)


        # test indicator
        test_indicator = X_test_subseq_indicator[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,0].reshape(-1)
        background_test_mask = np.where(test_indicator == 1, 1, np.nan).astype(int)

        # label
        label_y_list = np.repeat(np.array(label_y_per_sub), image_length).tolist()
        activity_labels = {0: "TRANSITION", 1: "WALKING", 2:"WALKING-DOWN", 3:"WALKING-UP", 4: "SITTING", 5: "STANDING", 6: "LAYING"}

        # y limits
        ymin, ymax = -2, 6.5

        if test_only:
            masked_plot_test_only =  [a for a, b in zip(masked_plot, test_indicator) if b == 1]
            x_prime_plot_test_only  = [a for a, b in zip(x_prime_plot, test_indicator) if b == 1]
            y_plot_test_only  = [a for a, b in zip(y_plot, test_indicator) if b == 1]
            label_y_list_test_only = [a for a, b in zip(label_y_list, test_indicator) if b == 1]

            # plot
            axs[i*2].plot(x_prime_plot_test_only, label="original (masked)", marker='o', color='blue', linestyle="-", markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked_plot_test_only != 0
            indices_x = np.where(masked_plot_test_only != 0)[0]
            axs[i*2].plot(indices_x, masked_plot_test_only[unmasked_x], label="original (unmasked)", color='orange', marker='o', linestyle="", markersize=ms)
            axs[i*2].legend()
            axs[i*2].set_title(f"{plot_labels[i]} original")
            axs[i*2].plot(label_y_list_test_only, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            axs[i*2].set_yticks(range(0,7))  # Set the range from 1 to 6
            axs[i*2].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names

            axs[i*2+1].plot(y_plot_test_only, label="reconstructed", color='green', marker='o', linestyle="-", markersize=ms, alpha=1)
            axs[i*2+1].plot(label_y_list_test_only, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            #axs[0].plot(paste[0,0,:,0], label="reconstruction + visible")
            axs[i*2+1].legend()
            axs[i*2+1].set_title(f"{plot_labels[i]} reconstructed")
            axs[i*2+1].set_ylim([ymin, ymax])
            axs[i*2+1].set_yticks(range(0,7))  # Set the range from 0 to 6
            axs[i*2+1].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names


        else:
            # plot
            axs[i*2].plot(x_prime_plot, label="original (masked)", marker='o', color='blue', linestyle="-", markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked_plot != 0
            indices_x = np.where(masked_plot != 0)[0]
            axs[i*2].plot(indices_x, masked_plot[unmasked_x], label="original (unmasked)", color='orange', marker='o', linestyle="", markersize=ms)
            axs[i*2].legend()
            axs[i*2].set_title(f"{plot_labels[i]} original")
            axs[i*2].plot(label_y_list, label="label", marker='o', linestyle="-", color='black', markersize=ms)
            axs[i*2].fill_between(np.arange(cutoff), ymin, ymax, where=background_test_mask, color='white', alpha=0.3)
            axs[i*2].fill_between(np.arange(cutoff), ymin, ymax, where=~background_test_mask, color='grey', alpha=0.3)
            axs[i*2].set_yticks(range(0,7))  # Set the range from 1 to 6
            axs[i*2].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names



            axs[i*2+1].plot(y_plot, label="reconstructed", color='green', marker='o', linestyle="-", markersize=ms, alpha=1)
            axs[i*2+1].plot(label_y_list, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            #axs[0].plot(paste[0,0,:,0], label="reconstruction + visible")
            axs[i*2+1].legend()
            axs[i*2+1].set_title(f"{plot_labels[i]} reconstructed")
            axs[i*2+1].set_ylim([ymin, ymax])
            axs[i*2+1].set_yticks(range(0,7))  # Set the range from 0 to 6
            axs[i*2+1].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names
            axs[i*2+1].fill_between(np.arange(cutoff), ymin, ymax, where=background_test_mask, color='white', alpha=0.3)
            axs[i*2+1].fill_between(np.arange(cutoff), ymin, ymax, where=~background_test_mask, color='grey', alpha=0.3)


    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.5, top=0.97)

    if plot_save:
        plt.savefig(f'{plot_save_name}_recon_plot.png')
    else:
        plt.show()




## downstream (ft) confusion matrix

def load_model_ft(checkpoint_path, LP=False, nb_classes=7, in_chans=1, input_size=[3,200], patch_size=[1,20],
                  finetune_interpolate_patches=False, mask_finetune=0, device='cuda', global_pool=False):

    if mask_finetune > 0:
        model = models_vit_mask.__dict__['vit_base_patch16'](
            img_size=input_size, patch_size=patch_size,
            num_classes=nb_classes, in_chans=in_chans, mask_finetune=mask_finetune,
        )
    if mask_finetune == 0:
        model = models_vit.__dict__['vit_base_patch16'](
            img_size=input_size, patch_size=patch_size, 
            num_classes=nb_classes, in_chans=in_chans
        )
    model.to(device)
    
    if LP:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True


    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    if finetune_interpolate_patches:
        interpolate_patch_shape = checkpoint_model['patch_embed.proj.weight'].shape
        interpolate_pos_embed(model, checkpoint_model, orig_size=(14,14), new_size=(int(input_size[0]/interpolate_patch_shape[-2]),int(input_size[1]/interpolate_patch_shape[-1]))) 
    else:
        interpolate_pos_embed(model, checkpoint_model, orig_size =(6,10), new_size=(6,10))
        
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False) # changed from strict=False
    print(msg)

    return model


def mae_classification_pass(dataset, model, batch_size=100, device='cuda', num_workers=1):
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=False)

    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in iter(data_loader):
            images = batch[0].to(device, non_blocking=True)
            target = batch[-1].to(device, non_blocking=True)

            output = model(images)  
            preds.append(output.cpu())
            targets.append(target.cpu())

    all_outputs = torch.cat(preds, dim=0)
    all_targets = torch.cat(targets, dim=0) 

    # Predictions
    preds_list = all_outputs.argmax(dim=1).tolist()
    targets_list = all_targets.tolist()

    # Accuracy (use logits, not argmax!)
    acc1, acc3 = accuracy(all_outputs, all_targets, topk=(1, 3))

    return preds_list, targets_list, acc1






