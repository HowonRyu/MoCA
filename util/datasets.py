# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class UCIHAR(Dataset):
    def __init__(self, data_path, alt, is_test=False, is_all_data=False, pre_mix_up = False, normalization_chan=False,
                 normalization=False, transform=False, padding_transform=False, mix_up=True, nb_classes=7):

        # only for off-line mix_up in pretrain stage
        if pre_mix_up:
            x_train_dir = os.path.join(data_path, 'X_train_aug_all.pt') #(N, H, W, C)
            self.X = torch.tensor(torch.load(x_train_dir),dtype=torch.float32)
            if alt:
                self.X = self.X.unsqueeze(dim=3)
            else:
                self.X = self.X.unsqueeze(dim=3).permute(0,3,2,1)

            # dummy variable
            self.y = torch.zeros(self.X.shape[0], dtype=torch.long)

        else:
            X_train_dir = os.path.join(data_path, 'X_train_all.pt')
            y_train_dir = os.path.join(data_path, 'y_train_all_mode.pt')
            X_test_dir = os.path.join(data_path, 'X_test_all.pt')
            y_test_dir = os.path.join(data_path, 'y_test_all_mode.pt')
            X_all_dir = os.path.join(data_path, 'X_all_all.pt')
            y_all_dir = os.path.join(data_path, 'y_all_all_mode.pt')

            # Load data based on whether it's a test set or training set
            if is_all_data:
                self.X = torch.tensor(torch.load(X_all_dir), dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_all_dir), dtype=torch.long)

            elif is_test:
                self.X = torch.tensor(torch.load(X_test_dir),dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_test_dir), dtype=torch.long)
            else:
                self.X = torch.tensor(torch.load(X_train_dir),dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_train_dir), dtype=torch.long)


            if normalization_chan:
                per_channel_mean = self.X.mean(dim=(0,1,2), keepdim=True)
                per_channel_std = self.X.std(dim=(0,1,2), keepdim=True)
                self.X = (self.X - per_channel_mean) / (per_channel_std)


            if alt:
                self.X = self.X.permute(0, 3, 2, 1) # (N, H', W, C') = (N, C, W, H) = (N, 6, L, 1)


            if nb_classes==6:
                # Filter out samples with label 0 and adjust labels 1-6 to 0-5
                mask = self.y != 0
                self.X = self.X[mask]
                self.y = self.y[mask] -1  

        self.X = self.X.permute(0,3,1,2) # conform to (N, 1, H, W) = (N, 1, 6, L)
        self.normalization = normalization
        self.transform = transform
        self.padding_transform = padding_transform
        self.mix_up = mix_up
        self.shape = [self.X.shape, self.y.shape]
        print(padding_transform)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        # Apply normalization if transform is set
        if self.normalization:
            sample_X = (sample_X - sample_X.mean(dim=1,keepdim=True)) / sample_X.std(dim=1,keepdim=True)

        if self.mix_up:
            # Randomly select another sample
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            sample_X = self.mix_time_series(sample_X, sample_X2)

        # only for vit_base line
        if self.transform:
            sample_X = sample_X.unsqueeze(0)
            #sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
            sample_X = F.interpolate(sample_X, size=(96, 160), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
        return sample_X, sample_y

        if self.padding_transform:
            pad_right = 224 - 200  # 24
            pad_bottom = 224 - 6   # 218
            print("sample x before shape:", sample_X.shape)
            sample_padded = F.pad(sample_X, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
            sample_X = sample_padded.unsqueeze(0).repeat(3, 1, 1)
            print("padding_transform shape:", sample_X.shape)
        return sample_X, sample_y

    def mix_time_series(self, sample1, sample2):

        ts_len = sample1.shape[1]
        lambada = torch.distributions.Uniform(0, 0.5).sample().item()
        sample1_size = int(ts_len * lambada)
        sample2_size = ts_len - sample1_size 

        chunk1 = sample1[:, :sample1_size, :]  
        chunk2 = sample2[:, sample1_size:, :]  
        result = torch.cat((chunk1, chunk2), dim=1)

        return result


class RISE(Dataset):
    def __init__(self, data_path, alt, is_test=False, normalization_chan=False,
                 normalization=False, transform=False, mix_up=False, RISE_hz = 30, hz_adjustment=False):

        X_train_dir = os.path.join(data_path, 'X_train.pt')  # X_train dim (N, H, W, C)
        y_train_dir = os.path.join(data_path, 'y_train.pt')
        X_test_dir = os.path.join(data_path, 'X_test.pt')
        y_test_dir = os.path.join(data_path, 'y_test.pt')

        # Load data based on whether it's a test set or training set
        if is_test:
            self.X = torch.tensor(torch.load(X_test_dir),dtype=torch.float32)
            self.y = torch.tensor(torch.load(y_test_dir), dtype=torch.long)
        else:
            self.X = torch.tensor(torch.load(X_train_dir),dtype=torch.float32)
            self.y = torch.tensor(torch.load(y_train_dir), dtype=torch.long)

        if normalization_chan:
            per_channel_mean = self.X.mean(dim=(0,1,2), keepdim=True)
            per_channel_std = self.X.std(dim=(0,1,2), keepdim=True)
            self.X = (self.X - per_channel_mean) / (per_channel_std)

        if alt:
            self.X = self.X.permute(0, 3, 2, 1)  # (N, H', W, C') = (N, C, W, H) = (N, 3, L, 1)


        self.X = self.X.permute(0,3,1,2) # conform to (N, 1, H, W) = (N, 1, 3, L)
        self.normalization = normalization
        self.transform = transform
        self.mix_up = mix_up
        self.shape = [self.X.shape, self.y.shape]
        self.RISE_hz = RISE_hz
        self.hz_adjustment = hz_adjustment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        # Apply normalization if transform is set
        if self.normalization:
            sample_X = (sample_X - sample_X.mean(dim=1,keepdim=True)) / sample_X.std(dim=1,keepdim=True)

        if self.mix_up:
            # Randomly select another sample
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            sample_X = self.mix_time_series(sample_X, sample_X2)

        # only for vit_base line
        if self.transform:
            sample_X = sample_X.unsqueeze(0)
            sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
        if self.hz_adjustment:
            new_length = int((sample_X.shape[-1]/self.RISE_hz) * 50 ) 
            sample_X = F.interpolate(sample_X, size=new_length, mode='linear', align_corners=True)
        return sample_X, sample_y

    def mix_time_series(self, sample1, sample2):

        ts_len = sample1.shape[1]
        lambada = torch.distributions.Uniform(0, 0.5).sample().item()
        sample1_size = int(ts_len * lambada)
        sample2_size = ts_len - sample1_size 

        chunk1 = sample1[:, :sample1_size, :]  
        chunk2 = sample2[:, sample1_size:, :]  
        result = torch.cat((chunk1, chunk2), dim=1)

        return result




def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
