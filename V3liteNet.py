import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
import numpy as np
from PIL import Image
import cv2
import h5py
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixnet import *
from V3litedataset import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder=MixNet(net_type='mixnet_l')
    def forward(self,x):
        x=self.encoder(x)
        return x

class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=8,stride=1)
        self.conv1=nn.Conv2d(in_channels=56,out_channels=56, kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=56,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(56)
        self.bn2=nn.BatchNorm2d(1)
        # self.conv1 = nn.Conv2d(in_channels=56, out_channels=128, kernel_size=(8, 8), stride=(1,1), padding=(0,0))
        # self.bn1=nn.BatchNorm2d(num_features=128)
        # self.conv2= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0,0))
        # self.conv3=nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), stride=(1,1), padding=(0,0))

    def forward(self,x):
        x=self.pool(x)
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        # x=self.conv3(x)
        return x

class Normalizer:
    @staticmethod
    def gpu_normalizer(x):
        _, _, rh, rw = x.size()
        normalize_ones = torch.cuda.FloatTensor(1, 1, rh, rw).fill_(0)
        filter = torch.cuda.FloatTensor(1, 1, 8, 8).fill_(1)
        for i in range(rh-7):
            for j in range(rw-7):
                normalize_ones[0][0][i:i+8,j:j+8]+=filter[0][0]
        x = x / normalize_ones
        return x.squeeze().cpu().detach().numpy()

class dynamic_unfolding(nn.Module):
    def __init__(self):
        super(dynamic_unfolding, self).__init__()
        pass

    def forward(self, x, local_count, output_stride):
        # print(x)
        # print(x.size())
        a,b,h,w=x.size()
        R=torch.zeros((a,1,h,w)).cuda()
        # harray=np.arange(0,h,output_stride)
        # warray=np.arange(0,w,output_stride)
        # print(R.size())
        for batch in range(a):
            for i in range(h-output_stride+1):
                for j in range(w-output_stride+1):
                    r=torch.mean(x,dim=1)
                    r=r[batch][i:i+output_stride,j:j+output_stride]
                    exp = torch.exp(r)
                    s = torch.sum(exp)
                    r = exp / s
                    R[batch][0][i:i+output_stride,j:j+output_stride]+=local_count[batch][0][i][j]*r
                    # R.unsqueeze(0).unsqueeze(0)
        return R

class CountingModels(nn.Module):
    def __init__(self, input_size=64, output_stride=8):
        super(CountingModels, self).__init__()
        self.counter = Counter()
        self.encoder = Encoder()
        self.dynamic_unfolding = dynamic_unfolding()
        self.normalizer = Normalizer.gpu_normalizer
        self.weight_init()

    def forward(self,x,is_normalize=False):
        imh, imw = x.size()[2:]
        x = self.encoder(x)
        C = self.counter(x)
        R = self.dynamic_unfolding(local_count=C, output_stride=8, x=x)

        if is_normalize==True:
            R=self.normalizer(R)

        return {'C':C,'R':R}

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight,
                #         mode='fan_in',
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__=='__main__':
    # net=CountingModels().cuda()
    # net.eval()
    # x=torch.randn(8,3,320,320).cuda()
    # y=net(x)
    # print(y['R'].size())
    # print(y['C'].size())

    # system-related parameters
    data_dir = "maize_tassels_counting_uav_dataset"
    train_list = "train.txt"
    val_list = "val.txt"
    image_scale = 1. / 255
    image_mean = [.3405, .4747, .2418]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    input_size = 64
    output_stride = 8
    resized_ratio = 0.125
    print_every = 1

    # model-related parameters
    optimizer = 'sgd'
    batch_size = 8
    crop_size = (320, 320)
    learning_rate = 0.01
    milestones = [200, 400]
    momentum = 0.95
    mult = 1
    num_epoch = 500
    weight_decay = 0.0005

    net = CountingModels()
    net = nn.DataParallel(net)
    net.cuda()

    # filter parameters
    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []

    # define loss function and optimizer
    criterion = nn.L1Loss(reduction='mean').cuda()

    optimizer = torch.optim.SGD(
        [
            {'params': learning_params},
            {'params': pretrained_params, 'lr': learning_rate / mult},
        ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # restore parameters
    start_epoch = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'mae': [],
        'mse': [],
        'rmae': [],
        'rmse': [],
        'r2': []
    }

    # define transform
    transform_train = [
        RandomCrop(crop_size),
        RandomFlip(),
        Normalize(
            image_scale,
            image_mean,
            image_std
        ),
        ToTensor(),
        ZeroPadding(output_stride)
    ]
    transform_val = [
        Normalize(
            image_scale,
            image_mean,
            image_std
        ),
        ToTensor(),
        ZeroPadding(output_stride)
    ]
    composed_transform_train = transforms.Compose(transform_train)
    composed_transform_val = transforms.Compose(transform_val)

    # define dataset loader
    trainset = MaizeTasselDataset(
        data_dir=data_dir,
        data_list=train_list,
        ratio=resized_ratio,
        train=True,
        transform=composed_transform_train
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    valset = MaizeTasselDataset(
        data_dir=data_dir,
        data_list=val_list,
        ratio=resized_ratio,
        train=False,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    def R_generator(D, p, o, Io):
        # Input:
        # D: density map
        # p: patch size
        # o: output stride
        # Io: full-one matrix of size oxo
        # Output:
        # R_hat: fine-resolution ground-truth map
        h, w = D.size()[2:]
        rh, rw, k = int(h / o), int(w / o), int(p / o)
        R_hat = F.conv2d(D, Io, stride=o)
        R_hat = F.unfold(R_hat, kernel_size=k)
        R_hat = F.fold(R_hat, (rh, rw), kernel_size=k)
        return R_hat


    Io = torch.FloatTensor(1, 1, 8, 8).fill_(1).cuda()

    net.train()

    if batch_size == 1:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    running_loss = 0.0
    in_sz = input_size
    os = output_stride
    target_filter = torch.cuda.FloatTensor(1, 1, in_sz, in_sz).fill_(1)
    for i, sample in enumerate(train_loader):

        inputs, targets = sample['image'], sample['target']
        inputs, targets = inputs.cuda(), targets.cuda()

            # zero the parameter gradients
        optimizer.zero_grad()
            # forward
        print("OK")
        y= net(inputs)
        C=y['C']
        R=y['R']
        print("OK")

        R_hat = R_generator(targets, 64, 8, Io)
            # generate targets
            # print(targets.size())

        targets = F.conv2d(targets, target_filter, stride=os)

            # print(targets.size())
            # compute loss
        loss1 = criterion(C, targets)
        loss2 = criterion(R, R_hat)

            # backward + optimize
        loss=loss1+loss2
        loss.backward()
        optimizer.step()

            # collect and print statistics
        running_loss += loss.item()

