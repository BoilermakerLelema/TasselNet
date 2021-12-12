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
from V3liteNet import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

class MaizeTasselDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_dir + "/" + data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        self.image_list = []

        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.target_blacks = {}
        self.gtcounts = {}
        self.dotimages = {}

    def pointlocation(self,data):
        # data = pd.read_csv(str)
        row, col = data.shape
        gtcount = data['region_count'][0]
        points=[]
        for i in range(row):
            location = data['region_shape_attributes'][i]
            if len(location) < 20:
                continue
            else:
                points.append([json.loads(location)['cx'],json.loads(location)['cy']])
        return gtcount, points

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name[0])
        if file_name[0] not in self.images:
            image = read_image(self.data_dir + file_name[0])
            annotation = pd.read_csv(self.data_dir + file_name[1])
            h, w = image.shape[:2]
            nh = int(np.ceil(h * self.ratio))
            nw = int(np.ceil(w * self.ratio))

            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            target_black = image * 0

            dotimage = image.copy()
            gtcount,points=self.pointlocation(annotation)

            for pt in points:
                pt[0], pt[1] = int(pt[0] * self.ratio), int(pt[1] * self.ratio)
                target[pt[1], pt[0]] = 1
                cv2.circle(dotimage, (pt[0], pt[1]), int(24 * self.ratio), (255,0, 0), -1)
                cv2.circle(target_black, (pt[0], pt[1]), int(24 * self.ratio), (255,0, 0), -1)

            target = gaussian_filter(target, 80 * self.ratio)
            # print(gtcount)
            # print(dotimage.shape)
            # plt.imshow(dotimage)
            # plt.imshow(target)
            # plt.show()
            # print(target.sum())

            self.images.update({file_name[0]: image.astype('float32')})
            self.targets.update({file_name[0]: target})
            self.target_blacks.update({file_name[0]: target_black})
            self.gtcounts.update({file_name[0]: gtcount})
            self.dotimages.update({file_name[0]: dotimage})

        sample = {
            'image': self.images[file_name[0]], ## original image
            'target': self.targets[file_name[0]], ## target density map
            'target_black': self.target_blacks[file_name[0]],  ## target density map
            'gtcount': self.gtcounts[file_name[0]] ## ground truth counting number
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # assert output_size is a integer or tuple
        self.output_size = output_size

    def __call__(self, sample):

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.shape[:2]

        if isinstance(self.output_size, tuple): # if outputsize is a tuple:
            new_h = min(self.output_size[0], h)
            new_w = min(self.output_size[1], w) # to avoid output_size less than or equal to h,w
            assert (new_h, new_w) == self.output_size
        else:
            crop_size = min(self.output_size, h, w) # if outputsize is a integer
            assert crop_size == self.output_size
            new_h = new_w = crop_size #square sample

        if gtcount > 0: #ground truth count
            mask = target > 0 # target??
            ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[ch:h - ch + 1, cw:w - cw + 1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            if len(idh) != 0:
                ids = random.choice(range(len(idh)))
                hc, wc = idh[ids], idw[ids]
                top, left = hc - ch, wc - cw
            else:
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
        else:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)

        image = image[top:top + new_h, left:left + new_w, :]
        target = target[top:top + new_h, left:left + new_w]
        return {'image': image, 'target': target, 'gtcount': gtcount}

class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target, 'gtcount': gtcount}


class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image, target = image.astype('float32'), target.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std

        image, target = image.astype('float32'), target.astype('float32')

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize = self.psize

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.size()[-2:]
        ph, pw = (psize - h % psize), (psize - w % psize)
        # print(ph,pw)

        (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
        (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)

        if (ph != psize) or (pw != psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image, tmp_pad)
            target = F.pad(target, tmp_pad)

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = image.transpose((2, 0, 1))
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        image, target = torch.from_numpy(image), torch.from_numpy(target)
        return {'image': image, 'target': target, 'gtcount': gtcount}

if __name__ == '__main__':
    optimizer = 'sgd'
    batch_size = 8
    crop_size = (320, 320)
    learning_rate = 0.01
    milestones = [200, 400]
    momentum = 0.95
    mult = 1
    num_epoch = 500
    weight_decay = 0.0005

    data_dir = "maize_tassels_counting_uav_dataset"
    train_list = "train.txt"
    val_list = "val.txt"
    image_scale = 1. / 255
    image_mean = [0.4463, 0.5352, 0.3247]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))
    input_size = 64
    output_stride = 8
    resized_ratio = 0.125
    print_every = 1

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

    composed_transform_train = transforms.Compose(transform_train)
    trainset=MaizeTasselDataset(
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

    transform_val = [
        Normalize(
            image_scale,
            image_mean,
            image_std
        ),
        ToTensor(),
        ZeroPadding(output_stride)
    ]
    composed_transform_val = transforms.Compose(transform_val)
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
    num=0
    for i, data in enumerate(train_loader):
        num+=1
    print(num)




