"""create dataset and dataloader"""
import importlib
import logging
import os
import os.path as osp
import numpy as np
import random
from functools import partial
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from prefetch_generator import BackgroundGenerator

from utils.registry import DATASET_REGISTRY

data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in os.listdir(data_folder)
    if v.endswith("_dataset.py")
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"data.{file_name}") for file_name in dataset_filenames
]


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SquarePad:
    def __call__(self, image):
        w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        # padding = (hp, vp, hp, vp)
        padding = ((hp, hp), (vp, vp))
        img = np.pad(image, padding, 'constant')
        return img


class PassiveDegrad():
    def __init__(self, src_size, fake_size):
        self.src_size = src_size
        self.fake_size = fake_size

        src_h, src_w, _ = src_size
        fake_h, fake_w, _ = fake_size
        self.to_gray = True if len(src_size) == 3 else False
        self.reversed_fake_size = (fake_w, fake_h)
         
        scale = max(fake_h / src_h, fake_w / src_w)
        self.scale = scale

        crop_h = int(fake_h / scale)
        crop_w = int(fake_w / scale)
        skip_h = (src_h - crop_h) // 2
        skip_w = (src_w - crop_w) // 2
        self.crop_h, self.crop_w, self.skip_h, self.skip_w = crop_h, crop_w, skip_h, skip_w

    def __call__(self, img):
        if self.to_gray:
            img = img.convert('L')
        img = img.crop((self.skip_w, self.skip_h, self.skip_w+self.crop_w, self.skip_h+self.crop_h))
        img = img.resize(self.reversed_fake_size, resample=Image.BICUBIC)
        return np.array(img)


def create_dataloader(dataset, dataset_opt, dist=False):
    phase = dataset_opt["phase"]
    if phase == "train":
        num_workers = dataset_opt["workers_per_gpu"]
        batch_size = dataset_opt["imgs_per_gpu"]
        return DataLoaderX(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=partial(worker_init_fn, num_workers=num_workers, rank=0, seed=0),
            drop_last=True,
            pin_memory=True
        )
    else:
        batch_size = dataset_opt["imgs_per_gpu"]
        return DataLoaderX(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=True
        )


def create_dataset(dataset_opt):
    """create dataset
    IMG: torch.Tensor, RGB, [0, 255]
    """
    name = dataset_opt["name"]
    
    is_train = (dataset_opt["phase"] == "train")

    # data preprocess, for pretraining
    if dataset_opt["is_pretrain"]:
        src_img_size = dataset_opt["src_img_size"]
        fake_img_size = dataset_opt["fake_img_size"]

        assert src_img_size != fake_img_size, print("The image size should be different")

        transform = transforms.Compose([
            PassiveDegrad(src_img_size, fake_img_size),
            SquarePad(),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # get dataset
    if name == "MNIST":
        dataset = datasets.MNIST(root=dataset_opt["dataroot"], train=is_train, transform=transform, download=True)
    elif name == "CIFAR10":
        dataset = datasets.CIFAR10(root=dataset_opt["dataroot"], train=is_train, transform=transform, download=True)
    elif name == "CIFAR100":
        dataset = datasets.CIFAR100(root=dataset_opt["dataroot"], train=is_train, transform=transform, download=True)
    else:
        raise Exception("Unknown dataset: %s" % (name))

    return dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
