#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from model import (
    get_autoencoder,
    get_pdn_small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
    InfiniteDataloader,
)
from sklearn.metrics import roc_auc_score
from pytorch_lightning import LightningModule, Trainer


class EfficientAD(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = 42
        self.on_gpu = torch.cuda.is_available()
        self.out_channels = 384
        self.image_size = 256
        self.default_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )
        self.teacher = None
        self.student = None
        self.autoencoder = None
        self.teacher_mean = None
        self.teacher_std = None

    def setup(self, stage):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        config = self.config

        if config.dataset == "mvtec_ad":
            dataset_path = config.mvtec_ad_path
        elif config.dataset == "mvtec_loco":
            dataset_path = config.mvtec_loco_path
        else:
            raise Exception("Unknown config.dataset")

        pretrain_penalty = True
        if config.imagenet_train_path == "none":
            pretrain_penalty = False

        # create output dir
        train_output_dir = os.path.join(
            config.output_dir, "trainings", config.dataset, config.subdataset
        )
        test_output_dir = os.path.join(
            config.output_dir, "anomaly_maps", config.dataset, config.subdataset, "test"
        )
        os.makedirs(train_output_dir)
        os.makedirs(test_output_dir)

        # load data
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, "train"),
            transform=transforms.Lambda(self.train_transform),
        )
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, "test")
        )
        if config.dataset == "mvtec_ad":
            # mvtec dataset paper recommend 10% validation set
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(self.seed)
            train_set, validation_set = torch.utils.data.random_split(
                full_train_set, [train_size, validation_size], rng
            )
        elif config.dataset == "mvtec_loco":
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, "validation"),
                transform=transforms.Lambda(self.train_transform),
            )
        else:
            raise Exception("Unknown config.dataset")

        self.train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
