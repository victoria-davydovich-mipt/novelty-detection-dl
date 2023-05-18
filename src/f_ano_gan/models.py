import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


from configs.globals import *


class Generator(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.opt.lr)


class Discriminator(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = nn.Sequential(
            self.discriminator_block(opt.channels, 16, bn=False),
            self.discriminator_block(16, 32),
            self.discriminator_block(32, 64),
            self.discriminator_block(64, 128),
        )
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1))

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        return features

    def discriminator_block(self, in_filters, out_filters, bn=True):
        block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        if bn:
            block.add_module("batch_norm", nn.BatchNorm2d(out_filters, 0.8))
        return block

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.opt.lr)


class Encoder(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = nn.Sequential(
            self.encoder_block(opt.channels, 16, bn=False),
            self.encoder_block(16, 32),
            self.encoder_block(32, 64),
            self.encoder_block(64, 128),
        )
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, opt.latent_dim), nn.Tanh()
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity

    def encoder_block(self, in_filters, out_filters, bn=True):
        block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        if bn:
            block.add_module("batch_norm", nn.BatchNorm2d(out_filters, 0.8))
        return block

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.opt.lr)
