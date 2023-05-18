import json
import argparse

import torch
from f_ano_gan.dataset import build_dataloader

from model_training.dataset import build_gan_train_transforms
from model_training.models import Discriminator, Generator


from configs.globals import *


import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class WGANGPLightning(pl.LightningModule):
    def __init__(self, generator, discriminator, dataloader, lambda_gp=10):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.lambda_gp = lambda_gp

    def configure_optimizers(self):
        optimizer_G = Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        optimizer_D = Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return [optimizer_G, optimizer_D], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        device = self.device

        # Configure input
        real_imgs = imgs.to(device)

        # Train Discriminator
        if optimizer_idx == 0:
            optimizer_D = self.optimizers()[optimizer_idx]
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], self.latent_dim, device=device)

            # Generate a batch of images
            fake_imgs = self.generator(z)

            # Real images
            real_validity = self.discriminator(real_imgs)
            # Fake images
            fake_validity = self.discriminator(fake_imgs.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                self.discriminator, real_imgs.data, fake_imgs.data, device
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.lambda_gp * gradient_penalty
            )

            d_loss.backward()
            optimizer_D.step()
            self.log("d_loss", d_loss)

        # Train Generator
        if optimizer_idx == 1:
            optimizer_G = self.optimizers()[optimizer_idx]
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], self.latent_dim, device=device)

            # Generate a batch of images
            fake_imgs = self.generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = self.discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()
            self.log("g_loss", g_loss)

    def train_dataloader(self):
        return self.dataloader

    def train_wgangp(self, opt):
        self.lr = opt.lr
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.latent_dim = opt.latent_dim

        trainer = pl.Trainer(
            max_epochs=opt.n_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar_refresh_rate=1,
        )
        trainer.fit(self)

        os.makedirs("results/images", exist_ok=True)
        z = torch.randn(25, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z)

        torch.save(self.generator.state_dict(), "results/generator")
        torch.save(self.discriminator.state_dict(), "results/discriminator")


def main(options):
    torch.manual_seed(options.seed)

    transforms = build_gan_train_transforms(options.img_size, options.channels)
    train_dataloader = build_dataloader(
        options.dataset, transforms, train=True, batch_size=options.batch_size
    )
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    wganp_lightning = WGANGPLightning(
        opt, generator, discriminator, train_dataloader, DEVICE
    )

    trainer = pl.Trainer()
    trainer.fit(wganp_lightning)


if __name__ == "__main__":
    with open("configs/datasets.json") as f:
        dataset_records = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(dataset_records),
        help="name of the dataset",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=300, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img-size", type=int, default=64, help="size of each image dimension"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="number of image channels (If set to 1, convert image to grayscale)",
    )
    parser.add_argument(
        "--n-critic",
        type=int,
        default=5,
        help="number of training steps for " "discriminator per iter",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=400, help="interval betwen image samples"
    )
    parser.add_argument("--seed", type=int, default=None, help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
