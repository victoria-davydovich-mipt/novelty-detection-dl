from configs.globals import *

import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import pytorch_lightning as pl


class EncoderIZIFLightning(pl.LightningModule):
    def __init__(self, generator, discriminator, encoder, dataloader, kappa=1.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.dataloader = dataloader
        self.kappa = kappa

    def configure_optimizers(self):
        optimizer_E = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return optimizer_E

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        device = self.device

        # Configure input
        real_imgs = imgs.to(device)

        # Train Encoder
        optimizer_E = self.optimizers()
        optimizer_E.zero_grad()

        # Generate a batch of latent variables
        z = self.encoder(real_imgs)

        # Generate a batch of images
        fake_imgs = self.generator(z)

        # Real features
        real_features = self.discriminator.forward_features(real_imgs)
        # Fake features
        fake_features = self.discriminator.forward_features(fake_imgs)

        # izif architecture
        criterion = nn.MSELoss()
        loss_imgs = criterion(fake_imgs, real_imgs)
        loss_features = criterion(fake_features, real_features)
        e_loss = loss_imgs + self.kappa * loss_features

        e_loss.backward()
        optimizer_E.step()
        self.log("e_loss", e_loss)

    def train_dataloader(self):
        return self.dataloader

    def train_encoder_izif(self, opt):
        self.lr = opt.lr
        self.b1 = opt.b1
        self.b2 = opt.b2

        trainer = pl.Trainer(
            max_epochs=opt.n_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar_refresh_rate=1,
        )
        trainer.fit(self)

        os.makedirs("results/images_e", exist_ok=True)
        z = self.encoder(
            self.generator(
                torch.randn(25, self.generator.latent_dim, device=self.device)
            )
        )
        reconfiguration_imgs = self.generator(z)
        save_image(
            reconfiguration_imgs.data,
            "results/images_e/final.png",
            nrow=5,
            normalize=True,
        )
        torch.save(self.encoder.state_dict(), "results/encoder")
