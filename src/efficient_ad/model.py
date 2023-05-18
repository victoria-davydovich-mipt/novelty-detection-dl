import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Autoencoder(pl.LightningModule):
    def __init__(self, out_channels: int = 384):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(size=3, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=8, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=15, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=32, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=63, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=127, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Upsample(size=56, mode="bilinear"),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        reconstruction = self.forward(x)
        loss = F.mse_loss(reconstruction, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
