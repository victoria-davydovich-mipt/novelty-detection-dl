import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from configs.globals import *


def build_gan_train_transforms(img_size, n_channels):
    transforms_list = [
        transforms.Resize([img_size] * 2),
        transforms.RandomHorizontalFlip(),
    ]
    if n_channels == 1:
        transforms_list.append(transforms.Grayscale())
    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * n_channels, [0.5] * n_channels),
        ]
    )

    train_transforms = transforms.Compose(transforms_list)

    return train_transforms


class Dataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        batch_size,
        n_workers=0,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def setup(self, stage=None):
        split = "train" if self.train else "test"
        self.dataset = ImageFolder(
            os.path.join(self.root, self.dataset_name, split),
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )


def build_dataloader(dataset_name, transformations, train, batch_size):
    dataset = Dataset(
        dataset_name, transform=transformations, train=train, batch_size=batch_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader
