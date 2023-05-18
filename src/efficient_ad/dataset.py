from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index: int) -> tuple:
        sample, _ = super().__getitem__(index)
        return sample


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index: int) -> tuple:
        path, _ = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


def InfiniteDataLoader(loader: DataLoader) -> iter:
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.dataset = ImageFolderWithPath(self.data_dir, transform=ToTensor())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
