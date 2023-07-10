from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from lightning.pytorch import LightningDataModule


class CLIC(LightningDataModule):
    def __init__(self, data_dir, num_workers, pin_memory, persistent_workers,
                 train_batch_size, train_patch_size,
                 valid_batch_size, valid_patch_size,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_batch_size = train_batch_size
        self.train_transform = transforms.Compose([transforms.RandomCrop(train_patch_size), transforms.ToTensor()])

        self.valid_batch_size = valid_batch_size
        self.valid_transform = transforms.Compose([transforms.CenterCrop(valid_patch_size), transforms.ToTensor()])

    def setup(self, stage):
        self.clic_train = ImageFolder(root=self.data_dir, split='train', transform=self.train_transform)
        self.clic_valid = ImageFolder(root=self.data_dir, split='valid', transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.clic_train, batch_size=self.train_batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.clic_valid, batch_size=self.valid_batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
