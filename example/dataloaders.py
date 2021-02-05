from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader


def get_dataloaders(batch_size):
    train_dataset = CIFAR10('data', transform=torchvision.transforms.ToTensor(), train=True, download=True)
    valid_dataset = CIFAR10('data', transform=torchvision.transforms.ToTensor(), train=False, download=True)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, valid_dataloader
