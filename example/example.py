from pytorch_trainer import PytorchTrainer
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import numpy as np


def get_dataloaders():
    train_dataset = CIFAR10('data', transform=torchvision.transforms.ToTensor(), train=True, download=True)
    valid_dataset = CIFAR10('data', transform=torchvision.transforms.ToTensor(), train=False, download=True)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,
        batch_size=32,
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=32,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, valid_dataloader


class Mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2()

        self.mobilenet = nn.Sequential(*(list(mobilenet.children())[:-1]))
        self.linear = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.mobilenet(x).squeeze()
        return self.linear(x)


def minibatch_fn(iteration, minibatch, models, optimizers,
                 grad_clip_thresh, train):
    x, y = minibatch

    model = models['model']
    y_pred = model(x)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    results = {'loss': loss.item()}

    if train:
        optimizer = optimizers['model']
        optimizer.zero_backward_step(loss, grad_clip_thresh)

    return results


if __name__ == '__main__':
    train_dataloader, valid_dataloader = get_dataloaders()
    model = Mobilenet()

    trainer = PytorchTrainer(
        model_names=['model'],
        model_list=[model],
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        minibatch_fn=minibatch_fn,
        total_iterations=1000,
        iterations_per_validation=200,
    )

    trainer.run()
