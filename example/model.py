import torchvision
import torch.nn as nn


class Mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2()

        self.mobilenet = nn.Sequential(*(list(mobilenet.children())[:-1]))
        self.linear = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.mobilenet(x).squeeze()
        return self.linear(x)