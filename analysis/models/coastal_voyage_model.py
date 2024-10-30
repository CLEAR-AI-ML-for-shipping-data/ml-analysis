import torch
from torch import nn


class CoastalVoyageModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dim_1: int = 32,
        dim_2: int = 32,
        dim_3: int = 32,
        dim_4: int = 64,
        dim_5: int = 128,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim_1,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1 = nn.BatchNorm2d(dim_1, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_2, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_3, eps=1e-5),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim_4, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_4, dim_5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_5, eps=1e-5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=dim_5, out_features=64)

    def forward(self, x):
        for name, child in self.named_children():
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, start_dim=1))
