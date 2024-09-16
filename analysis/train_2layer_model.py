import datetime as dt
import pickle
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from astromorph.astromorph.src.byol import ByolTrainer
from astromorph.astromorph.src.models import AstroMorphologyModel
from voyage_dataset import VoyageDataset, VoyageFilelistDataset


class CoastalVoyageModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-5),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        for name, child in self.named_children():
            if name != "fc":
                x = child(x)
            else:
                x = child(torch.flatten(x, start_dim=1))


full_dataset = VoyageFilelistDataset("multilayer_minsize_8h_images.txt")

rng = torch.Generator().manual_seed(42)  # seeded RNG for reproducibility
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [0.8, 0.2], generator=rng
)

# DataLoaders have batch_size=1, because images have different sizes
train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

augmentation_function = torch.nn.Sequential(
    # T.RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
    # T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=(0, 360)),
    # T.RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
    T.Normalize(
        mean=torch.tensor([0.485, 0.456]), 
                           # 0.406]),
        std=torch.tensor([0.229, 0.224]), 
                          # , 0.225]),
    ),
)

model = CoastalVoyageModel()
trainer = ByolTrainer(network=model, augmentation_function=augmentation_function)

start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")

trainer.train_model(
    train_data=train_data,
    test_data=test_data,
    epochs=2,
    save_file=f"trained_model_{start_time}.pt",
    log_dir=f"runs/multilayer_{start_time}"
)
