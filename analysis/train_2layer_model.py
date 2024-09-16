import datetime as dt

import torch
from astromorph.astromorph.src.byol import ByolTrainer
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import CoastalVoyageModel
from voyage_dataset import VoyageFilelistDataset

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
