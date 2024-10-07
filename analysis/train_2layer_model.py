import datetime as dt

import torch
from astromorph.astromorph.src.byol import ByolTrainer, MinMaxNorm
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import CoastalVoyageModel
from voyage_dataset import VoyageFilelistDataset

# full_dataset = VoyageFilelistDataset("multilayer_minsize_8h_images.txt")
# full_dataset = VoyageFilelistDataset("triple_layer_minsize_4h_images.txt")
data_file = "data/triple_layer_minsize_4h_images_full.txt"

start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")
logfile = f"logs/train_model_{start_time}.log"
logger.add(f"{logfile}")

logger.info(f"Writing logs to {logfile}")
logger.info(f"Using input data from {data_file}")

full_dataset = VoyageFilelistDataset(data_file)

rng = torch.Generator().manual_seed(42)  # seeded RNG for reproducibility
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [0.8, 0.2], generator=rng
)

# DataLoaders have batch_size=1, because images have different sizes
train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

augmentation_function = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=(0, 360)),
    # T.RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
    T.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ),
)

normalization_function = MinMaxNorm()

model = CoastalVoyageModel()
trainer = ByolTrainer(
    network=model,
    augmentation_function=augmentation_function,
    normalization_function=normalization_function,
)


trainer.train_model(
    train_data=train_data,
    test_data=test_data,
    epochs=2,
    save_file=f"trained_model_{start_time}.pt",
    log_dir=f"runs/multilayer_{start_time}",
)
