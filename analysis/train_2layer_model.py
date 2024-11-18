import argparse
import datetime as dt
import pprint
import tomllib

import torch
from astromorph.astromorph.src.byol import ByolTrainer, MinMaxNorm
from loguru import logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as T

from models import CoastalVoyageModel
from settings import TrainingSettings
from voyage_dataset import VoyageFilelistDataset
from voyage_hdf5_dataset import VoyageHDF5Dataset


def main(dataset: VoyageFilelistDataset, train_settings: TrainingSettings):

    start_time = dt.datetime.now().strftime("%Y%m%d_%H%M")
    logfile = f"logs/train_model_{start_time}.log"
    logger.info(f"Writing logs to {logfile}")
    logger.add(f"{logfile}")

    logger.info(
        "Starting training with settings:"
        + f"\n{pprint.pformat(train_settings.model_dump())}",
    )

    rng = torch.Generator().manual_seed(42)  # seeded RNG for reproducibility
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=rng
    )

    # DataLoaders have batch_size=1, because images have different sizes
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

    augmentation_function = torch.nn.Sequential(
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 360)),
        # T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
        # T.Normalize(
        #     mean=torch.tensor([0.485, 0.456, 0.406]),
        #     std=torch.tensor([0.229, 0.224, 0.225]),
        # ),
    )

    normalization_function = MinMaxNorm()

    lr_scheduler = ExponentialLR if train_settings.exponential_lr is True else None

    model = CoastalVoyageModel(**train_settings.network_settings)
    trainer = ByolTrainer(
        network=model,
        augmentation_function=augmentation_function,
        normalization_function=normalization_function,
        representation_size=train_settings.network_settings.get("dim_5", 128),
        lr_scheduler=lr_scheduler,
        lr_scheduler_options={"gamma": train_settings.gamma},
        learning_rate=train_settings.learning_rate,
    )

    trainer.train_model(
        train_data=train_data,
        test_data=test_data,
        epochs=train_settings.epochs,
        save_file=f"trained_model_{start_time}.pt",
        log_dir=f"runs/multilayer_{start_time}",
        batch_size=train_settings.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CLEAR training script")

    parser.add_argument(
        "-c", "--configfile", help="Specify a configfile", required=True
    )


    args = parser.parse_args()

    # Overriding settings are used to overwrite settings from the configfile
    overriding_settings = vars(args)
    configfile = overriding_settings.pop("configfile")
    with open(configfile, "rb") as file:
        config_dict = tomllib.load(file)
    # Overwrite the config file settings with command line settings
    for key, value in overriding_settings.items():
        if value is not None:
            config_dict.update({key: value})

    settings = TrainingSettings(**config_dict)

    if settings.core_limit:
        torch.set_num_threads(settings.core_limit)

    # dataset = VoyageFilelistDataset(settings.datafile)
    dataset = VoyageHDF5Dataset(settings.datafile)
    main(dataset, settings)
