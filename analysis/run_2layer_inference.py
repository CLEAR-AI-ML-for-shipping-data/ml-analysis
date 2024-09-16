import argparse
import os
import tomllib

from loguru import logger
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from sklearn import cluster

# Provide these to the namespace for the read models
from astromorph.astromorph.src.byol import ByolTrainer
from astromorph.astromorph.src.datasets import FilelistDataset
from astromorph.astromorph.src.settings import InferenceSettings


from voyage_dataset import VoyageDataset, VoyageFilelistDataset

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
from voyage_dataset import VoyageFilelistDataset
from models import CoastalVoyageModel


def normalize_image(image: torch.Tensor):
    """Ensure that an image has pixel values between 0 and 1

    Args:
        image: an image to be normalized
    """
    # image -= image.min()
    # image /= image.max()
    image *= 255
    return image


def create_thumbnail(image: torch.Tensor, thumbnail_size: int):
    # make sure the image is square
    # only use the unaugmented image
    # image = pad_image_to_square(image[0])
    image = np.array(image[0])
    image = np.concatenate([image, np.zeros((1, 256, 256))], axis=0)
    image = resize(np.array(image), (3, thumbnail_size, thumbnail_size))
    image = torch.from_numpy(image)[None]
    # image = torch.flip(image, [1, 2])
    return image


def main(
    dataset: FilelistDataset,
    model_name: str,
    export_embeddings: bool = False,
):
    """Run the inference.

    Args:
        dataset: dataset on which to run inference
        model_name: filename of the trained neural network
        export_embedding: whether to export embeddings
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = "cpu"
    logger.info("Using device {}", device)

    # images is a list of tensors with shape (1, 3, width, height)
    images = [image.to(device) for image in dataset.get_all_items()]

    # Loading model
    logger.info(f"Loading pretrained model {model_name}...")

    learner = torch.load(model_name)
    learner.eval()
    learner.to(device)

    logger.info("Calculating embeddings...")
    with torch.no_grad():
        dummy_embeddings = learner(images[0]) #, return_embedding=True)
        embeddings_dim = dummy_embeddings.shape[1]
        embeddings = torch.empty((0, embeddings_dim)).to(device)
        for image in tqdm(images):
            emb = learner(image) #, return_embedding=True)
            embeddings = torch.cat((embeddings, emb), dim=0)

    logger.info("Clustering embeddings...")
    clusterer = cluster.KMeans(n_clusters=10)
    cluster_labels = clusterer.fit_predict(embeddings.cpu())

    logger.info("Producing thumbnails...")
    plot_images = [normalize_image(image.cpu()) for image in images]

    # If thumbnails are too large, TensorBoard runs out of memory
    thumbnail_size = 81
    thumbnail_size = 39
    thumbnail_size = 215

    resized = [create_thumbnail(image, thumbnail_size) for image in plot_images]

    # Concatenate thumbnails into a single tensor for labelling the embeddings
    all_ims = torch.cat(resized)

    # Remove directory names, and remove the extension as well
    model_basename = os.path.basename(model_name).split(".")[0]
    writer = SummaryWriter(log_dir=f"runs/{model_basename}/")

    # If the data is stored in FITS files, retrieve extra metadata
    if isinstance(dataset, FilelistDataset):
        # Retrieve object name, RA, dec, rest frequency, and the filename
        names = dataset.get_object_property("OBJECT")
        right_ascension = dataset.get_object_property("OBSRA")
        declination = dataset.get_object_property("OBSDEC")
        rest_freq = dataset.get_object_property("RESTFRQ")
        filenames = dataset.filenames
        labels = list(
            zip(
                cluster_labels,
                names,
                right_ascension,
                declination,
                rest_freq,
                filenames,
            )
        )

        headers = [
            "cluster",
            "object",
            "right ascension",
            "declination",
            "rest freq",
            "filepath",
        ]

    else:
        labels = cluster_labels
        headers = None

    writer.add_embedding(
        embeddings, label_img=all_ims, metadata=labels, metadata_header=headers
    )

    if export_embeddings:
        exportdir = "exported/"
        if not os.path.exists(exportdir):
            os.mkdir(exportdir)

        if headers is None:
            headers = ["cluster_label"]
            labels = list(cluster_labels)

        embedding_columns = [f"emb_dim_{i}" for i in range(embeddings.shape[1])]
        df_embeddings = pd.DataFrame(columns=embedding_columns, data=embeddings.cpu())
        df_metadata = pd.DataFrame(columns=headers, data=labels)
        df_export = pd.concat([df_metadata, df_embeddings], axis=1)
        df_export.to_csv(f"exported/{model_basename}.csv", sep=";")


if __name__ == "__main__":
    # Options can either be provided by command line arguments, or a config file
    # Options from the command line will override those from the config file
    parser = argparse.ArgumentParser(
        prog="Astromorph pipeline", description=None, epilog=None
    )
    parser.add_argument("-d", "--datafile", help="Define a data file")
    parser.add_argument("-m", "--maskfile", help="Specify a mask file")
    parser.add_argument("-n", "--trained_network_name", help="Saved network model")
    parser.add_argument("-c", "--configfile", help="Specify a config file")
    args = parser.parse_args()

    # If there is a config file, load those settings first
    # Otherwise, only use settings from the command line
    if args.configfile:
        overriding_settings = vars(args)
        configfile = overriding_settings.pop("configfile")
        with open(configfile, "rb") as file:
            config_dict = tomllib.load(file)
        # Overwrite the config file settings with command line settings
        for key, value in overriding_settings.items():
            if value is not None:
                config_dict.update({key: value})
    else:
        config_dict = vars(args)

    # Use InferenceSettings to validate settings
    settings = InferenceSettings(**config_dict)

    logger.info("Reading data")
    dataset = VoyageFilelistDataset(settings.datafile, **(settings.data_settings))

    main(dataset, settings.trained_network_name, settings.export_to_csv)
