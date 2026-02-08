import pickle
from typing import List, Union

import numpy as np
import torch
from astromorph.datasets.base_dataset import BaseDataset
from astromorph.datasets.helpers import augment_image, make_4D


class VoyageDataset(BaseDataset):
    def __init__(self, image_file: str, stacksize: int = 1, *args, **kwargs) -> None:
        super().__init__(stacksize, *args, **kwargs)
        with open(image_file, "rb") as file:
            images: List[np.ndarray] = pickle.load(file=file)
            self.images: List[torch.Tensor] = [
                torch.from_numpy(image).float() for image in images
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        images = augment_image(image, stacksize=self.stacksize)
        return images

    def get_all_items(self):
        return [make_4D(image, stacksize=self.stacksize) for image in self.images]


class VoyageFilelistDataset(BaseDataset):
    """Dataset for when images are stored in a list of files.

    This way you can load distributed image data into a model or DataLoader.

    """

    def __init__(self, filelist: Union[str, List[str]]):
        super().__init__()
        self.input_file = filelist
        if isinstance(filelist, list):
            self.filenames: List[str] = filelist
        else:
            with open(filelist, "r") as file:
                # Make sure to remove the newline characters at the end of each filename
                self.filenames = [fname.strip("\n") for fname in file.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        image = self.read_numpy_array(self.filenames[index])
        image = image[None, :, :, :]
        images = self.augment_image(image)

        return images

    def read_numpy_array(self, filename):
        with open(filename, "rb") as file:
            image = pickle.load(file)
        return torch.from_numpy(image).float()

    def augment_image(self, image: torch.Tensor):
        im_e = image
        im_c = torch.rot90(im_e, k=2, dims=(2, 3))
        im_b = torch.flip(im_e, dims=(2, 3))
        im_bc = torch.rot90(im_b, k=2, dims=(2, 3))

        # Concatenate along axis 0 to produce a tensor of shape (4, 3, W, H)
        images = torch.concatenate(
            (
                im_e,
                im_c,
                im_b,
                im_bc,
            ),
            0,
        )

        return images

    def get_all_items(self):
        return [self.read_numpy_array(image)[None, :, :, :] for image in self.filenames]
