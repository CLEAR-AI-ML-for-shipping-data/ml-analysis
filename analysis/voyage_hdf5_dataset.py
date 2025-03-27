import h5py
import torch

from astromorph.astromorph.src.datasets.base_dataset import BaseDataset
from make_thicker_line_plots import convolve_image


class VoyageHDF5Dataset(BaseDataset):
    """Dataset for when voyage data is stored in an HDF5 file.

    Attributes:
        filename: name of the HDF5 archive
    """

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        # Store filenames in attribute to quickly look up filename belonging
        # to a certain index. Otherwise this needs to be done with list
        # comprehension in the __getitem__ method.
        with h5py.File(self.filename) as file:
            self.filenames = [key for key in file.keys()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        with h5py.File(self.filename) as file:
            # Use the [()] subscription to retrieve an individual data item
            # See h5py documentation for more information
            image = file[self.filenames[index]][()]

        image[0] = convolve_image(image[0], kernel_size=3)
        image = torch.from_numpy(image).float()
        image = image[None, :, :, :]
        images = self.augment_image(image)

        return images

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
        with h5py.File(self.filename) as file:
            all_items = [
                torch.from_numpy(file[a_name][()]).float()[None, :, :, :]
                for a_name in self.filenames
            ]
        return all_items
