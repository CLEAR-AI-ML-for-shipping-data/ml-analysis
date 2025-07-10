import numpy as np
from scipy.signal import convolve


def convolve_image(image: np.ndarray, kernel_size=5):
    image_kernel = np.ones((kernel_size, kernel_size))
    if image.ndim == 3:
        for i in range(image.shape[-1]):
            image[:, :, i] = convolve(
                image[:, :, i], image_kernel, mode="same", method="direct",
            )
    else:
        image = convolve(image, image_kernel, mode="same", method="direct")
    return image
