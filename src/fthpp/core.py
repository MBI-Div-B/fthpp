from functools import partial
import logging
log = logging.getLogger(__name__)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# log.addHandler(ch)

import numpy as np
from numpy.typing import ArrayLike
import scipy.fft as fft

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    fft.set_global_backend(cufft)
    GPU = cp.is_available()
    log.info("CUDA GPU available.")
except ImportError:
    log.warning("Could not import cupy module (is it installed?). "
                "Proceeding with CPU support only.")
    GPU = False
except Exception as ex:
    log.warning(f"Error determining GPU availability: {ex}. "
                "Proceeding with CPU support only.")
    GPU = False

if GPU:
    from cupyx.scipy.ndimage import fourier_shift
    _fftopts = {}
else:
    from scipy.ndimage import fourier_shift
    _fftopts = dict(workers=-1)


def crop_center(image: ArrayLike, center: tuple[int], square=True) -> ArrayLike:
    """Return a symmetric crop around given center coordinate.
    
    Parameters
    ----------
    image : Input image
    center : Coordinate of center pixel
    square : bool, optional, default=True
        Make cropped area square by using the smallest common size for
        all dimensions.
    """
    m0, m1 = [int(min(c, s - c)) for s, c in zip(image.shape, center)]
    if square:
        m0 = m1 = min(m0, m1)
    c0, c1 = [int(c) for c in center]
    return image[c0 - m0:c0 + m0, c1 - m1:c1 + m1]


def pad_for_fft(image: ArrayLike, fill_value=0) -> ArrayLike:
    """Zeropad image to next FFT-efficient shape.

    Uses scipy.fft.next_fast_len() to calculate new shape and pads with constant
    values accordingly. Causes half-pixel misalignments for axes with odd length.
    
    Parameters
    ----------
    image : input image to be zeropadded

    Returns
    -------
    padded_image : zero-padded image
    """
    fastshape = [fft.next_fast_len(s) for s in image.shape]
    quot_rem = [divmod(fs - s, 2) for fs, s in zip(fastshape, image.shape)]
    pad = [[q, q + r] for q, r in quot_rem]
    return np.pad(image, pad, mode="constant", constant_values=fill_value)


def shift_image(image: ArrayLike, shift: tuple[float]) -> ArrayLike:
    """Cyclically shifts image with sub-pixel precision using Fourier shift.
    
    Parameters
    ----------
    image: image to be shifted by shift vector
    shift: (x, y) translation in px
    
    Returns
    -------
    image_shifted: Shifted image
    -------
    author: CK 2021
    """
    shift_image = fourier_shift(fft.fft2(image, **_fftopts), shift)
    shift_image = fft.ifft2(shift_image, **_fftopts)
    return shift_image.real


def fft2(image: ArrayLike) -> ArrayLike:
    """Return 2D inverse Fourier transform of image."""
    return fft.ifftshift(fft.fft2(fft.fftshift(image), **_fftopts))

def ifft2(image: ArrayLike) -> ArrayLike:
    """Return 2D Fourier transform of image."""
    return fft.fftshift(fft.ifft2(fft.ifftshift(image), **_fftopts))

