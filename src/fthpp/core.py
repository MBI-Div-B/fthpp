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
import scipy.constants as cst


try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    GPU = cp.is_available()
    log.info("CUDA GPU available.")
except ImportError:
    log.warning(
        "Could not import cupy module (is it installed?). "
        "Proceeding with CPU support only."
    )
    GPU = False
except Exception as ex:
    log.warning(
        f"Error determining GPU availability: {ex}. "
        "Proceeding with CPU support only."
    )
    GPU = False

if GPU:
    from cupyx.scipy.ndimage import fourier_shift

    fft.set_global_backend(cufft)
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
        Use smallest common length for all dimensions.
    """
    m0, m1 = [int(min(c, s - c)) for s, c in zip(image.shape, center)]
    if square:
        m0 = m1 = min(m0, m1)
    c0, c1 = [int(c) for c in center]
    return image[c0 - m0 : c0 + m0, c1 - m1 : c1 + m1]


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


# TODO: 
def propagate(
    holo: ArrayLike,
    prop_l: float,
    experimental_setup: dict,
    integer_wl_multiple: bool = True,
) -> ArrayLike:
    """
    Propagate the hologram

    Parameters
    ----------
    holo : array
        input hologram
    prop_l: scalar
        distance of propagation in metre
    experimental_setup: dict
        experimental setup parameters in the following form: {'ccd_dist': [in metre], 'energy': [in eV], 'px_size': [in metre]}
    integer_wl_multiple: bool, optional
        Use a propagation, that is an integer multiple of the x-ray wave length, default is True.

    Returns
    -------
    prop_holo: array
        propagated hologram
    -------
    author: MS 2016
    """
    wl = cst.h * cst.c / (experimental_setup["energy"] * cst.e)
    if integer_wl_multiple:
        prop_l = np.round(prop_l / wl) * wl

    l1, l2 = holo.shape
    q0, p0 = [s / 2 for s in holo.shape]  # centre of the hologram
    q, p = np.mgrid[0:l1, 0:l2]  # grid over CCD pixel coordinates
    pq_grid = (q - q0) ** 2 + (
        p - p0
    ) ** 2  # grid over CCD pixel coordinates, (0,0) is the centre position
    dist_wl = 2 * prop_l * np.pi / wl
    phase = dist_wl * np.sqrt(
        1
        - (experimental_setup["px_size"] / experimental_setup["ccd_dist"]) ** 2
        * pq_grid
    )
    return np.exp(1j * phase) * holo


def shift_phase(arr: ArrayLike, phase: float) -> ArrayLike:
    """
    Multiply complex-valued arr with a global phase

    Parameters
    ----------
    arr: ArrayLike
        input array
    phase: float
        complex-phase angle in rad

    Returns
    -------
    phase_shifted: ArrayLike
        phase-shifted array
    -------
    """
    return arr * np.exp(1j * phase)


def binning(arr: ArrayLike, binning_factor: int) -> ArrayLike:
    """
    Bins images: new_shape = old_shape/binning_factor

    Parameter
    =========
    array : ArrayLike
        input array
    binning_factor : int
        new_shape = old_shape/binning_factor

    Output
    ======
    new_array : ArrayLike
        binned array
    ======
    author: ck 2023

    """
    new_shape = (np.array(arr.shape) / binning_factor).astype(int)

    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )

    new_array = arr.reshape(shape).mean(-1).mean(1)
    return new_array
