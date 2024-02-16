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

from scipy.optimize import curve_fit
import scipy.constants as cst

import matplotlib as plt

# Image registration
from skimage.registration import phase_cross_correlation
from dipy.align.transforms import AffineTransform2D, TranslationTransform2D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    fft.set_global_backend(cufft)
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


def image_registration(
    image_unproccessed: ArrayLike,
    image_background: ArrayLike,
    method: str = "phase_cross_correlation",
    static_mask: ArrayLike = None,
    moving_mask: ArrayLike = None,
    roi: ArrayLike = None,
) -> ArrayLike:
    """
    Calculates image translation between two images with sub-pixel precision
    through image registration. Currently, there are two different methods
    implemented which can be selected via methods.

    - 'phase_cross_correlation' is a fast and efficient subpixel image
    translation registration by cross-correlation. However, is lacks
    precision in case of sub-pixel translations. Ref:
    (https://scikit-image.org/docs/stable/api/skimage.registration.html)
    - 'dipy' applies the affine registration methods provided by the package
    dipy. The method is already setup with tested parameter. Can also process
    regions of interests defined by a mask.
    (https://docs.dipy.org/stable/examples_built/registration/affine_registration_masks)

    Parameters
    ----------
    image_unproccessed: ArrayLike
        Moving image, will be aligned with respect to image_background

    image_background: ArrayLike
        static reference image

    method: str
        Method used for image registration. Either 'phase_cross_correlation'
        or 'dipy'

    static_mask: ArrayLike
        ignore masked pixel in static image for calculation of shift. Works
        only if method == 'dipy'

    moving_mask: ArrayLike
        ignore masked pixel in moving image for calculation of shift. Works
        only if method == 'dipy'

    Returns
    -------
    shift: ArrayLike
        shift np.array(dy,dx)
    -------
    author: CK 2022/23
    """

    # Different method to calc image registration
    if method == "phase_cross_correlation":
        # Calculate Shift
        shift, error, diffphase = phase_cross_correlation(
            image_background, image_unproccessed, upsample_factor=100
        )
    elif method == "dipy":
        # Define your metric
        nbins = 32
        sampling_prop = None  # all pixels
        metric = MutualInformationMetric(nbins, sampling_prop)  # Gaussian pyramide

        # What is your transformation type?
        transform = TranslationTransform2D()

        # How many iterations per level?
        level_iters = [10000, 1000, 100]

        # Smoothing of each level
        sigmas = [2.0, 1.0, 0.0]

        # Subsampling
        factors = [2, 1, 1]

        # Bring it together
        affreg = AffineRegistration(
            metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors
        )

        # Calc your transformation
        affine = affreg.optimize(
            image_background,
            image_unproccessed,
            transform,
            static_mask=static_mask,
            moving_mask=moving_mask,
            params0=None,
        )

        # Take only translation from affine transformation
        shift = np.array([affine.get_affine()[0, 2], affine.get_affine()[1, 2]])

    else:
        raise ValueError("Method not support!")

    return np.round(shift, 2)


def scalar_norm(arr1, arr2):
    """Calculate normalized sum of element-wise product for two arrays."""
    # this einstein sum expression is equivalent to, but faster than,
    # (arr1 * arr2).sum() / (arr2 * arr2).sum()
    return np.einsum("ij, ij", arr1, arr2) / np.einsum("ij, ij", arr2, arr2)


def func_lin(x, a, b):
    """Basic linear function for fitting"""
    return a * x + b


def fitting_norm(
    arr1: ArrayLike, arr2: ArrayLike, plot: bool = True
) -> tuple[float, float]:
    """
    Find normalization factor, offset by fitting pixel values of two arrays.

    Parameters
    ----------
    arr1: array
        first array

    arr2: array
        reference array

    plot : bool
        Plot fit

    Returns
    -------
    factor: scalar
        Intensity correction factor
    offset: scalar
        Intensity offset
    -------
    author: CK 2023
    """

    # Create y, x data
    xdata = np.concatenate(arr2)
    ydata = np.concatenate(arr1)

    # Ignore all x,y = 0 values, e.g., if a mask is used
    ignore = np.logical_or((np.abs(xdata) <= 1e-5), (np.abs(ydata) <= 1e-5))
    xdata = xdata[np.argwhere(ignore == False)]
    ydata = ydata[np.argwhere(ignore == False)]

    xdata = np.squeeze(xdata, axis=1)
    ydata = np.squeeze(ydata, axis=1)

    # curve fit cannot handle cupy arrays
    if GPU is True:
        xdata, ydata = cp.asnumpy(xdata), cp.asnumpy(ydata)

    # Fitting
    popt, pcov = curve_fit(func_lin, xdata, ydata)
    factor = popt[0]
    offset = popt[1]

    # Enable plotting
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(xdata, ydata, s=5)
        ax.plot(xdata, func_lin(xdata, *popt), "r-")
        ax.set_xlabel("Intensity Ref")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Linear Fit: {factor:0.4f}*x + {offset:0.4f}")

    return factor, offset


def normalize(
    image: ArrayLike,
    image_ref: ArrayLike,
    method: str = "scalar_norm",
    plot: bool = False,
) -> tuple[float, float]:
    """
    Calculates intensity normalization factor between two images


    Parameters
    ----------
    image: array
        first image

    image_ref: array
        reference image

    method: str
        Method for calculating scaling factor (scalarproduct,correlation)

    plot : bool
        Plot fit if method is correlation

    Returns
    -------
    factor: scalar
        Intensity correction factor
    offset: scalar
        Intensity offset
    -------
    author: CK 2023
    """

    if method == "scalar_norm":
        factor = scalar_norm(image, image_ref)
        offset = 0

    elif method == "correlation":
        factor, offset = fitting_norm(image, image_ref, plot=plot)

    else:
        raise ValueError("Method not support!")

    return factor, offset


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
