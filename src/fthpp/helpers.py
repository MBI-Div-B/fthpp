import numpy as np
from numpy.typing import ArrayLike

# Image registration
from skimage.registration import phase_cross_correlation
from dipy.align.transforms import AffineTransform2D, TranslationTransform2D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

# scipy
from scipy.stats import linregress


def image_registration_phase_cross_correlation(
    image_background: ArrayLike,
    image_unproccessed: ArrayLike,
) -> ArrayLike:
    """
    Calculates image translation between two images with sub-pixel precision
    using the phase_cross_correlation for image registration. 'phase_cross_correlation'
    is a fast and efficient subpixel image translation registration from scikit-image.
    However, is lacks precision in case of sub-pixel translations. Ref:
    (https://scikit-image.org/docs/stable/api/skimage.registration.html)

    Parameters
    ----------
    image_unproccessed: ArrayLike
        Moving image, will be aligned with respect to image_background

    image_background: ArrayLike
        static reference image

    Returns
    -------
    shift: ArrayLike
        shift np.array(dy,dx)
    -------
    author: CK 2022/23
    """

    # Calculate Shift
    shift, error, diffphase = phase_cross_correlation(
        image_background, image_unproccessed, upsample_factor=100
    )

    return np.round(shift, 2)


def image_registration_dipy(
    image_background: ArrayLike,
    image_unproccessed: ArrayLike,
    static_mask: ArrayLike = None,
    moving_mask: ArrayLike = None,
    roi: ArrayLike = None,
) -> ArrayLike:
    """
    Calculates image translation between two images with sub-pixel precision
    the affine registration methods provided by the package
    dipy. The method is already setup with tested parameter. Can also process
    regions of interests defined by a mask.
    (https://docs.dipy.org/stable/examples_built/registration/affine_registration_masks)

    Parameters
    ----------
    image_unproccessed: ArrayLike
        Moving image, will be aligned with respect to image_background

    image_background: ArrayLike
        static reference image

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
    author: CK 2023
    """

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

    return np.round(shift, 2)


def scalar_norm(arr1, arr2):
    """Calculate normalized sum of element-wise product for two arrays."""
    # this einstein sum expression is equivalent to, but faster than,
    # (arr1 * arr2).sum() / (arr2 * arr2).sum()
    return np.einsum("ij, ij", arr1, arr2) / np.einsum("ij, ij", arr2, arr2)


def match_intensity_linreg(
    arr1: ArrayLike, arr2: ArrayLike, roi=None
) -> tuple[float, float]:
    """
    Find intensity normalization factor, offset by fitting pixel values of two arrays.

    Parameters
    ----------
    arr1: array
        first array

    arr2: array
        reference array

    Returns
    -------
    factor: scalar
        Intensity scaling factor
    offset: scalar
        Intensity offset
    -------
    author: CK 2023
    """

    if roi is None:
        roi = np.s_[()]

    # Create y, x data
    xdata = arr2[roi].flatten()
    ydata = arr1[roi].flatten()

    # Ignore all x,y = 0 values, e.g., if a mask is used
    ignore = np.logical_or((np.abs(xdata) <= 1e-5), (np.abs(ydata) <= 1e-5))
    xdata = xdata[np.argwhere(ignore == False)]
    ydata = ydata[np.argwhere(ignore == False)]

    xdata = np.squeeze(xdata, axis=1)
    ydata = np.squeeze(ydata, axis=1)

    # Calc lin regression
    reg = linregress(xdata, ydata)
    factor = getattr(reg, "slope")
    offset = getattr(reg, "intercept")

    return factor, offset
