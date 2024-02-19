import numpy as np
from numpy.typing import ArrayLike

# Image registration
from skimage.registration import phase_cross_correlation
from dipy.align.transforms import AffineTransform2D, TranslationTransform2D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration


# TODO: separate functions, magic numbers, rounding
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


# TODO: define in fit function or use lambda function
def func_lin(x, a, b):
    """Basic linear function for fitting"""
    return a * x + b

# TODO: name, separate out plot part (generalized), use linregress?
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

# TODO: do we need this? very general name
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