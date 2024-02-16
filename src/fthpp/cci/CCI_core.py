"""
# Python package for CCI correlation analysis.

CCI Paper Reference: 
Klose, C., Büttner, F., Hu, W. et al. Coherent correlation imaging 
for resolving fluctuating states of matter. Nature 614, 256–261 (2023). 
https://doi.org/10.1038/s41586-022-05537-9

2022-24
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

########## EXTERNAL DEPENDENCIES ##########

import sys, os

# Data
import numpy as np
from numpy.typing import ArrayLike

# Use cupy if there is a gpu
try:
    import cupy as xp

    GPU = cp.is_available()
except ImportError:
    import numpy as xp

    GPU = False

# Progress bar
from tqdm.auto import tqdm

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# scipy
from scipy.ndimage.filters import gaussian_filter

# Clustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances


########## INTERNAL DEPENDENCIES ##########

# self-written libraries
sys.path.append("..")
from core.py import fft2, ifft2

########## HELPER FUNCTIONS ##########


def circle_mask(shape: tuple[int], center: tuple[int], radius, sigma=None) -> ArrayLike:
    """
    Draws circle mask with option to apply gaussian filter for smoothing

    Parameters
    ----------
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter

    Returns
    -------
    mask: array
        binary mask, or smoothed binary mask
    -------
    author: ck 2022
    """

    # setup array
    x = np.linspace(0, shape[1] - 1, shape[1])
    y = np.linspace(0, shape[0] - 1, shape[0])
    X, Y = np.meshgrid(x, y)

    # define circle
    mask = np.sqrt(((X - center[1]) ** 2 + (Y - center[0]) ** 2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if sigma is not None:
        mask = gaussian_filter(mask, sigma)

    return mask


# ======================
# General
# ======================


def parula_cmap():
    """
    rgb values of Matlab 'parula' colormap

    Parameters
    ----------
    None


    Returns
    -------
    cm_data = rgb colormap as nested list
    -------
    author: ck 2022
    """

    cm_data = [
        [0.2081, 0.1663, 0.5292],
        [0.2116238095, 0.1897809524, 0.5776761905],
        [0.212252381, 0.2137714286, 0.6269714286],
        [0.2081, 0.2386, 0.6770857143],
        [0.1959047619, 0.2644571429, 0.7279],
        [0.1707285714, 0.2919380952, 0.779247619],
        [0.1252714286, 0.3242428571, 0.8302714286],
        [0.0591333333, 0.3598333333, 0.8683333333],
        [0.0116952381, 0.3875095238, 0.8819571429],
        [0.0059571429, 0.4086142857, 0.8828428571],
        [0.0165142857, 0.4266, 0.8786333333],
        [0.032852381, 0.4430428571, 0.8719571429],
        [0.0498142857, 0.4585714286, 0.8640571429],
        [0.0629333333, 0.4736904762, 0.8554380952],
        [0.0722666667, 0.4886666667, 0.8467],
        [0.0779428571, 0.5039857143, 0.8383714286],
        [0.079347619, 0.5200238095, 0.8311809524],
        [0.0749428571, 0.5375428571, 0.8262714286],
        [0.0640571429, 0.5569857143, 0.8239571429],
        [0.0487714286, 0.5772238095, 0.8228285714],
        [0.0343428571, 0.5965809524, 0.819852381],
        [0.0265, 0.6137, 0.8135],
        [0.0238904762, 0.6286619048, 0.8037619048],
        [0.0230904762, 0.6417857143, 0.7912666667],
        [0.0227714286, 0.6534857143, 0.7767571429],
        [0.0266619048, 0.6641952381, 0.7607190476],
        [0.0383714286, 0.6742714286, 0.743552381],
        [0.0589714286, 0.6837571429, 0.7253857143],
        [0.0843, 0.6928333333, 0.7061666667],
        [0.1132952381, 0.7015, 0.6858571429],
        [0.1452714286, 0.7097571429, 0.6646285714],
        [0.1801333333, 0.7176571429, 0.6424333333],
        [0.2178285714, 0.7250428571, 0.6192619048],
        [0.2586428571, 0.7317142857, 0.5954285714],
        [0.3021714286, 0.7376047619, 0.5711857143],
        [0.3481666667, 0.7424333333, 0.5472666667],
        [0.3952571429, 0.7459, 0.5244428571],
        [0.4420095238, 0.7480809524, 0.5033142857],
        [0.4871238095, 0.7490619048, 0.4839761905],
        [0.5300285714, 0.7491142857, 0.4661142857],
        [0.5708571429, 0.7485190476, 0.4493904762],
        [0.609852381, 0.7473142857, 0.4336857143],
        [0.6473, 0.7456, 0.4188],
        [0.6834190476, 0.7434761905, 0.4044333333],
        [0.7184095238, 0.7411333333, 0.3904761905],
        [0.7524857143, 0.7384, 0.3768142857],
        [0.7858428571, 0.7355666667, 0.3632714286],
        [0.8185047619, 0.7327333333, 0.3497904762],
        [0.8506571429, 0.7299, 0.3360285714],
        [0.8824333333, 0.7274333333, 0.3217],
        [0.9139333333, 0.7257857143, 0.3062761905],
        [0.9449571429, 0.7261142857, 0.2886428571],
        [0.9738952381, 0.7313952381, 0.266647619],
        [0.9937714286, 0.7454571429, 0.240347619],
        [0.9990428571, 0.7653142857, 0.2164142857],
        [0.9955333333, 0.7860571429, 0.196652381],
        [0.988, 0.8066, 0.1793666667],
        [0.9788571429, 0.8271428571, 0.1633142857],
        [0.9697, 0.8481380952, 0.147452381],
        [0.9625857143, 0.8705142857, 0.1309],
        [0.9588714286, 0.8949, 0.1132428571],
        [0.9598238095, 0.9218333333, 0.0948380952],
        [0.9661, 0.9514428571, 0.0755333333],
        [0.9763, 0.9831, 0.0538],
    ]

    return cm_data


def parula_map():
    """
    Create matplotlib colormap from rgb of Matlab 'parula' colormap

    Parameter
    ---------
    None


    Returns
    -------
    parula: colormap as matplotlib colormap
    -------
    author: ck 2022
    """

    cm_data = parula_cmap()
    parula = LinearSegmentedColormap.from_list("parula", cm_data)

    return parula


def filter_reference(holo: ArrayLike, mask: ArrayLike, diameter: int) -> ArrayLike:
    """
    Filter reference-induced modulations from fth holograms

    Parameters
    ----------
    holo : ArrayLike
        input fourier-transform holography hologram
    mask: array
        (smooth) mask to select cross correlation in Patterson map
    diameter: int
        crops array to size (diameter,diameter) symmetrically around array center

    Returns
    -------
    holo_filtered: ArrayLike
        reference-filtered "hologram"
    -------
    author: CK 2022
    """

    # convert to xp array
    holo = xp.array(holo)
    mask = xp.array(mask)

    # Transform to Patterson map
    tmp_array = xp.array(ifft2(holo))
    center = xp.array(tmp_array.shape) / 2

    # Crop Patterson map
    tmp_array = tmp_array[
        int(center[1] - diameter / 2) : int(center[1] + diameter / 2 + 1),
        int(center[0] - diameter / 2) : int(center[0] + diameter / 2 + 1),
    ]
    tmp_array = tmp_array * mask

    # Crop ROI of holograms
    tmp_array = fft2(tmp_array)
    holo_filtered = tmp_array.real

    # Back to numpy
    if GPU is True:
        holo_filtered = xp.asnumpy(holo_filtered)

    return holo_filtered


def seg_statistics(
    image: ArrayLike, mask: ArrayLike, NrStd: float = 1
) -> tuple[ArrayLike, float, float]:
    """
    Creates mask based on values outside of a noise intervall
    defined by the statistics of the array (intensity band-stop filter)

    Parameters
    ----------
    image : ArrayLike
        input imagegram
    mask : ArrayLike
        Predefined mask to calculate std and mean in roi
    NrStd: scalar
        Multiplication factor of the standard deviation to count a pixel as noise. Default is 1.

    Returns
    -------
    statistics mask: ArrayLike
        bool mask of values larger than noise level
    -------
    author: CK 2022
    """

    # Convert image array
    image = xp.array(image)

    # Flatten and select only non-zero elements
    temp = image[mask == 0]

    # Calc Mean and standard deviation of array
    MEAN = xp.mean(temp)
    STD = xp.std(temp)

    # Create band-stop mask based on image statistics
    statistics_mask = xp.abs(image) >= MEAN + NrStd * STD

    # Convert to numpy
    if GPU is True:
        statistics_mask = xp.asnumpy(statistics_mask)

    return statistics_mask, MEAN, STD


def create_ring_mask(
    shape: tuple[int], center: tuple[int], radi: list[float]
) -> tuple[ArrayLike, ArrayLike:bool]:
    """
    Creates mask of ring sements from given list of radi

    Parameters
    ----------
    shape : int tuple
        shape of output arrays
    radi: list of int
        list of radi in px to create centered rings in q-space radi=[r1,r2,r3,...].

    Returns
    -------
    mask_circ: ArrayLike
        2d array with labeled rings
    masks_ring: ArrayLike (bool)
        3d array containing boolean masks of each ring
    -------
    author: CK 2023
    """

    # Setup array
    mask_circ = np.zeros(shape)

    # Create array with labeled rings
    for radius in radi:
        mask_circ = mask_circ + circle_mask(
            mask_circ.shape, center, radius, sigma="none"
        )
    mask_circ = np.abs(mask_circ - len(radi))
    mask_circ[mask_circ == len(radi)] = 0

    # Create bool mask for each ring
    masks_ring = np.zeros((len(radi) - 1, shape[0], shape[1]), dtype=bool)
    for i in range(0, len(radi) - 1):
        masks_ring[i] = mask_circ == i + 1

    return mask_circ, masks_ring


def correlate_holograms(
    diff1: ArrayLike,
    diff2: ArrayLike,
    sum1: ArrayLike,
    sum2: ArrayLike,
    Statistics1: ArrayLike,
    Statistics2: ArrayLike,
) -> tuple[float, ArrayLike]:
    """
    Function to determine the heterodyne pearson cross-correlation
    of two difference holograms.

    Parameters
    ----------
    diff1 : ArrayLike
        difference hologram of the first data image
    diff2 : ArrayLike
        difference hologram of the second data image
    sum1: ArrayLike
        sum hologram corresponding to first data image to flatten
    sum2: ArrayLike
        sum hologram corresponding to second data image to flatten

    Returns
    -------
    c_val : scalar
        correlation value of the two holograms
    c_array: ArrayLike
        pixelwise correlation array of the two holograms
    -------
    author: CK 2022
    """

    # Convert to used array format
    diff1 = xp.array(diff1)
    diff2 = xp.array(diff2)
    sum1 = xp.array(sum1)
    sum2 = xp.array(sum2)
    Statistics1 = xp.array(Statistics1)
    Statistics2 = xp.array(Statistics2)

    # replace all zeros in sum1/sum2 with another value to avoid infinities
    sum1[sum1 == 0] = 1e-8
    sum2[sum2 == 0] = 1e-8

    # Combine Statistics Mask
    mask = xp.logical_or(Statistics1, Statistics2).astype(float)

    # Calc flattened holos called scattering images
    S1 = diff1 * mask / xp.sqrt(sum1)
    S2 = diff2 * mask / xp.sqrt(sum2)

    # normalization Factor
    norm = xp.sqrt(xp.sum(S1 * S1) * xp.sum(S2 * S2))

    # calculate the pixelwise correlation
    c_array = S1 * S2 / norm

    # average correlation
    c_val = xp.sum(c_array)

    return c_val, c_array


def correlation_map_with_valid_pixels(
    image_stack: ArrayLike, valid_pixel_mask: ArrayLike
) -> ArrayLike:
    """
    Function to calculate pearson cross-correlation map of 3d array along first axis.
    Considers binary masks which define valid pixels of images

    Parameters
    ----------
    image_stack : ArrayLike
        d1xd2xd3 array (d1: nr images, d2,d3: shape of single images)

    valid_pixel_mask : ArrayLike
        bool array which defines valid pixels for each image of image_stack.
        same dimensions as image stack

    Returns
    -------
    corr_map : ArrayLike
        correlation map where every image of image_stack is correlatetd with all other images
    -------
    author: CK 2022
    """

    # Convert to array
    image_stack = xp.array(image_stack, dtype=xp.float32)
    valid_pixel_mask = xp.array(valid_pixel_mask, dtype=xp.float32)

    # predefine array
    corr_map = xp.eye(image_stack.shape[0], dtype=xp.float32)

    # Varies first holo
    for ii in tqdm(range(image_stack.shape[0])):
        # Get holos and statistics of complete set for index ii
        holo_1 = image_stack[ii]
        statistics_1 = valid_pixel_mask[ii]

        # Get holo mask for all other holos
        holo_2 = image_stack[ii + 1 :]

        # Get combined statistics mask
        mask = xp.logical_or(statistics_1, valid_pixel_mask[ii + 1 :])

        # Apply mask
        holo_1 = holo_1 * mask
        holo_2 = holo_2 * mask

        # normalization Factor
        norm = xp.sqrt(
            xp.sum(holo_1 * holo_1, axis=(1, 2)) * xp.sum(holo_2 * holo_2, axis=(1, 2))
        )

        # Correlation array
        corr_array = (holo_1 * holo_2) / norm[:, None, None]
        corr_map[ii, ii + 1 :] = xp.sum(corr_array, axis=(1, 2))

    # Use symmetry to fill corr map
    corr_map = corr_map + xp.rot90(xp.fliplr(corr_map))
    corr_map = corr_map - xp.eye(corr_map.shape[0])

    # Convert to numpy
    if GPU is True:
        corr_map = xp.asnumpy(corr_map)

    return corr_map


def correlation_map(image_stack: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Function to calculate pearson cross-correlation map of array

    Parameters
    ----------
    image_stack : d1xd2 cupy array (d1: nr images, d2: image array (1d or 2d))
        array of all scattering images (stack)

    Returns
    -------
    corr_map : cupy array
        correlation map where every image is correlatetd with each other image
    -------
    author: CK 2023
    """

    # Convert to cupy
    image_stack = xp.array(image_stack)

    # Check array dimension
    if image_stack.ndim == 3:
        image_stack = image_stack.reshape(
            image_stack.shape[0], image_stack.shape[1] * image_stack.shape[2]
        )
    elif image_stack.ndim > 3:
        raise ValueError("Array shape does not match requirements!")

    # predefine array
    size = (image_stack.shape[0], image_stack.shape[0])
    corr_map_nonorm, corr_map_pearson, corr_map_sutton = (
        xp.zeros(size, dtype=xp.float32),
        xp.zeros(size, dtype=xp.float32),
        xp.zeros(size, dtype=xp.float32),
    )

    # Pure Multiplication
    corr_map_nonorm = (
        xp.dot(image_stack, image_stack.T) / image_stack.shape[1]
    )  # Averaged value

    # Calc correlation function (Sutton)
    # Normalization
    mean_counts = xp.mean(image_stack, axis=1)
    mean_counts = mean_counts.reshape(1, mean_counts.shape[0])
    norm = xp.dot(mean_counts.T, mean_counts)  # (nxn) Normalization array

    # Calc Corr
    corr_map_sutton = corr_map_nonorm / norm

    # Calc correlation function (Pearson)
    # Normalization
    cross_corr = xp.diag(corr_map_nonorm)
    cross_corr = cross_corr.reshape(1, cross_corr.shape[0])
    norm = xp.sqrt(xp.dot(cross_corr.T, cross_corr))  # (nxn) Normalization array

    # Calc corr
    corr_map_pearson = corr_map_nonorm / norm

    # Convert to numpy
    if GPU is True:
        corr_map_nonorm = xp.asnumpy(corr_map_nonorm)
        corr_map_pearson = xp.asnumpy(corr_map_pearson)
        corr_map_sutton = xp.asnumpy(corr_map_sutton)

    return corr_map_nonorm, corr_map_pearson, corr_map_sutton


def reconstruct_correlation_map(frames: ArrayLike, corr_array: ArrayLike) -> ArrayLike:
    """
    Selects smaller section of a large correlation map based on direct
    array indexing

    Parameters
    ----------
    frames: ArrayLike
        frame indexes
    corr_array: ArrayLike
        complete pair correlation map

    Returns
    -------
    corr: array
        section of the (complete) correlation map
    -------
    author: CK 2021
    """

    print(f"Reconstructing correlation map... (%d frames)" % len(frames))

    # Reshape frame array
    frames = np.reshape(frames, frames.shape[0])

    # Indexing of correlation array
    corr = corr_array[np.ix_(frames, frames)]

    return corr


def create_linkage(
    cluster_idx: int,
    corr_array: ArrayLike,
    metric: str = "correlation",
    order: int = 1,
    plot: bool = True,
) -> tuple[ArrayLike, ArrayLike]:
    """
    calculates distance metric, linkage and feedback plots

    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
    metric: str or callable, default = 'correlation'
        Metric to use when calculating distance metric from which clustering
        linkage will be calculated. Must be one of the options
        allowed by sklearn.metrics.pairwise_distances.
    order: int
        how many times the distance metric should be applied before the linkage
        will be calculated. order == 0 means there won't be any distance metric
        and the linkage will be calculated from the initial correlation map.
        order == 1 is the default case. order == 2 calculated the distance of the
        distance metric etc.
    plot: bool
        Enable/disable feedback plots
    Returns
    -------
    tlinkage: array
        clustering linkage array
    -------
    author: CK 2021
    """

    # Calc distance metric
    dist_metric = corr_array.copy()

    # Calculate higher orders of distance metrics
    if order > 0:
        for n in range(1, order + 1):
            dist_metric = pairwise_distances(dist_metric, metric=metric, n_jobs=-1)

    # Calculate Linkage
    tlinkage = linkage(dist_metric, method="average", metric=metric)

    # Output plots
    if plot is True:
        nr_cluster = 2
        temp_assignment = fcluster(tlinkage, nr_cluster, criterion="maxclust")

        fig, _ = plt.subplots(figsize=(8, 8))
        fig.suptitle(f"Cluster Index: {cluster_idx}")

        # Dist metric
        ax1 = plt.subplot(2, 2, 1)
        vmi, vma = np.percentile(dist_metric[dist_metric >= 1e-5], [1, 99])
        ax1.imshow(dist_metric, vmin=vmi, vmax=vma, cmap=parula_map(), aspect="auto")
        ax1.set_title("Distance metric")
        ax1.set_xlabel("Frame index k")
        ax1.set_ylabel("Frame index k")

        # Corr map
        ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
        vmi, vma = np.percentile(corr_array[corr_array <= 1 - 1e-5], [5, 95])
        ax2.imshow(corr_array, vmin=vmi, vmax=vma, cmap=parula_map(), aspect="auto")
        ax2.set_title("Correlation map")
        ax2.set_xlabel("Frame index k")
        ax2.set_ylabel("Frame index k")
        plt.gca().invert_yaxis()

        # Assignment plot
        ax3 = plt.subplot(2, 2, 3, sharex=ax1)
        ax3.plot(temp_assignment)
        ax3.set_title("Frame assignment")
        ax3.set_xlabel("Frame index k")
        ax3.set_ylabel("State")
        ax3.set_ylim((0.5, 2.5))
        ax3.set_yticks([1, 2])

        # Assignment plot
        ax4 = plt.subplot(2, 2, 4)
        dendrogram(tlinkage, p=100, truncate_mode="lastp")

    return tlinkage, dist_metric


def cluster_hierarchical(
    tlinkage: ArrayLike, parameter: float, clusteringOption: str = "maxclust"
) -> ArrayLike:
    """
    calculates distance metric, linkage and feedback plots

    Parameters
    ----------
    tlinkage: ArrayLike
        The hierarchical clustering encoded with the matrix returned by the linkage function
    parameter: scalar
        parameter of clustering option, e.g., nr of clusters or inconsistency
        threshold
    clusteringOption: string
        criterion used in forming flat clusters
        - 'inconsistent': cluster inconsistency threshold
        - 'maxcluster': number of total clusters

    Returns
    -------
    cluster_assignment: ArrayLike
        assignment of frames/elements to cluster
    -------
    author: CK 2021
    """

    # Options
    if clusteringOption == "maxclust":
        criterion_ = "maxclust"
    elif clusteringOption == "inconsistent":
        criterion_ = "inconsistent"
    else:
        raise ValueError("Error: clustering option not valid!")

    # Get cluster
    cluster_assignment = fcluster(tlinkage, parameter, criterion=criterion_)

    return cluster_assignment


def clustering_feedback(
    cluster_idx: int,
    nr: int,
    corr_array_large: ArrayLike,
    corr_array_small: ArrayLike,
    dist_metric_sq: ArrayLike,
    tlinkage: ArrayLike,
):
    """
    Creates feedback plots of clustering: Section of initial correlation map
    assigned to cluster, new reduced correlation map, distance metric,
    dendrogram of linkage tree

    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    nr: int
        index of subcluster
    corr_array_large: ArrayLike
        initial pair correlation map
    corr_array_small: ArrayLike
        pair correlation map of new subcluster
    dist_metric_sq: ArrayLike
        distance metric of pair correlation map in square format
    tlinkage: ArrayLike
        clustering linkage array

    Returns
    -------
    fig with plots
    -------
    author: CK 2021
    """

    # get colomap
    parula = parula_map()

    # Output plots
    fig, _ = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"Cluster Index: {cluster_idx}-{nr}")

    # section of Initial Corr map
    ax1 = plt.subplot(2, 2, 1)
    vmi, vma = np.percentile(corr_array_large[corr_array_large != 1], [5, 95])
    ax1.imshow(corr_array_large, vmin=vmi, vmax=vma, cmap=parula, aspect="auto")
    ax1.set_title("Section initial correlation map")
    ax1.set_xlabel("Frame index k")
    ax1.set_ylabel("Frame index k")
    plt.gca().invert_yaxis()

    # section of Initial Corr map
    ax2 = plt.subplot(2, 2, 2)
    vmi, vma = np.percentile(corr_array_small[corr_array_small != 1], [5, 95])
    ax2.imshow(corr_array_small, vmin=vmi, vmax=vma, cmap=parula, aspect="auto")
    ax2.set_title("New correlation map")
    ax2.set_xlabel("Frame index k")
    ax2.set_ylabel("Frame index k")

    # Dist metric
    ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
    vmi, vma = np.percentile(dist_metric_sq[dist_metric_sq != 0], [1, 99])
    ax3.imshow(dist_metric_sq, vmin=vmi, vmax=vma, cmap=parula, aspect="auto")
    ax3.set_title("Distance metric")
    ax3.set_xlabel("Frame index k")
    ax3.set_ylabel("Frame index k")
    plt.gca().invert_yaxis()

    # Assignment plot
    ax4 = plt.subplot(2, 2, 4)
    dendrogram(tlinkage, p=150, truncate_mode="lastp")

    plt.tight_layout()
    return


def process_cluster(
    cluster: list,
    cluster_idx: int,
    corr_array: ArrayLike,
    cluster_assignment: ArrayLike,
    order: int = 1,
    metric: str = "correlation",
    save: bool = False,
    plot: bool = True,
) -> list:
    """
    processes a given cluster assignment and adds new subclusters to
    'cluster'-dictionary

    Parameters
    ----------
    cluster: list of dictionaries
        stores relevant data of clusters, e.g., assigned frames
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: ArrayLike
        pair correlation map
    cluster_assignment: ArrayLike
        assignment of frames to cluster
    save: bool
        save new subclusters in "cluster"-list and delete current cluster from list
    plot: bool
        enable or disable feedback plots
    Returns
    -------
    cluster: list of dicts
        updated "cluster"-list
    -------
    author: CK 2022
    """

    length = len(cluster)

    # Get initial frames in cluster
    frames = cluster[cluster_idx]["Cluster_Frames"]
    frames = np.reshape(frames, frames.shape[0])

    # Get nr of new subclusters
    nr = np.unique(cluster_assignment)

    # Vary subclusters
    for ii in nr:
        print(f"Creating sub-cluster: {cluster_idx}-{ii}")

        # Get assignment
        tmp_assignment = np.argwhere(cluster_assignment == ii)
        tmp_assignment = np.reshape(tmp_assignment, tmp_assignment.shape[0])

        # Get subcluster correlation array
        tmp_corr_small = corr_array[np.ix_(tmp_assignment, tmp_assignment)]

        # Create mask which selects the section of the correlation that is assigned to sub-cluster ii
        tmp_mask = np.zeros([cluster_assignment.shape[0], cluster_assignment.shape[0]])
        tmp_mask[np.ix_(tmp_assignment, tmp_assignment)] = corr_array[
            np.ix_(tmp_assignment, tmp_assignment)
        ]
        tmp_corr_large = tmp_mask

        if plot is True:
            if len(tmp_assignment) > 1:
                # Calculate Linkage
                tlinkage, dist_metric = create_linkage(
                    cluster_idx, corr_array, metric=metric, order=order, plot=False
                )
                # Plots
                clustering_feedback(
                    cluster_idx,
                    ii,
                    tmp_corr_large,
                    tmp_corr_small,
                    dist_metric,
                    tlinkage,
                )
        # Save new cluster
        if save == True:
            print(f"Saving subcluster {cluster_idx}-{ii} as new cluster {length + ii}")
            cluster.append(
                {
                    "Cluster_Nr": length + ii,
                    "Cluster_Frames": frames[np.ix_(tmp_assignment)],
                }
            )

    # Del old cluster from 'cluster'-list
    if save == True:
        cluster[cluster_idx] = {}

    return cluster
