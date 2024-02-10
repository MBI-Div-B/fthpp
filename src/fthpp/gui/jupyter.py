########## EXTERNAL DEPENDENCIES ##########

import sys

import numpy as np

from numpy.typing import ArrayLike
from inspect import ismethod

import ipywidgets
from ipywidgets import FloatRangeSlider, FloatSlider, interact, IntSlider
from IPython.core.display_functions import display

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

from scipy.ndimage import gaussian_filter


########## INTERNAL DEPENDENCIES ##########

sys.path.append("..")
from core import reconstruct, shift_image, propagate, shift_phase

########## HELPER FUNCTIONS ##########


def cp_to_np(A: ArrayLike) -> ArrayLike:
    """
    Converts any array-like object to a numpy array. Specifically used to convert seamlessly between numpy and cupy arrays.

    Parameters
    ----------
    A : ArrayLike
        Array to convert

    Returns
    -------
    ArrayLike
        Converted numpy array
    """
    try:
        return np.array(A.get())
    except Exception:
        return np.array(A)


def circle_mask(shape, center, radius, sigma=None):
    """
    Draws circle mask with option to apply gaussian filter for smoothing

    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter

    Output
    ======
    mask: array
        binary mask, or smoothed binary mask
    ======
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


def axis_to_roi(axis: plt.Axes, labels=None) -> dict:
    """
    Generate numpy slice expression from bounds of matplotlib figure axis.

    If labels is not None, return a roi dictionary for xarray.

    Parameters
    ----------
    axis : plt.Axes

    labels : _type_, optional
        , by default None

    Returns
    -------
    roi: dict

    """
    x0, x1 = sorted(axis.get_xlim())
    y0, y1 = sorted(axis.get_ylim())
    if labels is None:
        roi = np.s_[int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))]
    else:
        roi = {
            labels[0]: slice(int(round(y0)), int(round(y1))),
            labels[1]: slice(int(round(x0)), int(round(x1))),
        }
    return roi


def intensity_scale():
    "DUMMY. Do we need this function?"
    pass


def stack_animate(
    fig: plt.Figure, artists: list, datas: list, fname: str = None, fps: int = 5
):
    """
    Animates an existing matplotlib figure with arbitrary layout by replacing any targeted "artist" with a matching entry-slice in "datas".

    An "artist" can be the handle of a line-plot, an imshow, a text-object or even the method of a matplotlib object (e.g., ax.set_xlim or plot.set_color).
    A "data"-entry must be a list of data-sets (e.g. 2d-numpy arrays for imshow) matching the corresponding artist. All data-sets in "datas" must be at least as long as datas[0].

    If no fname is given, creates a drop-down widget to change the viewed data-slice live.
    If fname is given, a GIF-file is created with the supplied data at the specified location andfps.

    Parameters
    ----------
    fig : plt.Figure
        figure object to manipulate
    artists : list
        Objects to animate in fig
    datas : list
        Changing data of artists
    fname : str, optional
        Specify a save location to create a gif, by default None
    fps : int, optional
        frame per second of gif, by default 5
    """

    if len(artists) != len(datas):
        print(
            "Number of animated artists does not match number of supplied arrays"
        )  # LOG!
        return

    i_max = len(datas[0])

    def update(index=0):
        for artist, data in zip(artists, datas):
            if ismethod(artist):
                print(artist, data[index])
                artist(*data[index])
            elif type(artist) == matplotlib.text.Text:
                artist.set_text(data[index])
            elif type(artist) == matplotlib.image.AxesImage:
                artist.set_data(data[index])
            elif type(artist) == matplotlib.lines.Line2D:
                data = np.array(data)
                if len((data).shape) == 3:
                    if data.shape[0] == 2:
                        artist.set_data(data[0, index], data[1, index])
                    elif data.shape[1] == 2:
                        artist.set_data(data[index, 0], data[index, 1])
                else:
                    artist.set_ydata(data[index])

        return

    if fps is None or fname is None:
        ipywidgets.interact(
            update,
            index=range(i_max),
        )
    else:
        # put gif export here...
        plt.ioff()
        ani = animation.FuncAnimation(
            fig, update, repeat=True, frames=i_max, interval=1000 / fps
        )
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(
            fps=fps, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(fname, writer=writer)
        plt.ion()

    return


########## INTERACTIVE JUPYTER WIDGETS ##########


def cimshow(im: ArrayLike, ax=None, **kwargs):
    """
    Simple 2d image plot with adjustable contrast.

    If no axis object is given, creates and returns new matplotlib figure and axis. Otherwise, fills given axis environment and returns imshow-object.

    Parameters
    ----------
    im : ArrayLike
        2 dim. real-valued array.
    ax : matpltolib axis, optional
        matplotlib axis object to insert the image in, by default None

    Returns
    -------
    fig,ax:
        new matplotlib figure and axis -or-
    imhandle:
        new imshow object created in ax
    """
    im = cp_to_np(im)
    new_fig = False
    if ax is None:
        new_fig = True
        fig, ax = plt.subplots(figsize=(7, 7))
    im0 = im[0] if len(im.shape) == 3 else im
    imhandle = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [0.1, 99.9, 0.001, 99.999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax),
        min=vmin,
        max=vmax,
        step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width="500px"),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        imhandle.set_clim(contrast)

    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)

        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            imhandle.set_data(im[nr])

    if new_fig:
        return fig, ax
    else:
        return imhandle


class InteractiveCenter:
    """
    Plot image with controls for contrast and beamstop alignment tools.
    """

    def __init__(
        self, im: ArrayLike, c0: int = None, c1: int = None, rBS: int = 15, **kwargs
    ):
        """_summary_

        Parameters
        ----------
        im : ArrayLike
            2d array, contains image data of which to determine center
        c0 : int, optional
            starting center coordinate y, defaults to the center of im, by default None
        c1 : int, optional
            starting center coordinate x, by default None
        rBS : int, optional
            starting beamstop radius, by default 15
        """
        im = cp_to_np(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]

        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2

        self.c0 = c0
        self.c1 = c1
        self.rBS = rBS

        self.circles = []
        for i in range(5):
            color = "g" if i == 1 else "r"
            circle = plt.Circle([c0, c1], 10 * (i + 1), ec=color, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

        w_c0 = ipywidgets.IntText(value=c0, step=0.5, description="c0 (vert)")
        w_c1 = ipywidgets.IntText(value=c1, step=0.5, description="c1 (hor)")
        w_rBS = ipywidgets.IntText(value=rBS, description="rBS")

        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1, r=w_rBS)

    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))


class InteractiveBeamstop:
    """
    Plot image with controls for image plotting contrast beamstop parameter.
    Use to find best radi and smoothing values.
    """

    def __init__(
        self,
        im: ArrayLike,
        c0: float = None,
        c1: float = None,
        rBS: int = 60,
        stdBS: int = 4,
        **kwargs
    ):
        """
        Parameters
        ----------
        im : ArrayLike
            2d real valued array
        c0 : float, optional
            starting center coordinate y, defaults to the center of im, by default None
        c1 : float, optional
            starting center coordinate x, defaults to the center of im, by default None
        rBS : int, optional
            radius of beamstop, by default 60
        stdBS : int, optional
            smoothing of beamstop edge (higher is smoother), by default 4
        """

        # Parameter coordinates
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        self.center = [c0, c1]

        # Beamstop parameter
        self.rBS = rBS
        self.stdBS = stdBS

        # Create beamstop mask
        im = cp_to_np(im)
        self.im = im
        self.mask_bs = 1 - circle_mask(
            im.shape, self.center, self.rBS, sigma=self.stdBS
        )
        self.image = np.array(im * self.mask_bs)

        # Plotting
        fig, ax = plt.subplots()
        self.mm = ax.imshow(self.image)
        cmin, cmax, vmin, vmax = np.nanpercentile(im, [0.1, 99, 0.1, 99.9])
        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )
        cim = ipywidgets.interact(self.update_plt, contrast=sl_contrast)

        # Change beamstop parameter
        w_rBS = ipywidgets.IntText(value=self.rBS, description="radius")
        w_std = ipywidgets.IntText(value=self.stdBS, description="smoothing")
        ipywidgets.interact(self.update_bs, r=w_rBS, std=w_std)

    # Update plot
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    # Update bs
    def update_bs(self, r, std):
        self.rBS = r
        self.stdBS = std
        self.mask_bs = 1 - circle_mask(self.mask_bs.shape, self.center, r, sigma=std)
        self.image = self.im * self.mask_bs
        self.mm.set_data(self.image)


class DrawPolygonMask:
    """Interactive drawing of polygon masks"""

    def __init__(self, image: ArrayLike):
        self.image = image
        self.image_plot = image
        self.full_mask = np.zeros(image.shape)
        self.coordinates = []
        self.masks = []
        self._create_widgets()
        self.draw_gui()

    def _create_widgets(self):
        self.button_add = ipywidgets.Button(
            description="Add mask",
            button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_add.on_click(self.add_mask)

        self.button_del = ipywidgets.Button(
            description="Delete last mask",
            # button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_del.on_click(self.del_mask)

    def draw_gui(self):
        """Create plot and control widgets"""

        # Plotting
        fig, self.ax = plt.subplots(figsize=(8, 8))
        self.mm = self.ax.imshow(self.image_plot)
        # self.overlay = self.ax.imshow(self.full_mask, alpha=0.2)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.1, 99, 0.1, 99.9])

        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )
        cim = ipywidgets.interact(self.update_plt, contrast=sl_contrast)

        # How to use
        print("Click on the figure to create a polygon corner.")
        print("Click `Add mask` to store coordinates and apply mask.")
        print("Press the 'esc' key to reset the polygon for new drawing.")
        print("")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")
        print("Button `Delete last mask` deletes the masks recursively.")

        self.reset_polygon_selector()
        self.output = ipywidgets.Output()
        display(self.button_add, self.button_del, self.output)

    # Update plot
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def reset_polygon_selector(self):
        self.selector = PolygonSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="r", linestyle="-", linewidth=2, alpha=0.9),
        )

    def create_polygon_mask(self, shape, coordinates):
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(coordinates)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        self.masks.append(mask)
        self.coordinates.append(coordinates)

    def combine_masks(self):
        if len(self.masks) == 0:
            self.full_mask = np.zeros(self.image.shape)
        if len(self.masks) == 1:
            self.full_mask = self.masks[0]
        elif len(self.masks) > 1:
            self.full_mask = np.sum(np.array(self.masks).astype(int), axis=0)

        self.full_mask[self.full_mask > 1] = 1

    def add_mask(self, change):
        self.create_polygon_mask(self.image.shape, self.selector.verts)
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)

    def del_mask(self, change):
        self.coordinates.pop()
        self.masks.pop()
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)


class InteractiveCircleCoordinates:
    """
    Creates overlay with circles on an image. Slider allow changing
    between circles, adjust circle position and radi. Usefull for
    creating support mask for holographically aided phase retrieval

    Return list of tuples with mask parameters (center, radius)
    """

    masks = []

    def __init__(self, image: ArrayLike, num_masks: int, coordinates: list = None):
        print("Right click to move circle to mouse position!")
        self.image = image
        self.num_masks = num_masks
        self.init_masks(coordinates)
        self.draw_gui()

    def init_masks(self, coordinates):
        if coordinates is None:
            coordinates = []
            for n in range(self.num_masks):
                coordinates.append(
                    [self.image.shape[0] / 2, self.image.shape[1] / 2, 10]
                )

        self.masks = [
            plt.Circle(
                (coordinates[n][1], coordinates[n][0]),
                coordinates[n][2],
                fill=False,
                ec="r",
            )
            for n in range(self.num_masks)
        ]

    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.mm = self.ax.imshow(self.image, vmin=vmin, vmax=vmax, cmap="gray")
        for mask in self.masks:
            self.ax.add_artist(mask)

        self.widgets = {
            "contrast": ipywidgets.FloatRangeSlider(
                value=(vmin, vmax),
                min=cmin,
                max=cmax,
                step=(vmax - vmin) / 500,
                layout=ipywidgets.Layout(width="500px"),
            ),
            "mask_index": ipywidgets.IntSlider(min=0, max=self.num_masks - 1, value=0),
            "radius": ipywidgets.FloatSlider(
                min=0,
                max=400,
                value=10,
                step=0.5,
                description="radius",
                layout=ipywidgets.Layout(width="350px"),
            ),
            "c0": ipywidgets.FloatSlider(
                min=0,
                max=2048,
                value=1024,
                step=0.5,
                description="x",
                layout=ipywidgets.Layout(width="400px"),
            ),
            "c1": ipywidgets.FloatSlider(
                min=0,
                max=2048,
                value=1024,
                step=0.5,
                description="y",
                layout=ipywidgets.Layout(width="400px"),
            ),
        }

        ipywidgets.interact(self.update_plt, contrast=self.widgets["contrast"])
        ipywidgets.interact(self.update_controls, index=self.widgets["mask_index"])
        ipywidgets.interact(
            self.update_circle,
            radius=self.widgets["radius"],
            c0=self.widgets["c0"],
            c1=self.widgets["c1"],
        )
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)

    # Update imshow plot colormap
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def update_controls(self, index):
        """Update control widget values with selected circle parameters."""
        circle = self.masks[index]
        r, (c0, c1) = circle.radius, circle.center

        self.widgets["radius"].value = r
        self.widgets["c0"].value = c0
        self.widgets["c1"].value = c1

        for c in self.masks:
            c.set_edgecolor("g")
        self.masks[index].set_edgecolor("r")

    def update_circle(self, radius, c0, c1):
        """Set center and size of active circle."""
        index = self.widgets["mask_index"].value
        self.masks[index].set_radius(radius)
        self.masks[index].set_center([c0, c1])

        print("Aperture Coordinates:")
        print(self.get_mask())

    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        index = self.widgets["mask_index"].value
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            self.masks[index].set_center([c0, c1])
            self.widgets["c0"].value = c0
            self.widgets["c1"].value = c1

    def get_mask(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return [
            (np.round(c.center[1], 1), np.round(c.center[0], 1), np.round(c.radius, 1))
            for c in self.masks
        ]


class InteractiveOptimizer:
    """
    Interactively adjust FTH parameters: center, propagation and phase shift.
    """

    params = {
        "phase": 0,
        "center": (0, 0),
        "propdist": 0,
        "pixelsize": 13.5e-6,
        "energy": 779,
        "detectordist": 0.2,
    }
    widgets = {}

    def __init__(self, hologram: ArrayLike, roi, params: dict = {}):
        """
        Parameters
        ----------
        hologram : ArrayLike
            2d array of hologram to reconstruct
        roi : _type_
            _description_
        params : dict, optional
            Starting reconstruction parameters, by default {"phase": 0, "center": (0, 0), "propdist": 0, "pixelsize": 13.5e-6,
              "energy": 779, "detectordist": 0.2}
        """
        self.params.update(params)
        self.holo = hologram  # .astype(np.single)
        self.holo_centered = hologram.copy()
        self.holo_prop = hologram.copy()
        self.roi = roi

        self.make_ui()

    def make_ui(self):
        self.fig, (self.axr, self.axi) = plt.subplots(
            ncols=2,
            figsize=(7, 3.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        self.reco = reconstruct(self.holo)[self.roi]
        vmin, vmax = np.percentile(self.reco.real, [0.01, 99.9])
        vlim = 2 * np.abs(self.reco.real).max()

        opt = dict(vmin=vmin, vmax=vmax, cmap="gray_r")
        self.mm_real = self.axr.imshow(self.reco.real, **opt)
        self.mm_imag = self.axi.imshow(self.reco.imag, **opt)

        self.widgets["clim"] = FloatRangeSlider(
            value=(vmin, vmax),
            min=-vlim,
            max=vlim,
        )
        self.widgets["phase"] = FloatSlider(
            value=self.params["phase"],
            min=-np.pi,
            max=np.pi,
        )
        self.widgets["c0"] = FloatSlider(
            value=self.params["center"][0], min=-5, max=5, step=0.01
        )
        self.widgets["c1"] = FloatSlider(
            value=self.params["center"][1], min=-5, max=5, step=0.01
        )
        self.widgets["propdist"] = FloatSlider(
            value=self.params["propdist"], min=-10, max=10, step=0.1
        )
        self.widgets["energy"] = ipywidgets.BoundedFloatText(
            value=self.params["energy"],
            min=1,
            max=10000,
        )
        self.widgets["detectordist"] = ipywidgets.BoundedFloatText(
            value=self.params["detectordist"], min=0.01
        )
        self.widgets["pixelsize"] = ipywidgets.BoundedFloatText(
            value=self.params["pixelsize"],
            min=1e-7,
        )

        interact(self.update_clim, clim=self.widgets["clim"])
        interact(self.update_phase, phase=self.widgets["phase"])
        interact(self.update_center, c0=self.widgets["c0"], c1=self.widgets["c1"])
        interact(
            self.update_propagation,
            dist=self.widgets["propdist"],
            det=self.widgets["detectordist"],
            pxs=self.widgets["pixelsize"],
            energy=self.widgets["energy"],
        )

    def update_clim(self, clim):
        self.mm_real.set_clim(clim)
        self.mm_imag.set_clim(clim)

    def update_phase(self, phase):
        self.params["phase"] = phase
        reco_shifted = shift_phase(self.reco, phase)
        self.mm_real.set_data(reco_shifted.real)
        self.mm_imag.set_data(reco_shifted.imag)

    def update_center(self, c0, c1):
        self.params["center"] = (c0, c1)
        self.holo_centered = shift_image(self.holo, [c0, c1])
        self.reco = reconstruct(self.holo_centered)[self.roi]
        self.update_phase(self.params["phase"])

    def update_propagation(self, dist, det, pxs, energy):
        dist *= 1e-6
        self.params.update(
            {"propdist": dist, "detectordist": det, "pixelsize": pxs, "energy": energy}
        )
        self.holo_prop = propagate(self.holo_centered, dist, det, pxs, energy)
        self.reco = reconstruct(self.holo_prop)[self.roi]
        self.update_phase(self.params["phase"])

    def get_full_reco(self):
        return shift_phase(reconstruct(self.holo_prop), self.params["phase"])
