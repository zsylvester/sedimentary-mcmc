"""
Sedimentary MCMC Analysis Module

This module provides functions for analyzing sedimentary sequences using
Markov Chain Monte Carlo (MCMC) methods, wavelet transforms, and dynamic time warping.

Key functionality includes:
- Log blocking using continuous wavelet transform
- Self-Organizing Map (SOM) clustering for facies analysis
- Transition probability matrix construction
- Markov chain generation for synthetic log creation
- Log correlation using dynamic time warping

Dependencies:
    numpy, matplotlib, scipy, librosa, skimage, minisom, tqdm
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from librosa.sequence import dtw
from skimage import measure
from tqdm import trange


# =============================================================================
# Wavelet Transform Functions
# =============================================================================

def ricker(points, a):
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet models the function:
        A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)
    where A = 2/(sqrt(3*a)*(pi**0.25)).

    Parameters
    ----------
    points : int
        Number of points in the output vector. The wavelet will be centered
        around the middle of the vector.
    a : float
        Width parameter of the wavelet, controlling its scale.

    Returns
    -------
    ndarray
        Array of shape (points,) containing the Ricker wavelet.

    Examples
    --------
    >>> wavelet = ricker(100, 4.0)
    >>> len(wavelet)
    100
    """
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec ** 2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


def cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Perform a continuous wavelet transform on input data.

    Computes the CWT by convolving the data with scaled versions of the
    wavelet function at each specified width.

    Parameters
    ----------
    data : ndarray
        1D array of data on which to perform the transform.
    wavelet : callable
        Wavelet function that takes (points, width, **kwargs) and returns
        a 1D array. See `ricker` for an example.
    widths : array_like
        Sequence of widths (scales) to use for the transform.
    dtype : dtype, optional
        Desired output data type. Defaults to float64 for real wavelets
        and complex128 for complex wavelets.
    **kwargs
        Additional keyword arguments passed to the wavelet function.

    Returns
    -------
    ndarray
        2D array of shape (len(widths), len(data)) containing the CWT
        coefficients.

    Notes
    -----
    For non-symmetric, complex-valued wavelets, the input signal is convolved
    with the time-reversed complex-conjugate of the wavelet data.

    References
    ----------
    S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)",
    Academic Press, 2009.
    """
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = signal.convolve(data, wavelet_data, mode='same')
    return output


# =============================================================================
# Log Blocking Functions
# =============================================================================

def log_blocking(log, scale, plot=True):
    """
    Block a log into discrete intervals using continuous wavelet transform.

    Uses the CWT with a Ricker wavelet to identify natural boundaries in the
    log signal, then computes the mean property value within each block.

    Parameters
    ----------
    log : ndarray
        1D array containing the log values to be blocked.
    scale : int
        Scale parameter for boundary detection. Lower values result in
        more boundaries (finer blocking).
    plot : bool, optional
        If True, display a plot of the blocked log. Default is True.

    Returns
    -------
    md : ndarray
        Measured depth indices (0 to len(log)-1).
    grn : ndarray
        Blocked log values, where each sample is assigned the mean value
        of its containing block.
    bounds : ndarray
        Array of boundary indices defining the blocks.
    prop : ndarray
        Mean property value for each block.

    Notes
    -----
    The function finds zero-crossings in the CWT at the specified scale
    to define block boundaries. Empty segments are filled with values
    from neighboring segments.
    """
    widths = np.arange(1, 1000)
    cwt_all = cwt(log, ricker, widths)
    contours = measure.find_contours(cwt_all.T, 0)

    bounds = []
    for i in range(len(contours)):
        x = contours[i][:, 0]
        y = contours[i][:, 1]
        dec, dummy = np.modf(y)
        x = x[dec == 0.0]
        y = y[dec == 0.0]
        if scale in y:
            if y[0] == 0.0:
                bounds.append(x[0])
            if y[-1] == 0.0:
                bounds.append(x[-1])

    bounds.append(0)
    bounds.sort()
    bounds.append(len(log) - 1)
    bounds = np.array(bounds).astype(int)

    d1 = np.arange(len(log))
    prop = np.zeros((len(bounds) - 1,))

    for i in range(len(bounds) - 1):
        wlog_segment = log[(d1 >= bounds[i]) & (d1 < bounds[i + 1])]
        if len(wlog_segment) > 0:
            prop[i] = np.nanmean(wlog_segment)
        else:
            if (i > 0) & (i < len(bounds) - 2):
                prop[i] = np.nanmean(log[(d1 >= bounds[i - 1]) & (d1 < bounds[i + 2])])
            else:
                prop[i] = np.nan

    if plot:
        plot_blocked_log(d1, bounds, prop)

    md = d1
    grn = np.zeros(np.shape(md))
    for i in range(len(prop)):
        grn[(md >= bounds[i]) & (md < bounds[i + 1])] = prop[i]
    grn[-1] = grn[-2]

    if plot:
        plt.plot(grn, md, 'k')

    return md, grn, bounds, prop


def plot_blocked_log(depths, bounds, prop, ax=None):
    """
    Plot a blocked log as colored intervals.

    Parameters
    ----------
    depths : ndarray
        Array of depth values.
    bounds : ndarray
        Array of boundary indices defining the blocks.
    prop : ndarray
        Property values for each block (used for coloring).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,10))

    for i in range(len(prop)):
        fillcolor = [1 - 0.4 * prop[i], 1 - 0.7 * prop[i], 0.6 - 0.6 * prop[i]]
        x = [0, 1, 1, 0]
        y = [bounds[i], bounds[i], bounds[i + 1], bounds[i + 1]]
        ax.fill(x, y, color=fillcolor)

    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(np.max(depths), np.min(depths))
    plt.tight_layout()

    return ax


# =============================================================================
# Image Processing Functions
# =============================================================================

def image_to_log(image, channel=0, invert=True, trim_start=0, trim_end=0):
    """
    Convert an image to a 1D log by averaging across columns.

    Parameters
    ----------
    image : ndarray
        Input image array with shape (height, width) or (height, width, channels).
    channel : int, optional
        Color channel to use if image has multiple channels. Default is 0 (red).
    invert : bool, optional
        If True, invert the values (1 - value). Default is True.
    trim_start : int, optional
        Number of rows to remove from the start. Default is 0.
    trim_end : int, optional
        Number of rows to remove from the end. Default is 0.

    Returns
    -------
    ndarray
        1D array containing the log values.
    """
    if len(image.shape) == 3:
        img_channel = image[:, :, channel]
    else:
        img_channel = image

    # Apply trimming
    if trim_start > 0:
        img_channel = img_channel[trim_start:]
    if trim_end > 0:
        img_channel = img_channel[:-trim_end]

    log = np.mean(img_channel, axis=1)

    if invert:
        log = 1 - log

    return log


# =============================================================================
# Markov Chain Functions
# =============================================================================

def transition_matrix(transitions):
    """
    Compute a transition probability matrix from a sequence of states.

    Parameters
    ----------
    transitions : array_like
        Sequence of integer state indices.

    Returns
    -------
    ndarray
        Square transition probability matrix where M[i,j] is the probability
        of transitioning from state i to state j.

    Notes
    -----
    Each row of the output matrix sums to 1 (or 0 if that state was never
    observed in the input sequence).

    References
    ----------
    https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
    """
    n = 1 + max(transitions)
    M = [[0] * n for _ in range(n)]

    for (i, j) in zip(transitions, transitions[1:]):
        M[i][j] += 1

    # Convert to probabilities
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]

    return np.array(M)


def plot_transition_matrix(M, colormap, som_shape, colormap_2d=None, ax=None,
                           cmap='viridis', title='Facies Transition Probability Matrix',
                           show_colormap_reference=True):
    """
    Plot a transition probability matrix with color-coded cluster labels.

    Parameters
    ----------
    M : ndarray
        Transition probability matrix of shape (n_clusters, n_clusters).
    colormap : ndarray
        Array of shape (n_clusters, 3) containing RGB colors for each cluster,
        as returned by generate_2d_colormap.
    som_shape : tuple
        Shape of the SOM grid as (rows, cols).
    colormap_2d : ndarray, optional
        Array of shape (rows, cols, 3) containing the 2D color grid,
        as returned by generate_2d_colormap. Required if show_colormap_reference is True.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure. If show_colormap_reference
        is True and ax is None, creates a figure with two subplots.
    cmap : str, optional
        Colormap for the matrix values. Default is 'viridis'.
    title : str, optional
        Title for the plot. Default is 'Facies Transition Probability Matrix'.
    show_colormap_reference : bool, optional
        If True, display the 2D colormap reference panel. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot (only if ax is None).
    axes : matplotlib.axes.Axes or tuple
        The axes containing the plot(s). If show_colormap_reference is True,
        returns (ax_matrix, ax_colormap).
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object.
    """
    n_clusters = som_shape[0] * som_shape[1]

    # Create figure if no axes provided
    if ax is None:
        if show_colormap_reference:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5),
                                     gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})
            ax_matrix = axes[0]
            ax_cmap = axes[1]
        else:
            fig, ax_matrix = plt.subplots(figsize=(7, 6))
            ax_cmap = None
    else:
        fig = None
        ax_matrix = ax
        ax_cmap = None
        show_colormap_reference = False  # Can't show reference if single axis provided

    # Plot the transition matrix
    im = ax_matrix.imshow(M, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax_matrix, label='Probability', shrink=0.8)

    # Add colored squares along axes to show cluster colors
    # Patches are placed directly adjacent to the matrix (no gap, no overlap)
    patch_size = 0.4

    for i in range(n_clusters):
        # Color patches on the left (y-axis)
        rect_y = plt.Rectangle((-0.5 - patch_size, i - 0.5),
                               patch_size, 1, color=colormap[i], clip_on=False)
        ax_matrix.add_patch(rect_y)
        # Color patches on the bottom (x-axis)
        rect_x = plt.Rectangle((i - 0.5, n_clusters - 0.5),
                               1, patch_size, color=colormap[i], clip_on=False)
        ax_matrix.add_patch(rect_x)

    # Set axis limits and labels
    ax_matrix.set_xlim(-0.5, n_clusters - 0.5)
    ax_matrix.set_ylim(n_clusters - 0.5, -0.5)
    ax_matrix.set_xlabel('To Cluster')
    ax_matrix.set_ylabel('From Cluster')
    ax_matrix.set_title(title, pad=10)

    # Plot the colormap reference panel
    if show_colormap_reference and ax_cmap is not None:
        if colormap_2d is None:
            # Generate colormap_2d if not provided
            _, colormap_2d = generate_2d_colormap(som_shape)

        ax_cmap.imshow(colormap_2d, interpolation='nearest')
        for i in range(som_shape[0]):
            for j in range(som_shape[1]):
                cluster_idx = i * som_shape[1] + j
                brightness = np.mean(colormap_2d[i, j])
                text_color = 'white' if brightness < 0.5 else 'black'
                ax_cmap.text(j, i, str(cluster_idx), ha='center', va='center',
                            fontsize=12, fontweight='bold', color=text_color)
        ax_cmap.set_xticks([])
        ax_cmap.set_yticks([])
        ax_cmap.set_title('Cluster Colors', pad=10)

    if fig is not None:
        if show_colormap_reference:
            return fig, (ax_matrix, ax_cmap), cbar
        else:
            return fig, ax_matrix, cbar
    else:
        return ax_matrix, cbar


def get_next_term(ind, M, n_states):
    """
    Sample the next state from the Markov chain.

    Parameters
    ----------
    ind : int
        Current state index.
    M : ndarray
        Transition probability matrix.
    n_states : int
        Total number of states.

    Returns
    -------
    int
        Index of the next state.
    """
    return random.choices(np.arange(n_states), M[ind, :])[0]


def make_chain(M, start_term, n):
    """
    Generate a Markov chain of specified length.

    Parameters
    ----------
    M : ndarray
        Transition probability matrix.
    start_term : int
        Initial state index.
    n : int
        Desired length of the chain.

    Returns
    -------
    list
        List of state indices representing the Markov chain.

    References
    ----------
    https://stackoverflow.com/questions/59858123/generating-markov-chain-from-transition-matrix
    """
    n_states = M.shape[0]
    chain = [start_term]
    for i in range(n - 1):
        chain.append(get_next_term(chain[-1], M, n_states))
    return chain


def pick_random_value_from_cluster(prop, cluster_index, cluster_n, bins):
    """
    Sample a random value from a cluster's distribution.

    Uses histogram-based sampling to pick a value consistent with the
    distribution of property values in the specified cluster.

    Parameters
    ----------
    prop : ndarray
        Array of property values.
    cluster_index : ndarray
        Array of cluster assignments for each property value.
    cluster_n : int
        Cluster number to sample from.
    bins : ndarray
        Bin edges for the histogram.

    Returns
    -------
    float
        Randomly sampled value from the cluster's distribution.
    """
    prob = np.diff(bins)[0] * np.histogram(
        prop[cluster_index == cluster_n], bins=bins, density=True
    )[0]
    bin_choice = random.choices(np.arange(len(bins) - 1), prob)[0]
    return np.random.uniform(low=bins[bin_choice], high=bins[bin_choice + 1], size=1)[0]


def facies_chain(M, vsh, ths, cluster_index, chain_length, vsh_bins=None, ths_bins=None):
    """
    Generate synthetic facies sequences using a Markov chain.

    Creates a sequence of facies (characterized by volume of shale and thickness)
    by walking through the transition probability matrix and sampling property
    values from each visited cluster.

    Parameters
    ----------
    M : ndarray
        Transition probability matrix.
    vsh : ndarray
        Volume of shale values for all observed facies.
    ths : ndarray
        Thickness values for all observed facies.
    cluster_index : ndarray
        Cluster assignments for each facies observation.
    chain_length : int
        Number of facies to generate.
    vsh_bins : ndarray, optional
        Bin edges for vsh histogram. Default is np.arange(0, 1.01, 0.05).
    ths_bins : ndarray, optional
        Bin edges for thickness histogram. Default is 21 bins spanning the
        range of ths values.

    Returns
    -------
    vsh_chain : list
        Volume of shale values for the synthetic sequence.
    ths_chain : list
        Thickness values for the synthetic sequence.
    """
    if vsh_bins is None:
        vsh_bins = np.arange(0, 1.01, 0.05)
    if ths_bins is None:
        ths_bins = np.linspace(np.floor(np.min(ths)), np.ceil(np.max(ths)), 21)

    chain = make_chain(M, 0, chain_length)
    vsh_chain = []
    ths_chain = []

    for cluster_n in chain:
        vsh_chain.append(pick_random_value_from_cluster(vsh, cluster_index, cluster_n, bins=vsh_bins))
        ths_chain.append(pick_random_value_from_cluster(ths, cluster_index, cluster_n, bins=ths_bins))

    return vsh_chain, ths_chain


def chain_to_log(vsh_chain, ths_chain):
    """
    Convert a facies chain to a blocked log.

    Parameters
    ----------
    vsh_chain : list or ndarray
        Volume of shale values for each facies.
    ths_chain : list or ndarray
        Thickness values for each facies.

    Returns
    -------
    md : ndarray
        Measured depth array.
    grn : ndarray
        Property values at each depth.
    depths : ndarray
        Depth at each facies boundary.
    """
    depths = np.hstack((0, np.cumsum(ths_chain[::-1])))
    md = np.arange(int(np.max(np.cumsum(ths_chain[::-1]))))
    grn = np.zeros(np.shape(md))

    for i in range(len(ths_chain)):
        grn[(md >= depths[i]) & (md < depths[i + 1])] = vsh_chain[::-1][i]

    return md, grn, depths


def plot_facies_column(vsh_chain, ths_chain, ax=None):
    """
    Plot a synthetic facies column as colored intervals.

    Parameters
    ----------
    vsh_chain : list or ndarray
        Volume of shale values for each facies.
    ths_chain : list or ndarray
        Thickness values for each facies.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    depths = np.hstack((0, np.cumsum(ths_chain[::-1])))[::-1]

    for i in range(len(ths_chain)):
        fillcolor = [1 - 0.4 * vsh_chain[i], 1 - 0.7 * vsh_chain[i], 0.6 - 0.6 * vsh_chain[i]]
        x = [0, 1, 1, 0]
        y = [depths[i], depths[i], depths[i + 1], depths[i + 1]]
        ax.fill(x, y, color=fillcolor)

    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(np.max(depths), np.min(depths))
    plt.tight_layout()

    return ax


# =============================================================================
# Log Correlation Functions
# =============================================================================

def correlate_logs(log1, log2, exponent=0.15):
    """
    Correlate two logs using dynamic time warping.

    Computes a similarity matrix based on the absolute difference between
    log values raised to a power, then uses DTW to find the optimal alignment.

    Parameters
    ----------
    log1 : ndarray
        First log (reference).
    log2 : ndarray
        Second log (to be correlated).
    exponent : float, optional
        Exponent for the cost function. Lower values make the cost more
        sensitive to small differences. Default is 0.15.

    Returns
    -------
    p : ndarray
        Correlation indices for the first log.
    q : ndarray
        Correlation indices for the second log.
    D : ndarray
        Accumulated cost matrix from DTW.

    Notes
    -----
    The function uses librosa's DTW implementation. The warping path
    (p, q) defines how samples in log1 map to samples in log2.
    """
    r = len(log1)
    c = len(log2)
    sm = np.zeros((r, c))

    for i in range(r):
        sm[i, :] = (np.abs(log2 - log1[i])) ** exponent

    D, wp = dtw(C=sm)
    p = np.array(wp[:, 0])
    q = np.array(wp[:, 1])

    return p, q, D


def plot_correlation_matrix(D, p, q, ax=None):
    """
    Plot the DTW cost matrix with the warping path.

    Parameters
    ----------
    D : ndarray
        Accumulated cost matrix from DTW.
    p : ndarray
        Row indices of the warping path.
    q : ndarray
        Column indices of the warping path.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(D)
    ax.plot(q, p, 'w')
    plt.colorbar(im, ax=ax)

    return ax


def compute_correlation_quality(log1, log2, p, q):
    """
    Compute the quality of a log correlation.

    Parameters
    ----------
    log1 : ndarray
        First log (reference).
    log2 : ndarray
        Second log (correlated).
    p : ndarray
        Correlation indices for log1.
    q : ndarray
        Correlation indices for log2.

    Returns
    -------
    r_value : float
        Pearson correlation coefficient between the aligned logs.
    slope : float
        Slope of the linear regression.
    intercept : float
        Intercept of the linear regression.
    """
    slope, intercept, r_value, p_value, std_error = stats.linregress(log1[p], log2[q])
    return r_value, slope, intercept


# =============================================================================
# SOM Clustering Functions
# =============================================================================

def generate_2d_colormap(som_shape, method='corners', corner_colors=None,
                         saturation=0.7, lightness=0.6):
    """
    Generate a 2D colormap suitable for visualizing SOM clustering results.

    Creates a smooth color gradient across a 2D grid where nearby cells have
    similar colors, making it easy to visually identify cluster relationships.

    Parameters
    ----------
    som_shape : tuple
        Shape of the SOM grid as (rows, cols).
    method : str, optional
        Color generation method. Options are:
        - 'corners': Interpolate between four corner colors (default)
        - 'hsv_gradient': Use HSV color space with hue varying by position
        - 'lab_gradient': Use perceptually uniform LAB-like interpolation
    corner_colors : dict, optional
        Dictionary with corner colors for 'corners' method. Keys are
        'top_left', 'top_right', 'bottom_left', 'bottom_right'.
        Values are RGB tuples (0-1 range). If None, uses default colors.
    saturation : float, optional
        Saturation level for 'hsv_gradient' method (0-1). Default is 0.7.
    lightness : float, optional
        Lightness/value level for 'hsv_gradient' method (0-1). Default is 0.6.

    Returns
    -------
    colormap : ndarray
        Array of shape (n_clusters, 3) containing RGB colors for each cluster.
        Clusters are indexed in row-major order (same as np.ravel_multi_index).
    colormap_2d : ndarray
        Array of shape (rows, cols, 3) containing the 2D color grid.

    Examples
    --------
    >>> colors, colors_2d = generate_2d_colormap((3, 3))
    >>> # Use colors[cluster_index] to get color for each data point
    >>> plt.scatter(x, y, c=[colors[i] for i in cluster_indices])

    >>> # Visualize the colormap itself
    >>> plt.imshow(colors_2d)
    """
    rows, cols = som_shape
    n_clusters = rows * cols

    if method == 'corners':
        # Bilinear interpolation between four corner colors
        if corner_colors is None:
            # Default: visually distinct corners that blend well
            corner_colors = {
                'top_left': np.array([0.2, 0.4, 0.8]),      # Blue
                'top_right': np.array([0.9, 0.3, 0.3]),     # Red
                'bottom_left': np.array([0.3, 0.8, 0.4]),   # Green
                'bottom_right': np.array([0.9, 0.7, 0.2])   # Orange/Yellow
            }
        else:
            corner_colors = {k: np.array(v) for k, v in corner_colors.items()}

        colormap_2d = np.zeros((rows, cols, 3))

        for i in range(rows):
            for j in range(cols):
                # Normalized coordinates (0-1)
                y_norm = i / max(rows - 1, 1)
                x_norm = j / max(cols - 1, 1)

                # Bilinear interpolation
                top = (1 - x_norm) * corner_colors['top_left'] + x_norm * corner_colors['top_right']
                bottom = (1 - x_norm) * corner_colors['bottom_left'] + x_norm * corner_colors['bottom_right']
                color = (1 - y_norm) * top + y_norm * bottom

                colormap_2d[i, j] = np.clip(color, 0, 1)

    elif method == 'hsv_gradient':
        # HSV-based gradient with hue varying across the 2D space
        colormap_2d = np.zeros((rows, cols, 3))

        for i in range(rows):
            for j in range(cols):
                # Map position to hue (0-1 range, wrapping)
                # Use both row and column to determine hue
                y_norm = i / max(rows - 1, 1)
                x_norm = j / max(cols - 1, 1)

                # Hue varies diagonally for maximum distinction
                hue = (x_norm * 0.5 + y_norm * 0.5) % 1.0

                # Slight variation in saturation and value based on position
                sat = saturation * (0.8 + 0.2 * (1 - y_norm))
                val = lightness * (0.85 + 0.15 * x_norm)

                # Convert HSV to RGB
                colormap_2d[i, j] = _hsv_to_rgb(hue, sat, val)

    elif method == 'lab_gradient':
        # Perceptually uniform gradient using LAB-like interpolation
        # Define anchor colors in a perceptually spaced manner
        colormap_2d = np.zeros((rows, cols, 3))

        # Create a set of perceptually distinct base colors
        base_hues = [0.0, 0.15, 0.33, 0.5, 0.66, 0.83]  # Red, Orange, Green, Cyan, Blue, Magenta

        for i in range(rows):
            for j in range(cols):
                y_norm = i / max(rows - 1, 1)
                x_norm = j / max(cols - 1, 1)

                # Select hue based on diagonal position
                diag = (x_norm + y_norm) / 2
                hue_idx = diag * (len(base_hues) - 1)
                hue_low = int(hue_idx)
                hue_high = min(hue_low + 1, len(base_hues) - 1)
                hue_frac = hue_idx - hue_low

                hue = base_hues[hue_low] * (1 - hue_frac) + base_hues[hue_high] * hue_frac

                # Vary saturation and lightness based on perpendicular direction
                perp = abs(x_norm - y_norm)
                sat = saturation * (0.7 + 0.3 * (1 - perp))
                val = lightness * (0.8 + 0.2 * perp)

                colormap_2d[i, j] = _hsv_to_rgb(hue, sat, val)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'corners', 'hsv_gradient', or 'lab_gradient'.")

    # Flatten to 1D array indexed by cluster number (row-major order)
    colormap = colormap_2d.reshape(-1, 3)

    return colormap, colormap_2d


def _hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB.

    Parameters
    ----------
    h : float
        Hue (0-1).
    s : float
        Saturation (0-1).
    v : float
        Value/brightness (0-1).

    Returns
    -------
    ndarray
        RGB color as array of shape (3,) with values in range 0-1.
    """
    if s == 0.0:
        return np.array([v, v, v])

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return np.array([v, t, p])
    elif i == 1:
        return np.array([q, v, p])
    elif i == 2:
        return np.array([p, v, t])
    elif i == 3:
        return np.array([p, q, v])
    elif i == 4:
        return np.array([t, p, v])
    else:
        return np.array([v, p, q])


def plot_som_colormap(som_shape, method='corners', **kwargs):
    """
    Display a 2D colormap grid for visualization.

    Parameters
    ----------
    som_shape : tuple
        Shape of the SOM grid as (rows, cols).
    method : str, optional
        Color generation method (see generate_2d_colormap).
    **kwargs
        Additional arguments passed to generate_2d_colormap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    colormap : ndarray
        The 1D colormap array (n_clusters, 3).
    """
    colormap, colormap_2d = generate_2d_colormap(som_shape, method=method, **kwargs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colormap_2d, interpolation='nearest')

    # Add cluster index labels
    rows, cols = som_shape
    for i in range(rows):
        for j in range(cols):
            cluster_idx = i * cols + j
            # Choose text color based on background brightness
            brightness = np.mean(colormap_2d[i, j])
            text_color = 'white' if brightness < 0.5 else 'black'
            ax.text(j, i, str(cluster_idx), ha='center', va='center',
                   fontsize=12, fontweight='bold', color=text_color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'2D Colormap for {rows}x{cols} SOM\n(method: {method})')

    return fig, ax, colormap


def plot_som_clusters(data_normalized, cluster_index, som, colormap, colormap_2d,
                      som_shape, ax=None, show_colormap_reference=True,
                      show_neighbor_links=True, xlabel='Normalized Vsh',
                      ylabel='Normalized Thickness', title='SOM Clustering Results'):
    """
    Plot SOM clustering results with colored clusters and optional neighbor links.

    Parameters
    ----------
    data_normalized : ndarray
        Normalized input data of shape (n_samples, 2).
    cluster_index : ndarray
        1D array of cluster assignments for each data point.
    som : MiniSom
        Trained SOM object.
    colormap : ndarray
        Array of shape (n_clusters, 3) containing RGB colors for each cluster.
    colormap_2d : ndarray
        Array of shape (rows, cols, 3) containing the 2D color grid.
    som_shape : tuple
        Shape of the SOM grid as (rows, cols).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    show_colormap_reference : bool, optional
        If True, display the 2D colormap reference panel. Default is True.
    show_neighbor_links : bool, optional
        If True, draw lines connecting neighboring clusters in the SOM grid.
        Default is True.
    xlabel : str, optional
        Label for x-axis. Default is 'Normalized Vsh'.
    ylabel : str, optional
        Label for y-axis. Default is 'Normalized Thickness'.
    title : str, optional
        Title for the plot. Default is 'SOM Clustering Results'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot (only if ax is None).
    axes : matplotlib.axes.Axes or tuple
        The axes containing the plot(s). If show_colormap_reference is True,
        returns (ax_scatter, ax_colormap).
    """
    rows, cols = som_shape
    n_clusters = rows * cols

    # Create figure if no axes provided
    if ax is None:
        if show_colormap_reference:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                     gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})
            ax_scatter = axes[0]
            ax_cmap = axes[1]
        else:
            fig, ax_scatter = plt.subplots(figsize=(8, 6))
            ax_cmap = None
    else:
        fig = None
        ax_scatter = ax
        ax_cmap = None
        show_colormap_reference = False

    # Get SOM weights (centroids) for plotting neighbor links
    weights = som.get_weights()  # Shape: (rows, cols, 2)

    # Draw neighbor links between adjacent clusters
    if show_neighbor_links:
        for i in range(rows):
            for j in range(cols):
                centroid = weights[i, j]
                # Connect to right neighbor
                if j < cols - 1:
                    neighbor = weights[i, j + 1]
                    ax_scatter.plot([centroid[0], neighbor[0]], [centroid[1], neighbor[1]],
                                   'k-', linewidth=1, alpha=0.5, zorder=1)
                # Connect to bottom neighbor
                if i < rows - 1:
                    neighbor = weights[i + 1, j]
                    ax_scatter.plot([centroid[0], neighbor[0]], [centroid[1], neighbor[1]],
                                   'k-', linewidth=1, alpha=0.5, zorder=1)

    # Plot scatter points for each cluster
    for c in np.unique(cluster_index):
        mask = cluster_index == c
        ax_scatter.scatter(data_normalized[mask, 0], data_normalized[mask, 1],
                          c=[colormap[c]], label=f'cluster {c}', alpha=0.7, s=50, zorder=2)

    # Plot SOM centroids with cluster numbers
    for i in range(rows):
        for j in range(cols):
            cluster_idx = i * cols + j
            centroid = weights[i, j]
            ax_scatter.scatter(centroid[0], centroid[1], marker='o',
                              s=250, color='white', edgecolors='k', linewidths=2, zorder=3)
            ax_scatter.text(centroid[0], centroid[1], str(cluster_idx),
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='k', zorder=4)

    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_title(title)

    # Plot the colormap reference panel
    if show_colormap_reference and ax_cmap is not None:
        ax_cmap.imshow(colormap_2d, interpolation='nearest')
        for i in range(rows):
            for j in range(cols):
                cluster_idx = i * cols + j
                brightness = np.mean(colormap_2d[i, j])
                text_color = 'white' if brightness < 0.5 else 'black'
                ax_cmap.text(j, i, str(cluster_idx), ha='center', va='center',
                            fontsize=12, fontweight='bold', color=text_color)
        ax_cmap.set_xticks([])
        ax_cmap.set_yticks([])
        ax_cmap.set_title('Cluster Colors', pad=10)

    if fig is not None:
        if show_colormap_reference:
            return fig, (ax_scatter, ax_cmap)
        else:
            return fig, ax_scatter
    else:
        return ax_scatter


def cluster_facies_data(vsh, ths, som_shape=(3, 3), sigma=0.3, learning_rate=0.5,
                        n_iterations=10000, init_method='pca', random_seed=None,
                        plot=True):
    """
    Cluster facies data using a Self-Organizing Map.

    Parameters
    ----------
    vsh : ndarray
        Volume of shale values.
    ths : ndarray
        Thickness values.
    som_shape : tuple, optional
        Shape of the SOM grid. Default is (3, 3).
    sigma : float, optional
        Spread of the neighborhood function. Controls the initial neighborhood
        radius. Larger values mean more neurons are updated together initially.
        Default is 0.3.
    learning_rate : float, optional
        Initial learning rate. Default is 0.5.
    n_iterations : int, optional
        Number of training iterations. More iterations generally lead to better
        convergence. Default is 10000.
    init_method : str, optional
        Weight initialization method. Options are:
        - 'pca': Initialize weights along principal components (recommended).
          This typically leads to better topological organization where
          neighboring clusters in the SOM grid are also neighbors in data space.
        - 'random': Random initialization from the data range.
        Default is 'pca'.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        If True, display a scatter plot of clusters. Default is True.

    Returns
    -------
    cluster_index : ndarray
        1D array of cluster assignments.
    som : MiniSom
        Trained SOM object.
    data_normalized : ndarray
        Normalized input data.

    Notes
    -----
    For well-organized SOM results where neighboring clusters are also
    neighbors in the data space, consider:
    1. Using PCA initialization (init_method='pca') - this aligns the initial
       weights along the principal components of the data.
    2. Increasing n_iterations for better convergence (e.g., 20000-50000).
    3. Adjusting sigma - larger values create more global organization initially.
    """
    from minisom import MiniSom

    data = np.vstack((vsh, ths)).T
    data_normalized = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)

    som = MiniSom(som_shape[0], som_shape[1], 2, sigma=sigma,
                  learning_rate=learning_rate, random_seed=random_seed)

    # Initialize weights
    if init_method == 'pca':
        som.pca_weights_init(data_normalized)
    elif init_method == 'random':
        som.random_weights_init(data_normalized)
    else:
        raise ValueError(f"Unknown init_method: {init_method}. Use 'pca' or 'random'.")

    # Train the SOM
    som.train(data_normalized, n_iterations)

    winner_coordinates = np.array([som.winner(x) for x in data_normalized]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    if plot:
        fig, ax = plt.subplots()
        for c in np.unique(cluster_index):
            ax.scatter(data_normalized[cluster_index == c, 0],
                      data_normalized[cluster_index == c, 1],
                      label=f'cluster={c}', alpha=0.7)

        for centroid in som.get_weights():
            ax.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                      s=30, linewidths=2, color='k', label='centroid')

    return cluster_index, som, data_normalized


# =============================================================================
# Monte Carlo Simulation Functions
# =============================================================================

def run_monte_carlo_correlation(reference_log, M, vsh, ths, cluster_index,
                                 n_simulations=1000, chain_length=370,
                                 exponent=0.15, noise_std=0.01, verbose=True):
    """
    Run Monte Carlo simulations to assess correlation significance.

    Generates synthetic logs using the Markov chain model and correlates
    each with the reference log to build a distribution of correlation
    quality metrics.

    Parameters
    ----------
    reference_log : ndarray
        Reference log to correlate against.
    M : ndarray
        Transition probability matrix.
    vsh : ndarray
        Volume of shale training data.
    ths : ndarray
        Thickness training data.
    cluster_index : ndarray
        Cluster assignments for training data.
    n_simulations : int, optional
        Number of synthetic logs to generate. Default is 1000.
    chain_length : int, optional
        Length of each synthetic chain. Default is 370.
    exponent : float, optional
        Cost function exponent for DTW. Default is 0.15.
    noise_std : float, optional
        Standard deviation of noise to add to logs. Default is 0.01.
    verbose : bool, optional
        If True, print progress every 100 iterations. Default is True.

    Returns
    -------
    r_values : list
        Correlation coefficients for each simulation.
    log_lengths : list
        Length of each synthetic log.
    """
    r_values = []
    log_lengths = []

    log1 = reference_log + np.random.normal(0, noise_std, len(reference_log))

    for count in trange(n_simulations):
        vsh_chain, ths_chain = facies_chain(M, vsh, ths, cluster_index, chain_length)
        md, grn, depths = chain_to_log(vsh_chain, ths_chain)

        log2 = grn + np.random.normal(0, noise_std, len(grn))
        p, q, D = correlate_logs(log1, log2, exponent)

        r_value, slope, intercept = compute_correlation_quality(log1, log2, p, q)
        r_values.append(r_value)
        log_lengths.append(len(log2))

    return r_values, log_lengths


def plot_monte_carlo_results(r_values, comparison_r_value=None, n_bins=50):
    """
    Plot histogram of Monte Carlo correlation results.

    Parameters
    ----------
    r_values : list or ndarray
        Correlation coefficients from Monte Carlo simulations.
    comparison_r_value : float, optional
        A reference correlation value to mark on the histogram (e.g., from
        a real correlation). If provided, a red vertical line is drawn.
    n_bins : int, optional
        Number of histogram bins. Default is 50.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    fig, ax = plt.subplots()
    hist_values = ax.hist(r_values, n_bins)

    if comparison_r_value is not None:
        max_count = max(hist_values[0])
        ax.plot([comparison_r_value, comparison_r_value],
                [0, 1.1 * max_count], 'r', linewidth=2)
        ax.set_ylim(0, 1.1 * max_count)

    ax.set_xlim(0, max(1.1 * max(hist_values[1]), 0.8))
    ax.set_xlabel('Correlation coefficient (r)')
    ax.set_ylabel('Count')
    ax.set_title('Monte Carlo Correlation Distribution')

    return fig, ax
