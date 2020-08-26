import numpy as np
import pandas as pd
import rasterio
import rasterio.transform

from .utils import get_centroid_dist
from .optim import optimize


def parse_sortie(sortie_dir, sortie_input_type):
    """Parses sortie plots and returns a data frame.

    This fills the data frame with images between images with digitized
    centroids or corners.

    Args:
        sortie_dir (str): file path to sortie plot csv
        sortie_input_type (str): type of input, in ['swath_corners',
            'swath_centroids']
    """
    if sortie_input_type in ['swath_corners', 'swath_centroids']:
        offset = 0.5 if sortie_input_type == 'swath_corners' else 0
        # process sortie information
        df_swath = pd.read_csv(sortie_dir)
        assert (set(['idx0', 'idx1', 'pixel_x', 'pixel_y', 'swath_id'])
                .issubset(df_swath.columns))
        df_swath = df_swath.sort_values(by=['idx0', 'idx1'])
        # initialize image level df
        df = []
        # for each swath
        for swath_id, g in df_swath.groupby('swath_id'):
            # take the center on each end
            g = g.groupby(['idx0', 'idx1']).agg(
                {'pixel_x': 'mean', 'pixel_y': 'mean'}).reset_index()
            # get idx0 (folder), error thrown if not unique
            idx0, = g.loc[:, 'idx0'].astype(int).unique().tolist()
            # list all images in between
            start_end = g.iloc[[0, -1], :]
            start, end = start_end.loc[:, 'idx1'].tolist()
            idx1 = np.linspace(start, end, end - start + 1, dtype=int)
            # interpolate linearly to roughly georef all image centroids
            df.append(pd.DataFrame({
                'swath_id': swath_id,
                'idx0': idx0,
                'idx1': idx1,
                # record centroids from linear interpolation of sortie info
                'sortie_x_centroid': np.interp(
                    x=idx1,
                    xp=(start - offset, end + offset),
                    fp=start_end['pixel_x'].values),
                'sortie_y_centroid': np.interp(
                    x=idx1,
                    xp=(start - offset, end + offset),
                    fp=start_end['pixel_y'].values),
            }))
        df = pd.concat(df, axis=0, ignore_index=True)
    else:
        raise NotImplementedError
    return df


def initialize_centroids(indices, links, widths, heights, x0, x1, y0, y1):
    """Within a swath, initialize centroids with relative transforms.

    Args:
        indices (list of tuple of int): each index is a unique image identifier
        links (dict): {i_idx, j_idx: affine.Affine}
        widths, heights (list of int): widths and heights of each image
        x0, x1, y0, y1 (float): start/end x, y values

    Returns:
        numpy.ndarray [N,]: centroids_x, centroids_y
    """
    assert len(indices) == len(widths) == len(heights)
    if len(indices) == 1:
        assert x0 == x1 and y0 == y1
        return np.array([x0]), np.array([y0])
    # identity transform, used to calculate distances between two
    # image centroids
    identity_trans = rasterio.transform.Affine.identity()
    # assuming that all images are at the same scale
    # calculate centroid positions
    # calculate relative distances to adjust image positions
    # and fit the swath into the swath box drawn on the sortie plot
    dist = []
    # iterate over image pairs
    for i in range(len(indices) - 1):
        if (indices[i], indices[i + 1]) in links.keys():
            relative_trans = links[(indices[i], indices[i + 1])]
            if relative_trans is not None:
                # append distances between centroids if links are made
                dist.append(get_centroid_dist(
                    transforms=(identity_trans, relative_trans),
                    widths=widths[i:(i + 2)],
                    heights=heights[i:(i + 2)]))
                continue
        # else, linearly interpolate if no links are made
        dist.append(np.nan)

    # collect distances
    dist = np.array(dist)
    # linearly interpolate to fill nan's
    if np.all(np.isnan(dist)):  # if no links are made
        dist = np.linspace(0, 1, len(indices))
    else:
        dist = np.interp(
            x=range(dist.shape[0]),
            xp=(~np.isnan(dist)).nonzero()[0],
            fp=dist[~np.isnan(dist)])
        dist /= dist.sum()
        # take cumulative sum
        dist = np.cumsum(dist)
        # append 0 to the start
        dist = np.hstack([np.array([0]), dist])

    # space the images out in proportion to their cumulative distances
    # to the start/end images
    centroids_x = (1 - dist) * x0 + dist * x1
    centroids_y = (1 - dist) * y0 + dist * y1
    return centroids_x, centroids_y


def initialize_theta_scale(n_thetas, scales, verbose=False, **kwargs):
    """Initialize images with the same theta and scale.

    Args:
        n_thetas (int): estimated rotation of images, number of angle
            to try
        scales (list of float): list of scales to try
        verbose (bool)
        **kwargs passed to src.optim.optimize()

    Returns:
        float, float: best theta, best scale
    """
    n_valid_links = len(
        [k for k, v in kwargs['links'].items() if v is not None])
    if len(kwargs['nodes']) == 1 or n_valid_links == 0:
        return 0, scales[0]  # return smallest scale
    # theta initialization possibilities
    thetas = np.linspace(0, 2 * np.pi, num=n_thetas, endpoint=False)
    # try all the possible pairs of thetas & scales
    min_loss = np.inf
    best_theta = np.nan
    best_scale = np.nan
    for theta in thetas:
        for scale in scales:
            # evaluate once with the initialization values
            _, (loss,), _ = optimize(
                thetas_init=[theta] * len(kwargs['nodes']),
                scales_init=[scale] * len(kwargs['nodes']),
                n_iter=1, lr_theta=0, lr_scale=0, lr_xy=0, **kwargs)
            # find the combination with lowest loss
            if loss < min_loss:
                min_loss = loss
                best_theta = theta
                best_scale = scale
    if verbose:
        print('Best theta: {:.3f}, Best scale: {:.3f}'
              .format(best_theta, best_scale))
    return best_theta, best_scale
