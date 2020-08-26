import os
import glob
import numpy as np
import rasterio
import rasterio.transform
import rasterio.control
import shapely
import shapely.geometry
import cv2 as cv

assert rasterio.__version__ >= '1.1.0'


def transform_coord(transform, to, xy=None, colrow=None):
    """Transforms x/y coord to/from col/row coord in a vectorized manner.

    Args:
        transform (affine.Affine): affine transformation
            (e.g., rasterio.io.DatasetReader.transform)
        to (str): in ['xy', 'colrow']
        xy (numpy.ndarray [N, 2]): x, y coords
        colrow (numpy.ndarray [N, 2]): col, row coords

    Returns:
        numpy.ndarray [N, 2]: transformed array
    """
    if to == 'colrow':
        t = np.array(~transform).reshape((3, 3))
        stacked = np.hstack((xy, np.ones((xy.shape[0], 1))))
        return t.dot(stacked.T).T[:, 0:2]
    elif to == 'xy':
        t = np.array(transform).reshape((3, 3))
        stacked = np.hstack((colrow, np.ones((colrow.shape[0], 1))))
        return t.dot(stacked.T).T[:, 0:2]
    else:
        raise NotImplementedError


def convert_affine(transform, to):
    """From/to affine.Affine() to/from world files.

    The world file convention is different from GDAL/rasterio/OpenCV
    in that world files record affine transformations with the
    centroid of the pixel at the upper left corner, while
    GDAL/rasterio/OpenCV records the upper left corner directly.

    Args:
        transform (affine.Affine or list of float): affine transform
            input
        to (str): in ['affine.Affine', 'world_file'], output format

    Returns:
        list of float or affine.Affine: output affine transform
    """
    if to == 'affine.Affine':
        output = np.array(transform).reshape((3, 2)).T
        output = np.vstack((output, [0, 0, 1]))
        output[:, 2] = output.dot([-0.5, -0.5, 1])
        output = output[0:2, :]
        return rasterio.transform.Affine(*output.flatten())
    elif to == 'world_file':
        output = list(transform.column_vectors)
        output[2] = transform * (0.5, 0.5)  # upper left corner
        output = [i for col in output for i in col]  # unpack
        return output
    else:
        raise NotImplementedError


def calculate_transform_from_gcp(pts,
                                 pixel_x='pixel_x', pixel_y='pixel_y',
                                 map_x='map_x', map_y='map_y'):
    """Calculates Helmert transform from ground control points.

    This includes y-axis flipping.

    Args:
        pts (pandas.DataFrame): storing gcp points
        pixel_x, pixel_y, map_x, map_y (str): keys to pixel/map coords

    Returns:
        affine.Affine: affine transformation describing raster georef
    """
    assert (set([pixel_x, pixel_y, map_x, map_y])
            .issubset(pts.columns))
    transform, _ = cv.estimateAffinePartial2D(
        pts.loc[:, [pixel_x, pixel_y]].values * np.array([1, -1]),  # from
        pts.loc[:, [map_x, map_y]].values,  # to
        method=cv.LMEDS)
    transform = (rasterio.transform.Affine(*transform.flatten()) *
                 rasterio.transform.Affine(1, 0, 0, 0, -1, 0))
    return transform


def create_batch_symlink(src, dst, suffix):
    """Creates symbolic links in batches.

    All files in `src` folders (and subfolders) with `suffix` will
    be traversed. A symbolic link to them (with the same relative
    path) will be created in `dst`.

    Args:
        src, dst (str): file directory
        suffix (str): file suffix with a leading dot
    """
    files = [os.path.relpath(f, src)
             for f in glob.glob(os.path.join(src, '**/*' + suffix),
                                recursive=True)]
    # prepare folders
    dst_dirs = set(os.path.dirname(os.path.join(dst, f))
                   for f in files)
    for dst_dir in dst_dirs:
        os.makedirs(dst_dir, exist_ok=True)
    # create symbolic links
    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        if os.path.lexists(dst_file):
            os.remove(dst_file)
        os.symlink(src=os.path.abspath(src_file),
                   dst=dst_file)


def convert_to_bbox(transform, width, height):
    """Outputs a shapely polygon bounding box.

    Args:
        transform (affine.Affine): from pixel space to world crs space
        width, height (int): in pixels

    Returns:
        shapely.geometry.Polygon: the bounding box
    """
    box = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]])
    box = transform_coord(transform, to='xy', colrow=box)
    box = shapely.geometry.Polygon(box)
    return box


def get_centroid_dist(transforms, widths, heights):
    """Get distance between centroids based on two sets of georef info.

    Example:
        >>> from affine import Affine
        >>> get_centroid_dist((Affine.identity(), Affine.translation(3, 4)),
        ...                   (10, 10), (20, 20))
        5

    Args:
        transforms (iterable of affine.Affine [2,]): corresponds to image 0/1
        widths, heights (iterable of int [2,]): corresponds to image 0/1

    Returns:
        float: distance between centroids
    """
    # unpack
    trans0, trans1 = transforms
    w0, w1 = widths
    h0, h1 = heights
    # get centroids
    centroid0_w, centroid0_h = trans0 * (w0 / 2, h0 / 2)
    centroid1_w, centroid1_h = trans1 * (w1 / 2, h1 / 2)
    # take distance
    return np.sqrt((centroid1_w - centroid0_w) ** 2 +
                   (centroid1_h - centroid0_h) ** 2)


def prepare_folder(files):
    """Make folders for files, if necessary.

    Args:
        files (list of str): all the files of interest
    """
    # drop empty strings
    dirs = set(os.path.dirname(f) for f in files) - {''}
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def snap_to_grid(bounds, zoom):
    """Snaps a bbox to XYZ tile grid.

    Args:
        bounds (tuple of float): xmin, ymin, xmax, ymax, in web mercator crs
            (EPSG: 3857)
        zoom (int): zoom level

    Returns:
        tuple of int: xmin, ymin, xmax, ymax, (tile number at the specified
            zoom level)
    """
    MAX_SIZE = 40075016.68  # meters, earth's circumsference
    xmin, ymin, xmax, ymax = bounds
    xtile_min = int(
        np.floor((xmin + MAX_SIZE / 2) / MAX_SIZE * (2 ** zoom)))
    xtile_max = int(
        np.ceil((xmax + MAX_SIZE / 2) / MAX_SIZE * (2 ** zoom))) - 1
    ytile_min = int(
        np.floor((MAX_SIZE / 2 - ymax) / MAX_SIZE * (2 ** zoom)))
    ytile_max = int(
        np.ceil((MAX_SIZE / 2 - ymin) / MAX_SIZE * (2 ** zoom))) - 1
    return xtile_min, ytile_min, xtile_max, ytile_max


def grid_to_bounds(tile_bounds, zoom):
    """Get bounding box (in Web Mercator EPSG: 3857 system) of XYZ tiles.

    Args:
        tile_bounds (tuple of int): xtile_min, ytile_min, xtile_max, ytile_max
        zoom (int): zoom level

    Returns:
        bounds (tuple of float): xmin, ymin, xmax, ymax, in web mercator crs
            (EPSG: 3857)
    """
    MAX_SIZE = 40075016.68  # meters, earth's circumsference
    xtile_min, ytile_min, xtile_max, ytile_max = tile_bounds
    xmin = xtile_min / (2 ** zoom) * MAX_SIZE - MAX_SIZE / 2
    xmax = (xtile_max + 1) / (2 ** zoom) * MAX_SIZE - MAX_SIZE / 2
    ymin = MAX_SIZE / 2 - (ytile_max + 1) / (2 ** zoom) * MAX_SIZE
    ymax = MAX_SIZE / 2 - (ytile_min) / (2 ** zoom) * MAX_SIZE
    return xmin, ymin, xmax, ymax


def cut_to_tiles(array, tile_bounds, zoom, folder):
    """Cut a large array into small 256x256 tiles and save them.

    Args:
        array (numpy.ndarray): array to be cut
        tile_bounds (tuple of int): xtile_min, ytile_min, xtile_max, ytile_max
            indices of x, y tiles that are in the array, used for naming files
        zoom (int): zoom level, used for naming files
        folder (str): directory to store tiles in. A tiles/ folder will be
            created here.
    """
    xtile_min, ytile_min, xtile_max, ytile_max = tile_bounds
    height, width = array.shape
    assert height == 256 * (ytile_max - ytile_min + 1)
    assert width == 256 * (xtile_max - xtile_min + 1)
    prepare_folder([
        os.path.join(folder, 'tiles/{z}/{x}/'.format(z=zoom, x=x))
        for x in range(xtile_min, xtile_max + 1)])
    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            cv.imwrite(
                os.path.join(folder,
                             'tiles/{z}/{x}/{y}.png'.format(z=zoom, x=x, y=y)),
                array[(y - ytile_min) * 256:(y - ytile_min + 1) * 256,
                      (x - xtile_min) * 256:(x - xtile_min + 1) * 256])


def generate_low_zoom_tiles(min_zoom, max_zoom, folder):
    """Given max_zoom tiles, generate a pyramid of tiles up to min_zoom level.

    Args:
        min_zoom, max_zoom (int): min/max zoom level
        folder (str): directory to store tiles in. A tiles/ folder will be
            assumed to exist here.
    """
    for zoom in np.arange(min_zoom, max_zoom)[::-1]:
        print('Generating tiles for zoom level {}'.format(zoom))
        input_tiles = set(glob.glob(
            os.path.join(folder, 'tiles/{z}/*/*.png'.format(z=zoom + 1))))
        while len(input_tiles) > 0:
            input_tile = input_tiles.pop()
            input_x, input_y = input_tile.replace('.png', '').split('/')[-2:]
            input_x, input_y = int(input_x), int(input_y)
            input_xs = ([input_x, input_x + 1] if input_x % 2 == 0 else
                        [input_x - 1, input_x])
            input_ys = ([input_y, input_y + 1] if input_y % 2 == 0 else
                        [input_y - 1, input_y])
            output_x, output_y = input_x // 2, input_y // 2
            output_img = np.zeros((256, 256))
            output_tile = os.path.join(folder, 'tiles/{z}/{x}/{y}.png'.format(
                z=zoom, x=output_x, y=output_y))
            for idx_x, input_x in enumerate(input_xs):
                for idx_y, input_y in enumerate(input_ys):
                    input_tile = os.path.join(
                        folder,
                        'tiles/{z}/{x}/{y}.png'.format(
                            z=zoom + 1, x=input_x, y=input_y))
                    input_tiles.discard(input_tile)
                    if os.path.isfile(input_tile):
                        input_img = cv.imread(input_tile, cv.IMREAD_GRAYSCALE)
                        input_img = cv.resize(input_img, (128, 128))
                        output_img[idx_y * 128:(idx_y + 1) * 128,
                                   idx_x * 128:(idx_x + 1) * 128] = input_img
            prepare_folder([output_tile])
            cv.imwrite(output_tile, output_img)
