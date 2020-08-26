import os
import warnings
import cv2 as cv
import rasterio
import rasterio.transform
import numpy as np


def preprocess(file, scale=None, crop=None, cache=False, cache_dir=None):
    """Loads the aerial photo. Cache to reduce loading time.

    Args:
        file (str): file path
        scale (float): scaling factor, <1 means down-resolutioning
            if None, no scaling
        crop (dict): 4 keys: 'top', 'bottom', 'left', 'right'
            each value is a float in (0, 1), representing the proportion
            of image width/height; if None, no cropping
        cache (bool): whether to cache images to speed up processing
        cache_dir (str): cannot be None if cache is True, NOTE: ALL CACHED
            IMAGES NEED TO HAVE A UNIQUE NAME

    Returns:
        numpy.ndarray [height, width]: loaded grayscale image
        affine.Affine: the geo transform from the loaded image to the
            original image, NOT to crs units
    """
    if scale is None:
        skip_scale = True
        scale = 1
    else:
        skip_scale = False
    if crop is None:
        skip_crop = True
        crop = {'top': 0, 'bottom': 1, 'left': 0, 'right': 1}
    else:
        skip_crop = False
    # read original image metadata
    with warnings.catch_warnings():
        warnings.simplefilter(
            action='ignore',
            category=rasterio.errors.NotGeoreferencedWarning)
        raw_ds = rasterio.open(file)
        raw_width, raw_height = raw_ds.width, raw_ds.height
    # compute transform and output size
    out_trans = rasterio.transform.Affine(
        1 / scale, 0., int(round(crop['left'] * raw_width)),
        0., 1 / scale, int(round(crop['top'] * raw_height)))
    out_width = int(round((crop['right'] - crop['left']) *
                          raw_width * scale))
    out_height = int(round((crop['bottom'] - crop['top']) *
                           raw_height * scale))
    # determine if cached images would be used
    if cache:
        assert cache_dir is not None
        # generate cache file path
        cache_file = os.path.join(
            cache_dir, os.path.splitext(os.path.basename(file))[0] + '.tif')
        if os.path.isfile(cache_file):
            # load cached file if it exists
            cache_ds = rasterio.open(cache_file)
            if (out_trans.almost_equals(cache_ds.transform) and
                    out_width == cache_ds.width and
                    out_height == cache_ds.height):
                return cache_ds.read().squeeze(0), cache_ds.transform
    # otherwise, process the raw file
    if raw_ds.count > 1:
        img = raw_ds.read().mean(0)
        img = img.astype(np.uint8)  # convert from float64 to uint8
    else:
        img = raw_ds.read().squeeze(0)
    if not skip_crop:
        # crop black border
        img = img[
            int(crop['top'] * raw_height):int(crop['bottom'] * raw_height),
            int(crop['left'] * raw_width):int(crop['right'] * raw_width)]
    if not skip_scale:
        # resize
        img = cv.resize(img, dsize=(out_width, out_height))
    # save if caching
    if cache:
        with rasterio.open(cache_file, 'w', driver='GTiff',
                           height=out_height, width=out_width, count=1,
                           dtype=rasterio.uint8,
                           transform=out_trans) as out_ds:
            out_ds.write(img, 1)
    return img, out_trans
