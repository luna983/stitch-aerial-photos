import pytest

import os
import time
import numpy as np
import cv2 as cv
import rasterio
import rasterio.transform
import rasterio.warp

from ..preprocess import preprocess


@pytest.fixture
def raw_file(data_dir):
    return str(data_dir / 'test_preprocess_main.jpg')


@pytest.fixture
def processed_file(data_dir):
    return str(data_dir / 'test_preprocess_processed.png')


@pytest.fixture
def reproj_file(data_dir):
    return str(data_dir / 'test_preprocess_reprojected.png')


@pytest.fixture
def cache_dir(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache_file(cache_dir):
    return str(cache_dir / 'test_preprocess_main.tif')


def test_preprocess_output(raw_file, processed_file, reproj_file):
    # check preprocessing output
    # from visual inspections I conclude that preprocessing is as expected
    img, trans = preprocess(file=raw_file, scale=0.5,
                            crop={'top': 0.5, 'bottom': 0.8,
                                  'left': 0.3, 'right': 1},
                            cache=False, cache_dir=None)
    # 1. test for image shape
    assert img.shape == (225, 455)  # height, width
    # 2. test for stable output, saved by cv.imwrite(processed_file, img)
    expected = cv.imread(processed_file, cv.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(img, expected)
    # 3. test for affine transform
    assert (rasterio.transform.Affine(2, 0, 390, 0, 2, 750) ==
            pytest.approx(trans))
    # 4. reproject to make sure transform is correct
    reproj_img = np.zeros((1500, 1300), np.uint8)  # same as raw file size
    rasterio.warp.reproject(
        source=img,
        destination=reproj_img,
        src_transform=trans,
        src_crs={'init': 'EPSG:3857'},  # place holder, no meaning
        dst_transform=rasterio.transform.Affine.identity(),
        dst_crs={'init': 'EPSG:3857'},  # place holder, no meaning
        resampling=rasterio.warp.Resampling.nearest)
    # test for stable output, saved by cv.imwrite(reproj_file, reproj_img)
    expected = cv.imread(reproj_file, cv.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(reproj_img, expected)


def test_preprocess_cache(raw_file, cache_dir, cache_file):

    # 1. check that cache file is saved
    tic = time.time()
    img0, trans0 = preprocess(file=raw_file, scale=0.5,
                              crop={'top': 0.5, 'bottom': 0.8,
                                    'left': 0.3, 'right': 1},
                              cache=True, cache_dir=str(cache_dir))
    toc = time.time()
    print('Preprocessing w/o caching: {:.3f}s'.format(toc - tic))
    assert os.path.isfile(cache_file)
    size = os.path.getsize(cache_file)

    # 2. check that cache file is loaded
    tic = time.time()
    img1, trans1 = preprocess(file=raw_file, scale=0.5,
                              crop={'top': 0.5, 'bottom': 0.8,
                                    'left': 0.3, 'right': 1},
                              cache=True, cache_dir=str(cache_dir))
    toc = time.time()
    print('Preprocessing w/ caching: {:.3f}s'.format(toc - tic))
    np.testing.assert_array_equal(img0, img1)
    assert trans0 == pytest.approx(trans1)
    assert size == os.path.getsize(cache_file)

    # 3. check that cache file is overwritten w/ new settings
    img2, trans2 = preprocess(file=raw_file, scale=0.1,
                              crop={'top': 0.2, 'bottom': 0.3,
                                    'left': 0.1, 'right': 1},
                              cache=True, cache_dir=str(cache_dir))
    # test for image shape
    assert img2.shape == (15, 117)  # height, width
    assert size != os.path.getsize(cache_file)
