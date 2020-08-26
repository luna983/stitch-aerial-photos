import pytest

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import shapely
import shapely.geometry
import cv2 as cv

from ..utils import (transform_coord,
                     convert_affine,
                     calculate_transform_from_gcp,
                     create_batch_symlink,
                     convert_to_bbox,
                     get_centroid_dist,
                     prepare_folder,
                     snap_to_grid,
                     grid_to_bounds,
                     cut_to_tiles,
                     generate_low_zoom_tiles)


@pytest.mark.parametrize(
    'transform,to,xy,colrow,expected',
    [
        pytest.param(
            rasterio.transform.Affine(2, 0, 5, 0, -2, 10),
            'colrow',
            np.array([[0, 0], [25, 10]]),
            None,
            np.array([[-2.5, 5], [10, 0]])
        ),
        pytest.param(
            rasterio.transform.Affine(1, 1, 4, 1, -1, 7),
            'xy',
            None,
            np.array([[0, 0], [20, 10]]),
            np.array([[4, 7], [34, 17]])
        ),
    ],
)
def test_transform_coord(transform, to, xy, colrow, expected):
    assert (
        transform_coord(transform, to, xy, colrow) == pytest.approx(expected))


@pytest.mark.parametrize(
    'transform,to,expected',
    [
        pytest.param(
            [1, 1, 1, -1, 5, 7],
            'affine.Affine',
            rasterio.transform.Affine(1, 1, 4, 1, -1, 7),
        ),
        pytest.param(
            rasterio.transform.Affine(1, 1, 4, 1, -1, 7),
            'world_file',
            [1, 1, 1, -1, 5, 7],
        ),
    ],
)
def test_convert_affine(transform, to, expected):
    # refer to this
    # https://www.perrygeo.com/python-affine-transforms.html
    assert convert_affine(transform, to) == pytest.approx(expected)


@pytest.mark.parametrize(
    'pts,expected',
    [
        pytest.param(
            pd.DataFrame({
                'pixel_x': [0, 0, 5, 5],
                'pixel_y': [0, 5, 0, 5],
                'map_x': [6, 1, 6, 1],
                'map_y': [6, 6, 1, 1],
            }),
            rasterio.transform.Affine(0, -1, 6, -1, 0, 6),
        ),
    ],
)
def test_calculate_transform_from_gcp(pts, expected):
    assert calculate_transform_from_gcp(pts) == pytest.approx(expected)


def test_create_batch_symlink(tmp_path):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    dst.mkdir()
    (src / 'sub0').mkdir()
    (src / 'sub1').mkdir()
    (src / 'sub0' / 'test.yes').touch()
    (src / 'sub0' / 'test.no').touch()
    create_batch_symlink(src=src, dst=dst, suffix='.yes')
    assert (dst / 'sub0').is_dir()
    assert (dst / 'sub0' / 'test.yes').is_file()
    assert (dst / 'sub0' / 'test.yes').is_symlink()
    assert not (dst / 'sub1').exists()
    assert not (dst / 'sub0' / 'test.no').exists()


@pytest.mark.parametrize(
    'transform,width,height,expected',
    [
        pytest.param(
            rasterio.transform.Affine(1, 1, 4, 1, -1, 7),
            20,
            30,
            shapely.geometry.Polygon(
                [[4, 7], [34, -23], [54, -3], [24, 27], [4, 7]])
        ),
    ],
)
def test_convert_to_bbox(transform, width, height, expected):
    assert convert_to_bbox(transform, width, height).contains(expected)
    assert expected.contains(convert_to_bbox(transform, width, height))


@pytest.mark.parametrize(
    'transforms,widths,heights,expected',
    [
        pytest.param(
            (rasterio.transform.Affine.identity(),
             rasterio.transform.Affine.translation(14, 2) *
             rasterio.transform.Affine.rotation(90)),
            (10, 4),
            (20, 2),
            10
        ),
    ],
)
def test_get_centroid_dist(transforms, widths, heights, expected):
    assert (get_centroid_dist(transforms, widths, heights) ==
            pytest.approx(expected))


def test_prepare_folder(tmp_path):
    (tmp_path / 'test0').mkdir()
    prepare_folder([
        str(tmp_path / 'test0/file0.test'),
        str(tmp_path / 'test0/file1.test'),
        str(tmp_path / 'test1/file0.test'),
        str(tmp_path / 'test1/file1.test'),
        str(tmp_path / 'file2.test')])
    dirs = [str(d.relative_to(tmp_path))
            for d in tmp_path.iterdir() if d.is_dir()]
    assert set(dirs) == {'test0', 'test1'}


@pytest.mark.parametrize(
    'bounds,zoom,expected',
    [
        pytest.param(
            (- 40075016.68 / 2, - 40075016.68 / 2,
             40075016.68 / 2, 40075016.68 / 2),
            0,
            (0, 0, 0, 0),
        ),
        pytest.param(
            (- 200, - 200, 0, 0),
            0,
            (0, 0, 0, 0),
        ),
        pytest.param(
            (- 200, 0, 0, 200),
            1,
            (0, 0, 0, 0),
        ),
        pytest.param(
            (- 200, - 200, 200, 200),
            1,
            (0, 0, 1, 1),
        ),
        pytest.param(
            (0, 0, 200, 200),
            1,
            (1, 0, 1, 0),
        ),
    ],
)
def test_snap_to_grid(bounds, zoom, expected):
    assert snap_to_grid(bounds, zoom) == expected


@pytest.mark.parametrize(
    'tile_bounds,zoom,expected',
    [
        pytest.param(
            (0, 0, 0, 0),
            0,
            (- 40075016.68 / 2, - 40075016.68 / 2,
             40075016.68 / 2, 40075016.68 / 2),
        ),
        pytest.param(
            (0, 0, 1, 1),
            1,
            (- 40075016.68 / 2, - 40075016.68 / 2,
             40075016.68 / 2, 40075016.68 / 2),
        ),
        pytest.param(
            (0, 1, 0, 1),
            1,
            (- 40075016.68 / 2, - 40075016.68 / 2, 0, 0),
        ),
        pytest.param(
            (0, 2, 1, 3),
            2,
            (- 40075016.68 / 2, - 40075016.68 / 2, 0, 0),
        ),
    ],
)
def test_grid_to_bounds(tile_bounds, zoom, expected):
    assert grid_to_bounds(tile_bounds, zoom) == pytest.approx(expected)


def test_cut_to_tiles(tmp_path):
    xx, yy = np.meshgrid(np.arange(512), np.arange(512))
    array = (xx >= 256).astype(np.uint8) + (yy >= 256).astype(np.uint8)
    cut_to_tiles(
        array=array,
        tile_bounds=(0, 0, 1, 1),
        zoom=1,
        folder=str(tmp_path))
    np.testing.assert_equal(
        np.full((256, 256), 0),
        cv.imread(str(tmp_path / 'tiles' / '1' / '0' / '0.png'),
                  cv.IMREAD_GRAYSCALE))
    np.testing.assert_equal(
        np.full((256, 256), 1),
        cv.imread(str(tmp_path / 'tiles' / '1' / '0' / '1.png'),
                  cv.IMREAD_GRAYSCALE))
    np.testing.assert_equal(
        np.full((256, 256), 1),
        cv.imread(str(tmp_path / 'tiles' / '1' / '1' / '0.png'),
                  cv.IMREAD_GRAYSCALE))
    np.testing.assert_equal(
        np.full((256, 256), 2),
        cv.imread(str(tmp_path / 'tiles' / '1' / '1' / '1.png'),
                  cv.IMREAD_GRAYSCALE))


def test_generate_low_zoom_tiles(tmp_path):
    (tmp_path / 'tiles').mkdir()
    (tmp_path / 'tiles' / '2').mkdir()
    (tmp_path / 'tiles' / '1').mkdir()
    (tmp_path / 'tiles' / '2' / '0').mkdir()
    (tmp_path / 'tiles' / '2' / '3').mkdir()
    cv.imwrite(str(tmp_path / 'tiles' / '2' / '0' / '0.png'),
               np.full((256, 256), 1, dtype=np.uint8))
    cv.imwrite(str(tmp_path / 'tiles' / '2' / '3' / '0.png'),
               np.full((256, 256), 10, dtype=np.uint8))
    generate_low_zoom_tiles(
        min_zoom=0, max_zoom=2, folder=tmp_path)
    assert cv.imread(
        str(tmp_path / 'tiles' / '1' / '1' / '0.png'),
        cv.IMREAD_GRAYSCALE).sum() == 128 * 128 * 10
    assert cv.imread(
        str(tmp_path / 'tiles' / '1' / '0' / '0.png'),
        cv.IMREAD_GRAYSCALE).sum() == 128 * 128
    assert cv.imread(
        str(tmp_path / 'tiles' / '0' / '0' / '0.png'),
        cv.IMREAD_GRAYSCALE).sum() == 64 * 64 * 11
