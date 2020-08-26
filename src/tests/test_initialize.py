import pytest

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform

from ..initialize import (parse_sortie, initialize_centroids,
                          initialize_theta_scale)


def test_parse_sortie_corners(tmp_path):
    sortie_dir = str(tmp_path / 'test.csv')
    in_df = pd.DataFrame({
        'idx0': [10, 10, 10, 10, 10, 10, 10, 10],
        'idx1': [5, 5, 3, 3, 61, 61, 61, 61],
        'pixel_x': [100., 100, 400, 400, -10, 0, -10, 0],
        'pixel_y': [0., 100, 100, 0, -10, -10, 0, 0],
        'swath_id': [0, 0, 0, 0, 2, 2, 2, 2],
        'extra': [1, 2, 3, 4, 5, 6, 7, 8],
    })
    in_df.to_csv(sortie_dir, index=False)
    out_df = parse_sortie(sortie_dir, sortie_input_type='swath_corners')
    out_df_true = pd.DataFrame({
        'swath_id': [0, 0, 0, 2],
        'idx0': [10, 10, 10, 10],
        'idx1': [3, 4, 5, 61],
        'sortie_x_centroid': [350., 250, 150, -5],
        'sortie_y_centroid': [50., 50, 50, -5],
    })
    pd.testing.assert_frame_equal(out_df, out_df_true, check_dtype=False)


def test_parse_sortie_centroids(tmp_path):
    sortie_dir = str(tmp_path / 'test.csv')
    in_df = pd.DataFrame({
        'idx0': [10, 10, 11, 11],
        'idx1': [5, 3, 61, 61],
        'pixel_x': [100, 300, -10, -10],
        'pixel_y': [0, 200, -10, -10],
        'swath_id': [0, 0, 2, 2],
        'extra': [1, 2, 3, 4],
    })
    in_df.to_csv(sortie_dir, index=False)
    out_df = parse_sortie(sortie_dir, sortie_input_type='swath_centroids')
    out_df_true = pd.DataFrame({
        'swath_id': [0, 0, 0, 2],
        'idx0': [10, 10, 10, 11],
        'idx1': [3, 4, 5, 61],
        'sortie_x_centroid': [300, 200, 100, -10],
        'sortie_y_centroid': [200, 100, 0, -10],
    })
    pd.testing.assert_frame_equal(out_df, out_df_true, check_dtype=False)


@pytest.mark.parametrize(
    'indices,links,widths,heights,x0,x1,y0,y1,expected_x,expected_y',
    [
        # vanilla
        pytest.param(
            [(10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8)],
            {((10, 3), (10, 4)): rasterio.transform.Affine.translation(10, 0),
             ((10, 4), (10, 5)): rasterio.transform.Affine.translation(10, 0),
             ((10, 5), (10, 6)): rasterio.transform.Affine.translation(10, 0),
             ((10, 6), (10, 7)): rasterio.transform.Affine.translation(20, 0),
             ((10, 7), (10, 8)): rasterio.transform.Affine.translation(30, 0)},
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            0, 80, 0, 40,
            np.array([0, 10, 20, 30, 50, 80]),
            np.array([0, 5, 10, 15, 25, 40]),
        ),
        # different sizes of images
        pytest.param(
            [(10, 3), (10, 4), (10, 5)],
            {((10, 3), (10, 4)): rasterio.transform.Affine.translation(10, 0),
             ((10, 4), (10, 5)): rasterio.transform.Affine.translation(10, 0)},
            [10, 10, 40],
            [10, 10, 10],
            0, 35, 0, 35,
            np.array([0, 10, 35]),
            np.array([0, 10, 35]),
        ),
        # with missing links
        pytest.param(
            [(10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8)],
            {((10, 3), (10, 4)): None,
             ((10, 4), (10, 5)): rasterio.transform.Affine.translation(10, 0),
             ((10, 5), (10, 6)): rasterio.transform.Affine.translation(10, 0),
             ((10, 6), (10, 7)): None,
             ((10, 7), (10, 8)): rasterio.transform.Affine.translation(30, 0)},
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            0, 80, 0, 40,
            np.array([0, 10, 20, 30, 50, 80]),
            np.array([0, 5, 10, 15, 25, 40]),
        ),
        # with all missing links
        pytest.param(
            [(10, 3), (10, 4), (10, 5)],
            {((10, 3), (10, 4)): None,
             ((10, 4), (10, 5)): None},
            [10, 10, 10],
            [10, 10, 10],
            0, 20, 0, 40,
            np.array([0, 10, 20]),
            np.array([0, 20, 40]),
        ),
        # with only one image
        pytest.param(
            [(10, 3)],
            {((10, 3), (10, 4)): rasterio.transform.Affine.translation(10, 0)},
            [10],
            [10],
            2, 2, 3, 3,
            np.array([2]),
            np.array([3]),
        ),
    ],
)
def test_initialize_centroids(indices, links, widths, heights,
                              x0, x1, y0, y1, expected_x, expected_y):
    output_x, output_y = initialize_centroids(
        indices, links, widths, heights, x0, x1, y0, y1)
    np.testing.assert_allclose(output_x, expected_x)
    np.testing.assert_allclose(output_y, expected_y)


@pytest.mark.parametrize(
    'n_thetas,scales,nodes,links,width,height,xs_init,ys_init,expected',
    [
        # perfect case
        pytest.param(
            36,
            [0.9, 1, 1.1],
            [(1, 1), (1, 2), (1, 3)],
            {((1, 1), (1, 2)): rasterio.transform.Affine.translation(10, 0),
             ((1, 2), (1, 3)): rasterio.transform.Affine.translation(10, 0)},
            [20, 20, 20],
            [30, 30, 30],
            [0, 10, 20],
            [0, 0, 0],
            (0., 1.),
        ),
        # theta selection
        pytest.param(
            360,
            [1],
            [(1, 1), (1, 2), (1, 3)],
            {((1, 1), (1, 2)): rasterio.transform.Affine.translation(10, 0),
             ((1, 2), (1, 3)): rasterio.transform.Affine.translation(10, 0)},
            [20, 20, 20],
            [20, 20, 20],
            [0, 10, 20],
            [0, 10, 20],
            (np.pi / 4, 1.),
        ),
        # scale selection
        pytest.param(
            36,
            [0.4, 0.5, 0.6, 1],
            [(1, 1), (1, 2), (1, 3)],
            {((1, 1), (1, 2)): rasterio.transform.Affine.translation(10, 0),
             ((1, 2), (1, 3)): rasterio.transform.Affine.translation(10, 0)},
            [20, 20, 20],
            [30, 30, 30],
            [0, 5, 10],  # squeeze the images together
            [0, 0, 0],
            (0., 0.5),
        ),
    ]
)
def test_initialize_theta_scale(n_thetas, scales,
                                nodes, links, width, height,
                                xs_init, ys_init, expected):
    assert initialize_theta_scale(
        n_thetas=n_thetas, scales=scales,
        nodes=nodes, links=links, width=width, height=height,
        xs_init=xs_init, ys_init=ys_init) == pytest.approx(expected, rel=1e-3)
