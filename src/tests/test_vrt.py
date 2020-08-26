import pytest

import os
import pickle
import numpy as np
import pandas as pd
import cv2 as cv
import rasterio
import rasterio.transform
import geopandas
import geopandas.testing
import shapely
import shapely.geometry

from ..vrt import VirtualRaster


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / 'img'
    d.mkdir()
    (d / 'TEST_0010').mkdir()
    (d / 'TEST_0011').mkdir()
    cv.imwrite(str(d / 'TEST_0010' / 'TEST_0010_0003.jpg'),
               np.ones((20, 40)) * 255)
    cv.imwrite(str(d / 'TEST_0010' / 'TEST_0010_0004.jpg'),
               np.ones((40, 40)) * 255)
    cv.imwrite(str(d / 'TEST_0011' / 'TEST_0011_0061.jpg'),
               np.zeros((20, 40)))
    return d


@pytest.fixture
def wld_dir_to(tmp_path):
    d = tmp_path / 'wld_to'
    d.mkdir()
    return d


@pytest.fixture
def wld_dir_from(tmp_path):
    d = tmp_path / 'wld_from'
    d.mkdir()
    (d / 'TEST_0010').mkdir()
    (d / 'TEST_0011').mkdir()
    with open(str(d / 'TEST_0010' / 'TEST_0010_0003.jgw'), 'w') as f:
        f.write('2.0\n0.0\n0.0\n-2.0\n1.0\n9.0')
    with open(str(d / 'TEST_0010' / 'TEST_0010_0004.jgw'), 'w') as f:
        f.write('1.0\n0.0\n0.0\n-1.0\n10.5\n9.5')
    with open(str(d / 'TEST_0011' / 'TEST_0011_0061.jgw'), 'w') as f:
        f.write('1.0\n0.0\n0.0\n-1.0\n20.5\n9.5')
    return d


@pytest.fixture
def mosaic_gcp_dir(tmp_path):
    d = tmp_path / 'gcp_mosaic'
    d.mkdir()
    mosaic_p = pd.DataFrame({
        'file_id': ['TEST_0011/TEST_0011_0061',
                    'TEST_0010/TEST_0010_0004',
                    'TEST_0010/TEST_0010_0003'],
        'world_trans': [rasterio.transform.Affine(1.25, 0, 25, 0, 1.25, 0),
                        rasterio.transform.Affine(1.25, 0, 12.5, 0, 1.25, 0),
                        rasterio.transform.Affine(2.5, 0, 0, 0, 2.5, 0)],
        'width': [40, 40, 40],
        'height': [20, 40, 20],
    })
    with open(str(d / 'mosaic.p'), 'wb') as f:
        pickle.dump(mosaic_p, f)
    mosaic_csv = pd.DataFrame({
        'pixel_x': [50, 50, 90],
        'pixel_y': [6.25, 37.5, 12.5],
        'map_x': [40, 40, 72],
        'map_y': [5, -20, 0],
    })
    mosaic_csv.to_csv(str(d / 'mosaic.csv'), index=False)
    return d


@pytest.fixture
def ind_gcp_dir(tmp_path):
    d = tmp_path / 'gcp_ind'
    d.mkdir()
    return d


@pytest.fixture
def ind_gcp_csv(ind_gcp_dir):
    (ind_gcp_dir / 'TEST_0010').mkdir()
    df = pd.DataFrame({'pixel_x': [0, 1], 'pixel_y': [0, 1],
                       'map_x': [0, 1], 'map_y': [0, -1]})
    df.to_csv(str(ind_gcp_dir / 'TEST_0010' / 'TEST_0010_0003.csv'),
              index=False)
    df.to_csv(str(ind_gcp_dir / 'TEST_0010' / 'TEST_0010_0004.csv'),
              index=False)


@pytest.fixture
def init_csv_dir(tmp_path):
    f = str(tmp_path / 'init.csv')
    df = pd.DataFrame({
        'swath_id': [0, 0, 2],
        'idx0': [10, 10, 11],
        'idx1': [3, 4, 61],
        'x_init': [350., 250, -5],
        'y_init': [50., 50, -5],
        'theta_init': [0, 0, 0],
        'scale_init': [1, 1, 1],
        'file_id': ['TEST_0010/TEST_0010_0003',
                    'TEST_0010/TEST_0010_0004',
                    'TEST_0011/TEST_0011_0061'],
    })
    df.to_csv(f, index=False)
    return f


@pytest.fixture
def v_from_csv(init_csv_dir, img_dir, wld_dir_to):
    v = VirtualRaster.from_csv(
        file=str(init_csv_dir),
        img_dir=str(img_dir),
        wld_dir=str(wld_dir_to),
        img_suffix='.jpg',
        wld_suffix='.jgw',
        crs='EPSG:3857',
        index_cols=['idx0', 'idx1'],
        verbose=True)
    return v


@pytest.fixture
def df_from_csv(img_dir, wld_dir_to):
    df = pd.DataFrame({
        'swath_id': [0, 0, 2],
        'idx0': [10, 10, 11],
        'idx1': [3, 4, 61],
        'x_init': [350., 250, -5],
        'y_init': [50., 50, -5],
        'theta_init': [0, 0, 0],
        'scale_init': [1, 1, 1],
        'file_id': ['TEST_0010/TEST_0010_0003',
                    'TEST_0010/TEST_0010_0004',
                    'TEST_0011/TEST_0011_0061'],
        'img_file': [str(img_dir / 'TEST_0010' / 'TEST_0010_0003.jpg'),
                     str(img_dir / 'TEST_0010' / 'TEST_0010_0004.jpg'),
                     str(img_dir / 'TEST_0011' / 'TEST_0011_0061.jpg')],
        'wld_file': [str(wld_dir_to / 'TEST_0010' / 'TEST_0010_0003.jgw'),
                     str(wld_dir_to / 'TEST_0010' / 'TEST_0010_0004.jgw'),
                     str(wld_dir_to / 'TEST_0011' / 'TEST_0011_0061.jgw')],
        'width': [40, 40, 40],
        'height': [20, 40, 20],
    }).set_index(['idx0', 'idx1'])
    return df


@pytest.fixture
def df_from_world_files(df_from_csv, wld_dir_from):
    df = df_from_csv.drop(
        columns=['x_init', 'y_init', 'swath_id'])
    df.loc[:, 'wld_file'] = [
        str(wld_dir_from / 'TEST_0010' / 'TEST_0010_0003.jgw'),
        str(wld_dir_from / 'TEST_0010' / 'TEST_0010_0004.jgw'),
        str(wld_dir_from / 'TEST_0011' / 'TEST_0011_0061.jgw')]
    df.loc[:, 'world_trans'] = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, -2, 10),
         rasterio.transform.Affine(1, 0, 10, 0, -1, 10),
         rasterio.transform.Affine(1, 0, 20, 0, -1, 10)],
        index=df.index)
    bboxes = [
        shapely.geometry.Polygon([[0, 10], [80, 10],
                                  [80, -30], [0, -30], [0, 10]]),
        shapely.geometry.Polygon([[10, 10], [50, 10],
                                  [50, -30], [10, -30], [10, 10]]),
        shapely.geometry.Polygon([[20, 10], [60, 10],
                                  [60, -10], [20, -10], [20, 10]])]
    df = geopandas.GeoDataFrame(df, geometry=bboxes)
    df = df.loc[:, ['width', 'height',
                    'file_id', 'img_file', 'wld_file', 'world_trans',
                    'geometry']]
    return df


def test_from_csv(v_from_csv, df_from_csv):
    pd.testing.assert_frame_equal(
        v_from_csv.df, df_from_csv,
        check_dtype=False, check_index_type=False)


def test_from_world_files(img_dir, wld_dir_from, df_from_world_files):
    v = VirtualRaster.from_world_files(
        img_dir=str(img_dir),
        wld_dir=str(wld_dir_from),
        img_suffix='.jpg',
        wld_suffix='.jgw',
        crs='EPSG:3857')
    geopandas.testing.assert_geodataframe_equal(v.df, df_from_world_files)


def test_to_world_files(wld_dir_to, v_from_csv):
    v_from_csv.df.loc[:, 'world_trans'] = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, -2, 10),
         rasterio.transform.Affine(1, 0, 10, 0, -1, 10),
         rasterio.transform.Affine(1, 0, 20, 0, -1, 10)],
        index=v_from_csv.df.index)
    # try different argument options, default
    v_from_csv.to_world_files()
    # check for file existence, errors will be raised if files do not exist
    (wld_dir_to / 'TEST_0010' / 'TEST_0010_0003.jgw').unlink()
    (wld_dir_to / 'TEST_0010' / 'TEST_0010_0004.jgw').unlink()
    (wld_dir_to / 'TEST_0011' / 'TEST_0011_0061.jgw').unlink()
    # specify directory
    v_from_csv.to_world_files(wld_dir=wld_dir_to)
    assert (wld_dir_to / 'TEST_0010' / 'TEST_0010_0003.jgw').is_file()
    assert (wld_dir_to / 'TEST_0010' / 'TEST_0010_0004.jgw').is_file()
    assert (wld_dir_to / 'TEST_0011' / 'TEST_0011_0061.jgw').is_file()


def test_to_from_world_files(img_dir, wld_dir_to, df_from_world_files):
    """Save to file and then reload, assert the transforms/geoms are equal."""
    v = VirtualRaster(
        df=df_from_world_files,
        img_dir=str(img_dir),
        wld_dir=str(wld_dir_to),
        img_suffix='.jpg',
        wld_suffix='.jgw',
        crs='EPSG:3857')
    v.to_world_files(wld_dir=wld_dir_to)
    v = VirtualRaster.from_world_files(
        img_dir=str(img_dir),
        wld_dir=str(wld_dir_to),
        img_suffix='.jpg',
        wld_suffix='.jgw',
        crs='EPSG:3857')
    geopandas.testing.assert_geoseries_equal(
        df_from_world_files.loc[:, 'geometry'],
        v.df.loc[:, 'geometry'])
    pd.testing.assert_series_equal(
        df_from_world_files.loc[:, 'world_trans'],
        v.df.loc[:, 'world_trans'])


@pytest.mark.parametrize(
    'mode,input_extent_type,input_extent,'
    'output_res,output_bounds,scale,crop,'
    'expected_img_file,expected_trans',
    [
        pytest.param(
            'stack',
            'all',
            None,
            None,
            None,
            None,
            None,
            'test_vrt_show_stack.png',
            rasterio.transform.Affine(0.8, 0, 0, 0, -0.8, 10)
        ),
        pytest.param(
            'stack',
            'bounds',
            (0, -10, 80, 10),
            None,
            None,
            None,
            None,
            'test_vrt_show_bounds.png',
            rasterio.transform.Affine(0.8, 0, 0, 0, -0.8, 10)
        ),
        pytest.param(
            'composite',
            'img_pos',
            [1, 2],
            None,
            None,
            None,
            None,
            'test_vrt_show_subset.png',
            rasterio.transform.Affine(0.5, 0, 10, 0, -0.5, 10)
        ),
        pytest.param(
            'composite',
            'img_name',
            [(10, 4), (11, 61)],
            None,
            None,
            None,
            None,
            'test_vrt_show_subset.png',
            rasterio.transform.Affine(0.5, 0, 10, 0, -0.5, 10)
        ),
        pytest.param(
            'stack',
            'all',
            None,
            1,
            None,
            None,
            None,
            'test_vrt_show_small.png',
            rasterio.transform.Affine(1, 0, 0, 0, -1, 10)
        ),
        pytest.param(
            'stack',
            'all',
            None,
            None,
            (0, -10, 80, 10),
            None,
            None,
            'test_vrt_show_bounds.png',
            rasterio.transform.Affine(0.8, 0, 0, 0, -0.8, 10)
        ),
        pytest.param(
            'stack',
            'all',
            None,
            None,
            None,
            0.5,
            {'top': 0, 'bottom': 0.8, 'left': 0, 'right': 1},
            'test_vrt_show_preprocess.png',
            rasterio.transform.Affine(0.8, 0, 0, 0, -0.8, 10)
        ),
        pytest.param(
            'overlay',
            'img_pos',
            [2, 1],
            None,
            None,
            None,
            None,
            'test_vrt_show_overlay.png',
            rasterio.transform.Affine(0.5, 0, 10, 0, -0.5, 10)
        ),
        pytest.param(
            'composite',
            'all',
            None,
            None,
            None,
            None,
            None,
            'test_vrt_show_composite.png',
            rasterio.transform.Affine(0.8, 0, 0, 0, -0.8, 10)
        ),
    ],
)
def test_show(mode, input_extent_type, input_extent,
              output_res, output_bounds, scale, crop,
              expected_img_file, expected_trans,
              img_dir, wld_dir_from, data_dir):
    # create new virtual raster
    v = VirtualRaster.from_world_files(
        img_dir=str(img_dir),
        wld_dir=str(wld_dir_from),
        img_suffix='.jpg',
        wld_suffix='.jgw',
        crs='EPSG:3857')
    # call show function
    output, trans = v.show(
        mode=mode,
        input_extent_type=input_extent_type, input_extent=input_extent,
        output_res=output_res, output_bounds=output_bounds,
        scale=scale, crop=crop)
    if mode == 'stack':
        output = output.transpose(1, 2, 0)
    # img saved by cv.imwrite(str(data_dir / expected_img_file), output)
    imread_mode = cv.IMREAD_COLOR if mode == 'stack' else cv.IMREAD_GRAYSCALE
    expected_img = cv.imread(str(data_dir / expected_img_file), imread_mode)
    # test image is correct
    np.testing.assert_equal(expected_img, output)
    # test transform is correct
    assert pytest.approx(trans) == expected_trans


@pytest.mark.parametrize(
    'f,graph,expected',
    [
        pytest.param(
            lambda i, j: rasterio.transform.Affine.scale(2),
            None,
            {((10, 3), (10, 4)): rasterio.transform.Affine.scale(2),
             ((10, 4), (10, 3)): rasterio.transform.Affine.scale(0.5),
             ((4, 4), (4, 5)): None},
        ),
        pytest.param(
            lambda i, j: [os.path.basename(i), os.path.basename(j)],
            {(10, 3): [(11, 61)]},
            {((10, 3), (11, 61)): ['TEST_0010_0003.jpg', 'TEST_0011_0061.jpg'],
             ((4, 4), (4, 5)): None},
        ),
    ],
)
def test_build_links(f, graph, expected, v_from_csv):
    v_from_csv.graph = {(10, 3): [(10, 4)], (10, 4): [(10, 3)]}
    v_from_csv.links = {((4, 4), (4, 5)): None}
    v_from_csv.build_links(f=f, graph=graph)
    assert v_from_csv.links == expected


def test_build_graph_links(v_from_csv):
    # test across
    v_from_csv.df.loc[:, 'x_init'] = [350, 250, -5]
    v_from_csv.df.loc[:, 'y_init'] = [50, 50, -5]
    v_from_csv.build_graph_links(
        f=lambda x, y: rasterio.transform.Affine.identity(),
        method='across', neighbor_across_swath=1, max_dist=270)
    assert v_from_csv.graph == {
        (10, 4): [(11, 61)], (11, 61): [(10, 4)]}
    assert v_from_csv.links == {
        ((10, 4), (11, 61)): rasterio.transform.Affine.identity(),
        ((11, 61), (10, 4)): rasterio.transform.Affine.identity()}


def test_global_optimize(v_from_csv):
    v_from_csv.graph = {(10, 3): [(10, 4)], (10, 4): [(10, 3)]}
    v_from_csv.links = {
        ((10, 3), (10, 4)): rasterio.transform.Affine.translation(0, -50),
        ((10, 4), (10, 3)): rasterio.transform.Affine.translation(0, 50)}
    v_from_csv.df.loc[:, 'x_init'] = [350, 250, -5]
    v_from_csv.df.loc[:, 'y_init'] = [50, 50, -5]
    v_from_csv.df.loc[:, 'theta_init'] = [np.pi * 3 / 2, np.pi * 3 / 2, 0]
    v_from_csv.df.loc[:, 'scale_init'] = [2, 2, .5]
    v_from_csv.global_optimize(
        n_iter=100, lr_theta=0.001, lr_scale=0.001, lr_xy=0.5)
    relative_trans = (~v_from_csv.df.at[(10, 3), 'relative_trans'] *
                      v_from_csv.df.at[(10, 4), 'relative_trans'])
    assert v_from_csv.df.at[(11, 61), 'relative_trans'] == pytest.approx(
        rasterio.transform.Affine(0.5, 0, -15, 0, 0.5, -10))
    assert v_from_csv.optim_losses[-1] == pytest.approx(0, abs=1e-2)
    assert (pytest.approx(relative_trans, abs=0.1) ==
            rasterio.transform.Affine.translation(0, -50))


def test_georef_joint(v_from_csv, mosaic_gcp_dir, ind_gcp_dir):
    v_from_csv.graph = {
        (10, 3): [(10, 4)],
        (10, 4): [(10, 3), (11, 61)],
        (11, 61): [(10, 4)]}
    v_from_csv.links = {
        # placeholders
        ((10, 3), (10, 4)): rasterio.transform.Affine.identity(),
        ((10, 4), (10, 3)): rasterio.transform.Affine.identity(),
        ((10, 3), (11, 61)): rasterio.transform.Affine.identity(),
        ((10, 4), (11, 61)): rasterio.transform.Affine.identity(),
        ((11, 61), (10, 3)): rasterio.transform.Affine.identity(),
        ((11, 61), (10, 4)): rasterio.transform.Affine.identity()}
    # create expected pandas Series
    expected = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, -2, 10),
         rasterio.transform.Affine(1, 0, 10, 0, -1, 10),
         rasterio.transform.Affine(1, 0, 20, 0, -1, 10)],
        index=v_from_csv.df.index)
    # this tests the simple flipping case
    v_from_csv.df.loc[:, 'relative_trans'] = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, 2, -10),
         rasterio.transform.Affine(1, 0, 10, 0, 1, -10),
         rasterio.transform.Affine(1, 0, 20, 0, 1, -10)],
        index=v_from_csv.df.index)
    v_from_csv.georef()
    pd.testing.assert_series_equal(
        expected, v_from_csv.df['world_trans'], check_names=False)
    # this tests the mosaic case
    v_from_csv.df.loc[:, 'relative_trans'] = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, 2, -30),
         rasterio.transform.Affine(1, 0, 10, 0, 1, -30),
         rasterio.transform.Affine(1, 0, 20, 0, 1, -30)],
        index=v_from_csv.df.index)
    v_from_csv.georef(
        mosaic_gcp_dir=mosaic_gcp_dir, ind_gcp_dir=ind_gcp_dir)
    pd.testing.assert_series_equal(
        expected, v_from_csv.df['world_trans'], check_names=False)
    # this tests the disjoint case: dropping images without georef
    v_from_csv.links.update({
        ((10, 3), (11, 61)): None,
        ((10, 4), (11, 61)): None,
        ((11, 61), (10, 3)): None,
        ((11, 61), (10, 4)): None})
    # create expected pandas Series
    expected = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, -2, 10),
         rasterio.transform.Affine(1, 0, 10, 0, -1, 10),
         None],
        index=v_from_csv.df.index)
    v_from_csv.georef(
        mosaic_gcp_dir=mosaic_gcp_dir, ind_gcp_dir=ind_gcp_dir)
    pd.testing.assert_series_equal(
        expected, v_from_csv.df['world_trans'],
        check_names=False)


def test_georef_separate(v_from_csv, mosaic_gcp_dir,
                         ind_gcp_dir, ind_gcp_csv):
    v_from_csv.graph = {
        (10, 3): [],
        (10, 4): [],
        (11, 61): []}
    # create expected pandas Series
    expected = pd.Series(
        [rasterio.transform.Affine(1, 0, 0, 0, -1, 0),
         rasterio.transform.Affine(1, 0, 0, 0, -1, 0),
         None],
        index=pd.MultiIndex.from_tuples([(10, 3), (10, 4), (11, 61)],
                                        names=['idx0', 'idx1']))
    # this tests the separate georef case
    v_from_csv.df.loc[:, 'relative_trans'] = pd.Series(
        [rasterio.transform.Affine(2, 0, 0, 0, 2, -10),
         rasterio.transform.Affine(1, 0, 10, 0, 1, -10),
         rasterio.transform.Affine(1, 0, 20, 0, 1, -10)],
        index=v_from_csv.df.index)
    v_from_csv.georef(
        mosaic_gcp_dir=mosaic_gcp_dir, ind_gcp_dir=ind_gcp_dir)
    pd.testing.assert_series_equal(
        expected, v_from_csv.df['world_trans'], check_names=False)
