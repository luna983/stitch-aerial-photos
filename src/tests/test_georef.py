import pytest

import pickle
import pandas as pd
import rasterio
import rasterio.transform

from ..georef import mosaic_to_individual, georef_by_gcp


@pytest.fixture
def mosaic_gcp_dir(tmp_path):
    d = tmp_path / 'mosaic'
    d.mkdir()
    return d


@pytest.fixture
def ind_gcp_to_dir(tmp_path):
    d = tmp_path / 'ind_to'
    d.mkdir()
    return d


@pytest.fixture
def ind_gcp_from_dir(tmp_path):
    d = tmp_path / 'ind_from'
    d.mkdir()
    (d / 'a').mkdir()
    (d / 'b').mkdir()
    (d / 'c').mkdir()
    return d


@pytest.fixture
def mosaic_p(mosaic_gcp_dir):
    mosaic_p = pd.DataFrame({
        'file_id': ['a/a', 'b/b', 'c/c'],
        'world_trans': [rasterio.transform.Affine(1.25, 0, 25, 0, 1.25, 0),
                        rasterio.transform.Affine(1.25, 0, 12.5, 0, 1.25, 0),
                        rasterio.transform.Affine(2.5, 0, 0, 0, 2.5, 0)],
        'width': [40, 40, 40],
        'height': [20, 40, 20],
    })
    # save test data to tmp_path
    with open(str(mosaic_gcp_dir / 'mosaic.p'), 'wb') as f:
        pickle.dump(mosaic_p, f)
    return mosaic_p


@pytest.fixture
def mosaic_csv(mosaic_gcp_dir):
    mosaic_csv = pd.DataFrame({
        'pixel_x': [50, 50, 90],
        'pixel_y': [6.25, 37.5, 12.5],
        'map_x': [40, 40, 72],
        'map_y': [5, -20, 0],
    })
    mosaic_csv.to_csv(str(mosaic_gcp_dir / 'mosaic.csv'), index=False)
    return mosaic_csv


@pytest.fixture
def a_csv(ind_gcp_from_dir):
    df = pd.DataFrame({'pixel_x': [20], 'pixel_y': [5],
                       'map_x': [40], 'map_y': [5]})
    df.to_csv(str(ind_gcp_from_dir / 'a' / 'a.csv'), index=False)
    return df


@pytest.fixture
def b_csv(ind_gcp_from_dir):
    df = pd.DataFrame({'pixel_x': [30], 'pixel_y': [30],
                       'map_x': [40], 'map_y': [-20]})
    df.to_csv(str(ind_gcp_from_dir / 'b' / 'b.csv'), index=False)
    return df


@pytest.fixture
def c_csv(ind_gcp_from_dir):
    df = pd.DataFrame({'pixel_x': [36], 'pixel_y': [5],
                       'map_x': [72], 'map_y': [0]})
    df.to_csv(str(ind_gcp_from_dir / 'c' / 'c.csv'), index=False)
    return df


@pytest.fixture
def relative_trans():
    return pd.Series(
        [rasterio.transform.Affine(1, 0, 20, 0, 1, 0),
         rasterio.transform.Affine(1, 0, 10, 0, 1, 0),
         rasterio.transform.Affine(2, 0, 0, 0, 2, 0)],
        index=['a/a', 'b/b', 'c/c'])


@pytest.fixture
def trans_expected():
    return rasterio.transform.Affine(1, 0, 0, 0, -1, 10)


def test_mosaic_to_individual(mosaic_gcp_dir, ind_gcp_to_dir,
                              mosaic_p, mosaic_csv,
                              a_csv, b_csv, c_csv):
    # call function
    mosaic_to_individual(mosaic_gcp_dir=str(mosaic_gcp_dir),
                         ind_gcp_dir=str(ind_gcp_to_dir))
    # check output
    pd.testing.assert_frame_equal(
        a_csv, pd.read_csv(str(ind_gcp_to_dir / 'a' / 'a.csv')),
        check_dtype=False)
    pd.testing.assert_frame_equal(
        b_csv, pd.read_csv(str(ind_gcp_to_dir / 'b' / 'b.csv')),
        check_dtype=False)
    pd.testing.assert_frame_equal(
        c_csv, pd.read_csv(str(ind_gcp_to_dir / 'c' / 'c.csv')),
        check_dtype=False)


def test_georef_by_gcp(a_csv, b_csv, c_csv,
                       ind_gcp_from_dir, relative_trans, trans_expected):
    assert (georef_by_gcp(ind_gcp_from_dir, relative_trans) ==
            pytest.approx(trans_expected))
    assert georef_by_gcp(ind_gcp_from_dir, relative_trans.iloc[0:1]) is None
