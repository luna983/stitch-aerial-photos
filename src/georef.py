import os
import glob
import pickle
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import geopandas as gpd

from .utils import (transform_coord,
                    calculate_transform_from_gcp,
                    convert_to_bbox,
                    prepare_folder)


def mosaic_to_individual(mosaic_gcp_dir, ind_gcp_dir):
    """Assign ground control points on a mosaic to individual images.

    Args:
        mosaic_gcp_dir (str): path to mosaic.csv + mosaic.p files
        ind_gcp_dir (str): path to directory storing gcp points on individual
            images, output saved to this directory
    """
    # clear csv files in directory
    for f in glob.glob(os.path.join(ind_gcp_dir, '**/*.csv')):
        os.remove(f)
    # load gcp points on the mosaic
    pts_mosaic = pd.read_csv(os.path.join(mosaic_gcp_dir, 'mosaic.csv'))
    pts_mosaic.loc[:, 'world_trans'] = None
    # load meta data
    with open(os.path.join(mosaic_gcp_dir, 'mosaic.p'), 'rb') as f:
        df = pickle.load(f)
    # compute geometries
    df.loc[:, 'geometry'] = df.apply(
        lambda x: convert_to_bbox(x['world_trans'], x['width'], x['height']),
        axis=1, result_type='reduce')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    # assign each point an image
    for i, row in pts_mosaic.iterrows():
        point = shapely.geometry.Point(
            *row.loc[['pixel_x', 'pixel_y']].values)
        # earlier images take precedence
        # argmax returns the first True instance
        idx = np.argmax(df['geometry'].contains(point).values)
        # link each point with an image, record file_id and transform
        for var in ['world_trans', 'file_id']:
            pts_mosaic.at[i, var] = df.iloc[idx, df.columns.get_loc(var)]
    # serialize to individual gcp csv files
    for file_id, g in pts_mosaic.groupby('file_id'):
        pts_img = g.copy()
        # construct file names
        img_file = os.path.join(ind_gcp_dir, file_id + '.csv')
        # prepare folders
        prepare_folder([img_file])
        # extract (unique) world trans
        trans, = pts_img['world_trans'].unique()
        # convert mosaic pixel coord to image pixel coord
        pts_img.loc[:, ['pixel_x', 'pixel_y']] = transform_coord(
            trans, to='colrow',
            xy=pts_img.loc[:, ['pixel_x', 'pixel_y']].values)
        # save with only four variables
        pts_img.loc[:, ['pixel_x', 'pixel_y',
                        'map_x', 'map_y']].to_csv(img_file, index=False)


def georef_by_gcp(gcp_dir, relative_trans, verbose=False):
    """Georeferences the raster by ground control points on individual images.

    Args:
        ind_gcp_dir (str): directory to ground control point csvs.
            ind_gcp_dir mirrors self.img_dir structure.
            both world file and geometry will be updated from the relative
            transforms.
        relative_trans (pandas.Series): with file_id (str) as index,
            relative transforms (affine.Affine) as values
        verbose (bool)

    Returns:
        affine.Affine or NoneType: the transform from relative_trans space
            to world crs, returns None if transform cannot be estimated
    """
    files = glob.glob(os.path.join(gcp_dir, '**/*.csv'))
    pts = []
    for file in files:
        file_id = os.path.relpath(file, gcp_dir).replace('.csv', '')
        if file_id not in relative_trans.index:
            if verbose:
                print('Ignored: ', file_id)
            continue
        pts_img = pd.read_csv(file)
        trans = relative_trans.at[file_id]
        # unify coord space for all gcp's
        pts_img.loc[:, ['pixel_x', 'pixel_y']] = transform_coord(
            trans, to='xy',
            colrow=pts_img.loc[:, ['pixel_x', 'pixel_y']].values)
        pts.append(pts_img)
    if len(pts) == 0:
        return None
    pts = pd.concat(pts)
    if pts.shape[0] < 2:
        return None
    # calculate transform from current (relative) space to world crs
    trans = calculate_transform_from_gcp(pts)
    return trans
