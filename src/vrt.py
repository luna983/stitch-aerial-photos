import os
import tqdm
import glob
import warnings
import collections
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import shapely.geometry
import rasterio
import rasterio.transform
import rasterio.warp

from .utils import (create_batch_symlink,
                    convert_affine,
                    convert_to_bbox,
                    prepare_folder)
from .graph import get_links, build_graph, get_subgraphs
from .optim import optimize
from .preprocess import preprocess
from .georef import mosaic_to_individual, georef_by_gcp


class VirtualRaster(object):
    """Virtual raster.

    Unlike GDAL, this handles rotations of source rasters.

    Caveat: when showing the virtual raster, the user can supply a pre-
    processing function that may involve cropping. The bounds of the loaded
    images may be different from the bounds stored in `self.df`. Note that
    `self.df` is intended to store georef info for the entire image.

    Args:
        df (geopandas.GeoDataFrame): contains at least the following key(s)
            index: integer indices for images
            'file_id': image file paths, should be concatenated with
                img_dir/wld_dir and a proper suffix in order to form complete
                paths
            'img_file' and 'wld_file': file paths to img/wld files
            'width', 'height': the width and height of an image (original),
                in pixels
        img_dir (str): directory to images
        wld_dir (str): directory to metadata (georef info)
        img_suffix, wld_suffix (str): suffix for image or world files
            including the leading dot
        graph (collections.defaultdict): {node index (idx0, idx1): [neighbors'
            indices (idx0, idx1)]}
        links (dict {((int, int), (int, int)): affine.Affine}):
            recording affine transforms estimated between pairs of
            images; links[((idx0_i, idx1_i), (idx0_j, idx1_j))] is the
                transform from j to i
        crs (str): string that represents the coord reference system
    """

    def __init__(self, df, img_dir, wld_dir, img_suffix, wld_suffix,
                 graph=None, links=None, crs=None):
        print('Initializing virtual raster.')
        # initialize self.df
        self.df = df
        # store paths and suffix
        assert img_dir != wld_dir
        self.img_dir = img_dir
        self.wld_dir = wld_dir
        self.img_suffix = img_suffix
        self.wld_suffix = wld_suffix
        # initialize graph and links
        self.graph = collections.defaultdict(list) if graph is None else graph
        self.links = {} if links is None else links
        # parse crs
        self.crs = rasterio.crs.CRS.from_string(crs)

    @classmethod
    def from_csv(cls, file, img_dir, wld_dir,
                 img_suffix, wld_suffix, crs, index_cols,
                 verbose=False):
        """From centroids recorded in a csv file.

        Args:
            file (str): path to csv file
            img_dir (str): path to images
            wld_dir (str): path to metadata (world files)
            img_suffix, wld_suffix (str): suffix for image or world files
                including the leading dot
            crs (str): string that represents the coord reference system
            index_cols (list of str): name of index columns
            verbose (bool)
        """
        df = pd.read_csv(file)
        df.loc[:, 'img_file'] = df.apply(
            lambda x: os.path.join(img_dir, x['file_id'] + img_suffix),
            axis=1)
        df.loc[:, 'wld_file'] = df.apply(
            lambda x: os.path.join(wld_dir, x['file_id'] + wld_suffix),
            axis=1)
        # check for img file existence
        file_exists = df.apply(
            lambda x: os.path.isfile(x['img_file']), axis=1)
        # drop non existent files
        if verbose:
            for f in df.loc[~file_exists, 'img_file'].tolist():
                print('No such file: {}'.format(f))
        df = df.loc[file_exists, :]
        # query and store height and width from img image metadata
        width = []
        height = []
        # iterate over each file
        for file in df.loc[:, 'img_file'].tolist():
            with warnings.catch_warnings():
                warnings.simplefilter(
                    action='ignore',
                    category=rasterio.errors.NotGeoreferencedWarning)
                # read metadata
                with rasterio.open(file, 'r') as ds:
                    width.append(ds.width)
                    height.append(ds.height)
        df.loc[:, 'width'] = width
        df.loc[:, 'height'] = height
        # sort and index
        df = df.sort_values(by=index_cols).set_index(index_cols)

        # return virtual raster object
        return cls(gpd.GeoDataFrame(df), img_dir=img_dir, wld_dir=wld_dir,
                   img_suffix=img_suffix, wld_suffix=wld_suffix,
                   crs=crs)

    @classmethod
    def from_world_files(cls, img_dir, wld_dir,
                         img_suffix, wld_suffix, crs):
        """Loads a virtual raster from wld_dir.

        Loads all files in wld_dir and overwrites existing symbolic links
        placed in img_dir. ALL SYMBOLIC LINKS THAT HAVE CORRESPONDING WORLD
        FILES IN wld_dir WILL BE OVERWRITTEN.
        NEVER SET img_dir AND wld_dir TO BE THE SAME.

        Args:
            img_dir (str): directory to images, NOT all files here will be
                loaded, only the ones with corresponding files in wld_dir
            wld_dir (str): directory to world files, all files here will be
                loaded. Directory structure under wld_dir and img_dir should be
                the same for all georeferenced images
            img_suffix, wld_suffix (str): suffix for image/world files
                including the leading dot
            crs (str): string that represents the coord reference system
        """
        # find all files
        wld_files = glob.glob(os.path.join(wld_dir, '**/*' + wld_suffix))
        file_ids = [os.path.relpath(f, wld_dir).replace(wld_suffix, '')
                    for f in wld_files]
        idx0 = [int(os.path.basename(f).split('_')[-2]) for f in file_ids]
        idx1 = [int(os.path.basename(f).split('_')[-1]) for f in file_ids]
        img_files = [os.path.join(img_dir, f + img_suffix) for f in file_ids]
        # assert all images exist
        for f in img_files:
            assert os.path.isfile(f)
        # create symbolic links, overwrite old ones
        # NOTICE THAT THIS METHOD HAS SIDE EFFECTS
        # IT OVERWRITES EXISTING SYMBOLIC LINKS FOR WORLD FILES
        create_batch_symlink(src=wld_dir, dst=img_dir, suffix=wld_suffix)
        # parse world files and collate into one geo dataframe
        width = []
        height = []
        world_trans = []
        bboxes = []
        # iterate over each file
        for file in img_files:
            # read metadata
            with rasterio.open(file, 'r') as ds:
                width.append(ds.width)
                height.append(ds.height)
                world_trans.append(ds.transform)
                bboxes.append(
                    convert_to_bbox(ds.transform, ds.width, ds.height))
        # construct a geo dataframe that can be quickly queried
        df = gpd.GeoDataFrame(
            {'idx0': idx0, 'idx1': idx1,
             'width': width, 'height': height,
             'file_id': file_ids,
             'img_file': img_files,
             'wld_file': wld_files,
             'world_trans': world_trans},
            geometry=bboxes
        ).sort_values(by=['idx0', 'idx1']).set_index(['idx0', 'idx1'])
        return cls(df, img_dir=img_dir, wld_dir=wld_dir,
                   img_suffix=img_suffix, wld_suffix=wld_suffix,
                   crs=crs)

    def to_world_files(self, wld_dir=None):
        """Saves world file georeference info.

        Args:
            wld_dir (str): directory to world files, this path, when
                concatenated with self.df['file_id'], plus the right suffix,
                will point to the world files
                Directory structure under wld_dir and self.img_dir will be
                the same for all georeferenced images.
                if None, self.df['wld_file'] will be used
        """
        df = self.df.loc[pd.notna(self.df['world_trans']), :]
        # extract outputs
        outputs = df.apply(
            lambda x: convert_affine(x['world_trans'], to='world_file'),
            axis=1).tolist()
        if wld_dir is None:
            wld_files = df.loc[:, 'wld_file'].tolist()
        else:
            # generate world file paths
            wld_files = df.apply(
                lambda x: os.path.join(wld_dir,
                                       x['file_id'] + self.wld_suffix),
                axis=1).tolist()
        # prepare folders
        prepare_folder(wld_files)

        # save world files
        for output, wld_file in zip(outputs, wld_files):
            with open(wld_file, 'w') as f:
                f.write('\n'.join(['{:.8f}'.format(x) for x in output]))

    def show(self, mode='stack',
             input_extent_type='all', input_extent=None,
             output_res=None, output_size=None, output_bounds=None,
             max_pixel=1e4, verbose=False, **kwargs):
        """Visualizes a virtual raster.

        Args:
            mode (str): in ['stack', 'overlay', 'composite']
                'stack': returns a [number of images, height, width] array
                    with each layer representing an original image
                'overlay': returns a [height, width] array with layers stacked
                    upon each other (some images will not be shown due to
                    overlap); images with smaller idx0, idx1 values take
                    priority
                'composite': returns an alpha composite of all layers
            input_extent_type (str): in ['bounds', 'all', 'img_pos',
                'img_name']
            input_extent: type depends on value of `extent_type`
                for 'bounds': (tuple) xmin, ymin, xmax, ymax (in specified crs)
                    bounds for the raster shown
                for 'all': (NoneType) show the full raster, this arg is ignored
                for 'img_pos': (list of int) list of images to be shown (int
                    positions), this can change the order of images (if
                    overlayed, which images get shown)
                for 'img_name': (list of tuple of int) list of images to be
                    shown, use indices that are in self.df.index, this can
                    change the order of images (if overlayed, which images
                    get shown)
            output_res (float): resolution (the width/height of a pixel, in
                crs units)
            output_size (tuple of int): height, width of output
            output_bounds (tuple of float): xmin, ymin, xmax, ymax
                (in specified crs) bounds for the output raster,
                this allows specification of the visualized output when
                not using input_extent_type = 'bounds'
            max_pixel (int): max number of pixels (for a single layer)
            verbose (bool)
            **kwargs: passed to src.preprocess.preprocess which loads the data
                implementing on-the-fly cropping and scaling

        Returns:
            tuple (numpy.ndarray, affine.Affine): output image and transform
        """
        df = self.df.loc[pd.notna(self.df['world_trans']), :].copy()
        # parse input extent type and extent
        if input_extent_type == 'bounds':
            xmin, ymin, xmax, ymax = input_extent
            # choose images that intersect with the bbox
            df = df.cx[xmin:xmax, ymin:ymax]
        elif input_extent_type in ['img_pos', 'img_name', 'all']:
            # position based indexing
            if input_extent_type == 'img_pos':
                df = df.iloc[input_extent, :]
            # label based indexing
            if input_extent_type == 'img_name':
                df = df.loc[input_extent, :]
            xmin, ymin, xmax, ymax = df.total_bounds
        else:
            raise NotImplementedError
        if df.shape[0] == 0:
            return None, None
        # reset index to integer positions
        df = df.reset_index(drop=True)
        # output_bounds override input_extent for output shape and position
        # if not None
        if output_bounds is not None:
            xmin, ymin, xmax, ymax = output_bounds

        # calculate output size
        assert output_res is None or output_size is None  # no conflicts
        if output_res is not None:
            # calculate output array size
            output_height = int((ymax - ymin) / output_res)
            output_width = int((xmax - xmin) / output_res)
            output_xres = output_yres = output_res
        if output_size is not None:
            output_height, output_width = output_size
            output_yres = (ymax - ymin) / output_height
            output_xres = (xmax - xmin) / output_width
        if output_res is None and output_size is None:
            # align longer edge with sqrt(max_pixel)
            output_xres = output_yres = (
                max((xmax - xmin), (ymax - ymin)) / np.sqrt(max_pixel))
            output_height = int((ymax - ymin) / output_yres)
            output_width = int((xmax - xmin) / output_xres)

        # check for file size, fail if it's too large
        assert output_height * output_width <= max_pixel

        # output transform
        dst_transform = rasterio.transform.Affine(
            output_xres, 0, xmin, 0, - output_yres, ymax)

        # initialize the outputs array
        if mode == 'stack':
            outputs = []
        elif mode in ['overlay', 'composite']:
            outputs = np.zeros((output_height, output_width),
                               dtype=np.float)
            if mode == 'overlay':
                alphas = np.zeros((output_height, output_width),
                                  dtype=bool)
            if mode == 'composite':
                counts = np.zeros((output_height, output_width),
                                  dtype=np.uint32)
        df_iterrows = tqdm.tqdm(df.iterrows()) if verbose else df.iterrows()
        # iterate over images
        for i, row in df_iterrows:
            # load images with the preprocess function
            img, trans_relative = preprocess(file=row['img_file'], **kwargs)
            # make sure that img does not have 0's (reserved for nodata)
            img = np.clip(img, 1, None)
            # reproject onto the new raster
            output = np.zeros((output_height, output_width),
                              dtype=np.uint8)
            rasterio.warp.reproject(
                source=img,
                destination=output,
                src_transform=row['world_trans'] * trans_relative,
                src_crs=self.crs,
                src_nodata=None,
                dst_transform=dst_transform,
                dst_crs=self.crs,
                dst_nodata=0)
            if mode == 'stack':
                outputs.append(output)
            elif mode in ['overlay', 'composite']:
                # compose transparency channels (alpha mask)
                alpha = output != 0  # nodata = 0
                if mode == 'overlay':
                    # alphas set to zero if valid data exist on outputs
                    # this indicates that earlier images take
                    # priority over later images
                    alpha = np.logical_and(alpha, np.logical_not(alphas))
                    alphas = np.logical_or(alpha, alphas)
                if mode == 'composite':
                    counts += alpha.astype(np.uint32)
                    # prevent division by zero
                    alpha = alpha / np.clip(counts, 1, None)
                outputs = output * alpha + outputs * (1 - alpha)
            else:
                raise NotImplementedError
        # return array and transform
        if mode == 'stack':
            # stack bands together as a numpy array
            outputs = np.array(outputs)
        return np.round(outputs).astype(np.uint8), dst_transform

    def build_links(self, f, graph=None, verbose=False):
        """Build links between nodes.

        Args:
            f (function): takes in two image file paths, img0 and img1
                returns an affine transformation from img1 to img0
            graph (collections.defaultdict(list)): if None, use self.graph
                {k: [v0, v1, ...]} indicating the images' neighbors
            verbose (bool)
        """
        graph = self.graph if graph is None else graph

        # prepare pairs of indices
        img_files = self.df.loc[:, 'img_file']
        pairs = [(i, j, img_files.loc[i], img_files.loc[j])
                 for i, js in graph.items() for j in js
                 # use tuple comparisons
                 if i < j and (i, j) not in self.links.keys()]

        # estimate transforms for every pair
        result = [((i, j), f(i_file, j_file))
                  for i, j, i_file, j_file in
                  tqdm.tqdm(pairs, desc='Building links.')]

        # collect into dictionary
        self.links.update(dict(result))

        # make links symmetric
        for i, js in graph.items():
            for j in js:
                if i > j:
                    self.links[(i, j)] = (None if self.links[(j, i)] is None
                                          else ~self.links[(j, i)])
        if verbose:
            print('Links: ', self.links)

    def build_graph_links(self, f, position_cols=['x_init', 'y_init'],
                          **kwargs):
        """Builds graph and corresponding links.

        Args:
            f (function): the function used for building links
                takes in two image file paths, img0 and img1
                returns an affine transformation from img1 to img0
                passed to self.build_links
            position_cols (list of str [2,]): names of columns that indicate
                x, y coordinates of images
            **kwargs: passed to src.graph.build_graph
        """
        indices = (self.df.groupby('swath_id')
                   .apply(lambda g: g.index.tolist()).tolist())
        if kwargs['method'] == 'across':
            assert position_cols is not None
            kwargs['positions'] = (
                self.df.groupby('swath_id')
                .apply(lambda g: g.loc[:, position_cols].values).tolist())
        # build graph
        graph = build_graph(indices, **kwargs)
        # update graph
        for k in graph.keys():
            self.graph[k] += graph[k]
        # build links
        self.build_links(f)

    def global_optimize(self, **kwargs):
        """Globally optimize to fit all images together.

        Args:
            **kwargs: passed to src.optim.optimize
        """
        print('Globally optimizing.')
        # globally optimize
        iters, losses, affines = optimize(
            nodes=self.df.index.tolist(),
            links=get_links(graph=self.graph, links=self.links),
            thetas_init=self.df.loc[:, 'theta_init'].tolist(),
            scales_init=self.df.loc[:, 'scale_init'].tolist(),
            xs_init=self.df.loc[:, 'x_init'].tolist(),
            ys_init=self.df.loc[:, 'y_init'].tolist(),
            width=self.df.loc[:, 'width'].tolist(),
            height=self.df.loc[:, 'height'].tolist(),
            **kwargs)
        # update transform
        self.df.loc[:, 'relative_trans'] = pd.Series(
            affines[-1], index=self.df.index)
        # collect output
        self.optim_iters = iters
        self.optim_losses = losses
        self.optim_affines = affines

    def georef(self, mosaic_gcp_dir=None, ind_gcp_dir=None):
        """Georeferences the raster by ground control points.

        Args:
            mosaic_gcp_dir, ind_gcp_dir (str or NoneType): directory to GCP
                csvs, ind_gcp_dir mirrors self.img_dir structure,
                both world file and geometry will be updated from the relative
                transforms.
                ind_gcp_dir takes precedence.
                If both=None or if both are empty/nonexistent, a simple y axis
                flipping is conducted.
        """
        if mosaic_gcp_dir is None and ind_gcp_dir is None:
            trans = rasterio.transform.Affine(1, 0, 0, 0, -1, 0)
            # update world transform from relative trans
            self.df.loc[:, 'world_trans'] = self.df.apply(
                lambda x: trans * x['relative_trans'],
                axis=1, result_type='reduce')
        else:
            files = list(glob.glob(os.path.join(ind_gcp_dir, '**/*.csv')))
            if len(files) == 0:
                mosaic_to_individual(
                    mosaic_gcp_dir=mosaic_gcp_dir, ind_gcp_dir=ind_gcp_dir)
            graph = {k: [v for v in vs if self.links[(k, v)] is not None]
                     for k, vs in self.graph.items()}
            subgraphs = get_subgraphs(graph)
            for subgraph in subgraphs:
                df = self.df.loc[list(subgraph),
                                 ['file_id', 'relative_trans']].copy()
                df.set_index('file_id', drop=True, inplace=True)
                trans = georef_by_gcp(gcp_dir=ind_gcp_dir,
                                      relative_trans=df['relative_trans'])
                if trans is None:
                    print('Dropping: ', list(subgraph))
                    self.df.loc[list(subgraph), 'world_trans'] = None
                else:
                    # update world transform from relative trans
                    self.df.loc[list(subgraph), 'world_trans'] = (
                        self.df.loc[list(subgraph), :].apply(
                            lambda x: trans * x['relative_trans'],
                            axis=1, result_type='reduce'))
        # update geometry from world_trans
        self.df.loc[:, 'geometry'] = self.df.apply(
            lambda x: (
                convert_to_bbox(
                    x['world_trans'], x['width'], x['height'])
                if x['world_trans'] is not None else
                shapely.geometry.Polygon([])),
            axis=1, result_type='reduce')
