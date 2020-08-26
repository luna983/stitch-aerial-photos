import numpy as np
import cv2 as cv
import rasterio
import rasterio.transform

from .preprocess import preprocess


# seeding
cv.setRNGSeed(0)


class Stitcher(object):
    """Stitches images together. Params default to standard ones.

    Args:
        scales (list of float): scaling factor for stitching
            this should be a list and low-res images will be used first
            if no match is found, high-res images will be used
        crop (dict): 4 keys: 'top', 'bottom', 'left', 'right'
            each value is a float in (0, 1), representing the proportion
            of image width/height
        cache_dir (str): directory for storing cache images, NOTE: ALL CACHED
            IMAGES NEED TO HAVE A UNIQUE NAME
        hessian_threshold (float): threshold for feature extraction (SURF)
            the higher, the fewer features get extracted, defaults to 100
        lowe_ratio (float): Lowe's ratio for discarding false matches
            the lower, the more false matches are discarded, defaults to 0.7
        min_inliers (int): minimum number of matches to attempt
            estimation of affine transform, the higher, the more high-quality
            the match, defaults to 200, this is also used for checking whether
            a higher resolution image should be used, higher res matching
            is attempted when no. of inliers from RANSAC < min_inliers
        ransac_reproj_threshold (float): max reprojection error in RANSAC
            to consider a point as an inlier, the higher, the more tolerant
            RANSAC is, defaults to 3.0
    """

    def __init__(self,
                 scales=None, crop=None, cache_dir=None,
                 hessian_threshold=100,
                 lowe_ratio=0.7,
                 min_inliers=200,
                 ransac_reproj_threshold=3.0):
        # record preprocessing params
        self.scales = [1] if scales is None else scales
        self.crop = crop
        self.cache_dir = cache_dir
        # create feature detector
        self.fd = cv.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold)
        # feature matcher
        FLANN_INDEX_KDTREE = 1
        self.mt = cv.FlannBasedMatcher({'algorithm': FLANN_INDEX_KDTREE})
        # Lowe's ratio for discarding false matches
        self.lowe_ratio = lowe_ratio
        # minimum feature matches to attempt transform estimation
        # if RANSAC inliers < min_inliers, higher resolution images are used
        self.min_inliers = min_inliers
        # RANSAC reprojection threshold
        # maximum reprojection error in the RANSAC algorithm
        # to consider a point as an inlier
        self.ransac_reproj_threshold = ransac_reproj_threshold

    def estimate_affine(self, img0, img1,
                        verbose=False, show=False, show_file=None):
        """Estimates the affine transformation.

        Args:
            img0, img1 (numpy.ndarray [height, width]): input images
            verbose (bool): return additional diagnostic values
            show (bool): whether to produce visualizations
            show_file (str): file for storing visualizations, contains
                directory but no suffix.
                the viz files will be stored as show_file + '_match.png' and
                show_file + '_overlay.png'

        Returns:
            affine.Affine or NoneType: affine transform to fit
                img1 onto img0, None if no match is found
            dict: diagnostics (if verbose)
        """
        # detect features, compute descriptors
        kp0, des0 = self.fd.detectAndCompute(img0, mask=None)
        kp1, des1 = self.fd.detectAndCompute(img1, mask=None)
        # match descriptors
        matches = self.mt.knnMatch(des0, des1, k=2)  # query, train
        # store all the good matches as per Lowe's ratio test
        good = []
        for m0, m1 in matches:
            if m0.distance < self.lowe_ratio * m1.distance:
                good.append(m0)
        if verbose:
            diag = {'n_match': len(good)}
        # visualize
        if show:
            color = (250, 128, 114)
            img_match = cv.drawMatches(
                img0, kp0, img1, kp1, good, outImg=None,
                matchColor=color, singlePointColor=color)
            cv.imwrite(show_file + '_match.png', img_match)
        # with all good matches, estimate affine transform w/ RANSAC
        if len(good) > self.min_inliers:
            pts0 = np.array([kp0[m.queryIdx].pt for m in good])
            pts1 = np.array([kp1[m.trainIdx].pt for m in good])
            transform, inliers = cv.estimateAffinePartial2D(
                pts1, pts0,
                method=cv.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold)
            if verbose:
                diag['n_inlier'] = inliers.sum()
            if inliers.sum() < self.min_inliers:
                return (None, diag) if verbose else None
            if show:
                # fit img1 onto img0
                img_overlay = cv.warpAffine(
                    # dsize = (width, height)
                    src=img1, M=transform, dsize=img0.shape[::-1])
                # overlay with 50% transparency
                img_overlay = cv.addWeighted(
                    img0, 0.5, img_overlay, 0.5, gamma=0.0)
                cv.imwrite(show_file + '_overlay.png', img_overlay)
            transform = rasterio.transform.Affine(*transform.flatten())
            return (transform, diag) if verbose else transform
        else:
            return (None, diag) if verbose else None

    def stitch_pair(self, img0, img1, verbose=False, **kwargs):
        """Stitch images together using a pyramid of resolutions.

        Args:
            img0, img1 (str): file path
            verbose (bool)
            **kwargs: passed to estimate_affine()

        Returns:
            affine.Affine or NoneType: affine transform to fit
                img1 onto img0, None if no match is found
                the relative transform is in terms of the original images
            dict: diagnostics (if verbose)
        """
        # iterate over resolutions
        params = {'crop': self.crop,
                  'cache': False if self.cache_dir is None else True,
                  'cache_dir': self.cache_dir}
        for scale in self.scales:
            img0_array, trans0 = preprocess(img0, scale=scale, **params)
            img1_array, trans1 = preprocess(img1, scale=scale, **params)
            output = self.estimate_affine(
                img0_array, img1_array, verbose=verbose, **kwargs)
            if verbose:
                relative_trans, diag = output
                diag['img0'] = img0
                diag['img1'] = img1
                diag['scale'] = scale
            else:
                relative_trans = output
            # return if transform is found
            if relative_trans is not None:
                # take into account the transforms between the original images
                # and the preprocessed images
                trans = trans0 * relative_trans * (~trans1)
                return (trans, diag) if verbose else trans
        return (None, diag) if verbose else None
