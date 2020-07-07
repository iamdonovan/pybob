#!/usr/bin/env python
import sys
import argparse
import os
import shutil
from itertools import chain
import numpy as np
import cv2
import gdal
import pandas as pd
import geopandas as gpd
from shapely.geometry.point import Point
from skimage.feature import peak_local_max
from skimage.measure import ransac
from skimage.transform import EuclideanTransform, AffineTransform, SimilarityTransform, warp
from pybob.bob_tools import mkdir_p
from pybob.image_tools import create_mask_from_shapefile
from pybob.GeoImg import GeoImg
import pybob.image_tools as imtools


def get_rough_geotransformation(prim, second, landmask=None):
    # prim_lowres = prim.resample(400, method=gdal.GRA_NearestNeighbour)
    second_lowres = second.resample(400, method=gdal.GRA_Lanczos)
    prim_lowres = prim.resample(400, method=gdal.GRA_Lanczos)

    second_lowres.img[np.isnan(second_lowres.img)] = 0
    prim_lowres.img[np.isnan(prim_lowres.img)] = 0

    if landmask is not None:
        lmask = create_mask_from_shapefile(prim_lowres, landmask)

    prim_rescale = imtools.stretch_image(prim_lowres.img, (0.05, 0.95))

    _mask = 255 * np.ones(prim_lowres.img.shape, dtype=np.uint8)

    if landmask is not None:
        _mask[np.logical_or(prim_rescale == 0, ~lmask)] = 0
    else:
        _mask[np.logical_or(np.isnan(prim_rescale), prim_rescale == 0)] = 0

    search_pts, match_pts, peak_corrs, z_corrs = imtools.gridded_matching(prim_lowres.img,
                                                                          second_lowres.img,
                                                                          _mask,
                                                                          spacing=20,
                                                                          tmpl_size=20,
                                                                          highpass=True)
    peak_corrs[peak_corrs < 0] = np.nan
    lowres_gcps = pd.DataFrame()
    lowres_gcps['pk_corr'] = peak_corrs.reshape(-1)
    lowres_gcps['z_corr'] = z_corrs.reshape(-1)
    lowres_gcps['match_j'] = match_pts[:, 0]
    lowres_gcps['match_i'] = match_pts[:, 1]
    lowres_gcps['src_j'] = search_pts[:, 1]
    lowres_gcps['src_i'] = search_pts[:, 0]
    lowres_gcps['dj'] = lowres_gcps['src_j'] - lowres_gcps['match_j']
    lowres_gcps['di'] = lowres_gcps['src_i'] - lowres_gcps['match_i']
    lowres_gcps.dropna(inplace=True)

    best_lowres = lowres_gcps[lowres_gcps.z_corr > lowres_gcps.z_corr.quantile(0.5)].copy()

    dst_pts = best_lowres[['match_j', 'match_i']].values
    src_pts = best_lowres[['src_j', 'src_i']].values

    dst_scale = (second_lowres.dx / second.dx)
    src_scale = (prim_lowres.dx / prim.dx)

    best_lowres['dj'] = best_lowres['dj'] * dst_scale
    best_lowres['di'] = best_lowres['di'] * dst_scale

    Minit, inliers = ransac((dst_pts * dst_scale, src_pts * src_scale), EuclideanTransform, min_samples=5,
                            residual_threshold=25, max_trials=5000)
    print('{} points used to find initial transformation'.format(np.count_nonzero(inliers)))

    return Minit, inliers, best_lowres


def qa_mask(img):
    return np.logical_not(np.logical_or.reduce((img == 672, img == 676, img == 680, img == 684,
                                                img == 1696, img == 1700, img == 1704, img == 1708)))


def get_mask(prim, _args):
    mask = 255 * np.ones(prim.img.shape, dtype=np.uint8)

    if _args.quality_mask is not None:
        second_qa = GeoImg(_args.quality_mask)
        quality_mask = qa_mask(second_qa.img)
        mask[quality_mask] = 0

    if _args.landmask is not None:
        lm = create_mask_from_shapefile(prim, _args.landmask)
        mask[lm == 0] = 0

    if _args.glacmask is not None:
        gm = create_mask_from_shapefile(prim, _args.glacmask, buffer=200)
        mask[gm > 0] = 0

    return mask


def _argparser():
    parser = argparse.ArgumentParser(description="Automatically register a Landsat L1GS image to a Landsat L1TP image",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('primary', action='store', type=str, help='non-referenced orthophoto mosaic')
    parser.add_argument('secondary', action='store', type=str, help='georeferenced satellite image')
    parser.add_argument('-q', '--quality_mask', action='store', type=str, default=None,
                        help='BQA band for the L1GS image (recommended)')
    parser.add_argument('-dem', action='store', type=str, default=None,
                        help='digital elevation model')
    parser.add_argument('-glacmask', action='store', type=str, default=None,
                        help='path to shapefile of glacier outlines')
    parser.add_argument('-landmask', action='store', type=str, default=None,
                        help='path to shapefile of land outlines')
    parser.add_argument('-no_lowres', action='store_true', default=False,
                        help="Don't do an initial low-res transformation")
    return parser


# def main():
parser_ = _argparser()
args = parser_.parse_args()

prim_fullres = GeoImg(args.primary)
second_fullres = GeoImg(args.secondary)

prim_fullres = prim_fullres.reproject(second_fullres)

if not args.no_lowres:
    Minit, inliers_init, lowres_gcps = get_rough_geotransformation(prim_fullres, second_fullres, landmask=args.landmask)

    # rough_tfm = warp(prim_fullres.img, Minit, output_shape=second_fullres.img.shape, preserve_range=True)
    # rough_tfm[np.isnan(rough_tfm)] = 0
    # shift the secondary image so that it is better aligned with the primary image
    second_fullres.shift(second_fullres.dx * lowres_gcps.loc[inliers_init, 'dj'].median(),
                      second_fullres.dy * lowres_gcps.loc[inliers_init, 'di'].median())

    prim_fullres = GeoImg(args.primary)  # reload primary, then re-project
    prim_fullres = prim_fullres.reproject(second_fullres)
else:
    print('skipping low-res transformation')

prim_fullres.img[np.isnan(prim_fullres.img)] = 0

# mask = get_mask(prim_fullres, second_fullres, Minit, args)
mask = get_mask(prim_fullres, args)
mask[prim_fullres.img == 0] = 0

search_pts, match_pts, peak_corr, z_corr = imtools.gridded_matching(prim_fullres.img,
                                                                    second_fullres.img,
                                                                    mask,
                                                                    spacing=50,
                                                                    tmpl_size=20,
                                                                    search_size=40,
                                                                    highpass=True)

xy = np.array([prim_fullres.ij2xy(pt) for pt in search_pts]).reshape(-1, 2)

gcps = gpd.GeoDataFrame()
gcps['geometry'] = [Point(pt) for pt in xy]
gcps['pk_corr'] = peak_corr
gcps['z_corr'] = z_corr
gcps['match_j'] = match_pts[:, 0]
gcps['match_i'] = match_pts[:, 1]
gcps['src_j'] = search_pts[:, 1]  # remember that search_pts is i, j, not j, i
gcps['src_i'] = search_pts[:, 0]
gcps['dj'] = gcps['src_j'] - gcps['match_j']
gcps['di'] = gcps['src_i'] - gcps['match_i']
gcps['elevation'] = 0
gcps.crs = prim_fullres.proj4

gcps.loc[gcps.z_corr == -1, 'z_corr'] = np.nan
gcps.dropna(inplace=True)

if args.dem is not None:
    dem = GeoImg(args.dem)
    for i, row in gcps.to_crs(crs=dem.proj4).iterrows():
        gcps.loc[i, 'elevation'] = dem.raster_points([(row.geometry.x, row.geometry.y)], nsize=9, mode='cubic')

best = gcps[gcps.z_corr > gcps.z_corr.quantile(0.5)]

Mfin, inliers_fin = ransac((best[['match_j', 'match_i']].values, best[['src_j', 'src_i']].values), SimilarityTransform,
                           min_samples=10, residual_threshold=4, max_trials=1000)
print('{} points used to find final transformation'.format(np.count_nonzero(inliers_fin)))
best = best[inliers_fin]

# out_inds = imtools.sliding_window_filter([second_fullres.img.shape[1], second_fullres.img.shape[0]], best, 200, mindist=100)

# best = best.loc[out_inds]

gcp_list = []
outname = os.path.splitext(os.path.basename(args.secondary))[0]
with open('{}_gcps.txt'.format(outname), 'w') as f:
    for i, row in best.iterrows():
        gcp_list.append(gdal.GCP(row.geometry.x, row.geometry.y, row.elevation, row.match_j, row.match_i))
        print(row.geometry.x, row.geometry.y, row.elevation, row.match_j, row.match_i, file=f)

shutil.copy(args.secondary, 'tmp.tif')
second_fullres.write('tmp.tif')
in_ds = gdal.Open('tmp.tif', gdal.GA_Update)

# unset the geotransform based on
# For now only the GTiff drivers understands full-zero as a hint
# to unset the geotransform
if in_ds.GetDriver().ShortName == 'GTiff':
    in_ds.SetGeoTransform([0, 0, 0, 0, 0, 0])
else:
    in_ds.SetGeoTransform([0, 1, 0, 0, 0, 1])

gcp_wkt = prim_fullres.proj_wkt
in_ds.SetGCPs(gcp_list, gcp_wkt)

del in_ds  # close the dataset, write to disk
in_ds = gdal.Open('tmp.tif')

print('warping image to new geometry')
mkdir_p('warped')
gdal.Warp(os.path.join('warped', os.path.basename(args.secondary)),
          in_ds, dstSRS=gcp_wkt,
          xRes=second_fullres.dx,
          yRes=second_fullres.dy,
          resampleAlg=gdal.GRA_Lanczos,
          outputType=gdal.GDT_Byte)

print('cleaning up.')

os.remove('tmp.tif')


# if __name__ == "__main__":
#     main()
