#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import numpy as np
import datetime as dt
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile
from operator import itemgetter

def parse_filename(fname):
    splitname = fname.split('_')
    if splitname[0] == 'AST':
        datestr = splitname[2][3:11]
        yy = int(datestr[4:])
        mm = int(datestr[0:2])
        dd = int(datestr[2:4])
    elif splitname[0] in ['SETSM', 'SDMI', 'SPOT', 'SRTM', 'Map']:
        datestr = splitname[2]
        yy = int(datestr[0:4])
        mm = int(datestr[4:6])
        dd = int(datestr[6:])
    else:
        print("I don't recognize how to parse date information from {}.".format(fname))
        return None
    return dt.datetime(yy, mm, dd)


def sort_chronologically(dates, data):
    datedict = list(zip(dates, data))
    if all(dates):
        datedict.sort(reverse=True, key=itemgetter(0))
    return datedict


def pretty_granules(fname):
    # splitname = fname.split('_')
    # if splitname[0] == 'AST':
    #     pname = '_'.join(splitname[0:3])
    # elif splitname[0] == 'SETSM':
    #     pname = '_'.join(splitname[0:-2]).replace('30m', '2m')
    # else:
    pname = os.path.splitext(fname)[0].rsplit('_adj', 1)[0]
    return pname


def rmsd(diffs):
    return np.sqrt(np.nanmean(diffs ** 2))


def write_meta_file(datedict, dDEM, gmask=None, slope=None, outlier=None, outfile=None, soutlier=None):
    # want to include: file 1, file 2, date 1, date 2, time separation
    # also RMSE, mean/std, slope category
    # get pretty crs and bounding box
    prettycrs = dDEM.spatialReference.ExportToPrettyWkt().split(',')[0].replace('PROJCS[', '').strip('"')
    bbox = [dDEM.xmin, dDEM.ymin, dDEM.xmax, dDEM.ymax]
    # get central date, date strings
    if all((datedict[0][0], datedict[1][0])):
        date1str = datedict[1][0].strftime('%Y-%m-%d')
        date0str = datedict[0][0].strftime('%Y-%m-%d')
        centerdatestr = (datedict[1][0] + (datedict[0][0] - datedict[1][0]) / 2).strftime('%Y-%m-%d')
        delta_t = '{:.4f}'.format((datedict[0][0]-datedict[1][0]).days / 365.2425)
    else:
        date1str = None
        date0str = None
        centerdatestr = None
        delta_t = None

    # get statistics
    if type(dDEM.img) is np.ma.core.MaskedArray:
        diff_rast = dDEM.img.data
        diff_rast[dDEM.img.mask] = np.nan
    else:
        diff_rast = dDEM.img
        
    if gmask is not None:
        glacier_mask = create_mask_from_shapefile(dDEM, gmask)
        diff_rast[glacier_mask] = np.nan
        stab_mean = np.nanmean(diff_rast)
        stab_median = np.nanmedian(diff_rast)
        stab_std = np.nanstd(diff_rast)
        stab_rms = rmsd(diff_rast)
        stab_nsamp = np.count_nonzero(~np.isnan(diff_rast))
    else:
        stab_mean = np.nan
        stab_median = np.nan
        stab_std = np.nan
        stab_rms = np.nan
        stab_nsamp = np.nan
    if slope is not None:
        srast = GeoImg(slope)
        srast = srast.reproject(dDEM)
        smask = srast.img < 20
        slope_mean = np.nanmean(diff_rast[smask])
        slope_median = np.nanmedian(diff_rast[smask])
        slope_std = np.nanstd(diff_rast[smask])
        slope_rms = rmsd(diff_rast[smask])
        slope_nsamp = np.count_nonzero(~np.isnan(diff_rast[smask]))
    else:
        slope_mean = np.nan
        slope_median = np.nan
        slope_std = np.nan
        slope_rms = np.nan
        slope_nsamp = np.nan

    if outfile is None:
        outfile = 'dH_' + pretty_granules(datedict[0][1].filename) + \
                          '_' + pretty_granules(datedict[1][1].filename)
    f = open(outfile + '.txt', 'w')
    print('DEM Differences provided as part of the ESA Glaciers_CCI project', file=f)
    print('---------------------------------------dem information---------------------------------------', file=f)
    print('dH filename:\t{}'.format(outfile + '.tif'), file=f)
    print('DEM1 source:\t{}'.format(pretty_granules(datedict[0][1].filename)), file=f)
    print('DEM1 date:\t{}'.format(date1str), file=f)
    print('DEM2 source:\t{}'.format(pretty_granules(datedict[1][1].filename)), file=f)
    print('DEM2 date:\t{}'.format(date0str), file=f)
    print('centerdate:\t{}'.format(centerdatestr), file=f)
    print('dt:\t\t{} years'.format(delta_t), file=f)
    print('---------------------------------------crs information---------------------------------------', file=f)
    print('coordinate reference system:\t{}'.format(prettycrs), file=f)
    print('bounding box:\t{}'.format(bbox), file=f)
    print('------------------------------------------statistics-----------------------------------------', file=f)
    if outlier is not None:
        print('outlier value:\t{}'.format(outlier), file=f)
    else:
        print('outlier value (statistics only):\t{}'.format(soutlier), file=f)
    print('stable terrain statistics, post-co-registration:', file=f)
    print('\t\tall stable terrain\tslope < 20 degrees', file=f)
    print('mean:\t\t{:.2f}\t\t\t{:.2f}'.format(stab_mean, slope_mean), file=f)
    print('median:\t\t{:.2f}\t\t\t{:.2f}'.format(stab_median, slope_median), file=f)
    print('std. dev.:\t{:.2f}\t\t\t{:.2f}'.format(stab_std, slope_std), file=f)
    print('rms diff.:\t{:.2f}\t\t\t{:.2f}'.format(stab_rms, slope_rms), file=f)
    print('n. samples:\t{:.0f}\t\t\t{:.0f}'.format(stab_nsamp, slope_nsamp), file=f)
    f.close()


def _argparser():
    parser = argparse.ArgumentParser(description="Difference co-registered DEM pairs, \
                                                  write metadata file with information.")
    parser.add_argument('--folder', action='store', type=str, help="Folder with two co-registered DEMs.")
    parser.add_argument('DEM1', action='store', type=str, help="Path to DEM 1")
    parser.add_argument('DEM2', action='store', type=str, help="Path to DEM 2")
    parser.add_argument('-mask', action='store', type=str, help="Glacier mask (optional)")
    parser.add_argument('-slope', action='store', type=str, help="Terrain slope (optional)")
    parser.add_argument('-s_outlier', action='store', type=float, default=100,
                        help="Set differences above/below to NaN to calculate statistics only.")
    parser.add_argument('-outlier', action='store', type=float, help="Set differences above/below to NaN")
    parser.add_argument('-o', '--outfile', action='store', type=str, help="Specify output filename")
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    if args.folder is None:
        args.folder = '.'
    print ('Starting program.')
    dateA = parse_filename(os.path.basename(args.DEM1))
    dateB = parse_filename(os.path.basename(args.DEM2))

    print(os.getcwd())

    demA = GeoImg(args.DEM1)
    print('Loaded {}'.format(args.DEM1))
    tmp_demB = GeoImg(args.DEM2)
    print('Loaded {}'.format(args.DEM2))
    
    # make sure that the images are lined up to their common extents
    demB = tmp_demB.reproject(demA)
    demB.filename = tmp_demB.filename    

    datedict = sort_chronologically((dateA, dateB), (demA, demB))
    if dateA is None or dateB is None:
        # if we can't get date info, assume they were passed chronologically
        print(dateA, dateB)
        dDEM = datedict[0][1].copy(new_raster=(demA.img - demB.img))
    else:
        print(datedict[0][0], datedict[1][0])
        dDEM = datedict[0][1].copy(new_raster=(datedict[0][1].img - datedict[1][1].img))

    # crop dDEM to valid extent to save on space
    valid_ext = dDEM.find_valid_bbox()
    dDEM = dDEM.crop_to_extent(valid_ext)


    if args.outlier is not None:
        dDEM.img[np.abs(dDEM.img) > args.outlier] = np.nan
    else:
        dDEM.mask(np.abs(dDEM.img) > args.s_outlier)
    
    if args.outfile is None:
        out1 = 'dH_' + pretty_granules(datedict[0][1].filename)
        out2 = '_' + pretty_granules(datedict[1][1].filename) + '.tif'
        outname = out1 + out2
    else:
        outname = args.outfile
    print('Writing dDEM to {}'.format(outname))
    dDEM.write(outname, out_folder=args.folder)

    metaname = os.path.splitext(outname)[0]
    print('Writing metadata to {}'.format(metaname + '.txt'))
    write_meta_file(datedict, dDEM, gmask=args.mask, slope=args.slope, 
                    outlier=args.outlier, outfile=metaname, soutlier=args.s_outlier)
    print('Finished.')


if __name__ == "__main__":
    main()
