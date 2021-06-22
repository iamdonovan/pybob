#!/usr/bin/env python
from __future__ import print_function, division
# from future_builtins import zip
import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from glob import glob
from gdal import GRA_NearestNeighbour
from pybob.GeoImg import GeoImg
from pybob.bob_tools import bin_data
import pybob.image_tools as it
#import pybob.ddem_tools as ddem_tools


def area_alt_dist(DEM, glacier_shapes, glacier_inds=None, bin_width=None):
    if glacier_inds is None:
        dem_data = DEM.img[glacier_shapes]
        if bin_width is None:
            z_range = np.nanmax(dem_data) - np.nanmin(dem_data)
            this_bin_width = min(100, int(z_range / 5))
        else:
            this_bin_width = bin_width
        min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % this_bin_width)
        max_el = np.nanmax(dem_data) + (this_bin_width - (np.nanmax(dem_data) % this_bin_width))
        bins = np.arange(min_el, max_el+1, this_bin_width)
        aads, _ = np.histogram(dem_data, bins=bins, range=(min_el, max_el))
        aads = aads * np.abs(DEM.dx) * np.abs(DEM.dy)
        bins = bins[:-1]  # remove the last element, because it's actually above the glacier range.
    else:
        bins = []
        aads = []

        for i in glacier_inds:
            dem_data = DEM.img[glacier_shapes == i]
            if bin_width is None:
                z_range = np.nanmax(dem_data) - np.nanmin(dem_data)
                this_bin_width = min(100, int(z_range / 5))
            else:
                this_bin_width = bin_width
            min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % this_bin_width)
            max_el = np.nanmax(dem_data) + (this_bin_width - (np.nanmax(dem_data) % this_bin_width))
            thisbin = np.arange(min_el, max_el+1, this_bin_width)
            thisaad, _ = np.histogram(dem_data, bins=thisbin, range=(min_el, max_el))

            bins.append(thisbin[:-1])  # remove last element.
            aads.append(thisaad * np.abs(DEM.dx) * np.abs(DEM.dy))  # make it an area!

    return bins, aads
    
    
def get_dH_curve(DEM, dDEM, glacier_mask, bins=None, mode='mean'):
    valid = np.logical_and(np.isfinite(DEM), np.isfinite(dDEM))
    dem_data = DEM[valid]
    ddem_data = dDEM[valid]

    if bins is None:
        bins = get_bins(DEM, glacier_mask)
    mean_dH, bin_areas = bin_data(bins, ddem_data, dem_data, mode=mode, nbinned=True)

    return bins, mean_dH, bin_areas

def get_bins(DEM, glacier_mask):
    dem_data = DEM[np.isfinite(DEM)]
    zmax = np.nanmax(dem_data)
    zmin = np.nanmin(dem_data)

    zrange = zmax - zmin
    bin_width = min(100, int(zrange / 5))

    min_el = zmin - (zmin % bin_width)
    max_el = zmax + (bin_width - (zmax % bin_width))
    bins = np.arange(min_el, max_el+1, bin_width)
    return bins


def plot_dH_curve(ddem_data, dem_data, bins, dH_curve, dH_median, fbin_area, mytitle):
    dbin = bins[1] - bins[0]
    fig = plt.figure(facecolor='w', figsize=(12, 8), dpi=200)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.bar(bins, fbin_area, dbin, align='edge', alpha=0.5, color='0.5', edgecolor='k')
    ax2.set_ylim(0, 100)
    
    ax.plot(dem_data, ddem_data, 'o', color='0.5', ms=10, fillstyle='full')
    ax.plot(bins+dbin/2, dH_curve, 'k^-', linewidth=2, ms=8)
    ax.plot(bins+dbin/2, dH_median, 'ro-', linewidth=2, ms=8)
    ax.plot([bins.min(), bins.max() + dbin], [0, 0], 'k--')

    plt.xlim(bins.min(), bins.max() + dbin)
    plt.title(mytitle)
    ax.set_xlabel('elevation (m)')
    ax.set_ylabel('elevation difference (m)')
    ax2.set_ylabel('percent data')
    return fig


def outlier_filter(bins, DEM, dDEM, nsig=3):
    new_ddem = np.zeros(dDEM.size)
    digitized = np.digitize(DEM, bins)
    for i, _ in enumerate(bins):
        this_bindata = dDEM[digitized == i+1]
        dstd = 1
        old_mean = np.nanmean(this_bindata)
        old_std = np.nanstd(this_bindata)
        while abs(dstd) > 0.5:
            thresh_up = old_mean + nsig * old_std
            thresh_dn = old_mean - nsig * old_std
            this_bindata[np.logical_or(this_bindata > thresh_up, this_bindata < thresh_dn)] = np.nan
            dstd = np.nanstd(this_bindata) - old_std
            old_mean = np.nanmean(this_bindata)
            old_std = np.nanstd(this_bindata)
        new_ddem[digitized == i+1] = this_bindata
    return new_ddem


def fill_curve(bins, curve):
    if np.isnan(curve[0]):
        curve[0] = 0
    if np.isnan(curve[-1]):
        curve[-1] = 0
    valid = np.isfinite(curve)
    tmp_curve = interp1d(bins[valid], curve[valid])
    return tmp_curve(bins)


def parse_filename(fname):
    splitname = fname.split('_')
    if splitname[0] == 'AST':
        datestr = splitname[2][3:11]
        yy = int(datestr[4:])
        mm = int(datestr[0:2])
        dd = int(datestr[2:4])
    elif splitname[0] == 'SETSM':
        datestr = splitname[2]
        yy = int(datestr[0:4])
        mm = int(datestr[4:6])
        dd = int(datestr[6:])
    elif splitname[0] == 'SDMI':
        datestr = splitname[2]
        yy = int(datestr[0:4])
        mm = int(datestr[4:6])
        dd = int(datestr[6:])
    elif splitname[0] == 'SPOT':
        datestr = splitname[2]
        yy = int(datestr[0:4])
        mm = int(datestr[4:6])
        dd = int(datestr[6:])
    elif splitname[0] == 'Map':
        datestr = splitname[2]
        yy = int(datestr[0:4])
        mm = int(datestr[4:6])
        dd = int(datestr[6:])
    else:
        print("I don't recognize how to parse date information from {}.".format(fname))
        return None
    return dt.date(yy, mm, dd)


def nice_split(fname):
    sname = fname.strip('.tif').split('_')
    sname.remove('dH')
    splitloc = [i+1 for i, x in enumerate(sname[1:]) if x in ['AST', 'SETSM', 'Map', 'SPOT', 'SDMI']][0]
    name1 = '_'.join(sname[0:splitloc])
    name2 = '_'.join(sname[splitloc:])
    return name1, name2


def mmdd2dec(year, month, day):
    yearstart = dt.datetime(year, 1, 1).toordinal()
    nextyearstart = dt.datetime(year+1, 1, 1).toordinal()
    datestruct = dt.datetime(year, month, day)
    datefrac = float(datestruct.toordinal() + 0.5 - yearstart) / (nextyearstart - yearstart)
    return year + datefrac


def _argparser():
    parser = argparse.ArgumentParser(description="Calculate dH curves from MMASTER dH images, based on \
    shapefiles of glacier outlines.")
    parser.add_argument('dH_folder', action='store', type=str, help="Path to folder with dH images.")
    parser.add_argument('basedem', action='store', type=str, help="Path to base DEM to use.")
    parser.add_argument('glac_outlines', action='store', type=str, help="Shapefile of glacier outlines")
    parser.add_argument('--glac_mask', action='store', type=str, help="(optional) raster of glacier outlines with \
                         unique values for each glacier")
    parser.add_argument('--outlier', action='store', type=float, help="Value to use as outlier [None]")
    parser.add_argument('--pct_comp', action='store', type=float, default=0.67, help="Coverage of glacier elevation \
                        range curve must cover to be included [0.67]")
    parser.add_argument('--namefield', action='store', type=str,
                        default='RGIId', help="Field with identifying glacier name [RGIId]")
    parser.add_argument('--out_folder', action='store', type=str, default='.',
                        help="Path to write csv files [.]")
    parser.add_argument('--plot_curves', action='store_true', default='.',
                        help="Plot curves of dH(z) for each dDEM for each glacier [False]")
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    if args.plot_curves:
        # set font stuff
        font = {'family': 'sans',
                'weight': 'normal',
                'size': 22}
        #    legend_font = {'family': 'sans',
        #                   'weight': 'normal',
        #                   'size': '16'}
        matplotlib.rc('font', **font)

    # load base dem
    print('Loading DEM {}'.format(args.basedem))
    basedem = GeoImg(args.basedem)
    print('DEM loaded.')
    # get glacier masks
    if args.glac_mask is None:
        print('Rasterizing glacier polygons to DEM extent.')
        master_mask, master_glacs = it.rasterize_polygons(basedem, args.glac_outlines, burn_handle='fid')
        master_mask[master_mask < 0] = np.nan
    else:
        print('Loading raster of glacier polygons {}'.format(args.glac_mask))
        master_mask_geo = GeoImg(args.glac_mask)
        master_mask = master_mask_geo.img
        master_glacs = np.unique(master_mask[np.isfinite(master_mask)])
    # master_mask = np.logical_and(master_mask, np.isfinite(basedem.img))
    # get names
    gshp = gpd.read_file(args.glac_outlines)
    print('Glacier masks loaded.')
    # create output folder if it doesn't already exist
    os.system('mkdir -p {}'.format(args.out_folder))

    # create folders to store glacier dH curve figures
    for g in gshp[args.namefield]:
        os.system('mkdir -p {}'.format(os.path.sep.join([args.out_folder, g])))

    print('Getting glacier AADs.')
    # get aad
    aad_bins, aads = area_alt_dist(basedem, master_mask, glacier_inds=master_glacs)
    # initialize pd dataframes for dH_curves
    df_list = [pd.DataFrame(aad_bin, columns=['elevation']) for aad_bin in aad_bins]
    g_list = [str(gshp[args.namefield][gshp['fid'] == glac].values[0]) for glac in master_glacs]
    df_dict = dict(zip(g_list, df_list))

    # turn aad_bins, aads into dicts with RGIId as keys
    bin_dict = dict(zip(g_list, aad_bins))
    aad_dict = dict(zip(g_list, aads))
    
    for i, df in enumerate(df_list):
        df['area'] = pd.Series(aads[i], index=df.index)
    
    # now that we have the AADs, make sure we preserve that distribution when we reproject.
    bin_widths = [np.diff(b)[0] for b in aad_bins]  
    basedem.img[np.isnan(master_mask)] = np.nan # remove all elevations outside of the glacier mask
    for i, g in enumerate(master_glacs):
        basedem.img[master_mask == g] = np.floor(basedem.img[master_mask == g] / bin_widths[i]) * bin_widths[i]

    # get a list of all dH
    dH_list = glob('{}/*.tif'.format(args.dH_folder))
    
    # initialize ur_dataframe
    ur_df = pd.DataFrame([os.path.basename(x) for x in dH_list], columns=['filename'])
    ur_df['dem1'] = [nice_split(x)[0] for x in ur_df['filename']]
    ur_df['dem2'] = [nice_split(x)[1] for x in ur_df['filename']]
    date1 = [parse_filename(x) for x in ur_df['dem1']]
    date2 = [parse_filename(x) for x in ur_df['dem2']]
    ur_df['date1'] = date1
    ur_df['date2'] = date2
    ur_df['delta_t'] = [(x - y).days / 365.2425 for x, y in list(zip(date1, date2))]
    ur_df['centerdate'] = [(y + dt.timedelta((x - y).days / 2)) for x, y in list(zip(date1, date2))]

    print('Found {} files in {}'.format(len(dH_list), args.dH_folder))
    print('Getting dH curves.')
    for i, dHfile in enumerate(dH_list):
        dH = GeoImg(dHfile)
        print('{} ({}/{})'.format(dH.filename, i+1, len(dH_list)))
        if args.glac_mask is None:
            dh_mask, dh_glacs = it.rasterize_polygons(dH, args.glac_outlines, burn_handle='fid')
        else:
            tmp_dh_mask = master_mask_geo.reproject(dH, method=GRA_NearestNeighbour)
            dh_mask = tmp_dh_mask.img
            dh_glacs = np.unique(dh_mask[np.isfinite(dh_mask)])
        tmp_basedem = basedem.reproject(dH, method=GRA_NearestNeighbour)
        deltat = ur_df.loc[i, 'delta_t']
        this_fname = ur_df.loc[i, 'filename']
        for i, glac in enumerate(dh_glacs):
            this_name = str(gshp[args.namefield][gshp['fid'] == glac].values[0])
            this_dem = tmp_basedem.img[dh_mask == glac]
            this_ddem = dH.img[dh_mask == glac]
            this_ddem[np.abs(this_ddem) > args.outlier] = np.nan
            if np.count_nonzero(np.isfinite(this_ddem)) / this_ddem.size < 0.25:
                continue
            # these_bins = get_bins(this_dem, dh_mask)
            filtered_ddem = outlier_filter(bin_dict[this_name], this_dem, this_ddem)
            # _, odH_curve = get_dH_curve(this_dem, this_ddem, dh_mask, bins=aad_bins)
            _, fdH_curve, fbin_area = get_dH_curve(this_dem, filtered_ddem, dh_mask, bins=bin_dict[this_name])
            _, fdH_median, _ = get_dH_curve(this_dem, filtered_ddem, dh_mask, bins=bin_dict[this_name], mode='median')
            fbin_area = 100 * fbin_area * np.abs(dH.dx) * np.abs(dH.dy) / aad_dict[this_name]
            if args.plot_curves:
                plot_dH_curve(this_ddem, this_dem, bin_dict[this_name], fdH_curve,
                              fdH_median, fbin_area, dH.filename.strip('.tif'))
                plt.savefig(os.path.join(args.out_folder, this_name, dH.filename.strip('.tif') + '.png'),
                            bbox_inches='tight', dpi=200)
                plt.close()
            # write dH curve in units of dH/dt (so divide by deltat)
            this_fname = this_fname.rsplit('.tif', 1)[0]
            df_dict[this_name][this_fname + '_mean'] = pd.Series(fdH_curve / deltat, index=df_dict[this_name].index)
            df_dict[this_name][this_fname + '_med'] = pd.Series(fdH_median / deltat, index=df_dict[this_name].index)
            df_dict[this_name][this_fname + '_pct'] = pd.Series(fbin_area, index=df_dict[this_name].index)

    print('Writing dH curves to {}'.format(args.out_folder))
    # write all dH_curves
    for g in df_dict.keys():
        print(g)
        df_dict[g].to_csv(os.path.sep.join([args.out_folder, '{}_dH_curves.csv'.format(g)]), index=False)


if __name__ == "__main__":
    main()
