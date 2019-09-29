from __future__ import print_function
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from descartes import PolygonPatch
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.polynomial import polyval, polyfit
from pybob.bob_tools import bin_data
from pybob.image_tools import hillshade


def set_pretty_fonts(font_size=24, legend_size=16):
    """
    sets matplotlib fonts to be nice and pretty for graphs that don't completely suck.
    """
    font = {'family': 'sans',
            'weight': 'normal',
            'size': font_size}
    legend_font = {'family': 'sans',
                   'weight': 'normal',
                   'size': legend_size}
    matplotlib.rc('font', **font)
    plt.ion()

def truncate_colormap(cmap, minval=0, maxval=1, n=100):
    """

    :param cmap:
    :param minval:
    :param maxval:
    :param n:
    :return:
    """
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n}, {a:.2f}, {b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def save_results_quicklook(img, raster, com_ext, outfilename, vmin=0, vmax=10, sfact=2, output_directory='.'):
    """

    :param img:
    :param raster:
    :param com_ext:
    :param outfilename:
    :param vmin:
    :param vmax:
    :param sfact:
    :param output_directory:
    :return:
    """
    xmin, ymin, xmax, ymax = com_ext
    ext = [xmin, xmax, ymin, ymax]

    fig = img.overlay(raster, extent=ext, sfact=sfact, vmin=vmin, vmax=vmax, showfig=False)

    savefig(os.path.join(output_directory, outfilename + '.png'), bbox_inches='tight', dpi=200)
    return fig


def plot_geoimg_sidebyside(img1, img2, com_extent=None, fig=None, cmap='gray', output_directory='.', filename=None):
    """

    :param img1:
    :param img2:
    :param com_extent:
    :param fig:
    :param cmap:
    :param output_directory:
    :param filename:
    :return:
    """
    if fig is None:
        fig = plt.figure(facecolor='w', figsize=(16.5, 16.5), dpi=200)
    elif isinstance(fig, Figure):
        fig
    else:
        fig = plt.figure(fig)

    ax1 = plt.subplot(121)
    if com_extent is not None:
        plt.imshow(img1, interpolation='nearest', cmap=cmap, extent=com_extent)
    else:
        plt.imshow(img1, interpolation='nearest', cmap=cmap)

    ax2 = plt.subplot(122)
    if com_extent is not None:
        plt.imshow(img2, interpolation='nearest', cmap=cmap, extent=com_extent)
    else:
        plt.imshow(img2, interpolation='nearest', cmap=cmap)

    if filename is not None:
        savefig(os.path.join(output_directory, filename), bbox_inches='tight', dpi=200)

    return fig, ax1, ax2


def plot_nice_histogram(data_values, outfilename, output_directory):
    """

    :param data_values:
    :param outfilename:
    :param output_directory:
    :return:
    """
    fig = plt.figure(facecolor='w', figsize=(16.5, 16.5), dpi=200)

    outputs = plt.hist(data_values, 25, alpha=0.5, normed=1)

    ylim = plt.gca().get_ylim()
    mu = data_values.mean()
    sigma = data_values.std()

    plt.plot([mu, mu], ylim, 'k--')
    plt.plot([mu+sigma, mu+sigma], ylim, 'r--')
    plt.plot([mu+2*sigma, mu+2*sigma], ylim, 'r--')

    plt.text(mu, np.mean(ylim), r'$\mu$', fontsize=24, color='k')
    plt.text(mu+sigma, np.mean(ylim), r'$\mu + \sigma$', fontsize=24, color='r')
    plt.text(mu+2*sigma, np.mean(ylim), r'$\mu + 2\sigma$', fontsize=24, color='r')
    # just to be sure
    plt.gca().set_ylim(ylim)

    savefig(output_directory + os.path.sep + outfilename + '_vdist.png', bbox_inches='tight', dpi=200)
    return fig, outputs


def plot_chips_corr_matrix(srcimg, destimg, corrmat):
    """

    :param srcimg:
    :param destimg:
    :param corrmat:
    :return:
    """
    fig = plt.figure()
    plt.ion()

    ax1 = plt.subplot(131)
    plt.imshow(srcimg, cmap='gray')
    ax1.set_title('source image')

    ax2 = plt.subplot(132)
    plt.imshow(destimg, cmap='gray')
    ax2.set_title('dest. image')

    peak1_loc = cv2.minMaxLoc(corrmat)[3]

    ax3 = plt.subplot(133)
    plt.imshow(corrmat, cmap='jet')
    plt.plot(peak1_loc[0], peak1_loc[1], 'w+')
    ax3.set_title('correlation')

    plt.show()

    return fig


def plot_ddem_results(img, colormap='seismic', caxsize="2.5%", clim=None, sfact=None):
    """

    :param img:
    :param colormap:
    :param caxsize:
    :param clim:
    :param sfact:
    :return:
    """
    if sfact is not None:
        fig = img.display(cmap=colormap, sfact=sfact)
    else:
        fig = img.display(cmap=colormap)

    if clim is not None:
        plt.clim(clim[0], clim[1])

    ax = plt.gca()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=caxsize, pad=0.1)
    plt.colorbar(cax=cax)

    return fig, ax, cax


def plot_dh_elevation(dDEM, DEM, glacier_mask=None, binning=None, bin_width=50, polyorder=None):
    """

    :param dDEM:
    :param DEM:
    :param glacier_mask:
    :param binning:
    :param bin_width:
    :param polyorder:
    :return:
    """
    fig = plt.figure()
    plt.ion()

    if glacier_mask is not None:
        ddem_data = dDEM.img[np.logical_and(np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img)), glacier_mask)]
        dem_data = DEM.img[np.logical_and(np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img)), glacier_mask)]
    else:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img))]
        dem_data = DEM.img[np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img))]

    plt.plot(dem_data, ddem_data, '+')

    if binning is not None:
        if binning not in ['mean', 'median', 'poly']:
            raise ValueError('binning must be one of mean, median, polyfit')
        min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % bin_width)
        max_el = np.nanmax(dem_data) - (bin_width - (np.nanmax(dem_data) % bin_width))
        bins = np.arange(min_el, max_el, bin_width)
        if binning in ['mean', 'median']:
            binned_data = bin_data(bins, ddem_data, dem_data, mode=binning)
            plt.plot(bins, binned_data, 'r', linewidth=3)
        else:
            if polyorder is None:
                raise ValueError('polyorder must be defined to use polynomial fitting.')
            pfit = polyfit(dem_data, ddem_data, polyorder)
            interp_points = polyval(bins, pfit)
            plt.plot(bins, interp_points, 'r', linewidth=3)
    return fig


def plot_polygon_df(polygon_df, fig=None, ax=None, mpl='mpl_polygon', **kwargs):
    """
    Plot a GeoDataFrame of polygons to figure.

    :param polygon_df: GeoDataFrame of polygon(s) to plot.
    :param fig: Optional existing figure to plot polygons to. If unset, creates a new figure.
    :param ax: Optional axis handle to plot polygons to. If unset, uses fig.gca()
    :param mpl: GeoDataFrame column name containing multipolygon indices. Default is mpl_polygon.
    :param kwargs: Keyword options to pass to matplotlib.collections.PatchCollection.

    :type polygon_df: geopandas.GeoDataFrame
    :type fig: matplotlib.pyplot.Figure
    :type ax: matplotlib.axes
    :type mpl: str

    :returns fig, polygon_df: Figure handle of the plot created, and updated geodataframe with plot geometries.

    Examples:

    >>> rgi = gpd.read_file('07_rgi60_Svalbard.shp')
    >>> f, rgi = plot_polygon_df(rgi, color='w', edgecolor='k', lw=.2, alpha=0.5)
    """

    if not isinstance(polygon_df, gpd.GeoDataFrame):
        raise Exception('polygon_df must be a GeoDataFrame.')
    
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()
    
    if mpl not in polygon_df.columns:
        polygon_df[mpl] = np.nan
        polygon_df[mpl] = polygon_df[mpl].astype(object)
        
        for i, row in polygon_df.iterrows():
            m_poly = row['geometry']
            poly = []
            if m_poly.geom_type == 'MultiPolygon':
                for pol in m_poly:
                    poly.append(PolygonPatch(pol))
            else:
                poly.append(PolygonPatch(m_poly))
            polygon_df.set_value(i, mpl, poly)
        
    mapindex = polygon_df[mpl].to_dict()
    for i, g in mapindex.items():
        p = PatchCollection(g, **kwargs)
        ax.add_collection(p)
    
    plt.axis('equal')

    return fig, polygon_df
    

def plot_shaded_dem(dem, azimuth=315, altitude=45, fig=None, extent=None, alpha=0.35, colormap='terrain', **kwargs):
    """
    Plot a shaded relief image of a DEM.

    :param dem: GeoImg representing a DEM.
    :param azimuth: Solar azimuth angle, in degress from North. Default 315.
    :param altitude: Solar altitude angle, in degrees from horizon. Default 45.
    :param fig: Figure to show image in. If not set, creates a new figure.
    :param extent: Spatial extent to limit the figure to, given as xmin, xmax, ymin, ymax.
    :param alpha: Alpha value to set DEM to. Default is 0.35.
    :param colormap: colormap style for matplotlib
    :param kwargs: Optional keyword arguments to pass to plt.imshow

    :type dem: pybob.GeoImg
    :type azimuth: float
    :type altitude: float
    :type fig: matplotlib.figure.Figure
    :type extent: array-like
    :type alpha: float
    :type colormap: str

    :returns fig: Handle pointing to the matplotlib Figure created (or passed to display).
    """
    if fig is None:
        fig = plt.figure()

    cmap = plt.get_cmap(colormap)
    new_cmap = truncate_colormap(cmap, 0.25, 0.75)

    shaded = hillshade(dem, azimuth=azimuth, altitude=altitude)
    if extent is None:
        extent = [dem.xmin, dem.xmax, dem.ymin, dem.ymax]

        mini = 0
        maxi = dem.npix_y
        minj = 0
        maxj = dem.npix_x
    else:
        xmin, xmax, ymin, ymax = extent
        mini, minj = dem.xy2ij((xmin, ymax))
        maxi, maxj = dem.xy2ij((xmax, ymin))

        mini += 0.5
        minj += 0.5

        maxi -= 0.5
        maxj -= 0.5  # subtract the .5 for half a pixel, add 1 for slice

    plt.imshow(255 * (shaded[int(mini):int(maxi+1), int(minj):int(maxj+1)] + 1) / 2, extent=extent, cmap='gray')
    cimg = plt.imshow(dem.img[int(mini):int(maxi+1), int(minj):int(maxj+1)],
                      cmap=new_cmap, extent=extent, alpha=alpha, **kwargs)

    return fig, cimg


def plot_aad_diagram():
    pass