import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.polynomial import polyval, polyfit
from pybob.bob_tools import bin_data


def save_results_quicklook(img, raster, com_ext, outfilename, vmin=0, vmax=10, sfact=2, output_directory='.'):
    xmin, ymin, xmax, ymax = com_ext
    ext = [xmin, xmax, ymin, ymax]

    fig = img.overlay(raster, extent=ext, sfact=sfact, vmin=vmin, vmax=vmax, showfig=False)

    savefig(os.path.join(output_directory, outfilename + '.png'), bbox_inches='tight', dpi=200)
    return fig


def plot_geoimg_sidebyside(img1, img2, com_extent=None, fig=None, cmap='gray', output_directory='.', filename=None):
    if fig is None:
        fig = plt.figure(facecolor='w', figsize=(16.5, 16.5), dpi=200)
    elif isinstance(fig, Figure):
        fig
    else:
        fig = plt.figure(fig)

    ax1 = plt.subplot(121, axisbg=(0.1, 0.15, 0.15))
    if com_extent is not None:
        plt.imshow(img1, interpolation='nearest', cmap=cmap, extent=com_extent)
    else:
        plt.imshow(img2, interpolation='nearest', cmap=cmap)

    ax2 = plt.subplot(122, axisbg=(0.1, 0.15, 0.15))
    if com_extent is not None:
        plt.imshow(img2, interpolation='nearest', cmap=cmap, extent=com_extent)
    else:
        plt.imshow(img2, interpolation='nearest', cmap=cmap)

    if filename is not None:
        savefig(os.path.join(output_directory, filename), bbox_inches='tight', dpi=200)

    return fig, ax1, ax2


def plot_nice_histogram(data_values, outfilename, output_directory):
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
