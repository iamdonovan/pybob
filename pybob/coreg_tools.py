"""
pybob.coreg_tools provides a toolset for coregistering DEMs, based on the method presented by `Nuth and Kääb (2011)`_.

.. _Nuth and Kääb (2011): https://www.the-cryosphere.net/5/271/2011/tc-5-271-2011.html
"""
# from __future__ import print_function
import os
import errno
from osgeo import gdal
import numpy as np
import matplotlib.pylab as plt
# plt.switch_backend('agg')
import scipy.optimize as optimize
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pybob.GeoImg import GeoImg
from pybob.ICESat import ICESat
from pybob.image_tools import create_mask_from_shapefile
from pybob.plot_tools import plot_shaded_dem
import gc


def get_slope(geoimg, alg='Horn'):
    """
    Wrapper function to calculate DEM slope using gdal.DEMProcessing.

    :param geoimg: GeoImg object of DEM to calculate slope
    :param alg: Algorithm for calculating Slope. One of 'ZevenbergenThorne' or 'Horn'. Default is 'Horn'.
    :type geoimg: pybob.GeoImg
    :type alg: str
    :returns geo_slope: new GeoImg object with slope raster
    """
    assert alg in ['ZevenbergenThorne', 'Horn'], "alg not recognized: {}".format(alg)
    slope_ = gdal.DEMProcessing('', geoimg.gd, 'slope', format='MEM', alg=alg)
    return GeoImg(slope_)


def get_aspect(geoimg, alg='Horn'):
    """
    Wrapper function to calculate DEM aspect using gdal.DEMProcessing.

    :param geoimg: GeoImg object of DEM to calculate aspect
    :param alg: Algorithm for calculating Aspect. One of 'ZevenbergenThorne' or 'Horn'. Default is 'Horn'.
    :type geoimg: pybob.GeoImg
    :type alg: str
    :returns geo_aspect: new GeoImg object with aspect raster
    """
    assert alg in ['ZevenbergenThorne', 'Horn'], "alg not recognized: {}".format(alg)
    aspect_ = gdal.DEMProcessing('', geoimg.gd, 'aspect', format='MEM', alg=alg)
    return GeoImg(aspect_)


def false_hillshade(dH, title, pp=None, clim=(-20, 20)):
    """
    Create a map figure showing the differences in black and white... 

    :param dh: GeoImg object of elevation differences 
    :param title: Title for plot
    :type geoimg: pybob.GeoImg
    :type title: str
    :returns fig: either prints to a pdf, or returns a figure
    """
    niceext = np.array([dH.xmin, dH.xmax, dH.ymin, dH.ymax]) / 1000.
    dHtemp = np.ma.masked_invalid(dH.img)
    mykeep = np.ma.less(np.ma.abs(dHtemp), np.ma.std(dHtemp) * 3)
    # mykeep = np.logical_and.reduce((np.isfinite(dH.img), (np.abs(dH.img) < np.nanstd(dH.img) * 3)))
    dH_vec = dHtemp[mykeep]

    if pp is not None:
        fig = plt.figure(figsize=(7, 5), dpi=300)
    else:
        fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()

    im1 = ax.imshow(dH.img, extent=niceext, origin='upper')

    ymin = np.ma.mean(dH_vec) - 2 * np.ma.std(dH_vec)
    ymax = np.ma.mean(dH_vec) + 2 * np.ma.std(dH_vec)

    im1.set_clim(ymin, ymax)
    im1.set_cmap('Greys')

    #    if np.sum(np.isfinite(dH_vec))<10:
    #        print("Error for statistics in false_hillshade")
    #    else:
    plt.title(title, fontsize=14)

    numwid = max([len('{:.1f} m'.format(np.ma.mean(dH_vec))),
                  len('{:.1f} m'.format(np.ma.median(dH_vec))), len('{:.1f} m'.format(np.ma.std(dH_vec)))])
    plt.annotate('MEAN:'.ljust(8) + ('{:.1f} m'.format(np.ma.mean(dH_vec))).rjust(numwid), xy=(0.65, 0.95),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.annotate('MEDIAN:'.ljust(8) + ('{:.1f} m'.format(np.ma.median(dH_vec))).rjust(numwid),
                 xy=(0.65, 0.90), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 color='red', family='monospace')
    plt.annotate('STD:'.ljust(8) + ('{:.1f} m'.format(np.ma.std(dH_vec))).rjust(numwid), xy=(0.65, 0.85),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    # plt.colorbar(im1)

    plt.tight_layout()
    gc.collect()
    if pp is not None:
        pp.savefig(fig, dpi=300)
        return
    else:
        return fig


def create_stable_mask(img, mask1, mask2):
    """
    Create mask representing stable terrain, given exclusion (i.e., glacier) and inclusion (i.e., land) masks.

    :param img: GeoImg to pull extents from
    :param mask1: filename for shapefile representing pixels to exclude from stable terrain (i.e., glaciers)
    :param mask2: filename for shapefile representing pixels to include in stable terrain (i.e., land)
    :type img: pybob.GeoImg
    :type mask1: str
    :type mask2: str

    :returns stable_mask: boolean array representing stable terrain
    """
    # if we have no masks, just return an array of true values
    if mask1 is None and mask2 is None:
        return np.ones(img.img.shape) == 0  # all false, so nothing will get masked.
    elif mask1 is not None and mask2 is None:  # we have a glacier mask, not land

        if mask1.split('.')[-1] == 'tif':
            mask_rast = myRaster = gdal.Open(mask1)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            mask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
        else:
            mask = create_mask_from_shapefile(img, mask1)
        return mask  # returns true where there's glacier, false everywhere else
    elif mask1 is None and mask2 is not None:
        if mask2.split('.')[-1] == 'tif':
            mask_rast = myRaster = gdal.Open(mask2)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            mask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
        else:
            mask = create_mask_from_shapefile(img, mask2)
        return np.logical_not(mask)  # false where there's land, true where there isn't
    else:  # if none of the above, we have two masks.
        # implement option if either or, or both mask are given as rasters. 

        if (mask1.split('.')[-1] == 'tif') & (mask2.split('.')[-1] == 'shp'):
            mask_rast = myRaster = gdal.Open(mask1)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            gmask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
            lmask = create_mask_from_shapefile(img, mask2)
        elif (mask2.split('.')[-1] == 'tif') & (mask1.split('.')[-1] == 'shp'):
            mask_rast = myRaster = gdal.Open(mask2)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            lmask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
            gmask = create_mask_from_shapefile(img, mask1)
        elif (mask1.split('.')[-1] == 'tif') & (mask2.split('.')[-1] == 'tif'):
            mask_rast = myRaster = gdal.Open(mask1)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            gmask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
            mask_rast = myRaster = gdal.Open(mask2)
            transform = myRaster.GetGeoTransform()
            dx = transform[1]
            dy = transform[5]
            Xsize = myRaster.RasterXSize
            Ysize = myRaster.RasterYSize
            lmask = myRaster.ReadAsArray(0, 0, Xsize, Ysize)
        else:
            gmask = create_mask_from_shapefile(img, mask1)
            lmask = create_mask_from_shapefile(img, mask2)
        return np.logical_or(gmask, np.logical_not(lmask))  # true where there's glacier or water


def preprocess(stable_mask, slope, aspect, primary, secondary):
    if isinstance(primary, GeoImg):
        stan = np.tan(np.radians(slope)).astype(np.float32)
        dH = primary.copy(new_raster=(primary.img - secondary.img))
        dH.img[stable_mask] = np.nan
        primary_mask = isinstance(primary.img, np.ma.masked_array)
        secondary_mask = isinstance(secondary.img, np.ma.masked_array)

        if primary_mask and secondary_mask:
            dH.mask(np.logical_or(primary.img.mask, secondary.img.mask))
        elif primary_mask:
            dH.mask(primary.img.mask)
        elif secondary_mask:
            dH.mask(secondary.img.mask)

        if dH.isfloat:
            dH.img[stable_mask] = np.nan

        # dHtan = dH.img / stan
        mykeep = np.logical_and.reduce((np.ma.less(np.absolute(np.ma.masked_invalid(dH.img)), 200.0),
                                        np.isfinite(dH.img),
                                        np.isfinite(aspect),
                                        np.isfinite(stan),
                                        np.ma.greater(np.ma.masked_invalid(slope), 3.0),
                                        np.ma.masked_invalid(dH.img) != 0.0,
                                        np.ma.greater_equal(np.ma.masked_invalid(aspect), 0),
                                        np.ma.less(np.ma.masked_invalid(stan), 100)))

        dH.img[np.invert(mykeep)] = np.nan
        xdata = aspect[mykeep]
        #        ydata = dHtan[mykeep]
        ydata = dH.img[mykeep]
        sdata = stan[mykeep]

    elif isinstance(primary, ICESat):
        secondary_pts = secondary.raster_points2(primary.xy, nsize=5, mode='cubic')
        dH = primary.elev - secondary_pts

        slope_pts = slope.raster_points2(primary.xy, nsize=5, mode='cubic')
        stan = np.tan(np.radians(slope_pts))

        aspect_pts = aspect.raster_points2(primary.xy, nsize=5, mode='cubic')

        smask = stable_mask.raster_points2(primary.xy) > 0

        dH[smask] = np.nan

        # dHtan = dH / stan
        mykeep = np.logical_and.reduce((np.ma.less(np.absolute(np.ma.masked_invalid(dH)), 200.0),
                                        np.isfinite(dH),
                                        np.isfinite(aspect_pts),
                                        np.isfinite(stan),
                                        np.ma.greater(np.ma.masked_invalid(slope_pts), 3.0),
                                        np.ma.masked_invalid(dH) != 0.0,
                                        np.ma.greater_equal(np.ma.masked_invalid(aspect_pts), 0),
                                        np.ma.less(np.ma.masked_invalid(stan), 100)))
        # mykeep = ((np.absolute(dH) < 100.0) & np.isfinite(dH) &
        #           (slope_pts > 3.0) & (dH != 0.0) & (aspect_pts >= 0) & (stan < 100))

        dH[np.invert(mykeep)] = np.nan
        xdata = aspect_pts[mykeep]
        # ydata = dHtan[mykeep]
        ydata = dH[mykeep]
        sdata = stan[mykeep]

    gc.collect()
    return dH, xdata, ydata, sdata


def coreg_fitting(xdata, ydata, sdata, title, pp=None):
    xdata = xdata.astype(np.float64)  # float64 truly necessary?
    ydata = ydata.astype(np.float64)
    sdata = sdata.astype(np.float64)
    ydata2 = np.divide(ydata, sdata)

    # fit using equation 3 of Nuth and Kaeaeb, 2011
    def fitfun(p, x, s):
        return p[0] * np.cos(np.radians(p[1] - x)) * s + p[2]

    def errfun(p, x, s, y):
        return fitfun(p, x, s) - y

    if xdata.size > 20000:
        mysamp = np.random.randint(0, xdata.size, 20000)
    else:
        mysamp = np.arange(0, xdata.size)

    mysamp = mysamp.astype(np.int64)

    # embed()
    # print("soft_l1")
    # lb = [-200, 0, -300]
    # ub = [200, 180, 300]
    p0 = [1, 1, -1]
    # p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]),
    #                                        method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
    # myresults = optimize.least_squares(errfun, p0, args=(xdata, ydata), method='trf', loss='soft_l1', f_scale=0.5)
    # myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], sdata[mysamp],
    #                                                     ydata[mysamp], method='trf', loss='soft_l1',
    #                                                     f_scale=0.1,ftol=1E-4,xtol=1E-4)
    # myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], ydata[mysamp]),
    #                                   method='trf', bounds=([lb,ub]), loss='soft_l1',
    #                                   f_scale=0.1,ftol=1E-8,xtol=1E-8)    
    myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], sdata[mysamp], ydata[mysamp]), method='trf',
                                       loss='soft_l1', f_scale=0.1, ftol=1E-8, xtol=1E-8)

    p1 = myresults.x
    # success = myresults.success # commented because it wasn't actually being used.
    # print success
    # print p1
    # convert to shift parameters in cartesian coordinates
    xadj = p1[0] * np.sin(np.radians(p1[1]))
    yadj = p1[0] * np.cos(np.radians(p1[1]))
    zadj = p1[2]  # * sdata.mean(axis=0)

    xp = np.linspace(0, 360, 361)
    # sp = np.zeros(xp.size) + 0.785398
    # sp = np.zeros(xp.size) + np.tan(np.radians(5)).astype(np.float32)
    sp = np.ones(xp.size) + np.nanmean(sdata[mysamp])
    p1[2] = np.divide(p1[2], np.nanmean(sdata[mysamp]))
    yp = fitfun(p1, xp, sp)

    if pp is not None:
        fig = plt.figure(figsize=(7, 5), dpi=300)
    else:
        fig = plt.figure(figsize=(7, 5))
    # fig.suptitle(title, fontsize=14)
    plt.title(title, fontsize=14)
    plt.plot(xdata[mysamp], ydata2[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
    plt.plot(xp, np.zeros(xp.size), 'k', ms=3)
    plt.plot(xp, np.divide(yp, sp), 'r-', ms=2)

    plt.xlim(0, 360)
    ymin, ymax = plt.ylim((np.nanmean(ydata2[mysamp])) - 2 * np.nanstd(ydata2[mysamp]),
                          (np.nanmean(ydata2[mysamp])) + 2 * np.nanstd(ydata2[mysamp]))

    # plt.axis([0, 360, -200, 200])
    plt.xlabel('Aspect [degrees]')
    plt.ylabel('dH / tan(slope)')
    numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
    plt.text(0.05, 0.15, '$\Delta$x: ' + ('{:.1f} m'.format(xadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.1, '$\Delta$y: ' + ('{:.1f} m'.format(yadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, '$\Delta$z: ' + ('{:.1f} m'.format(zadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)

    gc.collect()
    if pp is not None:
        pp.savefig(fig, dpi=200)
        return xadj, yadj, zadj
    else:
        return xadj, yadj, zadj, fig


def final_histogram(dH0, dHfinal, pp=None):
    if pp is not None:
        fig = plt.figure(figsize=(7, 5), dpi=200)
    else:
        fig = plt.figure(figsize=(7, 5))
    plt.title('Elevation difference histograms', fontsize=14)

    dH0 = np.ma.masked_invalid(dH0)
    dHfinal = np.ma.masked_invalid(dHfinal)

    dH0 = np.squeeze(np.ma.masked_invalid(dH0[np.ma.less(np.ma.abs(dH0), np.ma.std(dH0) * 3)]))
    dHfinal = np.squeeze(np.ma.masked_invalid(dHfinal[np.ma.less(np.ma.abs(dHfinal), np.ma.std(dHfinal) * 3)]))

    # dH0=dH0[np.isfinite(dH0)]
    # dHfinal=dHfinal[np.isfinite(dHfinal)]

    stats0 = [np.ma.mean(dH0), np.ma.median(dH0), np.ma.std(dH0), RMSE(dH0), np.ma.sum(np.isfinite(dH0))]
    stats_fin = [np.ma.mean(dHfinal), np.ma.median(dHfinal), np.ma.std(dHfinal), RMSE(dHfinal),
                 np.ma.sum(np.isfinite(dHfinal))]

    if (np.less(stats0[2], 1)):
        myrange = (-4, 4)
    elif np.logical_and(np.greater(stats0[2], 1), np.less(stats0[2], 5)):
        myrange = (-25, 25)
    else:
        myrange = (-60, 60)

    j1, j2 = np.histogram(dH0.compressed(), bins=100, range=myrange)
    k1, k2 = np.histogram(dHfinal.compressed(), bins=100, range=myrange)

    plt.plot(j2[1:], j1, 'k-', linewidth=2)
    plt.plot(k2[1:], k1, 'r-', linewidth=2)
    # plt.legend(['Original', 'Coregistered'])

    plt.xlabel('Elevation difference [meters]')
    plt.ylabel('Number of samples')
    plt.xlim(myrange[0], myrange[1])

    # numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
    plt.text(0.05, 0.90, 'Mean: ' + ('{:.1f} m'.format(stats0[0])),
             fontsize=12, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, 'Median: ' + ('{:.1f} m'.format(stats0[1])),
             fontsize=12, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, 'Std dev.: ' + ('{:.1f} m'.format(stats0[2])),
             fontsize=12, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, 'RMSE: ' + ('{:.1f} m'.format(stats0[3])),
             fontsize=12, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)

    plt.text(0.05, 0.65, 'Mean: ' + ('{:.1f} m'.format(stats_fin[0])),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.60, 'Median: ' + ('{:.1f} m'.format(stats_fin[1])),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.55, 'Std dev.: ' + ('{:.1f} m'.format(stats_fin[2])),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.50, 'RMSE: ' + ('{:.1f} m'.format(stats_fin[3])),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    if pp is not None:
        pp.savefig(fig, bbox_inches='tight', dpi=200)

    return stats_fin, stats0


def RMSE(indata):
    """ Return root mean square of indata.

    :param indata: differences to calculate root mean square of
    :type indata: array-like

    :returns myrmse: RMSE of indata.
    """
    myrmse = np.sqrt(np.nanmean(np.asarray(indata) ** 2))
    return myrmse


def get_geoimg(indata):
    if type(indata) is str or type(indata) is gdal.Dataset:
        return GeoImg(indata)
    elif type(indata) is GeoImg:
        return indata
    else:
        raise TypeError('input data must be a string pointing to a gdal dataset, or a GeoImg object.')


def dem_coregistration(primaryDEM, secondaryDEM, glaciermask=None, landmask=None, outdir='.',
                       pts=False, full_ext=False, return_var=True, alg='Horn', magnlimit=2, inmem=False):
    """
    Iteratively co-register elevation data.

    :param primaryDEM: Path to filename or GeoImg dataset representing "primary" DEM.
    :param secondaryDEM: Path to filename or GeoImg dataset representing "secondary" DEM.
    :param glaciermask: Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    :param landmask: Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    :param outdir: Location to save co-registration outputs.
    :param pts: If True, program assumes that primaryDEM represents point data (i.e., ICESat),
        as opposed to raster data. Slope/aspect are then calculated from secondaryDEM.
        primaryDEM should be a string representing an HDF5 file continaing ICESat data.
    :param full_ext: If True, program writes full extents of input DEMs. If False, program writes
        input DEMs cropped to their common extent. Default is False.
    :param return_var: return variables representing co-registered DEMs and offsets (default).
    :param alg: Algorithm for calculating Slope, Aspect. One of 'ZevenbergenThorne' or 'Horn'. Default is 'Horn'.
    :param magnlimit: Magnitude threshold for determining termination of co-registration algorithm, calculated as
        sum in quadrature of dx, dy, dz shifts. Default is 2 m.
    :param inmem: Don't write anything to disk

    :type primaryDEM: str, pybob.GeoImg
    :type secondaryDEM: str, pybob.GeoImg
    :type glaciermask: str
    :type landmask: str
    :type outdir: str
    :type pts: bool
    :type full_ext: bool
    :type return_var: bool
    :type alg: str
    :type magnlimit: float
    :type inmem: bool

    :returns primaryDEM, outsecondary, out_offs, stats: if return_var=True, returns primary DEM, co-registered
        secondary DEM x,y,z shifts removed from secondary DEM, and before/after comparison stats.

        If co-registration fails (i.e., there are too few acceptable points to perform co-registration), then returns
        original primary and secondary DEMs, with offsets set to -1.
    """

    # if the output directory does not exist, create it.
    outdir = os.path.abspath(outdir)
    try:
        os.makedirs(outdir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(outdir):
            pass
        else:
            raise
    # make a file to save the coregistration parameters and statistics to.
    paramf = open(outdir + os.path.sep + 'coreg_params.txt', 'w')
    statsf = open(outdir + os.path.sep + 'stats.txt', 'w')
    # create the output pdf
    pp = PdfPages(outdir + os.path.sep + 'CoRegistration_Results.pdf')

    if full_ext:
        print('Writing full extents of output DEMs.')
    else:
        print('Writing DEMs cropped to common extent.')

    if type(primaryDEM) is str:
        mfilename = os.path.basename(primaryDEM)
        mfiledir = os.path.dirname(primaryDEM)
        if mfiledir == '':
            mfiledir = os.path.abspath('.')
    else:
        mfilename = primaryDEM.filename
        mfiledir = primaryDEM.in_dir_abs_path

    if type(secondaryDEM) is str:
        sfilename = os.path.basename(secondaryDEM)
        sfiledir = os.path.dirname(secondaryDEM)
        if sfiledir == '':
            sfiledir = os.path.abspath('.')
    else:
        sfilename = secondaryDEM.filename
        sfiledir = secondaryDEM.in_dir_abs_path

    secondaryDEM = get_geoimg(secondaryDEM)
    #    secondaryDEM.mask(np.less(secondaryDEM.img.data,-30))
    # we assume that we are working with 'area' pixels (i.e., pixel x,y corresponds to corner, not center)
    if secondaryDEM.is_point():
        secondaryDEM.to_area()

    # if we're dealing with ICESat/pt data, change how we load primaryDEM data
    if pts:
        if not isinstance(primaryDEM, ICESat):
            primaryDEM = ICESat(primaryDEM)

        primaryDEM.project('epsg:{}'.format(secondaryDEM.epsg))
        mybounds = [secondaryDEM.xmin, secondaryDEM.xmax, secondaryDEM.ymin, secondaryDEM.ymax]
        primaryDEM.clip(mybounds)
        primaryDEM.clean()
        slope_geo = get_slope(secondaryDEM, alg)
        aspect_geo = get_aspect(secondaryDEM, alg)

        if not inmem:
            slope_geo.write('tmp_slope.tif', out_folder=outdir)
            aspect_geo.write('tmp_aspect.tif', out_folder=outdir)

        smask = create_stable_mask(secondaryDEM, glaciermask, landmask)
        secondaryDEM.mask(smask)
        stable_mask = secondaryDEM.copy(new_raster=smask)  # make the mask a geoimg

        # Create initial plot of where stable terrain is, including ICESat pts
        fig1, _ = plot_shaded_dem(secondaryDEM)
        plt.plot(primaryDEM.x[~np.isnan(primaryDEM.elev)], primaryDEM.y[~np.isnan(primaryDEM.elev)], 'k.')
        pp.savefig(fig1, bbox_inches='tight', dpi=200)

    else:
        orig_primaryDEM = get_geoimg(primaryDEM)

        # orig_primaryDEM.mask(np.less(orig_primaryDEM.img.data,-10))
        if orig_primaryDEM.is_point():
            orig_primaryDEM.to_area()
        primaryDEM = orig_primaryDEM.reproject(secondaryDEM)  # need to resample primaryDEM to cell size, extent of secondary.

        stable_mask = create_stable_mask(primaryDEM, glaciermask, landmask)

        slope_geo = get_slope(primaryDEM, alg)
        aspect_geo = get_aspect(primaryDEM, alg)
        if not inmem:
            slope_geo.write('tmp_slope.tif', out_folder=outdir)
            aspect_geo.write('tmp_aspect.tif', out_folder=outdir)
        primaryDEM.mask(stable_mask)

    slope = np.ma.masked_invalid(slope_geo.img)
    aspect = np.ma.masked_invalid(aspect_geo.img)

    if np.ma.sum(np.ma.greater(slope.flatten(), 3)) < 500:
        print("Exiting: Fewer than 500 valid slope points")
        if return_var:
            pp.close()
            return primaryDEM, secondaryDEM, -1
        else:
            pp.close()
            return -1

    mythresh = np.float64(200)  # float64 really necessary?
    mystd = np.float64(200)
    mycount = 0
    tot_dx = np.float64(0)
    tot_dy = np.float64(0)
    tot_dz = np.float64(0)
    magnthresh = 200

    mytitle = 'DEM difference: pre-coregistration'

    if pts:
        this_secondary = secondaryDEM
        this_secondary.mask(stable_mask.img)
    else:
        this_secondary = secondaryDEM.reproject(primaryDEM)
        this_secondary.mask(stable_mask)

    plt.close('all')
    while mythresh > 2 and magnthresh > magnlimit:
        mycount += 1
        print("Running iteration #{}".format(mycount))
        print("Running iteration #{}".format(mycount), file=paramf)

        # if we don't have two DEMs, showing the false hillshade doesn't work.
        if not pts:
            dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, primaryDEM, this_secondary)
            # if np.logical_or(np.sum(np.isfinite(xdata.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100):
            if np.logical_or.reduce((np.sum(np.isfinite(xdata.flatten())) < 100,
                                     np.sum(np.isfinite(ydata.flatten())) < 100,
                                     np.sum(np.isfinite(sdata.flatten())) < 100)):
                print("Exiting: Fewer than 100 data points")
                if return_var:
                    pp.close()
                    return primaryDEM, secondaryDEM, -1
                else:
                    pp.close()
                    return -1
            if mycount == 1:
                dH0 = np.copy(dH.img)
                dH0mean = np.nanmean(ydata)
                ydata -= dH0mean
                dH.img -= dH0mean
                mytitle = "DEM difference: pre-coregistration (dz0={:+.2f})".format(dH0mean)
            else:
                mytitle = "DEM difference: After Iteration {}".format(mycount - 1)
            false_hillshade(dH, mytitle, pp)
            dH_img = dH.img

        else:
            dH, xdata, ydata, sdata = preprocess(stable_mask, slope_geo, aspect_geo, primaryDEM, this_secondary)
            dH_img = dH
            # if np.logical_or(np.sum(np.isfinite(dH.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100):
            if np.logical_or.reduce((np.sum(np.isfinite(xdata.flatten())) < 100,
                                     np.sum(np.isfinite(ydata.flatten())) < 100,
                                     np.sum(np.isfinite(sdata.flatten())) < 100)):
                print("Exiting: Not enough data points")
                if return_var:
                    pp.close()
                    return primaryDEM, secondaryDEM, -1
                else:
                    pp.close()
                    return -1
            if mycount == 1:
                dH0 = dH
                dH0mean = np.nanmean(ydata)
                ydata -= dH0mean
                dH_img -= dH0mean

        # calculate threshold, standard deviation of dH
        # mythresh = 100 * (mystd-np.nanstd(dH_img))/mystd
        # mystd = np.nanstd(dH_img)
        # USE RMSE instead ( this is to make su that there is improvement in the spread)
        mythresh = 100 * (mystd - RMSE(dH_img)) / mystd
        mystd = RMSE(dH_img)

        mytitle2 = "Co-registration: Iteration {}".format(mycount)
        dx, dy, dz = coreg_fitting(xdata, ydata, sdata, mytitle2, pp)
        if mycount == 1:
            dz += dH0mean
        tot_dx += dx
        tot_dy += dy
        tot_dz += dz
        magnthresh = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        print(tot_dx, tot_dy, tot_dz)
        print(tot_dx, tot_dy, tot_dz, file=paramf)
        # print np.nanmean(secondarys[-1].img)

        # print secondarys[-1].xmin, secondarys[-1].ymin

        # shift most recent secondary DEM
        this_secondary.shift(dx, dy)  # shift in x,y
        # print tot_dx, tot_dy
        # no idea why secondarys[-1].img += dz doesn't work, but the below seems to.
        zupdate = np.ma.array(this_secondary.img.data + dz, mask=this_secondary.img.mask)  # shift in z
        this_secondary = this_secondary.copy(new_raster=zupdate)
        if pts:
            this_secondary.mask(stable_mask.img)
            slope_geo.shift(dx, dy)
            aspect_geo.shift(dx, dy)
            stable_mask.shift(dx, dy)
        else:
            this_secondary = this_secondary.reproject(primaryDEM)
            this_secondary.mask(stable_mask)

        print("Percent-improvement threshold: " + str(mythresh))
        print("Magnitude threshold: " + str(magnthresh))

        # secondarys[-1].display()
        if mythresh > 2 and magnthresh > magnlimit:
            dH = None
            dx = None
            dy = None
            dz = None
            xdata = None
            ydata = None
            sdata = None
        else:
            if not pts:
                dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, primaryDEM, this_secondary)
                mytitle = "DEM difference: After Iteration {}".format(mycount)
                # adjust final dH
                # myfadj=np.nanmean([np.nanmean(dH.img),np.nanmedian(dH.img)])
                # myfadj=np.nanmedian(dH.img)
                # tot_dz += myfadj
                # dH.img = dH.img-myfadj

                false_hillshade(dH, mytitle, pp)
                dHfinal = dH.img
            else:
                mytitle2 = "Co-registration: FINAL"
                dH, xdata, ydata, sdata = preprocess(stable_mask, slope_geo, aspect_geo, primaryDEM, this_secondary)
                dx, dy, dz = coreg_fitting(xdata, ydata, sdata, mytitle2, pp)
                dHfinal = dH
        plt.close('all')
    # Create final histograms pre and post coregistration
    # shift = [tot_dx, tot_dy, tot_dz]  # commented because it wasn't actually used.
    stats_final, stats_init = final_histogram(dH0, dHfinal, pp=pp)
    print("MEAN, MEDIAN, STD, RMSE, COUNT", file=statsf)
    print(stats_init, file=statsf)
    print(stats_final, file=statsf)

    # create new raster with dH sample used for co-registration as the band
    # dH.write(outdir + os.path.sep + 'dHpost.tif') # have to fill these in!
    # save full dH output
    # dHfinal.write('dHpost.tif', out_folder=outdir)
    # save adjusted secondary dem
    if sfilename is not None:
        secondaryoutfile = '.'.join(sfilename.split('.')[0:-1]) + '_adj.tif'
    else:
        secondaryoutfile = 'secondary_adj.tif'

    if pts:
        outsecondary = secondaryDEM.copy()
    else:
        if full_ext:
            outsecondary = get_geoimg(secondaryDEM)
            # outsecondary = secondaryDEM.copy()
        else:
            outsecondary = secondaryDEM.reproject(primaryDEM)
    # outsecondary = this_secondary.copy()
    # outsecondary.unmask()
    outsecondary.shift(tot_dx, tot_dy)
    outsecondary.img = outsecondary.img + tot_dz

    # if not pts and not full_ext:
    #    outsecondary = outsecondary.reproject(primaryDEM)
    if not inmem:
        outsecondary.write(secondaryoutfile, out_folder=outdir)

    if pts:
        if not inmem:
            slope_geo.write('tmp_slope.tif', out_folder=outdir)
            aspect_geo.write('tmp_aspect.tif', out_folder=outdir)

    # Final Check --- for debug
    if not pts:
        print("FinalCHECK")
        # outsecondary = outsecondary.reproject(primaryDEM)
        primaryDEM = orig_primaryDEM.reproject(outsecondary)

        dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, primaryDEM, outsecondary)
        false_hillshade(dH, 'FINAL CHECK', pp)
        # dx, dy, dz = coreg_fitting(xdata, ydata, sdata, "Final Check", pp)

        if mfilename is not None:
            mastoutfile = '.'.join(mfilename.split('.')[0:-1]) + '_adj.tif'
        else:
            mastoutfile = 'primary_adj.tif'

        if full_ext:
            primaryDEM = orig_primaryDEM
            outsecondary = outsecondary.reproject(primaryDEM)
        primaryDEM.write(mastoutfile, out_folder=outdir)
    if not inmem:
        outsecondary.write(secondaryoutfile, out_folder=outdir)
    pp.close()
    print("Fin.")
    print("Fin.", file=paramf)
    paramf.close()
    statsf.close()
    plt.close("all")

    out_offs = [tot_dx, tot_dy, tot_dz]

    gc.collect()  # try releasing memory!
    if return_var:
        return primaryDEM, outsecondary, out_offs, stats_final
