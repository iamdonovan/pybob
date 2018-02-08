from __future__ import print_function
import os
import subprocess
import gdal
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as optimize
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile


def false_hillshade(dH, title, pp):
    niceext = np.array([dH.xmin, dH.xmax, dH.ymin, dH.ymax])/1000.
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    im1 = ax.imshow(dH.img, extent=niceext)
    im1.set_clim(-20, 20)
    im1.set_cmap('Greys')
    fig.suptitle(title, fontsize=14)
    numwid = max([len('{:.1f} m'.format(np.nanmean(dH.img))),
                  len('{:.1f} m'.format(np.nanmedian(dH.img))), len('{:.1f} m'.format(np.nanstd(dH.img)))])
    plt.annotate('MEAN:'.ljust(8) + ('{:.1f} m'.format(np.nanmean(dH.img))).rjust(numwid), xy=(0.65, 0.89),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.annotate('MEDIAN:'.ljust(8) + ('{:.1f} m'.format(np.nanmedian(dH.img))).rjust(numwid),
                 xy=(0.65, 0.80), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 color='red', family='monospace')
    plt.annotate('STD:'.ljust(8) + ('{:.1f} m'.format(np.nanstd(dH.img))).rjust(numwid), xy=(0.65, 0.71),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    plt.tight_layout()
    pp.savefig(fig)
    return


def create_stable_mask(img, mask1, mask2):
    # if we have no masks, just return an array of true values
    if mask1 is None and mask2 is None:
        return np.ones(img.img.shape) == 0  # all false, so nothing will get masked.
    elif mask1 is not None and mask2 is None:  # we have a glacier mask, not land
        mask = create_mask_from_shapefile(img, mask1)
        return mask  # returns true where there's glacier, false everywhere else
    elif mask1 is None and mask2 is not None:
        mask = create_mask_from_shapefile(img, mask2)
        return np.logical_not(mask)  # false where there's land, true where there isn't
    else:  # if none of the above, we have two masks.
        gmask = create_mask_from_shapefile(img, mask1)
        lmask = create_mask_from_shapefile(img, mask2)
        return np.logical_or(gmask, np.logical_not(lmask))  # true where there's glacier or water


def preprocess(stable_mask, slope, aspect, master, slave):
    stan = np.tan(np.radians(slope)).astype(np.float32)
    # create a new raster with all the same properties as the masterDEM
    # but replace the raster data with the dH data.
    dH = master.copy(new_raster=(master.img-slave.img).astype(np.float32))
    dH.img[stable_mask] = np.nan

    dHtan = dH.img / stan
    mykeep = ((np.absolute(dH.img) < 60.0) & np.isfinite(dH.img) & (slope > 7.0) & (dH.img != 0.0) & (aspect >= 0))
    dH.img[np.invert(mykeep)] = np.nan
    xdata = aspect[mykeep]
    ydata = dHtan[mykeep]
    sdata = stan[mykeep]

    return dH, xdata, ydata, sdata


def coreg_fitting(xdata, ydata, sdata, title, pp):
    xdata = xdata.astype(np.float64)  # float64 truly necessary?
    ydata = ydata.astype(np.float64)
    sdata = sdata.astype(np.float64)
    # fit using equation 3 of Nuth and Kaeaeb, 2011

    def fitfun(p, x): return p[0] * np.cos(np.radians(p[1] - x)) + p[2]

    def errfun(p, x, y): return fitfun(p, x) - y

    p0 = [-1, 1, -1]
    p1, success = optimize.leastsq(errfun, p0[:], args=(xdata, ydata))
    # print success
    # print p1
    # convert to shift parameters in cartesian coordinates
    xadj = p1[0] * np.sin(np.radians(p1[1]))
    yadj = p1[0] * np.cos(np.radians(p1[1]))
    zadj = p1[2] * sdata.mean(axis=0)

    xp = np.linspace(0, 360, 361)
    yp = fitfun(p1, xp)

    if xdata.size > 50000:
        mysamp = np.random.randint(0, xdata.size, 50000)
    else:
        mysamp = np.arange(0, xdata.size)

    fig = plt.figure(figsize=(7, 5), dpi=600)
    fig.suptitle(title, fontsize=14)
    plt.plot(xdata[mysamp], ydata[mysamp], '.', ms=0.5, color='0.5', rasterized=True)
    plt.plot(xp, np.zeros(xp.size), 'k', ms=3)
    plt.plot(xp, yp, 'r-', ms=2)

    plt.axis([0, 360, -200, 200])
    plt.xlabel('Aspect [degrees]')
    plt.ylabel('dH / tan(slope)')
    numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
    plt.text(20, -125, '$\Delta$x: ' + ('{:.1f} m'.format(xadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.text(20, -150, '$\Delta$y: ' + ('{:.1f} m'.format(yadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.text(20, -175, '$\Delta$z: ' + ('{:.1f} m'.format(zadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace')
    pp.savefig(fig, rasterized=True)

    return xadj, yadj, zadj


def get_geoimg(indata):
    if type(indata) is str or type(indata) is gdal.Dataset:
        return GeoImg(indata)
    elif type(indata) is GeoImg:
        return indata
    else:
        raise TypeError('input data must be a string pointing to a gdal dataset, or a GeoImg object.')


def dem_coregistration(masterDEM, slaveDEM, glaciermask=None, landmask=None, outdir='.'):
    # if the output directory does not exist, create it.
    subprocess.call(["mkdir", "-p", outdir])
    outdir = os.path.abspath(outdir)
    # make a file to save the coregistration parameters to.
    paramf = open(outdir + os.path.sep + 'coreg_params.txt', 'w')
    # create the output pdf
    pp = PdfPages(outdir + os.path.sep + 'CoRegistration_Results.pdf')

    if type(masterDEM) is str:
        mfilename = os.path.basename(masterDEM)
    else:
        mfilename = masterDEM.filename

    if type(slaveDEM) is str:
        sfilename = os.path.basename(slaveDEM)
    else:
        sfilename = slaveDEM.filename

    masterDEM = get_geoimg(masterDEM)
    slaveDEM = get_geoimg(slaveDEM)

    masterDEM = masterDEM.reproject(slaveDEM)  # need to resample masterDEM to cell size of slave.

    # get the mask set up
    stable_mask = create_stable_mask(masterDEM, glaciermask, landmask)

    # masterDEM.write('tmp_master.tif', out_folder=outdir)
    # masterDEM.mask(stable_mask)

    # create slope, aspect rasters
    # print('calling gdal within python!')
    slope_ = gdal.DEMProcessing('', masterDEM.gd, 'slope', format='MEM')
    slope_geo = GeoImg(slope_)
    slope_geo.write('tmp_slope.tif', out_folder=outdir)
    slope_geo = slope_geo.reproject(slaveDEM)
    slope = slope_geo.img

    aspect_ = gdal.DEMProcessing('', masterDEM.gd, 'aspect', format='MEM')
    aspect_geo = GeoImg(aspect_)
    aspect_geo.write('tmp_aspect.tif', out_folder=outdir)
    aspect_geo = aspect_geo.reproject(slaveDEM)
    aspect = aspect_geo.img

    masterDEM.mask(stable_mask)

    # while loop!
    mythresh = np.float64(100)  # float64 really necessary?
    mystd = np.float64(100)
    mycount = 0
    tot_dx = np.float64(0)
    tot_dy = np.float64(0)
    tot_dz = np.float64(0)
    magnthresh = 100
    mytitle = 'DEM difference: pre-coregistration'
    slaves = []
    slaves.append(slaveDEM.reproject(masterDEM))
    slaves[-1].mask(stable_mask)

    while mythresh > 2 and magnthresh > 1:
        if mycount != 0:
            slaves.append(slaves[-1].reproject(masterDEM))
            slaves[-1].mask(stable_mask)
            mytitle = "DEM difference: After Iteration {}".format(mycount)
        mycount += 1
        print("Running iteration #{}".format(mycount))
        print("Running iteration #{}".format(mycount), file=paramf)
        dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, masterDEM, slaves[-1])
        false_hillshade(dH, mytitle, pp)

        mythresh = 100 * (mystd-np.nanstd(dH.img))/mystd
        mystd = np.nanstd(dH.img)

        mytitle2 = "Co-registration: Iteration {}".format(mycount)
        dx, dy, dz = coreg_fitting(xdata, ydata, sdata, mytitle2, pp)
        tot_dx += dx
        tot_dy += dy
        tot_dz += dz
        magnthresh = np.sqrt(np.square(dx)+np.square(dy)+np.square(dz))
        print(tot_dx, tot_dy, tot_dz)
        print(tot_dx, tot_dy, tot_dz, file=paramf)
        # print np.nanmean(slaves[-1].img)

        # print slaves[-1].xmin, slaves[-1].ymin

        # shift most recent slave DEM
        slaves[-1].shift(dx, dy)  # shift in x,y
        # print tot_dx, tot_dy
        # no idea why slaves[-1].img += dz doesn't work, but the below seems to.
        zupdate = np.ma.array(slaves[-1].img.data + dz, mask=slaves[-1].img.mask)  # shift in z
        slaves[-1] = slaves[-1].copy(new_raster=zupdate)

        # print np.nanmean(slaves[-1].img)

        slaves[-1] = slaves[-1].reproject(masterDEM)
        # slaves[-1].display()
        if mythresh > 2 and magnthresh > 1:
            dH = None
            dx = None
            dy = None
            dz = None
            xdata = None
            ydata = None
            sdata = None
        else:
            dHfinal = dH
            mytitle = "DEM difference: After Iteration {}".format(mycount)
            false_hillshade(dH, mytitle, pp)
            # slaves.pop(-1)

    # create new raster with dH sample used for co-registration as the band
    # dHSample = dH.copy(new_raster=dHpost_sample)
    # dHSample.write(outdir + os.path.sep + 'dHpost_sample.tif') # have to fill these in!
    # save full dH output
    dHfinal.write('dHpost.tif', out_folder=outdir)
    # save adjusted slave dem
    if sfilename is not None:
        slaveoutfile = '.'.join(sfilename.split('.')[0:-1]) + '_adj.tif'
    else:
        slaveoutfile = 'slave_adj.tif'
    outslave = slaveDEM.reproject(masterDEM)
    outslave.shift(tot_dx, tot_dy)
    outslave.img = outslave.img + tot_dz
    outslave.write(slaveoutfile, out_folder=outdir)

    if mfilename is not None:
        mastoutfile = '.'.join(mfilename.split('.')[0:-1]) + '_adj.tif'
    else:
        mastoutfile = 'master_adj.tif'
    masterDEM.write(mastoutfile, out_folder=outdir)

    pp.close()
    print("Fin.")
    print("Fin.", file=paramf)
    paramf.close()

    return masterDEM, slaves[-1]
