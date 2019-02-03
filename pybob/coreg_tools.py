#from __future__ import print_function
import os
import errno
import gdal
import numpy as np
import matplotlib.pylab as plt
#plt.switch_backend('agg')
import scipy.optimize as optimize
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pybob.GeoImg import GeoImg
from pybob.ICESat import ICESat
from pybob.image_tools import create_mask_from_shapefile
from pybob.plot_tools import plot_shaded_dem


def get_slope(geoimg):
    slope_ = gdal.DEMProcessing('', geoimg.gd, 'slope', format='MEM')
    return GeoImg(slope_)


def get_aspect(geoimg):
    aspect_ = gdal.DEMProcessing('', geoimg.gd, 'aspect', format='MEM')
    return GeoImg(aspect_)


def false_hillshade(dH, title, pp):
    niceext = np.array([dH.xmin, dH.xmax, dH.ymin, dH.ymax])/1000.
    mykeep = np.logical_and.reduce((np.isfinite(dH.img), (np.abs(dH.img) < np.nanstd(dH.img) * 3)))
    dH_vec = dH.img[mykeep]

    fig = plt.figure(figsize=(7, 5), dpi=300)
    ax = plt.gca()

    im1 = ax.imshow(dH.img, extent=niceext)
    im1.set_clim(-20, 20)
    im1.set_cmap('Greys')
#    if np.sum(np.isfinite(dH_vec))<10:
#        print("Error for statistics in false_hillshade")
#    else: 
    plt.title(title, fontsize=14)

    numwid = max([len('{:.1f} m'.format(np.mean(dH_vec))),
                  len('{:.1f} m'.format(np.median(dH_vec))), len('{:.1f} m'.format(np.std(dH_vec)))])
    plt.annotate('MEAN:'.ljust(8) + ('{:.1f} m'.format(np.mean(dH_vec))).rjust(numwid), xy=(0.65, 0.95),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.annotate('MEDIAN:'.ljust(8) + ('{:.1f} m'.format(np.median(dH_vec))).rjust(numwid),
                 xy=(0.65, 0.90), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 color='red', family='monospace')
    plt.annotate('STD:'.ljust(8) + ('{:.1f} m'.format(np.std(dH_vec))).rjust(numwid), xy=(0.65, 0.85),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    #plt.colorbar(im1)

    plt.tight_layout()
    pp.savefig(fig, dpi=200)
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

    if isinstance(master, GeoImg):
        stan = np.tan(np.radians(slope)).astype(np.float32)
        dH = master.copy(new_raster=(master.img-slave.img))
        dH.img[stable_mask] = np.nan
        master_mask = isinstance(master.img, np.ma.masked_array)
        slave_mask = isinstance(slave.img, np.ma.masked_array)

        if master_mask and slave_mask:
            dH.mask(np.logical_or(master.img.mask, slave.img.mask))
        elif master_mask:
            dH.mask(master.img.mask)
        elif slave_mask:
            dH.mask(slave.img.mask)
        
        if dH.isfloat:
            dH.img[stable_mask] = np.nan

        dHtan = dH.img / stan
        mykeep = ((np.absolute(dH.img) < 200.0) & np.isfinite(dH.img) &
                  (slope > 3.0) & (dH.img != 0.0) & (aspect >= 0))
        dH.img[np.invert(mykeep)] = np.nan
        xdata = aspect[mykeep]
#        ydata = dHtan[mykeep]
        ydata = dH.img[mykeep]
        sdata = stan[mykeep]

    elif isinstance(master, ICESat):
        slave_pts = slave.raster_points(master.xy)
        dH = master.elev - slave_pts
        
        slope_pts = slope.raster_points(master.xy)
        stan = np.tan(np.radians(slope_pts))
        
        aspect_pts = aspect.raster_points(master.xy)
        smask = stable_mask.raster_points(master.xy) > 0
        
        dH[smask] = np.nan

        dHtan = dH / stan

        mykeep = ((np.absolute(dH) < 200.0) & np.isfinite(dH) &
                  (slope_pts > 3.0) & (dH != 0.0) & (aspect_pts >= 0))
        
        dH[np.invert(mykeep)] = np.nan
        xdata = aspect_pts[mykeep]
        ydata = dHtan[mykeep]
        sdata = stan[mykeep]


    return dH, xdata, ydata, sdata


def coreg_fitting(xdata, ydata, sdata, title, pp):
    xdata = xdata.astype(np.float64)  # float64 truly necessary?
    ydata = ydata.astype(np.float64)
    sdata = sdata.astype(np.float64)
    ydata2 = np.divide(ydata,sdata)
    # fit using equation 3 of Nuth and Kaeaeb, 2011

    def fitfun(p, x, s): return p[0] * np.cos(np.radians(p[1] - x)) * s + p[2]

    def errfun(p, x, s, y): return fitfun(p, x, s) - y
    
    if xdata.size > 20000:
        mysamp = np.random.randint(0, xdata.size, 20000)
    else:
        mysamp = np.arange(0, xdata.size)
    mysamp=mysamp.astype(np.int64)
    
    
    #embed()
    #print("soft_l1")
    lb = [-200, 0, -300]
    ub = [200, 180, 300]
    p0 = [1, 1, -1]
    #p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]), method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
    #myresults = optimize.least_squares(errfun, p0, args=(xdata, ydata), method='trf', loss='soft_l1', f_scale=0.5)    
    myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], sdata[mysamp], ydata[mysamp]), method='trf', loss='soft_l1', f_scale=0.1,ftol=1E-4,xtol=1E-4)    
    #myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], ydata[mysamp]), method='trf', bounds=([lb,ub]), loss='soft_l1', f_scale=0.1,ftol=1E-8,xtol=1E-8)    
    p1 = myresults.x
    # success = myresults.success # commented because it wasn't actually being used.
    # print success
    # print p1
    # convert to shift parameters in cartesian coordinates
    xadj = p1[0] * np.sin(np.radians(p1[1]))
    yadj = p1[0] * np.cos(np.radians(p1[1]))
    zadj = p1[2] #* sdata.mean(axis=0)

    xp = np.linspace(0, 360, 361)
    sp = np.zeros(xp.size) + 0.785398
    yp = fitfun(p1, xp, sp)    
    
    fig = plt.figure(figsize=(7, 5), dpi=300)
    #fig.suptitle(title, fontsize=14)
    plt.title(title, fontsize=14)
    plt.plot(xdata[mysamp], ydata2[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
    plt.plot(xp, np.zeros(xp.size), 'k', ms=3)
    plt.plot(xp, np.divide(yp,np.tan(sp)), 'r-', ms=2)

    plt.xlim(0, 360)
    #plt.ylim(-200,200)
    ymin, ymax = plt.ylim((np.nanmean(ydata2[mysamp]))-2*np.nanstd(ydata2[mysamp]),
                          (np.nanmean(ydata2[mysamp]))+2*np.nanstd(ydata2[mysamp]))
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
    pp.savefig(fig, dpi=200)

    return xadj, yadj, zadj

def final_histogram(dH0, dHfinal, pp):
    fig = plt.figure(figsize=(7, 5), dpi=200)
    plt.title('Elevation difference histograms', fontsize=14)
    
    dH0 = np.squeeze(np.asarray(dH0[ np.logical_and.reduce((np.isfinite(dH0), (np.abs(dH0) < np.nanstd(dH0) * 3)))]))
    dHfinal = np.squeeze(np.asarray(dHfinal[ np.logical_and.reduce((np.isfinite(dHfinal), (np.abs(dHfinal) < np.nanstd(dHfinal) * 3)))]))
    
    dH0=dH0[np.isfinite(dH0)]
    dHfinal=dHfinal[np.isfinite(dHfinal)]
    
    j1, j2 = np.histogram(dH0, bins=100, range=(-60, 60))
    k1, k2 = np.histogram(dHfinal, bins=100, range=(-60, 60))
    
    stats0 = [np.mean(dH0), np.median(dH0), np.std(dH0), RMSE(dH0), np.sum(np.isfinite(dH0))]
    stats_fin = [np.mean(dHfinal), np.median(dHfinal), np.std(dHfinal), RMSE(dHfinal), np.sum(np.isfinite(dHfinal))]    
    
    plt.plot(j2[1:], j1,'k-', linewidth=2)
    plt.plot(k2[1:], k1,'r-', linewidth=2)
    #plt.legend(['Original', 'Coregistered'])
    
    plt.xlabel('Elevation difference [meters]')
    plt.ylabel('Number of samples')
    plt.xlim(-50,50)
    
    #numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
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
    pp.savefig(fig, dpi=200)
    
    return stats_fin, stats0
    
def RMSE(indata): 
    """ Return root mean square of indata."""
    myrmse = np.sqrt(np.nanmean(np.asarray(indata)**2))
    return myrmse
    
    
def get_geoimg(indata):
    if type(indata) is str or type(indata) is gdal.Dataset:
        return GeoImg(indata)
    elif type(indata) is GeoImg:
        return indata
    else:
        raise TypeError('input data must be a string pointing to a gdal dataset, or a GeoImg object.')


def dem_coregistration(masterDEM, slaveDEM, glaciermask=None, landmask=None, outdir='.', pts=False, full_ext=False, return_var=True):
    """
    Iteratively co-register elevation data, based on routines described in Nuth and Kaeaeb, 2011.

    Parameters
    ----------
    masterDEM : string or GeoImg
        Path to filename or GeoImg dataset representing "master" DEM.
    slaveDEM : string or GeoImg
        Path to filename or GeoImg dataset representing "slave" DEM.
    glaciermask : string, optional
        Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    landmask : string, optional
        Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    outdir : string, optional
        Location to save co-registration outputs.
    pts : bool, optional
        If True, program assumes that masterDEM represents point data (i.e., ICESat),
        as opposed to raster data. Slope/aspect are then calculated from slaveDEM.
        masterDEM should be a string representing an HDF5 file continaing ICESat data.
    full_ext : bool, optional
        If True, program writes full extents of input DEMs. If False, program writes
        input DEMs cropped to their common extent. Default is False.
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

    if type(masterDEM) is str:
        mfilename = os.path.basename(masterDEM)
        mfiledir = os.path.dirname(masterDEM)
        if mfiledir == '':
            mfiledir = os.path.abspath('.')
    else:
        mfilename = masterDEM.filename
        mfiledir = masterDEM.in_dir_abs_path

    if type(slaveDEM) is str:
        sfilename = os.path.basename(slaveDEM)
        sfiledir = os.path.dirname(slaveDEM)
        if sfiledir == '':
            sfiledir = os.path.abspath('.')
    else:
        sfilename = slaveDEM.filename
        sfiledir = slaveDEM.in_dir_abs_path

    slaveDEM = get_geoimg(slaveDEM)
    # we assume that we are working with 'area' pixels (i.e., pixel x,y corresponds to corner)
    if slaveDEM.is_point():
        slaveDEM.to_area()

    # if we're dealing with ICESat/pt data, change how we load masterDEM data
    if pts:
        masterDEM = ICESat(masterDEM)
        masterDEM.project('epsg:{}'.format(slaveDEM.epsg))
        mybounds=[slaveDEM.xmin, slaveDEM.xmax, slaveDEM.ymin, slaveDEM.ymax]
        masterDEM.clip(mybounds)
        masterDEM.clean()
        slope_geo = get_slope(slaveDEM)
        aspect_geo = get_aspect(slaveDEM)

        slope_geo.write('tmp_slope.tif', out_folder=outdir)
        aspect_geo.write('tmp_aspect.tif', out_folder=outdir)

        smask = create_stable_mask(slaveDEM, glaciermask, landmask)
        slaveDEM.mask(smask)
        stable_mask = slaveDEM.copy(new_raster=smask)  # make the mask a geoimg
        
        ### Create initial plot of where stable terrain is, including ICESat pts
        fig1 = plot_shaded_dem(slaveDEM)
        plt.plot(masterDEM.x[~np.isnan(masterDEM.elev)], masterDEM.y[~np.isnan(masterDEM.elev)], 'k.')
        pp.savefig(fig1, bbox_inches='tight', dpi=200)

    else:
        orig_masterDEM = get_geoimg(masterDEM)
        if orig_masterDEM.is_point():
            orig_masterDEM.to_area()
        masterDEM = orig_masterDEM.reproject(slaveDEM)  # need to resample masterDEM to cell size, extent of slave.
        stable_mask = create_stable_mask(masterDEM, glaciermask, landmask)

        slope_geo = get_slope(masterDEM)
        aspect_geo = get_aspect(masterDEM)
        slope_geo.write('tmp_slope.tif', out_folder=outdir)
        aspect_geo.write('tmp_aspect.tif', out_folder=outdir)
        masterDEM.mask(stable_mask)

    slope = slope_geo.img
    aspect = aspect_geo.img
    
    if np.sum(np.greater(slope.flatten(),3))<500:
        return
    
    mythresh = np.float64(200)  # float64 really necessary?
    mystd = np.float64(200)
    mycount = 0
    tot_dx = np.float64(0)
    tot_dy = np.float64(0)
    tot_dz = np.float64(0)
    magnthresh = 200
    magnlimit=1
    mytitle = 'DEM difference: pre-coregistration'
    if pts:
        this_slave = slaveDEM
        this_slave.mask(stable_mask.img)
    else:
        this_slave = slaveDEM.reproject(masterDEM)
        this_slave.mask(stable_mask)

    while mythresh > 2 and magnthresh > magnlimit:
        if mycount != 0:
            # slaves.append(slaves[-1].reproject(masterDEM))
            # slaves[-1].mask(stable_mask)
            mytitle = "DEM difference: After Iteration {}".format(mycount)
        mycount += 1
        print("Running iteration #{}".format(mycount))
        print("Running iteration #{}".format(mycount), file=paramf)
        
        # if we don't have two DEMs, showing the false hillshade doesn't work.
        if not pts:
            dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, masterDEM, this_slave)
            #if np.logical_or(np.sum(np.isfinite(xdata.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100):
            if np.logical_or.reduce((np.sum(np.isfinite(xdata.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100, np.sum(np.isfinite(sdata.flatten()))<100)):
                print("Exiting: Less than 100 data points")
                return
            false_hillshade(dH, mytitle, pp)
            dH_img = dH.img

        else:
            dH, xdata, ydata, sdata = preprocess(stable_mask, slope_geo, aspect_geo, masterDEM, this_slave)
            dH_img = dH
            #if np.logical_or(np.sum(np.isfinite(dH.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100):
            if np.logical_or.reduce((np.sum(np.isfinite(xdata.flatten()))<100, np.sum(np.isfinite(ydata.flatten()))<100, np.sum(np.isfinite(sdata.flatten()))<100)):
                print("Exiting: Not enough data points")
                return 
        
        
        if mycount == 1:
            dH0 = dH_img

        # calculate threshold, standard deviation of dH
        #mythresh = 100 * (mystd-np.nanstd(dH_img))/mystd
        #mystd = np.nanstd(dH_img)
        # USE RMSE instead ( this is to make su that there is improvement in the spread)
        mythresh = 100 * (mystd-RMSE(dH_img))/mystd
        mystd = RMSE(dH_img)

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
        this_slave.shift(dx, dy)  # shift in x,y
        # print tot_dx, tot_dy
        # no idea why slaves[-1].img += dz doesn't work, but the below seems to.
        zupdate = np.ma.array(this_slave.img.data + dz, mask=this_slave.img.mask)  # shift in z
        this_slave = this_slave.copy(new_raster=zupdate)
        if pts:
            this_slave.mask(stable_mask.img)
            slope_geo.shift(dx, dy)
            aspect_geo.shift(dx, dy)
            stable_mask.shift(dx, dy)
        else:
            this_slave = this_slave.reproject(masterDEM)
            this_slave.mask(stable_mask)

        print("Percent-improvement threshold and Magnitute threshold:")
        print(mythresh, magnthresh)
        
        # slaves[-1].display()
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
                dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, masterDEM, this_slave)
                mytitle = "DEM difference: After Iteration {}".format(mycount)
                #adjust final dH
                #myfadj=np.nanmean([np.nanmean(dH.img),np.nanmedian(dH.img)])
                #myfadj=np.nanmedian(dH.img)
                #tot_dz += myfadj
                #dH.img = dH.img-myfadj
                
                false_hillshade(dH, mytitle, pp)
                dHfinal = dH.img
            else:
                mytitle2 = "Co-registration: FINAL"
                dH, xdata, ydata, sdata = preprocess(stable_mask, slope_geo, aspect_geo, masterDEM, this_slave)
                dx, dy, dz = coreg_fitting(xdata, ydata, sdata, mytitle2, pp)
                dHfinal = dH
 
    # Create final histograms pre and post coregistration
    # shift = [tot_dx, tot_dy, tot_dz]  # commented because it wasn't actually used.
    stats_final, stats_init = final_histogram(dH0, dHfinal, pp)
    print("MEAN, MEDIAN, STD, RMSE, COUNT", file=statsf)
    print(stats_init, file=statsf)
    print(stats_final, file=statsf)

    # create new raster with dH sample used for co-registration as the band
    # dH.write(outdir + os.path.sep + 'dHpost.tif') # have to fill these in!
    # save full dH output
    # dHfinal.write('dHpost.tif', out_folder=outdir)
    # save adjusted slave dem
    if sfilename is not None:
        slaveoutfile = '.'.join(sfilename.split('.')[0:-1]) + '_adj.tif'
    else:
        slaveoutfile = 'slave_adj.tif'
    
    #if pts:
    #    outslave = slaveDEM.copy()
    #else:
    #    if full_ext:
    #        #outslave = get_geoimg(slaveDEM)
    #        outslave = slaveDEM.copy()
    #    else:
    #        outslave = slaveDEM.reproject(masterDEM)
    outslave = this_slave.copy()
    outslave.unmask()
    #outslave.shift(tot_dx, tot_dy)
    #outslave.img = outslave.img + tot_dz
    if not pts and not full_ext:
        outslave = outslave.reproject(masterDEM)
    outslave.write(slaveoutfile, out_folder=outdir)
    if pts:
        slope_geo.write('tmp_slope.tif', out_folder=outdir)
        aspect_geo.write('tmp_aspect.tif', out_folder=outdir)

    # Final Check --- for debug
    if not pts:
        print("FinalCHECK")
       #outslave = outslave.reproject(masterDEM)
        masterDEM = orig_masterDEM.reproject(outslave)
        dH, xdata, ydata, sdata = preprocess(stable_mask, slope, aspect, masterDEM, outslave)
        false_hillshade(dH, 'FINAL CHECK', pp)
        #dx, dy, dz = coreg_fitting(xdata, ydata, sdata, "Final Check", pp)

        if mfilename is not None:
            mastoutfile = '.'.join(mfilename.split('.')[0:-1]) + '_adj.tif'
        else:
            mastoutfile = 'master_adj.tif'

        if full_ext:
            masterDEM = orig_masterDEM
        masterDEM.write(mastoutfile, out_folder=outdir)

    pp.close()
    print("Fin.")
    print("Fin.", file=paramf)
    paramf.close()
    statsf.close()
    plt.close("all")

    out_offs = [tot_dx, tot_dy, tot_dz]

    if return_var:
        return masterDEM, outslave, out_offs
