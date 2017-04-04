# import multiprocessing as mp
import numpy as np
from scipy.interpolate import griddata, interp1d, Rbf, RectBivariateSpline as RBS
from numpy.polynomial.polynomial import polyval, polyfit
import pybob.coreg_tools as ct
import pybob.image_tools as it
from pybob.bob_tools import bin_data
from pybob.GeoImg import GeoImg


def difference(dem1, dem2, glaciermask=None, landmask=None, outdir='.'):
    master, slave = ct.dem_coregistration(dem1, dem2, glaciermask, landmask, outdir)
    master.unmask()
    slave.unmask()

    return master.copy(new_raster=(master.img - slave.img))


def fill_holes(dDEM, method, **kwargs):
    if type(dDEM) is not GeoImg:
        raise TypeError('dDEM must be a GeoImg')
    else:
        return method(dDEM, **kwargs)


# the following (linear, elevation_mean, elevation_median, elevation_poly, neighborhood_average)
# are all methods to pass to fill_holes. E.g., filled_DEM = fill_holes(DEM, linear)
# will use linear interpolation to fill any holes in the DEM (or GeoImg raster).
# linear and neighborhood_average can be used on either a DEM or a dDEM, but elevation_* requires
# both a DEM and a dDEM to work correctly.
def linear(DEM, **kwargs):
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    X, Y = DEM.xy()

    interp_points = DEM.img[np.isfinite(DEM.img)]
    interpX = X[np.isfinite(DEM.img)]
    interpY = Y[np.isfinite(DEM.img)]

    filled_DEM = griddata((interpX, interpY), interp_points, (X, Y))
    return DEM.copy(new_raster=filled_DEM)


def elevation_function(dDEM, **kwargs):
    if 'functype' not in kwargs:
        raise ValueError('Have to specify functype to use elevation_function.')

    return parse_elev_func_args(kwargs['functype'], dDEM, **kwargs)


def parse_elev_func_args(func_name, dDEM, **kwargs):
    # first, make sure we use one of the right function names
    if func_name not in ['mean', 'median', 'poly']:
        raise ValueError('{} not one of mean, median, poly'.format(func_name))

    if 'DEM' not in kwargs:
        raise ValueError('to use {}, you must supply a base DEM'.format(func_name))
    else:  # if we've been supplied a DEM, we have to check that it's the right size.
        DEM = kwargs['DEM']
        if type(DEM) is not GeoImg:
            raise TypeError('DEM must be a GeoImg')
        if DEM.img.shape != dDEM.img.shape:
            raise ValueError('dDEM and DEM must have the same shape')

    if 'glacier_mask' in kwargs:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
        dem_data = DEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
    else:
        ddem_data = dDEM.img[np.isfinite(dDEM.img)]
        dem_data = DEM.img[np.isfinite(dDEM.img)]

    # now, figure out which of mean, median, polyfit we're using, and do it.
    if func_name == 'poly':
        if 'poly_order' not in kwargs:
            raise ValueError('poly_order must be defined to use polynomial fitting')
        # now we fit a polynomial of order poly_order to the data:
        pfit = polyfit(dem_data, ddem_data, kwargs['poly_order'])

        # need a way to put hole values in the right place
        hole_inds = np.where(np.isnan(dDEM.img))
        dem_holes = DEM.img[hole_inds]
        interp_points = polyval(dem_holes, pfit)

        dDEM.img[dem_holes] = interp_points
        return dDEM
    elif func_name == 'mean' or func_name == 'median':
        if 'bins' not in kwargs:
            # if we haven't been handed bins, we have to make our own.
            if 'bin_width' not in kwargs:
                bin_width = 50  # if we haven't been told what bin width to use, default to 50.
            else:
                bin_width = kwargs['bin_width']
            min_el = DEM.img.min() - (DEM.img.min() % bin_width)
            max_el = DEM.img.max() + (bin_width - (DEM.img.max() % bin_width))
            bins = np.arange(min_el, max_el, bin_width)
        else:
            bins = kwargs['bins']
        binned_dH = bin_data(bins, ddem_data, dem_data, mode=func_name)
        # now, we interpolate the missing DH values to the binned values.
        f_elev = interp1d(bins, binned_dH)
        # pull out missing values from dDEM
        hole_inds = np.where(np.isnan(dDEM.img))
        hole_elevs = DEM.img[hole_inds]

        filled = f_elev(hole_elevs)
        dDEM.img[hole_inds] = filled

        return dDEM
    else:
        raise ValueError('Somehow we made it this far without naming a correct fitting function. Oops.')


def neighborhood_average(dDEM, **kwargs):
    if type(dDEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if 'neighborhood_size' not in kwargs:
        raise ValueError('neighborhood_size must be defined to use ddem_tools.neighborhood_poly()')

    nradius = int(kwargs['neighborhood_size'] / dDEM.dx)

    tmp_ddem = dDEM.copy().img
    out_ddem = dDEM.copy()

    missing = np.where(np.isnan(tmp_ddem))
    missing_inds = zip(missing[0], missing[1])

    tmp_ddem[np.logical_not(kwargs['glacier_mask'])] = np.nan

    for ind in missing_inds:
        nmask = neighborhood_mask(ind, nradius, tmp_ddem)
        out_ddem.img[ind] = np.nanmean(tmp_ddem[nmask])

    return out_ddem


# kriging is like linear, can be used with a dDEM or DEM.
def kriging(DEM, **kwargs):
    pass


# RBF is simliar to linear
def radial_basis(DEM, **kwargs):
    rbfunclist = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']

    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if 'rbfunc' not in kwargs:
        # default to linear radial basis function
        rbfunc = 'linear'
    elif kwargs['rbfunc'] not in rbfunclist:
        raise ValueError('unknown radial basis function {} requested. \
                          Check scipy.interpolate.Rbf docs for details'.format(kwargs['rbfunc']))
    else:
        rbfunc = kwargs['rbfunc']

    X, Y = DEM.xy()
    vals = ~np.isnan(DEM.img)
    f = Rbf(X[vals], Y[vals], DEM.img[vals], function=rbfunc)
    out_dem = f(X, Y)

    return DEM.copy(new_raster=out_dem)


# RBS
def RectBivariateSpline(DEM, **kwargs):
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    X, Y = DEM.xy()
    f = RBS(Y[0:, 0], X[0, 0:], DEM.img)
    out_dem = f.ev(Y, X)  # might have to flip upside-down?

    return DEM.copy(new_raster=out_dem)


# some helper functions
def neighborhood_mask(centerind, radius, array):
    a, b = centerind
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx-a, -b:ny-b]
    mask = x*x + y*y <= radius*radius

    return mask


# interpolate on a glacier-by-glacier basis, rather than on the DEM as a whole.
def fill_holes_individually(dDEM, glacshapes, functype, burn_handle=None, **kwargs):
    # first, get the individual glacier mask.
    ind_glac_mask, ind_glac_vals = it.rasterize_polygons(dDEM, glacshapes, burn_handle)

    filled_ddem = dDEM.copy()

    # if we already have a glacier mask defined, remove it.
    if 'glacier_mask' in kwargs:
        del kwargs['glacier_mask']

    for glac in ind_glac_vals:
        tmp_mask = ind_glac_mask == glac
        tmp_dem = fill_holes(dDEM, elevation_function, glacier_mask=tmp_mask, functype=functype, **kwargs)

        filled_ddem.img[tmp_mask] = tmp_dem.img[tmp_mask]
    return filled_ddem


# normalize glacier elevations given a shapefile of glacier elevations
def normalize_glacier_elevations(DEM, glacshapes, burn_handle=None):
    ind_glac_mask, ind_glac_vals = it.rasterize_polygons(DEM, glacshapes, burn_handle)
    normed_els = DEM.copy().img

    for i in ind_glac_vals:
        glac_inds = np.where(ind_glac_mask == i)
        glac_els = DEM.img[glac_inds]
        if glac_els.size > 0:
            max_el = np.nanmax(glac_els)
            min_el = np.nanmin(glac_els)
            normed_els[glac_inds] = (glac_els - min_el) / (max_el - min_el)

    normDEM = DEM.copy(new_raster=normed_els)

    return normDEM, ind_glac_mask, ind_glac_vals


# now we actually sum up the ddem values that fall within each glacier
# input from a shapefile
def calculate_volume_changes(dDEM, glacier_shapes, burn_handle=None):
    ind_glac_mask, ind_glac_vals = it.rasterize_polygons(dDEM, glacier_shapes, burn_handle=burn_handle)
    ind_vol_chgs = np.zeros(ind_glac_vals.shape)

    for i, ind in ind_glac_vals:
        glac_inds = np.where(ind_glac_mask == ind)
        glac_chgs = dDEM.img[glac_inds]
        # get the volume change by summing dh/dt, multiplying by cell
        ind_vol_chgs[i] = np.nansum(glac_chgs) * np.abs(dDEM.dx) * np.abs(dDEM.dy)

    return ind_glac_vals, ind_vol_chgs


# create an area-altitude distribution for a DEM and a glacier shapefile
def area_alt_dist(DEM, glacier_shapes, glacier_inds=None, bin_width=50):
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if glacier_inds is None:
        dem_data = DEM.img[glacier_shapes]

        min_el = DEM.img.min() - (DEM.img.min() % bin_width)
        max_el = DEM.img.max() + (bin_width - (DEM.img.max() % bin_width))
        bins = np.arange(min_el, max_el, bin_width)
        aads, _ = np.histogram(dem_data, bins=bins)
        aads = aads * np.abs(DEM.dx) * np.abs(DEM.dy)
    else:
        bins = []
        aads = []

        for i in glacier_inds:
            dem_data = DEM.img[glacier_shapes == i]
            min_el = dem_data.min() - (dem_data.min() % bin_width)
            max_el = dem_data.max() + (bin_width - (dem_data.max() % bin_width))
            thisbin = np.arange(min_el, max_el, bin_width)
            thisaad, _ = np.histogram(dem_data, bins=thisbin)

            bins.append(thisbin)
            aads.append(thisaad * np.abs(DEM.dx) * np.abs(DEM.dy))  # make it an area!

    return bins, aads
