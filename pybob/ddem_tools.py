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

    if 'valid_area' in kwargs:
        interp_mask = np.logical_and(kwargs['valid_area'], np.isfinite(DEM.img))
    else:
        interp_mask = np.isfinite(DEM.img)
    interp_points = DEM.img[interp_mask]
    interpX = X[interp_mask]
    interpY = Y[interp_mask]

    filled_DEM = griddata((interpX, interpY), interp_points, (X, Y))
    if 'valid_area' in kwargs:
        filled_DEM[np.logical_not(kwargs['valid_area'])] = np.nan
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
        ddem_data = dDEM.img[np.logical_and(np.logical_and(np.isfinite(dDEM.img),
                                            np.isfinite(DEM.img)), kwargs['glacier_mask'])]
        dem_data = DEM.img[np.logical_and(np.logical_and(np.isfinite(dDEM.img),
                                          np.isfinite(DEM.img)), kwargs['glacier_mask'])]
    else:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img))]
        dem_data = DEM.img[np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img))]

    # now, figure out which of mean, median, polyfit we're using, and do it.
    if func_name == 'poly':
        if 'poly_order' not in kwargs:
            raise ValueError('poly_order must be defined to use polynomial fitting')
        # now we fit a polynomial of order poly_order to the data:
        pfit = polyfit(dem_data, ddem_data, kwargs['poly_order'])

        # need a way to put hole values in the right place
        if 'glacier_mask' in kwargs:
            hole_inds = np.where(np.logical_and(np.logical_and(np.isnan(dDEM.img),
                                 np.isfinite(DEM.img)), kwargs['glacier_mask']))
        else:
            hole_inds = np.where(np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img)))

        hole_elevs = DEM.img[hole_inds]
        interp_points = polyval(hole_elevs, pfit)

        dDEM.img[hole_inds] = interp_points

        if 'valid_area' in kwargs:
            dDEM.img[np.logical_not(kwargs['valid_area'])] = np.nan

        return dDEM
    elif func_name == 'mean' or func_name == 'median':
        if 'bins' not in kwargs:
            # if we haven't been handed bins, we have to make our own.
            if 'bin_width' not in kwargs:
                bin_width = 50  # if we haven't been told what bin width to use, default to 50.
            else:
                bin_width = kwargs['bin_width']
            min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % bin_width)
            max_el = np.nanmax(dem_data) + (bin_width - (np.nanmax(dem_data) % bin_width))
            bins = np.arange(min_el, max_el+1, bin_width)
        else:
            bins = kwargs['bins']
        binned_dH = bin_data(bins, ddem_data, dem_data, mode=func_name)
        # now, we interpolate the missing DH values to the binned values.
        f_elev = interp1d(bins, binned_dH)
        # pull out missing values from dDEM
        if 'glacier_mask' in kwargs:
            hole_inds = np.where(np.logical_and(np.logical_and(np.isnan(dDEM.img),
                                 np.isfinite(DEM.img)), kwargs['glacier_mask']))
        else:
            hole_inds = np.where(np.logical_and(np.isfinite(dDEM.img), np.isfinite(DEM.img)))
        hole_elevs = DEM.img[hole_inds]

        filled = f_elev(hole_elevs)
        tmp_dDEM = dDEM.copy()
        tmp_dDEM.img[hole_inds] = filled

        if 'valid_area' in kwargs:
            tmp_dDEM.img[np.logical_not(kwargs['valid_area'])] = np.nan
        return tmp_dDEM
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

    if 'valid_area' in kwargs:
        if 'glacier_mask' in kwargs:
            valid_mask = np.logical_and(kwargs['valid_area'], kwargs['glacier_mask'])
        else:
            valid_mask = kwargs['valid_area']
        missing_mask = np.logical_and(valid_mask, np.isnan(tmp_ddem))
    else:
        missing_mask = np.isnan(tmp_ddem)

    missing = np.where(missing_mask)
    missing_inds = zip(missing[0], missing[1])

    if 'glacier_mask' in kwargs:
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


def circular_kernel(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    centerind = (radius, radius)
    mask = neighborhood_mask(centerind, radius, kernel)
    kernel[mask] = 1
    return kernel


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
        try:
            tmp_dem = fill_holes(dDEM, elevation_function, glacier_mask=tmp_mask, functype=functype, **kwargs)
            filled_ddem.img[tmp_mask] = tmp_dem.img[tmp_mask]
        except:
            continue
    return filled_ddem


# normalize glacier elevations given a shapefile of glacier elevations
def normalize_glacier_elevations(DEM, glacshapes, burn_handle=None):
    ind_glac_mask, raw_inds = it.rasterize_polygons(DEM, glacshapes, burn_handle)
    normed_els = DEM.copy().img

    ind_glac_vals = clean_glacier_indices(DEM, ind_glac_mask, raw_inds)

    for i in ind_glac_vals:
        glac_inds = np.where(ind_glac_mask == i)
        glac_els = DEM.img[glac_inds]
        if glac_els.size > 0:
            max_el = np.nanmax(glac_els)
            min_el = np.nanmin(glac_els)
            normed_els[glac_inds] = (glac_els - min_el) / (max_el - min_el)

    normDEM = DEM.copy(new_raster=normed_els)

    return normDEM, ind_glac_mask, ind_glac_vals


def clean_glacier_indices(geoimg, glacier_mask, raw_inds):
    clean_glaciers = []
    for glac in raw_inds:
        imask = glacier_mask == glac
        is_here = len(np.where(imask)[0]) > 0
        is_finite = len(np.where(np.isfinite(geoimg.img[imask]))[0]) > 0
        if is_here and is_finite:
            clean_glaciers.append(glac)
    return clean_glaciers


# now we actually sum up the ddem values that fall within each glacier
# input from a shapefile
def calculate_volume_changes(dDEM, glacier_shapes, burn_handle=None, ind_glac_vals=None):
    if type(glacier_shapes) is str:
        ind_glac_mask, ind_glac_vals = it.rasterize_polygons(dDEM, glacier_shapes, burn_handle=burn_handle)
    elif type(glacier_shapes) is np.array:
        if ind_glac_vals is None:
            ind_glac_vals = np.unique(glacier_shapes)
        elif type(ind_glac_vals) is list:
            ind_glac_vals = np.array(ind_glac_vals)
        ind_glac_mask = glacier_shapes

    ind_vol_chgs = np.nan * np.zeros(ind_glac_vals.shape)

    for i, ind in enumerate(ind_glac_vals):
        glac_inds = np.where(ind_glac_mask == ind)
        if glac_inds[0].size == 0:
            continue
        glac_chgs = dDEM.img[glac_inds]
        # get the volume change by summing dh/dt, multiplying by cell
        ind_vol_chgs[i] = np.nansum(glac_chgs) * np.abs(dDEM.dx) * np.abs(dDEM.dy)

    return ind_glac_vals, ind_vol_chgs


# create an area-altitude distribution for a DEM and a glacier shapefile
def area_alt_dist(DEM, glacier_shapes, glacier_inds=None, bin_width=None):
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if glacier_inds is None:
        dem_data = DEM.img[glacier_shapes]

        min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % bin_width)
        max_el = np.nanmax(dem_data) + (bin_width - (np.nanmax(dem_data) % bin_width))
        bins = np.arange(min_el, max_el+1, bin_width)
        aads, _ = np.histogram(dem_data, bins=bins, range=(min_el, max_el))
        aads = aads * np.abs(DEM.dx) * np.abs(DEM.dy)
    else:
        bins = []
        aads = []

        for i in glacier_inds:
            dem_data = DEM.img[glacier_shapes == i]
            if bin_width is None:
                z_range = np.nanmax(dem_data) - np.nanmin(dem_data)
                this_bin_width = min(50, int(z_range / 10))
            else:
                this_bin_width = bin_width
            min_el = np.nanmin(dem_data) - (np.nanmin(dem_data) % this_bin_width)
            max_el = np.nanmax(dem_data) + (this_bin_width - (np.nanmax(dem_data) % this_bin_width))
            thisbin = np.arange(min_el, max_el+1, this_bin_width)
            thisaad, _ = np.histogram(dem_data, bins=thisbin, range=(min_el, max_el))

            bins.append(thisbin)
            aads.append(thisaad * np.abs(DEM.dx) * np.abs(DEM.dy))  # make it an area!

    return bins, aads


