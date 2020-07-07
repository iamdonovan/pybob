"""
pybob.ddem_tools provides a number of tools for working with DEM differencing and calculating volume changes. Primarily
    designed for glaciers, but could be useful for calculating other volume changes as well.
"""
from __future__ import print_function
# import multiprocessing as mp
import numpy as np
import datetime as dt
from scipy.interpolate import griddata, interp1d, Rbf, RectBivariateSpline as RBS
from numpy.polynomial.polynomial import polyval, polyfit
import pybob.coreg_tools as ct
import pybob.image_tools as it
from pybob.bob_tools import bin_data
from pybob.GeoImg import GeoImg


sensor_names = ['AST', 'SETSM', 'Map', 'SPOT', 'SDMI', 'HMA']


def difference(dem1, dem2, glaciermask=None, landmask=None, outdir='.'):
    primary, secondary, _ = ct.dem_coregistration(dem1, dem2, glaciermask, landmask, outdir)
    primary.unmask()
    secondary.unmask()

    return primary.copy(new_raster=(primary.img - secondary.img))


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
        f_elev = interp1d(bins, binned_dH, fill_value="extrapolate")
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
    """
    Calculate an Area-Altitude Distribution for a glacier outline(s), given an input DEM.

    :param DEM: input DEM.
    :param glacier_shapes: mask representing glacier outlines. Can be boolean or integer depending on whether one
        or several AADs should be calculated.
    :param glacier_inds: array representing glacier indices in glacier_shapes. If unspecified, only one AAD is returned.
    :param bin_width: width of elevation bands to calculate area distribution in. If unspecified, result will be
        the minimum of 50m or 10% of the elevation range.

    :type DEM: pybob.GeoImg
    :type glacier_shapes: array-like
    :type glacier_inds: array-like
    :type bin_width: float

    :returns bins, aads: array, or list of arrays, representing elevation bands and glacier area (in DEM horizontal
        units) per elevation band.
    """
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if glacier_inds is None:
        dem_data = DEM.img[glacier_shapes]

        bins = get_bins(dem_data, bin_width=bin_width)
        min_el = bins[0]
        max_el = bins[-1]

        aads, _ = np.histogram(dem_data, bins=bins, range=(min_el, max_el))
        aads = aads * np.abs(DEM.dx) * np.abs(DEM.dy)
        bins = bins[:-1]  # remove the last element, because it's actually above the glacier range.
    else:
        bins = []
        aads = []

        for i in glacier_inds:
            dem_data = DEM.img[glacier_shapes == i]
            thisbin = get_bins(dem_data, bin_width=bin_width)

            min_el = thisbin[0]
            max_el = thisbin[-1]

            thisaad, _ = np.histogram(dem_data, bins=thisbin, range=(min_el, max_el))

            bins.append(thisbin[:-1])  # remove last element.
            aads.append(thisaad * np.abs(DEM.dx) * np.abs(DEM.dy))  # make it an area!

    return bins, aads


def nmad(data, nfact=1.4826):
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.

    :param data: input data
    :param nfact: normalization factor for the data; default is 1.4826

    :type data: array-like
    :type nfact: float
    :returns nmad: (normalized) median absolute deviation of data.
    """
    m = np.nanmedian(data)
    return nfact * np.nanmedian(np.abs(data - m))


def get_bins(DEM, glacier_mask=None, bin_width=None):
    """
    Get elevation bins for a DEM, given an optional glacier mask and an optional width. If unspecified,
        bin_width is calculated as the minimum of 50 units, or 10% of the DEM (or glacier, if mask provided) elevation
        range. Bin values represent the lower bound of the elevation band, and are rounded to be a multiple of the bin
        width.

    :param DEM: The DEM to get elevation bins for.
    :param glacier_mask: mask representing glacier outline, or region of interest.
    :param bin_width: width of bins to calculate.

    :type DEM: array-like
    :type glacier_mask: array-like
    :type bin_width: float

    :returns bins: the elevation bins.
    """
    if glacier_mask is not None:
        dem_data = DEM[np.logical_and(np.isfinite(DEM), glacier_mask)]
    else:
        dem_data = DEM[np.isfinite(DEM)]

    zmax = np.nanmax(dem_data)
    zmin = np.nanmin(dem_data)

    zrange = zmax - zmin

    if bin_width is None:
        bin_width = min(50, int(zrange / 10))

    min_el = zmin - (zmin % bin_width)
    max_el = zmax + (bin_width - (zmax % bin_width))
    bins = np.arange(min_el, max_el+1, bin_width)

    return bins


def get_elev_curve(DEM, dDEM, glacier_mask=None, bins=None, mode='mean', outlier=False, fill=False, poly_order=3):
    """
    Get a dh(z) curve for a glacier/region of interest, given a DEM and a difference DEM (dDEM). Available modes are
        'mean'/'median', calculating the mean(median) of each elevation bin, or poly, fitting a polynomial (default
        third-order) to the means of each elevation bin.

    :param DEM: DEM to determine z in dh(z)
    :param dDEM: difference DEM to determine dh in dh(z)
    :param glacier_mask: mask representing glacier outline
    :param bins: values representing the lower edge of elevation bins
    :param mode: how to determine the dh(z) relationship
    :param outlier: filter outliers using an iterative 3-sigma filter
    :param fill: fill missing bins using a polynomial fit (default third order)
    :param poly_order: order for any polynomial fitting

    :type DEM: array-like
    :type dDEM: array-like
    :type glacier_mask: array-like
    :type bins: array-like
    :type mode: str
    :type outlier: bool
    :type fill: bool
    :type poly_order: int

    :returns bins, curve, bin_areas: elevation bins, dh(z) curve, and number of pixels per elevation bin.
    """
    assert mode in ['mean', 'median', 'poly'], "mode not recognized: {}".format(mode)
    if glacier_mask is None:
        valid = np.logical_and(np.isfinite(DEM), np.isfinite(dDEM))
    else:
        valid = np.logical_and(glacier_mask, np.logical_and(np.isfinite(DEM), np.isfinite(dDEM)))

    dem_data = DEM[valid]
    ddem_data = dDEM[valid]

    if bins is None:
        bins = get_bins(DEM, glacier_mask)

    if outlier:
        # ddem_data = outlier_removal(bins, dem_data, ddem_data)
        ddem_data = nmad_outlier_removal(bins, dem_data, ddem_data)

    if mode in ['mean', 'median']:
        curve, bin_areas = bin_data(bins, ddem_data, dem_data, mode=mode, nbinned=True)
        if fill:
            _bins = bins[np.isfinite(curve)]
            _curve = bins[np.isfinite(curve)]
            p = polyfit(bins, curve, poly_order)
            fill_ = polyval(bins, p)
            curve[np.isnan(curve)] = fill_[np.isnan(curve)]

    else:
        _mean, bin_areas = bin_data(bins, ddem_data, dem_data, mode='mean', nbinned=True)
        p = polyfit(bins, _mean, poly_order)
        curve = polyval(bins, p)

    return bins, curve, bin_areas


def nmad_outlier_removal(bins, DEM, dDEM, nfact=3):
    new_ddem = np.zeros(dDEM.size)
    digitized = np.digitize(DEM, bins)
    for i, _ in enumerate(bins):
        this_bindata = dDEM[digitized == i]
        this_nmad = nmad(this_bindata)
        this_bindata[np.abs(this_bindata) > nfact * this_nmad] = np.nan
        new_ddem[digitized == i] = this_bindata
    return new_ddem


def outlier_removal(bins, DEM, dDEM, nsig=3):
    """
    Iteratively remove outliers in an elevation bin using a 3-sigma filter.

    :param bins: lower bound of elevation bins to use
    :param DEM: DEM to determine grouping for outlier values
    :param dDEM: elevation differences to filter outliers from
    :param nsig: number of standard deviations before a value is considered an outlier.

    :type bins: array-like
    :type DEM: array-like
    :type dDEM: array-like
    :type nsig: float

    :returns new_ddem: ddem with outliers removed (set to NaN)
    """
    new_ddem = np.zeros(dDEM.size)
    digitized = np.digitize(DEM, bins)
    for i, _ in enumerate(bins):
        this_bindata = dDEM[digitized == i]
        nout = 1
        old_mean = np.nanmean(this_bindata)
        old_std = np.nanstd(this_bindata)
        while nout > 1:
            thresh_up = old_mean + nsig * old_std
            thresh_dn = old_mean - nsig * old_std

            isout = np.logical_or(this_bindata > thresh_up, this_bindata < thresh_dn)
            nout = np.count_nonzero(isout)
            this_bindata[isout] = np.nan

            old_mean = np.nanmean(this_bindata)
            old_std = np.nanstd(this_bindata)
        new_ddem[digitized == i] = this_bindata
    return new_ddem


def calculate_dV_map(dDEM, ind_glac_mask, ind_glac_vals):
    """
    Calculate glacier volume changes from a dDEM using a map of glacier outlines.

    :param dDEM: difference DEM to get elevation changes from
    :param ind_glac_mask: glacier mask with different values for each glacier
    :param ind_glac_vals: list of unique index values in glacier mask for which to calculate volume changes.

    :type dDEM: pybob.GeoImg
    :type ind_glac_mask: array-like
    :type ind_glac_vals: array-like
    :returns ind_glac_vals, ind_vol_chgs: index and volume changes for the input indices.
    """
    ind_glac_vals = np.array(ind_glac_vals)
    ind_vol_chgs = np.nan * np.zeros(ind_glac_vals.shape)

    for i, ind in enumerate(ind_glac_vals):
        glac_inds = np.where(ind_glac_mask == ind)
        if glac_inds[0].size == 0:
            continue
        glac_chgs = dDEM.img[glac_inds]
        # get the volume change by summing dh/dt, multiplying by cell
        ind_vol_chgs[i] = np.nansum(glac_chgs) * np.abs(dDEM.dx) * np.abs(dDEM.dy)

    return ind_glac_vals, ind_vol_chgs


def calculate_dV_curve(aad, dh_curve):
    """
    Given a dh(z) curve and AAD values, calculate volume change.

    :param aad: Area-Altitude Distribution/Hypsometry with which to calculate volume change.
    :param dh_curve: dh(z) curve to use.

    :type aad: np.array
    :type dh_curve: np.array

    :returns dV: glacier volume change given input curves.
    """
    assert aad.size == dh_curve.size, "AAD and dh(z) curve must have same size."

    return np.nansum(aad * dh_curve)


def name_parser(fname):
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
    elif splitname[0] == 'HMA':
        datestr = splitname[1]
    else:
        raise ValueError("Could not parse {} based on sensor name. {} is not one of \
                          AST, SETSM, SDMI, SPOT, SRTM, Map, HMA.".format(fname, fname))
    yy = int(datestr[0:4])
    mm = int(datestr[4:6])
    dd = int(datestr[6:])

    return dt.date(yy, mm, dd)


def nice_split(fname):
    """
    Given a filename of the form dH_DEM1_DEM2, return DEM1, DEM2.

    :param fname: filename to split
    :type fname: str

    :returns name1, name2: DEM names parsed from input filename.
    """
    sname = fname.rsplit('.tif', 1)[0].split('_')
    sname.remove('dH')
    splitloc = [i+1 for i, x in enumerate(sname[1:]) if x in sensor_names][0]

    name1 = '_'.join(sname[0:splitloc])
    name2 = '_'.join(sname[splitloc:])
    return name1, name2
