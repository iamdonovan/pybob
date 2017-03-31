import numpy as np
from scipy.interpolate import griddata, interp1d
from numpy.polynomial.polynomial import polyval, polyfit
import pybob.coreg_tools as ct
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


def elevation_mean(dDEM, **kwargs):
    if type(dDEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')

    if 'DEM' not in kwargs:
        raise ValueError('to use elevation_mean, you must supply a base DEM')
    else:  # if we've been supplied a DEM, we have to check that it's the right size.
        DEM = kwargs['DEM']
        if type(DEM) is not GeoImg:
            raise TypeError('DEM must be a GeoImg')
        if DEM.img.shape != dDEM.img.shape:
            raise ValueError('dDEM and DEM must have the same shape')

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

    if 'glacier_mask' in kwargs:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
        dem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
    else:
        ddem_data = dDEM.img[np.isfinite(dDEM.img)]
        dem_data = DEM.img[np.isfinite(dDEM.img)]

    binned_dH = bin_data(bins, ddem_data, dem_data, mode='mean')

    # now, we interpolate the missing DH values to the binned values.
    f_elev = interp1d(bins, binned_dH)
    # pull out missing values
    hole_inds = np.where(np.isnan(dDEM.img))
    hole_elevs = DEM.img[hole_inds]

    filled = f_elev(hole_elevs)
    dDEM.img[hole_inds] = filled

    return dDEM


def elevation_median(dDEM, **kwargs):
    if type(dDEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if 'DEM' not in kwargs:
        raise ValueError('to use elevation_mean, you must supply a base DEM')
    else:  # if we've been supplied a DEM, we have to check that it's the right size.
        DEM = kwargs['DEM']
        if DEM.img.shape != dDEM.img.shape:
            raise ValueError('dDEM and DEM must have the same shape')

    if 'bins' not in kwargs:
        # if we haven't been handed bins, we have to make our own.
        if 'bin_width' not in kwargs:
            bin_width = 50  # if we haven't been told what bin width to use, default to 50.
        else:
            bin_width = kwargs['bin_width']
        bins = np.arange(np.nanmin(DEM.img), np.nanmax(DEM.img)+1, bin_width)
    else:
        bins = kwargs['bins']

    if 'glacier_mask' in kwargs:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
        dem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
    else:
        ddem_data = dDEM.img[np.isfinite(dDEM.img)]
        dem_data = DEM.img[np.isfinite(dDEM.img)]

    binned_dH = bin_data(bins, ddem_data, dem_data, mode='median')

    # now, we interpolate the missing DH values to the binned values.
    f_elev = interp1d(bins, binned_dH)
    # pull out missing values from dDEM
    hole_inds = np.where(np.isnan(dDEM.img))
    hole_elevs = DEM.img[hole_inds]

    filled = f_elev(hole_elevs)
    dDEM.img[hole_inds] = filled

    return dDEM


def elevation_poly(dDEM, **kwargs):
    if type(dDEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if 'poly_order' in kwargs:
        poly_order = kwargs['poly_order']
    else:
        raise ValueError('poly_order must be defined to use ddem_tools.elevation_poly()')
    if 'DEM' not in kwargs:
        raise ValueError('to use elevation_mean, you must supply a base DEM')
    else:  # if we've been supplied a DEM, we have to check that it's the right size.
        DEM = kwargs['DEM']
        if DEM.img.shape != dDEM.img.shape:
            raise ValueError('dDEM and DEM must have the same shape')

    if 'glacier_mask' in kwargs:
        ddem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
        dem_data = dDEM.img[np.logical_and(np.isfinite(dDEM.img), kwargs['glacier_mask'])]
    else:
        ddem_data = dDEM.img[np.isfinite(dDEM.img)]
        dem_data = DEM.img[np.isfinite(dDEM.img)]

    # now we fit a polynomial of order poly_order to the data:
    pfit = polyfit(dem_data, ddem_data, poly_order)

    # need a way to put hole values in the right place
    hole_inds = np.where(np.isnan(dDEM.img))
    dem_holes = DEM.img[hole_inds]
    interp_points = polyval(dem_holes, pfit)

    dDEM.img[dem_holes] = interp_points
    return dDEM


def neighborhood_average(DEM, **kwargs):
    if type(DEM) is not GeoImg:
        raise TypeError('DEM must be a GeoImg')
    if 'neighborhood_size' in kwargs:
        pass
    else:
        raise ValueError('neighborhood_size must be defined to use ddem_tools.neighborhood_poly()')


# kriging is like linear, can be used with a dDEM or DEM.
def kriging(DEM, **kwargs):
    pass


# normalize glacier elevations given a shapefile of glacier elevations
def normalize_glacier_elevations():
    pass


# now we actually sum up the ddem values that fall within each glacier
# input from a shapefile
def calculate_volume_change(dDEM, glacier_shapes):
    if type(glacier_shapes) is str:
        masksource = ogr.Open(shapefile)
        masklayer = masksource.GetLayer()
    else:
        pass
    pass
