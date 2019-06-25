"""
pybob.bob_tools is a collection of the tools that didn't really fit other places.
"""
from __future__ import print_function
import os
import random
import errno
import datetime as dt
from osgeo import ogr, osr
import numpy as np
from shapely.geometry import Point


def mkdir_p(out_dir):
    """
    Add bash mkdir -p functionality to os.makedirs.

    :param out_dir: directory to create.
    """
    try:
        os.makedirs(out_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise


def standard_landsat(instring):
    """
    Given a string of a landsat scenename, make a standard (pre-Collection) filename, of the form LSSPPPRRRYYYYDDDXXX01.
    """
    strsplit = instring.split('_')
    if len(strsplit) < 3:  # if we only have 1 (or fewer) underscores, it's already fine
        return strsplit[0]
    else:
        outstring = 'L'
        sensor = strsplit[0][1]
        # figure out what sensor we're using, and generate the appropriate string
        if sensor == '5':
            outstring += 'T5'
        elif sensor == '4':
            outstring += 'T4'
        elif sensor == '7':
            outstring += 'E7'
        elif sensor == '8':  # chances are, we won't ever see this one.
            outstring += 'C8'
        outstring += strsplit[0][2:]  # this is the path/row information
        # now it gets fun: getting the date right.
        year = strsplit[1][3:7]
        month = strsplit[1][7:9]
        day = strsplit[1][9:]
        # make sure we have 3 digits here
        doy = str(dt.datetime(int(year), int(month), int(day)).timetuple().tm_yday).zfill(3)
        outstring += year + doy
        # now, spoof the last bits so it's got the right size:
        outstring += 'XXX01'
        return outstring


def doy2mmdd(year, doy, string_out=True, outform='%Y/%m/%d'):
    """
    Return a string or a datetime object given a year and a day of year.

    :param year: Year of the input date, e.g., ``2018``.
    :param doy: Day of the year of the input date, e.g., ``1`` for 1 Jan.
    :param string_out: Return a string (True) or a datetime object (False). Default is True.
    :param outform: Format for string to return. Default is `%Y/%m/%d`, or 2018/01/01 for
        1 January 2018.

    :type year: int
    :type doy: int
    :type string_out: bool
    :type outform: str

    :returns date: datetime.datetime or string representation of the input date.

    >>> bt.doy2mmdd(2018, 1, string_out=True)
    `2018/01/01`

    >>> bt.doy2mmdd(2018, 1, string_out=False)
    datetime.datetime(2018, 1, 1, 0, 0)
    """
    datestruct = dt.datetime(year, 1, 1) + dt.timedelta(doy-1)
    if string_out:
        return datestruct.strftime(outform)
    else:
        return datestruct


def mmdd2doy(year, month, day, string_out=True):
    """
    Return a string or an int representing a day of year, given a date.

    :param year: Year of the input date, e.g., ``2018``.
    :param mm: Month of the year of the input date, e.g., ``1`` for Jan.
    :param dd: Day of the month of the input date.
    :param string_out: Return a string (True) or an int (False). Default is True.
    :type year: int
    :type mm: int
    :type dd: int
    :type string_out: bool
    :returns doy: day of year representation of year, month, day

    >>> bt.mmdd2doy(2018, 1, 1, string_out=True)
    `1`
    """
    doy = dt.datetime(year, month, day).timetuple().tm_yday
    if string_out:
        return str(doy)
    else:
        return doy


def mmdd2dec(year, month, day):
    yearstart = dt.datetime(year, 1, 1).toordinal()
    nextyearstart = dt.datetime(year+1, 1, 1).toordinal()
    datestruct = dt.datetime(year, month, day)
    datefrac = float(datestruct.toordinal() + 0.5 - yearstart) / (nextyearstart - yearstart)
    return year + datefrac


def doy2dec(year, doy):
    pass


def dec2mmdd(decdate):
    year = int(np.floor(decdate))
    datefrac = decdate - year
    
    yearstart = dt.datetime(year, 1, 1).toordinal()
    nextyearstart = dt.datetime(year+1, 1, 1).toordinal()
    
    days = np.round((nextyearstart - yearstart) * datefrac - 0.5)
    return dt.datetime(year, 1, 1) + dt.timedelta(days)


def dec2doy(decdate):
    year = int(np.floor(decdate))
    datefrac = decdate - year
    
    yearstart = dt.datetime(year, 1, 1).toordinal()
    nextyearstart = dt.datetime(year+1, 1, 1).toordinal()
    
    days = np.round((nextyearstart - yearstart) * datefrac - 0.5)
    return year, days+1

    pass


def parse_lsat_scene(scenename, string_out=True):
    sensor = scenename[0:3]
    path = int(scenename[3:6])
    row = int(scenename[6:9])
    year = int(scenename[9:13])
    doy = int(scenename[13:16])

    if string_out:
        return sensor, str(path), str(row), str(year), str(doy)
    else:
        return sensor, path, row, year, doy


def bin_data(bins, data2bin, bindata, mode='mean', nbinned=False):
    """
    Place data into bins based on a secondary dataset, and calculate statistics on them.

    :param bins: array-like structure indicating the bins into which data should be placed.
    :param data2bin: data that should be binned.
    :param bindata: secondary dataset that decides how data2bin should be binned. Should have same size/shape
        as data2bin.
    :param mode: How to calculate statistics of binned data. One of 'mean', 'median', 'std', 'max', or 'min'.
    :param nbinned: Return a second array, nbinned, with number of data points that fit into each bin.
        Default is False.
    :type bins: array-like
    :type data2bin: array-like
    :type bindata: array-like
    :type mode: str
    :type nbinned: bool

    :returns binned, nbinned: calculated, binned data with same size as bins input. If nbinned is True, returns a second
        array with the number of inputs for each bin.
    """
    assert mode in ['mean', 'median', 'std', 'max', 'min'], "mode not recognized: {}".format(mode)
    digitized = np.digitize(bindata, bins)
    binned = np.zeros(len(bins)) * np.nan
    if nbinned:  
        numbinned = np.zeros(len(bins))

    if mode == 'mean':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmean(data2bin[np.logical_and(np.isfinite(bindata), digitized == i+1)])
            if nbinned:
                numbinned[i] = np.count_nonzero(np.logical_and(np.isfinite(data2bin), digitized == i+1))
    elif mode == 'median':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmedian(data2bin[np.logical_and(np.isfinite(bindata), digitized == i+1)])
            if nbinned:
                numbinned[i] = np.count_nonzero(np.logical_and(np.isfinite(data2bin), digitized == i+1))
    elif mode == 'std':
        for i, _ in enumerate(bins):
            binned[i] = np.nanstd(data2bin[np.logical_and(np.isfinite(bindata), digitized == i+1)])
            if nbinned:
                numbinned[i] = np.count_nonzero(np.logical_and(np.isfinite(data2bin), digitized == i+1))
    elif mode == 'max':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmax(data2bin[np.logical_and(np.isfinite(bindata), digitized == i+1)])
            if nbinned:
                numbinned[i] = np.count_nonzero(np.logical_and(np.isfinite(data2bin), digitized == i+1))
    elif mode == 'min':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmin(data2bin[np.logical_and(np.isfinite(bindata), digitized == i+1)])
            if nbinned:
                numbinned[i] = np.count_nonzero(np.logical_and(np.isfinite(data2bin), digitized == i+1))
    else:
        raise ValueError('mode must be mean, median, std, max, or min')
    
    if nbinned:
        return np.array(binned), np.array(numbinned)
    else:
        return np.array(binned)


def random_points_in_polygon(poly, npts=1):
    xmin, ymin, xmax, ymax = poly.bounds
    rand_points = []

    for pt in range(npts):
        thisptin = False
        while not thisptin:
            rand_point = Point(xmin + (random.random() * (xmax-xmin)), ymin + (random.random() * (ymax-ymin)))
            thisptin = poly.contains(rand_point)
        rand_points.append(rand_point)

    return rand_points


def reproject_layer(inLayer, targetSRS):
    srcSRS = inLayer.GetSpatialRef()
    coordTrans = osr.CoordinateTransformation(srcSRS, targetSRS)

    outLayer = ogr.GetDriverByName('Memory').CreateDataSource('').CreateLayer(inLayer.GetName(),
                                                                              geom_type=inLayer.GetGeomType())
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    outLayerDefn = outLayer.GetLayerDefn()
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        geom = inFeature.GetGeometryRef()
        geom.Transform(coordTrans)
        outFeature = ogr.Feature(outLayerDefn)
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))

        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    return outLayer


def round_down(num, divisor):
    return num - (num % divisor)
