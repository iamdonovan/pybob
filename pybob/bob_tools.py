import random
import datetime as dt
import ogr
import osr
import numpy as np
from shapely.geometry import Point


def standard_landsat(instring):
    """Given a string of a landsat scenename, make a standard (pre-Collection) filename,
        of the form LSSPPPRRRYYYYDDDXXX01.
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
    """Return a string or a datetime object given a year and a day of year.

    Parameters
    ----------
    year : int
        Year of the input date, e.g., ``2018``.
    doy : int
        Day of the year of the input date, e.g., ``1`` for 1 Jan.
    string_out : bool
        Return a string (True) or a datetime object (False). Default is True.
    outform : str
        Format for string to return. Default is `%Y/%m/%d`, or 2018/01/01 for
        1 January 2018.

    Examples
    --------
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
    pass


def dec2doy(decdate):
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


def bin_data(bins, data2bin, bindata, mode='mean'):
    digitized = np.digitize(bindata, bins)
    binned = np.zeros(len(bins)) * np.nan
    if mode == 'mean':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmean(data2bin[np.logical_and(np.isfinite(bindata), digitized == i)])
    elif mode == 'median':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmedian(data2bin[np.logical_and(np.isfinite(bindata), digitized == i)])
    elif mode == 'std':
        for i, _ in enumerate(bins):
            binned[i] = np.nanstd(data2bin[np.logical_and(np.isfinite(bindata), digitized == i)])
    elif mode == 'max':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmax(data2bin[np.logical_and(np.isfinite(bindata), digitized == i)])
    elif mode == 'min':
        for i, _ in enumerate(bins):
            binned[i] = np.nanmin(data2bin[np.logical_and(np.isfinite(bindata), digitized == i)])
    else:
        raise ValueError('mode must be mean, median, or std')
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
