"""
pybob.GeoImg is a class to handle geospatial imagery, in particular satellite images and digital elevation models.
"""
import os
import re
import random
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from multiprocessing import Pool

numpy2gdal = {np.uint8: gdal.GDT_Byte, np.uint16: gdal.GDT_UInt16, np.int16: gdal.GDT_Int16,
              np.float32: gdal.GDT_Float32, np.float64: gdal.GDT_Float64, 
              np.uint32: gdal.GDT_UInt32, np.int32: gdal.GDT_Int32, 
              np.complex64: gdal.GDT_CFloat64}

gdal2numpy = {1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
              6: np.float32, 7: np.float64, 11: np.complex64}

lsat_sensor = {'C': 'OLI/TIRS', 'E': 'ETM+', 'T': 'TM', 'M': 'MSS', 'O': 'OLI', 'TI': 'TIRS'}


def parse_landsat(gname):
    attrs = []
    if len(gname.split('_')) == 1:
        attrs.append(lsat_sensor[gname[1]])
        attrs.append('Landsat {}'.format(int(gname[2])))
        attrs.append((int(gname[3:6]), int(gname[6:9])))
        year = int(gname[9:13])
        doy = int(gname[13:16])
        attrs.append(dt.datetime.fromordinal(dt.date(year-1, 12, 31).toordinal()+doy))
        attrs.append(attrs[3].date())
    elif re.match('L[COTEM][0-9]{2}', gname.split('_')[0]):
        split_name = gname.split('_')
        attrs.append(lsat_sensor[split_name[0][1]])
        attrs.append('Landsat {}'.format(int(split_name[0][2:4])))
        attrs.append((int(split_name[2][0:3]), int(split_name[2][3:6])))
        attrs.append(dt.datetime.strptime(split_name[3], '%Y%m%d'))
        attrs.append(attrs[3].date())
    return attrs


def get_file_info(in_filestring):
    in_filename = os.path.basename(in_filestring)
    in_dir = os.path.dirname(in_filestring)
    if in_dir == '':
        in_dir = '.'
    return in_filename, in_dir


def int_pts(myins):
    pt, ij, X, Y, z, mode = myins            
    try:
        zint = griddata((X.flatten(), Y.flatten()), z.flatten(), pt, method=mode)
        if zint.shape == (1,):
            zint = zint[0]
    except:
        zint = np.nan
    return zint


class GeoImg(object):
    """
    Create a GeoImg object from a GDAL-supported raster dataset.
    """
    def __init__(self, in_filename, in_dir=None, datestr=None,
                 datefmt='%m/%d/%y', dtype=None, attrs=None, update=False):
        """
        :param in_filename:  Filename or object to read in. If in_filename is a string, the GeoImg is created by
            reading the file corresponding to that filename. If in_filename is a gdal object, the
            GeoImg is created by operating on the corresponding object.
        :param in_dir: (optional) directory where in_filename is located. If not given, the directory
            will be determined from the input filename.
        :param datestr: (optional) string to pass to GeoImg, representing the date the image was acquired.
        :param datefmt: Format of datestr that datetime.datetime should use to parse datestr.
            Default is %m/%d/%y.
        :param dtype: numpy datatype to read input data as. Default is np.float32. See numpy docs for more details.
        :param update: Open GeoImg in using gdal.GA_Update (will overwrite information on disk)

        :type in_filename: str, gdal.Dataset
        :type in_dir: str
        :type datestr: str
        :type dtype: numpy datatype
        :type update: bool
        """
        def check_geotransform(gd):
            # Replace geotransform if origin not located at upper right... 
            gt = list(gd.GetGeoTransform())
            if gt[5]>0: # if origin is lower left coordinate, then replace with upper left
                gt[3] = gt[3] + gd.RasterXSize * gt[4] + gd.RasterYSize * gt[5]
                gt[5] = -gt[5]                                                
            return tuple(gt)

        if update:
            gdal_mode = gdal.GA_Update
        else:
            gdal_mode = gdal.GA_ReadOnly

        if type(in_filename) is gdal.Dataset:
            self.filename = None
            self.in_dir_path = None
            self.in_dir_abs_path = None
            self.gd = in_filename
        elif type(in_filename) is str:
            if in_dir is None:
                in_filename, in_dir = get_file_info(in_filename)
            self.filename = in_filename
            self.in_dir_path = in_dir
            self.in_dir_abs_path = os.path.abspath(in_dir)
            self.gd = gdal.Open(os.path.join(self.in_dir_path, self.filename), gdal_mode)
            if self.gd is None:
                raise RuntimeError('Unable to open file {}'.format(os.path.join(self.in_dir_path, self.filename)))
        else:
            raise Exception('in_filename must be a string or a gdal Dataset')

        self.gt = check_geotransform(self.gd)
        self.proj_wkt = self.gd.GetProjection()
        crs = osr.SpatialReference()
        crs.ImportFromWkt(self.proj_wkt)
        self.proj4 = crs.ExportToProj4()

        try:   # Starting to implement the possibility to load raster without ESPG pre-assigned
            self.epsg = int(''.join(filter(lambda x: x.isdigit(), self.proj_wkt.split(',')[-1])))
        except:
            self.epsg = None

        self.spatialReference = crs

        self.intype = self.gd.GetDriver().ShortName
        self.npix_x = self.gd.RasterXSize
        self.npix_y = self.gd.RasterYSize
        self.shape = (self.npix_y, self.npix_x)
        self.xmin = self.gt[0]
        self.xmax = self.gt[0] + self.npix_x * self.gt[1] + self.npix_y * self.gt[2]
        self.ymin = self.gt[3] + self.npix_x * self.gt[4] + self.npix_y * self.gt[5]
        self.ymax = self.gt[3]
        self.dx = self.gt[1]
        self.dy = self.gt[5]
        self.UTMtfm = [self.xmin, self.ymax, self.dx, self.dy]
        self.NDV = self.gd.GetRasterBand(1).GetNoDataValue()
        if dtype is None:  # if no datatype is given, try to read the one given by gdal
            try:
                dtype = gdal2numpy[self.gd.GetRasterBand(1).DataType]
            except KeyError:  # default to float32 if we haven't explicitly programmed the dtype yet.
                dtype = np.float32
        self.img = self.gd.ReadAsArray().astype(dtype)
        self.dtype = dtype
        self.px_loc = self.gd.GetMetadataItem('AREA_OR_POINT')
        if dtype in [np.float32, np.float64, np.complex64, np.floating, float]:
            self.isfloat = True
        else:
            self.isfloat = False

        if self.NDV is not None and self.isfloat:
            self.img[self.img == self.NDV] = np.nan
        elif self.NDV is not None:
            self.img = np.ma.masked_where(self.img == self.NDV, self.img)

        if self.filename is not None:
            self.match_sensor(in_filename, datestr=datestr, datefmt=datefmt)
        elif attrs is not None:
            self.sensor_name = attrs.sensor_name
            self.satellite = attrs.satellite
            self.tile = attrs.tile
            self.datetime = attrs.datetime
            self.date = attrs.date
        else:
            self.sensor_name = None
            self.satellite = None
            self.tile = None
            self.datetime = None
            self.date = None

        self.img_ov2 = self.img[0::2, 0::2]
        self.img_ov10 = self.img[0::10, 0::10]

    def match_sensor(self, fname, datestr=None, datefmt=''):
        """
        Attempts to pull metadata (e.g., sensor, date information) from fname, setting sensor_name, satellite,
        tile, datetime, and date attributes of GeoImg object.

        :param fname: filename of image to parse
        :param datestr: optional datestring to set date attributes
        :param datefmt: optional datetime format for datestr
        :type fname: str
        :type datestr: str
        :type datefmt: str
        """
        bname = os.path.splitext(os.path.basename(fname))[0]
        # assumes that the filename has a form GRANULE_BXX.ext
        if '_' in bname:
            gname = '_'.join(bname.split('_')[:-1])
        else:
            #print("I don't recognize this filename format.")
            #print("Make sure to specify a date and format if you need date info,")
            #print("  and your filename is not a standard filename.")
            print("No date information read from filename.")
            self.sensor_name = None
            self.satellite = None
            self.tile = None
            self.datetime = None
            self.date = None
            return

        if len(gname.split('_')) == 1:
            self.sensor_name = None
            self.satellite = None
            self.tile = None
            self.datetime = None
            self.date = None
        # first, check if we've been given a date
        elif datestr is not None:
            self.sensor_name = None
            self.satellite = None
            self.tile = None
            self.datetime = dt.datetime.strptime(datestr, datefmt)
            self.date = self.datetime.date()
        elif re.match('L[COTEM][0-9]{2}', gname.split('_')[0]):
            attrs = parse_landsat(gname)
            self.sensor_name = attrs[0]
            self.satellite = attrs[1]
            self.tile = attrs[2]
            self.datetime = attrs[3]
            self.date = attrs[4]
        elif gname.split('_')[0][0] == 'L' and len(gname.split('_')) == 1:
            attrs = parse_landsat(gname)
            self.sensor_name = attrs[0]
            self.satellite = attrs[1]
            self.tile = attrs[2]
            self.datetime = attrs[3]
            self.date = attrs[4]
        # next, sentinel 2
        elif re.match('T[0-9]{2}[A-Z]{3}', gname.split('_')[0]):
            # sentinel 2 tiles have form
            self.sensor_name = 'MSI'
            self.satellite = 'Sentinel-2'
            self.tile = gname.split('_')[0][1:]
            self.datetime = dt.datetime.strptime(gname.split('_')[1], '%Y%m%dT%H%M%S')
            self.date = self.datetime.date()
        # next, aster
        elif gname.split('_')[0] == 'AST':
            self.sensor_name = 'ASTER'
            self.satellite = 'Terra'
            self.tile = None
            self.datetime = dt.datetime.strptime(bname.split('_')[2][3:], '%m%d%Y%H%M%S')
            self.date = self.datetime.date()
        elif gname.split('_')[0] == 'SETSM':
            self.sensor_name = 'SETSM'
            self.satellite = gname.split('_')[1]
            self.tile = None
            self.datetime = dt.datetime.strptime(gname.split('_')[2], '%Y%m%d')
            self.date = self.datetime.date()
        elif gname.split('_')[0]== 'SPOT':
            self.sensor_name = 'HFS'
            self.satellite = 'SPOT5'
            self.tile = None
            self.datetime = dt.datetime.strptime(bname.split('_')[2], '%Y%m%d')
            self.date = self.datetime.date()
        else:
            # print("No date information read from filename.")
            self.sensor_name = None
            self.satellite = None
            self.tile = None
            self.datetime = None
            self.date = None

    def info(self):
        """ Prints information about the GeoImg (filename, coordinate system, number of columns/rows, etc.)."""
        print('Driver:             {}'.format(self.gd.GetDriver().LongName))
        if self.intype not in ['MEM', 'VRT']:
            print('File:               {}'.format(self.in_dir_path + os.path.sep + self.filename))
        else:
            print('File:               {}'.format('in memory'))
        print('Size:               {}, {}'.format(self.npix_x, self.npix_y))
        print('Coordinate System:  EPSG:{}'.format(self.epsg))
        print('NoData Value:       {}'.format(self.NDV))
        print('Pixel Size:         {}, {}'.format(self.dx, self.dy))
        print('Upper Left Corner:  {}, {}'.format(self.xmin, self.ymax))
        print('Lower Right Corner: {}, {}'.format(self.xmax, self.ymin))
        print('[MAXIMUM]:          {}'.format(np.nanmax(self.img)))
        print('[MINIMUM]:          {}'.format(np.nanmin(self.img)))
        # print('[MEAN]:             {}'.format(np.nanmean(self.img)))
        # print('[MEDIAN]:           {}'.format(np.nanmedian(self.img)))

    def display(self, fig=None, cmap='gray', extent=None, sfact=None, showfig=True, band=[0, 1, 2], **kwargs):
        """
        Display GeoImg in a matplotlib figure.

        :param fig: figure handle to show image in. If not set, creates a new figure.
        :param cmap: colormap to use for the image. Default is gray.
        :param extent: spatial extent to limit the figure to, given as xmin, xmax, ymin, ymax.
        :param sfact: Factor by which to reduce the number of pixels plotted.
            Default is 1 (i.e., all pixels are displayed).
        :param showfig: Open the figure window. Default is True.
        :param band: Image bands to use, if GeoImg represents a multi-band image.
        :param kwargs: Optional keyword arguments to pass to matplotlib.pyplot.imshow
        :type fig: matplotlib.figure.Figure
        :type cmap: matplotlib colormap
        :type extent: array-like
        :type sfact: int
        :type showfig: bool
        :type band: array-like
        :returns fig: Handle pointing to the matplotlib Figure created (or passed to display).
        """
        if fig is None:
            fig = plt.figure(facecolor='w')
            # fig.hold(True)
        # else:
            # fig.hold(True)

        if extent is None:

            extent = [self.xmin, self.xmax, self.ymin, self.ymax]
            disp_ext = extent
            mini = 0
            maxi = self.npix_y
            minj = 0
            maxj = self.npix_x
        else:
            xmin, xmax, ymin, ymax = extent
            mini, minj = self.xy2ij((xmin, ymax))
            maxi, maxj = self.xy2ij((xmax, ymin))
                       
            mini += 0.5
            minj += 0.5

            maxi -= 0.5
            maxj -= 0.5  # subtract the .5 for half a pixel, add 1 for slice

            mini = max(0, mini)
            maxi = min(maxi, self.npix_y)

            minj = max(0, minj)
            maxj = min(maxj, self.npix_x)
            disp_ext = [max(xmin, self.xmin), min(xmax, self.xmax), max(ymin, self.ymin), min(ymax, self.ymax)]
            
        # if we only have one band, plot it.
        if self.gd.RasterCount == 1:
            if sfact is None:
                showimg = self.img[int(mini):int(maxi+1), int(minj):int(maxj+1)]
            else:
                showimg = self.img[int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
            plt.imshow(showimg, extent=disp_ext, cmap=cmap, **kwargs)
        elif type(band) is int:
            if sfact is None:
                showimg = self.img[band][int(mini):int(maxi+1), int(minj):int(maxj+1)]
            else:
                showimg = self.img[band][int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
            plt.imshow(showimg, extent=disp_ext, cmap=cmap, **kwargs)
        else:  # if we have more than one band and we've asked to display them all, do it.
            if sfact is None:
                b1 = self.img[band[0]][int(mini):int(maxi+1), int(minj):int(maxj+1)]
                b2 = self.img[band[1]][int(mini):int(maxi+1), int(minj):int(maxj+1)]
                b3 = self.img[band[2]][int(mini):int(maxi+1), int(minj):int(maxj+1)]
            else:
                b1 = self.img[band[0]][int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
                b2 = self.img[band[1]][int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
                b3 = self.img[band[2]][int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
            rgb = np.dstack([b1, b2, b3]).astype(self.dtype)
            plt.imshow(rgb, extent=disp_ext, **kwargs)

        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        ax = fig.gca()  # get current axes
        ax.set_aspect('equal')    # set equal aspect
        ax.autoscale(tight=True)  # set axes tight

        if showfig:
            fig.show()  # don't forget this one!

        return fig

    def write(self, outfilename, out_folder=None, driver='GTiff', dtype=None, bands=None):
        """
        Write GeoImg to a gdal-supported raster file.
        
        :param outfilename: string representing the filename to be written to.
        :param out_folder: optional string representing the folder to be written to. If not set,
            folder is either guessed from outfilename, or assumed to be the current folder.
        :param driver: optional string representing the gdal driver to use to write the raster file. Default is GTiff.
            Options include: HDF4, HDF5, JPEG, PNG, JPEG2000 (if enabled). See gdal docs for more options.
        :param datatype: Type of data to write the raster as. Check GeoImg.numpy2gdal.keys() to see numpy data types implemented.
        :param bands: Specify band(s) to write to file. Default behavior is all bands.
        :type outfilename: str
        :type out_folder: str
        :type driver: str
        :type datatype: numpy datatype
        :type bands: array-like
        """
        if dtype is None:
            dtype = self.dtype
        # if we don't specify which bands, we're going to write all of them
        if bands is None:
            nband = self.gd.RasterCount
        else:
            nband = len(bands)
        
        driver = gdal.GetDriverByName(driver)

        ncols = self.npix_x
        nrows = self.npix_y

        if out_folder is None:
            outfilename, out_folder = get_file_info(outfilename)
            
        out = driver.Create(os.path.sep.join([out_folder, outfilename]), ncols, nrows, nband, numpy2gdal[dtype])

        setgeo = out.SetGeoTransform(self.gt)
        setproj = out.SetProjection(self.proj_wkt)
        nanmask = np.isnan(self.img)

        if self.NDV is not None:
            self.img[nanmask] = self.NDV
        elif np.count_nonzero(nanmask) > 0:
            self.img[nanmask] = -9999
            self.NDV = -9999

        if bands is None:
            if nband == 1:
                write = out.GetRasterBand(1).WriteArray(self.img)
                if self.NDV is not None:
                    out.GetRasterBand(1).SetNoDataValue(self.NDV)                
            else:
                for i in range(nband):
                    write = out.GetRasterBand(i+1).WriteArray(self.img[i, :, :])
                    if self.NDV is not None:
                        out.GetRasterBand(i+1).SetNoDataValue(self.NDV)
        else:
            if nband == 1:
                write = out.GetRasterBand(1).WriteArray(self.img[bands[0], :, :])
                if self.NDV is not None:
                    out.GetRasterBand(1).SetNoDataValue(self.NDV)
            else:
                for i, b in enumerate(bands):
                    write = out.GetRasterBand(i+1).WriteArray(self.img[b, :, :])
                    if self.NDV is not None:
                        out.GetRasterBand(i+1).SetNoDataValue(self.NDV)

        out.FlushCache()

        if self.NDV is not None:
            if self.isfloat:
                self.img[nanmask] = np.nan

        del setgeo, setproj, write

    def copy(self, new_raster=None, new_extent=None, driver='MEM', filename='',
             newproj=None, datatype=gdal.GDT_Float32):
        """
        Copy the GeoImg, creating a new GeoImg, optionally updating the extent and raster.

        :param new_raster: New raster to use. If not set, the new GeoImg will have the same raster
            as the old image.
        :param new_extent: New extent to use, given as xmin, xmax, ymin, ymax. If not set,
            the old extent is used. If set, you must also include a new raster.
        :param driver: gdal driver to use to create the new GeoImg. Default is 'MEM' (in-memory).
            See gdal docs for more options. If a different driver is used, filename must
            also be specified.
        :param filename: Filename corresponding to the new image, if not created in memory.

        :type new_raster: array-like
        :type new_extent: array_like
        :type driver: str
        :type filename: str

        :returns new_geo: A new GeoImg with the specified extent and raster.
        """
        drv = gdal.GetDriverByName(driver)
        if driver == 'MEM':
            filename = ''
        elif filename == '':
            raise Exception('must specify an output filename with driver {}'.format(driver))

        if (new_raster is None and new_extent is None):
            npix_y, npix_x = self.img.shape
            new_raster = self.img  # give the same raster
            newgt = self.gt
        elif (new_raster is not None and new_extent is None):
            # copy the geoimg and replace the raster with new_raster
            npix_y, npix_x = np.array(new_raster).shape

            dx = (self.xmax - self.xmin) / float(npix_x)
            dy = (self.ymin - self.ymax) / float(npix_y)
            newgt = (self.xmin, dx, 0, self.ymax, 0, dy)
        elif (new_raster is not None and new_extent is not None):
            # copy the geoimg and replace the raster with new_raster
            npix_y, npix_x = np.array(new_raster).shape
            xmin, xmax, ymin, ymax = new_extent
            dx = (xmax - xmin) / float(npix_x)
            dy = (ymin - ymax) / float(npix_y)
            newgt = (new_extent[0], dx, 0, new_extent[3], 0, dy)
        else:
            raise Exception('If new extent is specified, you must also specify the new raster to be used!')
        newGdal = drv.Create(filename, npix_x, npix_y, 1, datatype)
        wa = newGdal.GetRasterBand(1).WriteArray(new_raster)
        sg = newGdal.SetGeoTransform(newgt)

        if newproj is None:
            newproj = self.proj_wkt

        sp = newGdal.SetProjection(newproj)
        md = newGdal.SetMetadata(self.gd.GetMetadata())
        if self.NDV is not None:
            newGdal.GetRasterBand(1).SetNoDataValue(self.NDV)

        del wa, sg, sp, md

        out = GeoImg(newGdal, attrs=self)
        # make sure to apply mask
        if isinstance(new_raster, np.ma.MaskedArray):
            out.mask(new_raster.mask)
        elif isinstance(self.img, np.ma.MaskedArray):
            out.mask(self.img.mask)
        return out

    # return X,Y grids of coordinates for each pixel    
    def xy(self, ctype='corner', grid=True):
        """
        Get x,y coordinates of all pixels in the GeoImg.

        :param ctype: coordinate type. If 'corner', returns corner coordinates of pixels.
            If 'center', returns center coordinates. Default is corner.
        :param grid: Return gridded coordinates. Default is True.
        :type ctype: str
        :type grid: bool
        :returns x,y: numpy arrays corresponding to the x,y coordinates of each pixel.
        """
        assert ctype in ['corner', 'center'], "ctype is not one of 'corner', 'center': {}".format(ctype)

        xx = np.linspace(self.xmin, self.xmax, self.npix_x+1)

        if self.dy < 0:
            yy = np.linspace(self.ymax, self.ymin, self.npix_y+1)
        else:
            yy = np.linspace(self.ymin, self.ymax, self.npix_y+1)

        if ctype == 'center':
            xx += self.dx / 2  # shift by half a pixel
            yy += self.dy / 2
        if grid:
            return np.meshgrid(xx[:-1], yy[:-1])  # drop the last element
        else:
            return xx[:-1], yy[:-1]

    def reproject(self, dst_raster, driver='MEM', filename='', method=gdal.GRA_Bilinear):
        """
        Reproject the GeoImg to the same extent and coordinate system as another GeoImg.

        :param dst_raster: GeoImg to project given raster to.
        :param driver: gdal driver to use to create the new GeoImg. Default is 'MEM' (in-memory).
            See gdal docs for more options. If a different driver is used, filename must
            also be specified.
        :param filename: Filename corresponding to the new image, if not created in memory.
        :param method: gdal resampling algorithm to use. Default is GRA_Bilinear.
            Other options include: GRA_Average, GRA_Cubic, GRA_CubicSpline, GRA_NearestNeighbour.
            See gdal docs for more options and details.
        :type dst_raster: GeoImg
        :type driver: str
        :type filename: str
        :type method: gdal_GRA
        :returns new_geo: reprojected GeoImg.
        """
        drv = gdal.GetDriverByName(driver)
        if driver == 'MEM':
            filename = ''
        elif driver == 'GTiff' and filename == '':
            raise Exception('must specify an output filename')

        dest = drv.Create('', dst_raster.npix_x, dst_raster.npix_y,
                          1, gdal.GDT_Float32)
        dest.SetProjection(dst_raster.proj_wkt)
        dest.SetGeoTransform(dst_raster.gt)
        # copy the metadata of the current GeoImg to the new GeoImg
        dest.SetMetadata(self.gd.GetMetadata())
        if dst_raster.NDV is not None:
            dest.GetRasterBand(1).SetNoDataValue(dst_raster.NDV)
            dest.GetRasterBand(1).Fill(dst_raster.NDV)
        elif self.NDV is not None:
            dest.GetRasterBand(1).SetNoDataValue(self.NDV)
            dest.GetRasterBand(1).Fill(self.NDV)
        else:
            dest.GetRasterBand(1).Fill(0)

        gdal.ReprojectImage(self.gd, dest, self.proj_wkt, dst_raster.proj_wkt, method)

        out = GeoImg(dest, attrs=self)
        if out.NDV is not None and out.isfloat:
            out.img[out.img == out.NDV] = np.nan
        elif out.NDV is not None:
            out.img = np.ma.masked_where(out.img == out.NDV, out.img)

        return out

    def shift(self, xshift, yshift):
        """
        Shift the GeoImg in space by a given x,y offset.
        
        :param xshift: x offset to shift GeoImg by.
        :param yshift: y offset to shift GeoImg by.
        :type xshift: float
        :type yshift: float
        """
        gtl = list(self.gt)
        gtl[0] += xshift
        gtl[3] += yshift
        self.gt = tuple(gtl)
        self.gd.SetGeoTransform(self.gt)
        self.xmin = self.gt[0]
        self.xmax = self.gt[0] + self.npix_x * self.gt[1] + self.npix_y * self.gt[2]
        self.ymin = self.gt[3] + self.npix_x * self.gt[4] + self.npix_y * self.gt[5]
        self.ymax = self.gt[3]
        self.UTMtfm = [self.xmin, self.ymax, self.dx, self.dy]

    def ij2xy(self, ij):
        """
        Return x,y coordinates for a given row, column index pair.
        
        :param ij: row (i) and column (j) index of pixel.
        :type ij: float

        :returns xy: x,y coordinates of i,j in the GeoImg's spatial reference system.
        """
        if self.is_point():
            delta_p = 0.5
        else:
            delta_p = 0

        x = self.UTMtfm[0]+((ij[1]+delta_p)*self.UTMtfm[2])
        y = self.UTMtfm[1]+((ij[0]+delta_p)*self.UTMtfm[3])

        return x, y

    def xy2ij(self, xy):
        """
        Return row, column indices for a given x,y coordinate pair.
        
        :param xy: x, y coordinates in the GeoImg's spatial reference system.
        :type xy: float

        :returns ij: i,j indices of x,y in the image.
        
        """
        x = xy[0]
        y = xy[1]
        if self.is_point():
            delta_p = 0.5
        else:
            delta_p = 0

        j = (x - self.UTMtfm[0])/self.UTMtfm[2] - delta_p  # if python started at 1, + 0.5
        i = (y - self.UTMtfm[1])/self.UTMtfm[3] - delta_p  # if python started at 1, + 0.5

        return i, j

    def is_rotated(self):
        """Determine whether GeoImg is rotated with respect to North."""
        if len(self.img) == 3:
            # if we have multiple bands, find the smallest index
            # and sum along that (i.e., collapse the bands into one)
            bnum = self.img.shape.index(min(self.img.shape))
            tmpimg = self.img
            # but, we want to make sure we don't mess up non-nan nodata
            if not np.isnan(self.NDV):
                tmpimg[tmpimg == self.NDV] = np.nan
            testband = np.sum(tmpimg, bnum)
        else:
            testband = self.img

        _, ncols = testband.shape
        goodinds = np.where(np.isfinite(testband))
        uli = goodinds[0][np.argmin(goodinds[0])]
        ulj = np.min(goodinds[1][goodinds[0] == uli])
        llj = goodinds[1][np.argmin(goodinds[1])]
        return ~(np.abs(llj-ulj)/float(ncols) < 0.02)

    def find_corners(self, nodata=np.nan, mode='ij'):
        """
        Find corner coordinates of valid image area.
        
        :param nodata: nodata value to use. Default is numpy.nan
        :param mode: Type of coordinates to return. Options are 'ij', row/column index,
            or 'xy', x,y coordinate. Default is 'ij'.
        :type nodata: numeric
        :type mode: str

        :returns corners: Array corresponding to the corner coordinates, estimated from the convex hull
            of the valid data.
        """
        assert mode in ['ij', 'xy'], "mode is not one of 'ij', 'xy': {}".format(mode)
        
        # if we have more than one band, have to pick one or merge them.
        if len(self.img) == 3:
            # if we have multiple bands, find the smallest index
            # and sum along that (i.e., collapse the bands into one)
            bnum = self.img.shape.index(min(self.img.shape))
            tmpimg = self.img
            # but, we want to make sure we don't mess up non-nan nodata
            if not np.isnan(nodata):
                tmpimg[tmpimg == nodata] = np.nan
            testband = np.sum(tmpimg, bnum)
        else:
            testband = self.img
        # now we actually get the good indices
        if np.isnan(nodata):
            goodinds = np.where(np.isfinite(testband))
        else:
            goodinds = np.where(np.logical_not(testband == nodata))

        goodpoints = np.vstack((goodinds[0], goodinds[1])).transpose()
        hull = ConvexHull(goodpoints)

        iverts = goodpoints[hull.vertices, 0]
        jverts = goodpoints[hull.vertices, 1]
        corners = list(zip(iverts, jverts))

        if mode == 'xy':
            xycorners = [self.ij2xy(corner) for corner in corners]
            return np.array(xycorners)
        return np.array(corners)

    def find_valid_bbox(self, nodata=np.nan):
        """
        Find bounding box for valid data.

        :param nodata: nodata value to use for the image. Default is numpy.nan
        :type nodata: numeric
            
        :returns bbox: xmin, xmax, ymin, ymax of valid image area.
        """
        if np.isnan(nodata):
            goodinds = np.where(np.isfinite(self.img))
        else:
            goodinds = np.where(np.logical_not(self.img == nodata))

        # get the max, min of x,y that are valid.
        xmin, ymin = self.ij2xy((goodinds[0].min(), goodinds[1].min()))
        xmax, ymax = self.ij2xy((goodinds[0].max(), goodinds[1].max()))

        return [xmin, xmax, min(ymin, ymax), max(ymin, ymax)]

    def set_NDV(self, NDV):
        """
        Set nodata value to given value.

        :param NDV: value to set to nodata.
        :type NDV: numeric
        """
        self.NDV = NDV
        self.gd.GetRasterBand(1).SetNoDataValue(NDV)
        self.img[self.img == self.NDV] = np.nan

    def subimages(self, N, Ny=None, sBuffer=0):
        """
        Split the GeoImg into sub-images.
        
        :param N: number of column cells to split the image into.
        :param Ny: number of row cells to split image into. Default is same as N.
        :param sBuffer: number of pixels to overlap subimages by. Default is 0.

        :type N: int
        :type Ny: int
        :type sBuffer: int

        :returns sub_images: list of GeoImg tiles.
        """
        Nx = N
        if Ny is None:
            Ny = N

        new_width = int(np.floor(self.npix_x / float(Nx)))
        new_height = int(np.floor(self.npix_y / float(Ny)))

        simages = []
        for j in range(Nx):
            for i in range(Ny):
                lind = max(0, j*new_width-sBuffer)
                rind = min(self.npix_x, (j+1)*new_width + sBuffer)
                tind = max(0, i*new_height-sBuffer)
                bind = min(self.npix_y, (i+1)*new_height + sBuffer)

                imgN = self.img[tind:bind, lind:rind]
                xmin, ymin = self.ij2xy((bind, lind))
                xmax, ymax = self.ij2xy((tind, rind))
                extN = [xmin, xmax, ymin, ymax]
                newGimg = self.copy(new_raster=imgN, new_extent=extN)
                simages.append(newGimg)

        return simages

    def crop_to_extent(self, extent, pixel_size=None, bands=None):
        """
        Crop image to given extent.
        
        :param extent: Extent to which image should be cropped. If extent is a matplotlib figure handle, the image
            extent is taken from the x and y limits of the current figure axes. If extent is array-like, 
            it is assumed to be [xmin, xmax, ymin, ymax]
        :param pixel_size: Set pixel size of output raster. Default is calculated based on current
            pixel size and extent.
        :param bands: Image band(s) to crop - default assumes first (only) band.
            Remember that numpy indices start at 0 - i.e., the first band is band 0.
        :type extent: matplotlib.figure.Figure or array-like
        :type pixel_size: float
        :type bands: array-like
        :returns cropped_img: new GeoImg object resampled to the given image extent.
        """
        if isinstance(extent, plt.Figure):
            xmin, xmax = extent.gca().get_xlim()
            ymin, ymax = extent.gca().get_ylim()
        else:
            xmin, xmax, ymin, ymax = extent

        if bands is None:
            bands = [0]

        nbands = len(bands)
        
        if pixel_size is None:
            npix_x = int(np.round((xmax - xmin) / float(self.dx)))
            npix_y = int(np.round((ymin - ymax) / float(self.dy)))

            dx = (xmax - xmin) / float(npix_x)
            dy = (ymin - ymax) / float(npix_y)
        else:
            dx = pixel_size
            dy = -pixel_size            
            npix_x = int(np.round((xmax - xmin) / float(dx)))
            npix_y = int(np.round((ymin - ymax) / float(dy)))

        drv = gdal.GetDriverByName('MEM')
        dest = drv.Create('', npix_x, npix_y, nbands, gdal.GDT_Float32)
        dest.SetProjection(self.proj_wkt)
        newgt = (xmin, dx, 0.0, ymax, 0.0, dy)
        dest.SetGeoTransform(newgt)
        if self.NDV is not None:
            for i in range(len(bands)):
                dest.GetRasterBand(i+1).SetNoDataValue(self.NDV)
                dest.GetRasterBand(i+1).Fill(self.NDV)
        else:
            for i in range(len(bands)):
                dest.GetRasterBand(i+1).Fill(0)

        gdal.ReprojectImage(self.gd, dest, self.proj_wkt, self.proj_wkt, gdal.GRA_Bilinear)
        out = GeoImg(dest, attrs=self)
        if out.NDV is not None and out.isfloat:
            out.img[out.img == out.NDV] = np.nan
        elif out.NDV is not None:
            out.img = np.ma.masked_where(out.img == out.NDV, out.img)

        return out

    def resample(self, xres, yres=None, method=gdal.GRA_Bilinear):
        '''

        :param xres:
        :param yres:
        :param method:
        :return:
        '''
        if yres is None:
            yres = xres

        ds = gdal.Warp('', self.gd, xRes=xres, yRes=yres, format='VRT', resampleAlg=method)
        return GeoImg(ds)

    def overlay(self, raster, extent=None, vmin=0, vmax=10, sfact=None, showfig=True, alpha=0.25, cmap='jet'):
        """
        Overlay raster on top of GeoImg.
        
        :param raster: raster to display on top of the GeoImg.
        :param extent: Spatial of the raster. If not set, assumed to be same as the extent of the GeoImg.
            Given as xmin, xmax, ymin, ymax.
        :param vmin: minimum color value for the raster. Default is 0.
        :param vmax: maximum color value for the raster. Default is 10.
        :param sfact: Factor by which to reduce the number of points plotted.
            Default is 1 (i.e., all points are plotted).
        :param showfig: Open the figure window. Default is True.
        :param alpha: Alpha value to use for the overlay. Default is 0.25
        :param cmap: matplotlib.pyplot colormap to use for the image. Default is jet.
        :type raster: array-like
        :type extent: array-like
        :type vmin: float
        :type vmax: float
        :type sfact: int
        :type showfig: bool
        :type alpha: float
        :type cmap: str

        :returns fig: Handle pointing to the matplotlib Figure created (or passed to display).
        """
        fig = self.display(extent=extent, sfact=sfact, showfig=showfig)

        if showfig:
            plt.ion()

        oimg = plt.imshow(raster, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

        ax = plt.gca()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(oimg, cax=cax)

        # plt.show()

        return fig

    def mask(self, mask, mask_value=True):
        """
        Mask image values.

        :param mask: Array of same size as self.img corresponding to values that should be masked.
        :param mask_value: Value within mask to mask. If True, masks image where mask is True. If numeric, masks image
            where mask == mask_value.
        :type mask: array-like
        :type mask_value: bool or numeric
        """
        if mask_value is bool:
            if mask_value:
                self.img = np.ma.masked_where(mask, self.img)
        else:
            self.img = np.ma.masked_where(mask == mask_value, self.img)

    def unmask(self):
        """ Remove mask from image. If mask is not set, has no effect."""
        if isinstance(self.img, np.ma.masked_array):
            self.img = self.img.data
        else:
            pass

    def random_points(self, Npts, edge_buffer=None):
        """ 
        Generate a random sample of points within the image.
        
        :param Npts: number of random points to sample.
        :param edge_buffer: Optional buffer around edge of image, where pixels shouldn't be sampled. Default is zero.
        :type Npts: int
        :type edge_buffer: int
        :returns rand_pts: array of N random points from within the image.
        """
        # first, if we don't have an edge buffer and don't have a mask, everything is easy.
        if edge_buffer is None:
            indices = np.arange(self.img.size)  # a list of indices
            if not isinstance(self.img, np.ma.MaskedArray):
                goodinds = indices[np.isfinite(self.img.reshape(-1))]
            else:
                goodinds = indices[np.logical_and(np.invert(self.img.mask).reshape(-1),
                                   np.isfinite(self.img.data.reshape(-1)))]
            return np.array([np.array(np.unravel_index(x, self.img.shape)) for x in random.sample(list(goodinds), Npts)])
        elif edge_buffer is not None:
            tmp_img = self.img.data[edge_buffer:-edge_buffer, edge_buffer:-edge_buffer]
            indices = np.arange(tmp_img.size)
            if isinstance(self.img, np.ma.MaskedArray):
                tmp_mask = self.img.mask[edge_buffer:-edge_buffer, edge_buffer:-edge_buffer]
                goodinds = indices[np.logical_and(np.invert(tmp_mask).reshape(-1), np.isfinite(tmp_img.reshape(-1)))]
            else:
                goodinds = indices[np.isfinite(tmp_img.reshape(-1))]
            # return a random list as above, but remember to shift everything by the edge buffer.
            return np.array([np.array(np.unravel_index(x, tmp_img.shape))+edge_buffer for x in random.sample(list(goodinds), Npts)])

    def raster_points(self, pts, nsize=1, mode='linear'):
        """Interpolate raster values at a given point, or sets of points. 

        :param pts: Point(s) at which to interpolate raster value. If points fall outside
            of image, value returned is nan.'
        :param nsize: Number of neighboring points to include in the interpolation. Default is 1.
        :param mode: One of 'linear', 'cubic', or 'quintic'. Determines what type of spline is
            used to interpolate the raster value at each point. For more information, see
            scipy.interpolate.interp2d. Default is linear.
        :type pts: array-like
        :type nsize: int
        :type mode: str

        :returns rpts: Array of raster value(s) for the given points.
        """
        assert mode in ['linear', 'cubic', 'quintic'],"mode must be linear, cubic, or quintic."

        rpts = []
        # if we're given only one point, corresponding array
        # should have a size of two. in which case, we wrap it in a list.
        #TODO: this breaks for a single point when passing a list of tuples
        # if np.array(pts).size == 2:
        #     pts = [pts]

        if self.is_area():
            self.to_point()
            
        xx, yy = self.xy(ctype='center', grid=False)
        for pt in pts:
            ij = self.xy2ij(pt)
            try:
                ij = (np.int16(ij[0]+0.5), np.int16(ij[1]+0.5))
            except ValueError as e:
                print(ij)
                raise e
            if self.outside_image(ij, index=True):
                rpts.append(np.nan)
                continue
            else:
                # print("not outside!")
                x = xx[ij[1]-nsize:ij[1]+nsize+1]
                y = yy[ij[0]-nsize:ij[0]+nsize+1]
                z = self.img[ij[0]-nsize:ij[0]+nsize+1, ij[1]-nsize:ij[1]+nsize+1]
                X, Y = np.meshgrid(x, y)
                try:
                    zint = griddata((X.flatten(), Y.flatten()), z.flatten(), pt, method=mode)
                    if zint.shape == (1,):
                        zint = zint[0]
                except:
                    zint = np.nan

                rpts.append(zint)
        return np.array(rpts)

    def raster_points2(self, pts, nsize=1, mode='linear'):
        """Interpolate raster values at a given point, or sets of points using multiprocessing for speed.

        :param pts: Point(s) at which to interpolate raster value. If points fall outside
            of image, value returned is nan.'
        :param nsize: Number of neighboring points to include in the interpolation. Default is 1.
        :param mode: One of 'linear', 'cubic', or 'quintic'. Determines what type of spline is
            used to interpolate the raster value at each point. For more information, see 
            scipy.interpolate.interp2d.
        :type pts: array-like
        :type nsize: int
        :type mode: str

        :returns rpts: Array of raster value(s) for the given points.
        """
        assert mode in ['linear', 'cubic', 'quintic'],"mode must be linear, cubic, or quintic."
        # if we're given only one point, corresponding array
        # should have a size of two. in which case, we wrap it in a list.
        if np.array(pts).size == 2:
            pts = [pts]

        if self.is_area():
            self.to_point()            
            
        xx, yy = self.xy(ctype='center', grid=False)
        
        def getgrids(a):
            myimg, pt, nsize, mode = a
            ij = myimg.xy2ij(pt)
            ij = (np.int16(ij[0]+0.5), np.int16(ij[1]+0.5))

            xlow = np.int16(max(0, ij[1]-nsize))
            xhigh = np.int16(min(xx.size, ij[1]+nsize+1))

            ylow = np.int16(max(0, ij[0]-nsize))
            yhigh = np.int16(min(yy.size, ij[0]+nsize+1))

            try:
                x = xx[xlow:xhigh]
                y = yy[ylow:yhigh]

                X, Y = np.meshgrid(x, y)
                z = myimg.img[ylow:yhigh, xlow:xhigh]
            except MaskError as e:
                return pt, ij, np.nan * np.ones(nsize), np.nan * np.ones(nsize), np.nan * np.ones(nsize), mode
            return pt, ij, X, Y, z, mode

        myins = [getgrids((self, pt, nsize, mode)) for pt in pts]
        # print("half way")
        # myout = np.asarray([int_pts(myin,nsize,mode) for myin in myins])
        pool = Pool(6)
        # return np.asarray([int_pts(pt,self,nsize,mode) for pt in pts])
        return np.asarray(pool.map(int_pts, myins))

    def outside_image(self, ij, index=True):
        """
        Check whether a given point falls outside of the image.
        
        :param ij: Indices (or coordinates) of point to check.
        :param index: Interpret ij as raster indices (default is True). If False, assumes ij is coordinates.
        :type ij: array-like
        :type index: bool

        :returns is_outside: True if ij is outside of the image.
        """
        if not index:
            ij = self.xy2ij(ij)

        if np.any(np.array(ij) < 0):
            return True
        elif ij[0] > self.npix_y or ij[1] > self.npix_x:
            return True
        else:
            return False

    def std(self):
        """ Returns standard deviation (ignoring NaNs) of the image."""
        return np.nanstd(self.img)

    def mean(self):
        """ Returns mean (ignoring NaNs) of the image."""
        return np.nanmean(self.img)

    def median(self):
        """ Returns median (ignoring NaNs) of the image."""
        return np.nanmedian(self.img)

    def to_point(self):
        """
        Change pixel location from corner ('Area') to center ('Point'). Shifts raster by half pixel in the +x, -y direction.
        """
        if self.px_loc == 'Area':
            self.px_loc = 'Point'
            self.gd.SetMetadataItem('AREA_OR_POINT', self.px_loc)
            self.shift(self.dx/2, self.dy/2)
        else:
            pass

    def to_area(self):
        """
        Change pixel location from center ('Point') to corner ('Area'). Shifts raster by half pixel in the -x, +y direction.
        """
        if self.px_loc == 'Point':
            self.px_loc = 'Area'
            self.gd.SetMetadataItem('AREA_OR_POINT', self.px_loc)
            self.shift(-self.dx/2, -self.dy/2)
        else:
            pass

    def is_point(self):
        """
        Check if pixel coordinates correspond to pixel centers.

        :returns is_point: True if pixel coordinates correspond to pixel centers.
        """
        return self.px_loc == 'Point'

    def is_area(self):
        """
        Check if pixel coordinates correspond to pixel corners.

        :returns is_area: True if pixel coordinates correspond to pixel corners.
        """
        return self.px_loc == 'Area'
