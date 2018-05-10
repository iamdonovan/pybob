from __future__ import print_function
import os
import random
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import gdal
import osr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pybob.bob_tools import standard_landsat
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull


def get_file_info(in_filestring):
    in_filename = os.path.basename(in_filestring)
    in_dir = os.path.dirname(in_filestring)
    if in_dir == '':
        in_dir = '.'
    return in_filename, in_dir


class GeoImg(object):
    """Create a GeoImg object from a GDAL-supported raster dataset.

    Parameters
    ----------
    in_filename : str or gdal object.
        If in_filename is a string, the GeoImg is created by reading the file 
        corresponding to that filename. If in_filename is a gdal object, the 
        GeoImg is created by operating on the corresponding object.
    in_dir : str, optional
        Directory where in_filename is located. If not given, the directory
        will be determined from the input filename.
    datestr : str, optional
        Optional string to pass to GeoImg, representing the date the image was acquired.
    datefmt : str, optional
        Format of datestr that datetime.datetime should use to parse datestr.
        Default is %m/%d/%y.
        
    """
    def __init__(self, in_filename, in_dir=None, datestr=None, datefmt='%m/%d/%y'):

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
            self.gd = gdal.Open(os.path.join(self.in_dir_path, self.filename))
        else:
            raise Exception('in_filename must be a string or a gdal Dataset')

        self.gt = self.gd.GetGeoTransform()
        self.proj = self.gd.GetProjection()
        self.epsg = int(''.join(filter(lambda x: x.isdigit(), self.proj.split(',')[-1])))
        self.spatialReference = osr.SpatialReference()
        dump = self.spatialReference.ImportFromEPSG(self.epsg)  # do this to make our SRS nice and easy for osr later.
        del dump
        self.intype = self.gd.GetDriver().ShortName
        self.npix_x = self.gd.RasterXSize
        self.npix_y = self.gd.RasterYSize
        self.xmin = self.gt[0]
        self.xmax = self.gt[0] + self.npix_x * self.gt[1] + self.npix_y * self.gt[2]
        self.ymin = self.gt[3] + self.npix_x * self.gt[4] + self.npix_y * self.gt[5]
        self.ymax = self.gt[3]
        self.dx = self.gt[1]
        self.dy = self.gt[5]
        self.UTMtfm = [self.xmin, self.ymax, self.dx, self.dy]
        self.NDV = self.gd.GetRasterBand(1).GetNoDataValue()
        self.img = self.gd.ReadAsArray().astype(np.float32)
        self.dtype = self.gd.ReadAsArray().dtype
        if self.NDV is not None:
            self.img[self.img == self.NDV] = np.nan

        if self.filename is not None:
            if (datestr is not None):
                self.imagedatetime = dt.datetime.strptime(datestr, datefmt)
            elif (self.filename[0] == 'L'):  # if it looks like a Landsat file
                try:
                    self.filename = standard_landsat(self.filename)
                    self.sensor = self.filename[0:3]
                    self.path = int(self.filename[3:6])
                    self.row = int(self.filename[6:9])
                    self.year = int(self.filename[9:13])
                    self.doy = int(self.filename[13:16])
                    self.imagedatetime = dt.date.fromordinal(dt.date(self.year-1, 12, 31).toordinal()+self.doy)
                except:
                    print("This doesn't actually look like a Landsat file.")
            else:
                self.datetime = None
        else:
            self. datetime = None

        self.img_ov2 = self.img[0::2, 0::2]
        self.img_ov10 = self.img[0::10, 0::10]

    def info(self):
        """ Prints information about the GeoImg (filename, coordinate system, number of columns/rows, etc.)."""
        print('Driver:             {}'.format(self.gd.GetDriver().LongName))
        if self.intype != 'MEM':
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

    def display(self, fig=None, cmap='gray', extent=None, sfact=None, showfig=True, band=[0, 1, 2]):
        """
        Display GeoImg in a figure.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to show image in. If not set, creates a new figure.
        cmap : str, optional
            matplotlib.pyplot colormap to use for the image. Default is gray.
        extent : array-like, optional
            Spatial extent to limit the figure to, given as xmin, xmax, ymin, ymax.
        sfact : int, optional
            Factor by which to reduce the number of points plotted.
            Default is 1 (i.e., all points are plotted).
        showfig : bool, optional
            Open the figure window. Default is True.
        band : array-like, optional
            Image bands to use, if GeoImg represents a multi-band image.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle pointing to the matplotlib Figure created (or passed to display).
        """
        if fig is None:
            fig = plt.figure(facecolor='w')
            # fig.hold(True)
        # else:
            # fig.hold(True)

        if extent is None:
            extent = [self.xmin, self.xmax, self.ymin, self.ymax]

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

        # if we only have one band, plot it.
        if self.gd.RasterCount == 1:
            if sfact is None:
                showimg = self.img[int(mini):int(maxi+1), int(minj):int(maxj+1)]
            else:
                showimg = self.img[int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
            plt.imshow(showimg, extent=extent, cmap=cmap)
        elif type(band) is int:
            if sfact is None:
                showimg = self.img[band][int(mini):int(maxi+1), int(minj):int(maxj+1)]
            else:
                showimg = self.img[band][int(mini):int(maxi+1):sfact, int(minj):int(maxj+1):sfact]
            plt.imshow(showimg, extent=extent, cmap=cmap)
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
            plt.imshow(rgb, extent=extent)

        ax = fig.gca()  # get current axes
        ax.set_aspect('equal')    # set equal aspect
        ax.autoscale(tight=True)  # set axes tight

        if showfig:
            fig.show()  # don't forget this one!

        return fig

    def write(self, outfilename, out_folder=None, driver='GTiff', datatype=gdal.GDT_Float32):
        """
        Write GeoImg to a gdal-supported raster file.
        
        Parameters
        ----------
        outfilename : str
            String representing the filename to be written to.
        out_folder : str, optional
            String representing the folder to be written to. If not set,
            folder is either guessed from outfilename, or assumed to be the current folder.
        driver : str, optional
            String representing the gdal driver to use to write the raster file. Default is GTiff.
            Options include: HDF4, HDF5, JPEG, PNG, JPEG2000 (if enabled). See gdal docs for more options.
        datatype : gdal datatype
            Type of data to write the raster as. Default is GDT_Float32 (32-bit float).
            Other common options include: GDT_Float64 (64-bit float), GDT_Byte (8-bit integer),
                GDT_Int16 (16-bit integer), GDT_UInt16 (16-bit unsigned integer). See gdal docs for
                more options.    
        """
        driver = gdal.GetDriverByName(driver)

        ncols = self.npix_x
        nrows = self.npix_y
        nband = 1

        if out_folder is None:
            outfilename, out_folder = get_file_info(outfilename)
            
        out = driver.Create(os.path.sep.join([out_folder, outfilename]), ncols, nrows, nband, datatype)

        setgeo = out.SetGeoTransform(self.gt)
        setproj = out.SetProjection(self.proj)
        nanmask = np.isnan(self.img)

        if self.NDV is not None:
            self.img[nanmask] = self.NDV

        write = out.GetRasterBand(1).WriteArray(self.img)
        if self.NDV is not None:
            out.GetRasterBand(1).SetNoDataValue(self.NDV)

        out.GetRasterBand(1).FlushCache()

        if self.NDV is not None:
            self.img[nanmask] = np.nan

        del setgeo, setproj, write

    def copy(self, new_raster=None, new_extent=None, driver='MEM', filename='',
             newproj=None, datatype=gdal.GDT_Float32):
        """
        Copy the GeoImg, creating a new GeoImg, optionally updating the extent and raster.

        Parameters
        ----------
        new_raster : array-like, optional
            New raster to use. If not set, the new GeoImg will have the same raster
            as the old image.
        new_extent : array-like, optional
            New extent to use, given as xmin, xmax, ymin, ymax. If not set, 
            the old extent is used. If set, you must also include a new raster.
        driver : str, optional
            gdal driver to use to create the new GeoImg. Default is 'MEM' (in-memory).
            See gdal docs for more options. If a different driver is used, filename must
            also be specified.
        filename : str, optional
            Filename corresponding to the new image, if not created in memory.
            
        Returns
        -------
        new_geo : A new GeoImg with the specified extent and raster.
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
            newproj = self.proj

        sp = newGdal.SetProjection(newproj)

        if self.NDV is not None:
            newGdal.GetRasterBand(1).SetNoDataValue(self.NDV)

        del wa, sg, sp

        return GeoImg(newGdal)

    # return X,Y grids of coordinates for each pixel    
    def xy(self, ctype='corner', grid=True):
        """
        Get x,y coordinates of all pixels in the GeoImg.
        
        Parameters
        ----------
        ctype : str, optional
            coordinate type. If 'corner', returns corner coordinates of pixels. 
            If 'center', returns center coordinates. Default is center.
        grid : bool, optional
            Return gridded coordinates. Default is True.
            
        Returns
        -------
        x, y : array-like
            numpy arrays corresponding to the x,y coordinates of each pixel.
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
        
        Parameters
        ----------
        dst_raster : GeoImg
            GeoImg to project given raster to.
        driver : str, optional
            gdal driver to use to create the new GeoImg. Default is 'MEM' (in-memory).
            See gdal docs for more options. If a different driver is used, filename must
            also be specified.
        filename : str, optional
            Filename corresponding to the new image, if not created in memory.
        method : gdal GRA
            gdal resampling algorithm to use. Default is GRA_Bilinear. 
            Other options include: GRA_Average, GRA_Cubic, GRA_CubicSpline, GRA_NearestNeighbour.
            See gdal docs for more options and details.
            
        Returns
        -------
        new_geo : GeoImg
            reprojected GeoImg.
        """
        drv = gdal.GetDriverByName(driver)
        if driver == 'MEM':
            filename = ''
        elif driver == 'GTiff' and filename == '':
            raise Exception('must specify an output filename')

        dest = drv.Create('', dst_raster.npix_x, dst_raster.npix_y, 1, gdal.GDT_Float32)
        dest.SetProjection(dst_raster.proj)
        dest.SetGeoTransform(dst_raster.gt)
        if dst_raster.NDV is not None:
            dest.GetRasterBand(1).SetNoDataValue(dst_raster.NDV)
            dest.GetRasterBand(1).Fill(dst_raster.NDV)
        elif self.NDV is not None:
            dest.GetRasterBand(1).SetNoDataValue(self.NDV)
            dest.GetRasterBand(1).Fill(dst_raster.NDV)

        gdal.ReprojectImage(self.gd, dest, self.proj, dst_raster.proj, method)

        out = GeoImg(dest)
        # out.img[out.img == self.NDV] = np.nan

        return out

    def shift(self, xshift, yshift):
        """
        Shift the GeoImg in space by a given x,y offset.
        
        Parameters
        ----------
        xshift : float
            x offset to shift GeoImg by.
        yshift : float
            y offset to shift GeoImg by.
        
        Returns
        -------
        None
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
        
        Parameters
        ----------
        ij : array-like
            row (i) and column (j) index of pixel.
        
        Returns
        -------
        x, y : float
            x, y coordinates in the GeoImg's spatial reference system.
        """
        x = self.UTMtfm[0]+((ij[1]+0.5)*self.UTMtfm[2])
        y = self.UTMtfm[1]+((ij[0]+0.5)*self.UTMtfm[3])

        return x, y

    def xy2ij(self, xy):
        """
        Return row, column indices for a given x,y coordinate pair.
        
        Parameters
        ----------
        xy : array-like
            x, y coordinates in the GeoImg's spatial reference system.

        Returns
        -------
        ij : int
            row (i) and column (j) index of pixel.
        
        """
        x = xy[0]
        y = xy[1]
        j = int((x-self.UTMtfm[0])/self.UTMtfm[2]) - 0.5  # if python started at 1, + 0.5
        i = int((y-self.UTMtfm[1])/self.UTMtfm[3]) - 0.5  # if python started at 1, + 0.5

        return i, j

    def is_rotated(self):
        """ Determine whether GeoImg is rotated with respect to North."""
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
        
        Parameters
        ----------
        nodata : float, optional
            nodata value to use. Default is numpy.nan
        mode : str, optional
            Type of coordinates to return. Options are 'ij', row/column index,
            or 'xy', x,y coordinate. Default is 'ij'.
            
        Returns
        -------
        corners : array-like
            Array corresponding to the corner coordinates, estimated from the convex hull
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
        corners = zip(iverts, jverts)

        if mode == 'xy':
            xycorners = [self.ij2xy(corner) for corner in corners]
            return np.array(xycorners)
        return np.array(corners)

    def find_valid_bbox(self, nodata=np.nan):
        """
        Find bounding box for valid data.
        
        Parameters
        ----------
        nodata : float, optional
            nodata value to use for the image. Default is numpy.nan
            
        Returns
        -------
        bbox : array-like
            xmin, xmax, ymin, ymax of valid image area.
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
        """ Set nodata value to given value

        Parameters
        ----------
        NDV : numeric
            value to set to nodata.        
        """
        self.NDV = NDV
        self.gd.GetRasterBand(1).SetNoDataValue(NDV)
        self.img[self.img == self.NDV] = np.nan

    def subimages(self, N, Ny=None, sBuffer=0):
        """
        Split the GeoImg into sub-images.
        
        Parameters
        ----------
        N : int
            number of column cells to split the image into.
        Ny : int, optional
            number of row cells to split image into. Default is same as N.
        sBuffer : int, optional
            number of pixels to overlap subimages by. Default is 0.
            
        Returns
        -------
        sub_images : list
            list of GeoImg tiles.
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

    def crop_to_extent(self, extent):
        """
        Crop image to given extent.
        
        Parameters
        ----------
        extent : matplotlib.figure.Figure or array-like
            Extent to which image should be cropped. If extent is a matplotlib figure handle, the image
            extent is taken from the x and y limits of the current figure axes. If extent is array-like, 
            it is assumed to be [xmin, xmax, ymin, ymax]
        
        Returns
        -------
        cropped_img : GeoImg
            new GeoImg resampled to the given image extent.
        """
        if isinstance(extent, plt.Figure):
            xmin, xmax = extent.gca().get_xlim()
            ymin, ymax = extent.gca().get_ylim()
        else:
            xmin, xmax, ymin, ymax = extent

        npix_x = int(np.round((xmax - xmin) / float(self.dx)))
        npix_y = int(np.round((ymin - ymax) / float(self.dy)))

        dx = (xmax - xmin) / float(npix_x)
        dy = (ymin - ymax) / float(npix_y)

        drv = gdal.GetDriverByName('MEM')
        dest = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)
        dest.SetProjection(self.proj)
        newgt = (xmin, dx, 0.0, ymax, 0.0, dy)
        dest.SetGeoTransform(newgt)
        gdal.ReprojectImage(self.gd, dest, self.proj, self.proj, gdal.GRA_Bilinear)

        if self.NDV is not None:
            dest.GetRasterBand(1).SetNoDataValue(self.NDV)

        return GeoImg(dest)

    def overlay(self, raster, extent=None, vmin=0, vmax=10, sfact=None, showfig=True, alpha=0.25, cmap='jet'):
        """
        Overlay raster on top of GeoImg.
        
        Parameters
        ----------
        raster : array-like
            raster to display on top of the GeoImg.
        extent : array-like, optional
            Spatial of the raster. If not set, assumed to be same as the extent of the GeoImg.
            Given as xmin, xmax, ymin, ymax.
        vmin : float, optional
            minimum color value for the raster. Default is 0.
        vmax : float, optional
            maximum color value for the raster. Default is 10.
        sfact : int, optional
            Factor by which to reduce the number of points plotted.
            Default is 1 (i.e., all points are plotted).
        showfig : bool, optional
            Open the figure window. Default is True.
        alpha : float, optionall
            Alpha value to use for the overlay. Default is 0.25
        cmap : str, optional
            matplotlib.pyplot colormap to use for the image. Default is jet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle pointing to the matplotlib Figure created (or passed to display).
        """
        fig = self.display(extent=extent, sfact=sfact, showfig=showfig)

        if showfig:
            plt.ion()

        oimg = plt.imshow(raster, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

        ax = plt.gca()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(oimg, cax=cax)

#        plt.show()

        return fig

    def mask(self, mask, mask_value=True):
        """
        Mask array, given a mask and a value to mask.

        Parameters
        ----------
                
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
        Randomly sample points within the image.
        
        Parameters
        ----------
        Npts : int
            number of random points to sample.
        edge_buffer: int, optional
            Optional buffer around edge of image, where pixels shouldn't be sampled. Default is zero.
        
        Returns
        -------
        rand_pts : array-like
            array of N random points from within the image.
        """
        # first, if we don't have an edge buffer and don't have a mask, everything is easy.
        if edge_buffer is None:
            indices = np.arange(self.img.size)  # a list of indices
            if not isinstance(self.img, np.ma.MaskedArray):
                goodinds = indices[np.isfinite(self.img.reshape(-1))]
            else:
                goodinds = indices[np.logical_and(np.invert(self.img.mask).reshape(-1),
                                   np.isfinite(self.img.data.reshape(-1)))]
            return np.array([np.array(np.unravel_index(x, self.img.shape)) for x in random.sample(goodinds, Npts)])
        elif edge_buffer is not None:
            tmp_img = self.img.data[edge_buffer:-edge_buffer, edge_buffer:-edge_buffer]
            indices = np.arange(tmp_img.size)
            if isinstance(self.img, np.ma.MaskedArray):
                tmp_mask = self.img.mask[edge_buffer:-edge_buffer, edge_buffer:-edge_buffer]
                goodinds = indices[np.logical_and(np.invert(tmp_mask).reshape(-1), np.isfinite(tmp_img.reshape(-1)))]
            else:
                goodinds = indices[np.isfinite(tmp_img.reshape(-1))]
            # return a random list as above, but remember to shift everything by the edge buffer.
            return np.array([np.array(np.unravel_index(x, tmp_img.shape))+edge_buffer for x in random.sample(goodinds, Npts)])

    def raster_points(self, pts, mode='linear'):
        """Interpolate raster values at a given point, or sets of points. 
        
        Parameters
        ----------
        
        pts : array-like
            Point(s) at which to interpolate raster value. If points fall outside
            of image, value returned is nan.
        mode : str, optional
            One of 'linear', 'cubic', or 'quintic'. Determines what type of spline is
            used to interpolate the raster value at each point. For more information, see 
            scipy.interpolate.interp2d.
            
        Returns
        -------
        
        rpts : array-like
            Array of raster value(s) for the given points.
        """
        rpts = []
        # if we're given only one point, corresponding array
        # should have a size of two. in which case, we wrap it in a list.
        if np.array(pts).size == 2:
            pts = [pts]

        xx, yy = self.xy(ctype='center', grid=False)
        for pt in pts:
            ij = self.xy2ij(pt)
            ij = (int(ij[0]+0.5), int(ij[1]+0.5))
            if self.outside_image(ij, index=True):
                rpts.append(np.nan)
                continue
            else:
                # print("not outside!")
                x = xx[ij[1]-1:ij[1]+2]
                y = yy[ij[0]-1:ij[0]+2]
                z = self.img[ij[0]-1:ij[0]+2, ij[1]-1:ij[1]+2]
                X, Y = np.meshgrid(x, y)
                try:
                    zint = griddata((X.flatten(), Y.flatten()), z.flatten(), pt)
                except:
                    zint = np.nan
                rpts.append(zint)
        return np.array(rpts)

    def outside_image(self, ij, index=True):
        """Check whether a given point falls outside of the image.
        
        Parameters
        ----------
        ij : array-like
            Indices (or coordinates) of point to check.
        index: bool, optional
            Interpret ij as raster indices (default is True). If False,
            assumes ij is coordinates.
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
