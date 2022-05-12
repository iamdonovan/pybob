"""
pybob.image_tools is a collection of tools related to working with images.
"""
from __future__ import print_function
import os
import numpy as np
from osgeo import gdal, ogr, osr
import multiprocessing as mp
from llc import jit_filter_function
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage.exposure import match_histograms
from skimage.feature import greycomatrix, greycoprops
from pybob.bob_tools import parse_lsat_scene, round_down
from pybob.GeoImg import GeoImg


def hillshade(dem, azimuth=315, altitude=45):
    """
    Create a hillshade image of a DEM, given the azimuth and altitude.

    :param dem: GeoImg representing a DEM.
    :param azimuth: Solar azimuth angle, in degress from North. Default 315.
    :param altitude: Solar altitude angle, in degrees from horizon. Default 45.

    :type dem: pybob.GeoImg
    :type azimuth: float
    :type altitude: float

    :returns shade: numpy array of the same size as dem.img, representing the hillshade image.
    """
    x, y = np.gradient(dem.img, dem.dx, dem.dy)
    slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180
    altituderad = altitude * np.pi / 180
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) *\
             np.cos(slope) * np.cos(azimuthrad - aspect)

    return shaded


def nanmedian_filter(img, **kwargs):
    """
    Calculate a multi-dimensional median filter that respects NaN values
    and masked arrays.

    :param img: image on which to calculate the median filter
    :param kwargs: additional arguments to ndimage.generic_filter
        Note that either size or footprint must be defined. size gives the shape
        that is taken from the input array, at every element position, to define
        the input to the filter function. footprint is a boolean array that
        specifies (implicitly) a shape, but also which of the elements within
        this shape will get passed to the filter function. Thus size=(n,m) is
        equivalent to footprint=np.ones((n,m)). We adjust size to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and size is 2, then the actual size used is (2,2,2).
    :type img: array-like

    :returns filtered: Filtered array of same shape as input.
    """
    # set up the wrapper function to call generic filter
    @jit_filter_function
    def nanmed(a):
        return np.nanmedian(a)
    
    return ndimage.filters.generic_filter(img, nanmed, **kwargs)
    

def generate_panchrome(imgname, outname=None, out_dir='.', interactive=False):
    if outname is None:
        outname = imgname + '_B8.TIF'

    B5 = GeoImg(imgname + '_B5.TIF')
    B4 = GeoImg(imgname + '_B4.TIF')
    B3 = GeoImg(imgname + '_B3.TIF')
    B2 = GeoImg(imgname + '_B2.TIF')

    B8sim = 0.45 * B4.img + 0.2 * B3.img + 0.25 * B2.img + 0.1 * B5.img
    B8 = B4.copy(new_raster=B8sim)
    B8.write(outname, out_folder=out_dir)

    if interactive:
        return B8


def composite_raster(band1name, band2name, band3name, outname, out_dir='.', in_dir='.', driver='GTiff'):
    band1 = GeoImg(band1name, in_dir=in_dir)
    band2 = GeoImg(band2name, in_dir=in_dir)
    band3 = GeoImg(band3name, in_dir=in_dir)

    driver = gdal.GetDriverByName(driver)

    ncols = band1.npix_x
    nrows = band1.npix_y
    nband = 3
    datatype = band1.gd.GetRasterBand(1).DataType

    out = driver.Create(out_dir + os.path.sep + outname, ncols, nrows, nband, datatype)

    out.SetGeoTransform(band1.gt)
    out.SetProjection(band1.proj_wkt)

    out.GetRasterBand(1).WriteArray(band1.gd.ReadAsArray())
    out.GetRasterBand(2).WriteArray(band2.gd.ReadAsArray())
    out.GetRasterBand(3).WriteArray(band3.gd.ReadAsArray())

    for i in range(3):
        out.GetRasterBand(i+1).FlushCache()


def write_landsat_rgb(scenename, outname, out_dir='.', in_dir='.', driver='GTiff'):
    # takes the scenename, finds bands 1-3 (tm, etm), or 2-4 (oli), then writes outfilename.TIF, the RGB composite.
    sensor = parse_lsat_scene(scenename)[0]

    if (sensor == 'LC8' or sensor == 'LO8'):
        band1name = scenename + '_B4.TIF'
        band2name = scenename + '_B3.TIF'
        band3name = scenename + '_B2.TIF'
    else:
        band1name = scenename + '_B3.TIF'
        band2name = scenename + '_B2.TIF'
        band3name = scenename + '_B1.TIF'

    composite_raster(band1name, band2name, band3name, outname, out_dir, in_dir, driver)


def find_peaks(image, neighborhood_size=5, threshold=1500):
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    ij = np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects+1)))

    return ij


def create_mask_from_shapefile(geoimg, shapefile, buffer=None):
    """
    Create a boolean mask representing the polygons in a shapefile.

    :param geoimg: input GeoImg to pull spatial extents from
    :param shapefile: path to polygon shapefile
    :type geoimg: pybob.GeoImg
    :type shapefile: str

    :returns mask: boolean array corresponding to rasterized polygons, of same shape as geoimg.img
    """
    # load shapefile, rasterize to raster's extent
    masksource = ogr.Open(shapefile)
    inlayer = masksource.GetLayer()

    if buffer is not None:
        source_srs = inlayer.GetSpatialRef()
        target_srs = geoimg.spatialReference
        transform = osr.CoordinateTransformation(source_srs, target_srs)

        drv = ogr.GetDriverByName('Memory')
        dst_ds = drv.CreateDataSource('out')
        bufflayer = dst_ds.CreateLayer('', geom_type=ogr.wkbPolygon, srs=target_srs)
        feature_def = bufflayer.GetLayerDefn()

        for feature in inlayer:
            geom = feature.GetGeometryRef()
            _ = geom.Transform(transform)
            geom_buff = geom.Buffer(buffer)
            out_feature = ogr.Feature(feature_def)
            out_feature.SetGeometry(geom_buff)
            bufflayer.CreateFeature(out_feature)
            out_feature = None
        masklayer = bufflayer
    else:
        masklayer = inlayer
    # masksrs = masklayer.GetSpatialRef()

    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    masktarget.SetProjection(geoimg.proj_wkt)
    masktarget.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(masktarget, [1], masklayer)
    mask = masktarget.GetRasterBand(1).ReadAsArray()
    mask[mask != 0] = 1

    return mask == 1


def rasterize_polygons(geoimg, shapefile, burn_handle=None, dtype=gdal.GDT_Int16):
    """
    Create rasterized polygons given a GeoImg and a shapefile. Useful for creating an index raster with corresponding
    to polygon IDs.

    :param geoimg: input GeoImg to pull spatial extents from.
    :param shapefile: path to polygon shapefile
    :param burn_handle: field to pull values to rasterize. Default looks for the FID field in the shapefile.
    :param dtype: gdal datatype of rasterized layer. Default is gdal.GDT_Int16.
    :type geoimg: pybob.GeoImg
    :type shapefile: str
    :type burn_handle: str
    :type dtype: gdal.GDT

    :returns rasterized, inds: rasterized polygon array and values corresponding to polygons that were rasterized.
    """
    polysource = ogr.Open(shapefile, 0)  # read-only
    polylayer = polysource.GetLayer()

    if burn_handle is None:
        if polylayer.GetFIDColumn() == '':
            polylayer = add_fid_column(polylayer)
            burn_handle = 'TmpFID'
        else:
            burn_handle = polylayer.GetFIDColumn()

    target = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, dtype)
    target.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    target.SetProjection(geoimg.proj_wkt)
    target.GetRasterBand(1).Fill(-1)

    gdal.RasterizeLayer(target, [1], polylayer, options=["Attribute={}".format(burn_handle)])
    thisrast = target.GetRasterBand(1).ReadAsArray()

    thesevals = [feat.GetField(burn_handle) for feat in polylayer if feat.GetField(burn_handle) in np.unique(thisrast)]

    return thisrast, np.array(thesevals)


def add_fid_column(layer):
    tmpdrv = ogr.GetDriverByName('MEMORY')
    tmpsrc = tmpdrv.CreateDataSource('memData')
    tmpdrv.Open('memData', 1)
    newlayer = tmpsrc.CopyLayer(layer, '', ['OVERWRITE=YES'])

    fid_column = ogr.FieldDefn('TmpFID', ogr.OFTInteger)
    newlayer.CreateField(fid_column)

    for feat in newlayer:
        feat.SetField('TmpFID', feat.GetFID())

    return newlayer


def get_common_xy(bbox, img1, img2):

    if (bbox is None):
        #     find common x,y (using original UTM or other geo cartesian coordinates) box between images
        max_x = min([img1.xmax, img2.xmax])
        max_y = min([img1.ymax, img2.ymax])
        min_x = max([img1.xmin, img2.xmin])
        min_y = max([img1.ymin, img2.ymin])
    else:  # read area from command line (still UTM etc.)
        max_x = bbox[2]
        max_y = bbox[3]
        min_x = bbox[0]
        min_y = bbox[1]

    return max_x, max_y, min_x, min_y


def calc_glcm_params(img):
    glcmcontr = np.zeros(img.shape)
    glcmdissim = np.zeros(img.shape)
    glcmhomog = np.zeros(img.shape)
    glcmenergy = np.zeros(img.shape)
    glcmcorrel = np.zeros(img.shape)
    glcmASM = np.zeros(img.shape)

    # have to make sure that the raster is uint16 or uint8
    tmpimg = img.astype(np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i < 3) or (j < 3):
                continue
            if (i > (img.shape[0] - 4)) or (j > (img.shape[1] - 4)):
                continue
            glcm_window = tmpimg[i-3:i+4, j-3:j+4]
            glcm = greycomatrix(glcm_window, [1], [0], symmetric=True, normed=True)

            glcmcontr[i, j] = greycoprops(glcm, 'contrast')
            glcmdissim[i, j] = greycoprops(glcm, 'dissimilarity')
            glcmhomog[i, j] = greycoprops(glcm, 'homogeneity')
            glcmenergy[i, j] = greycoprops(glcm, 'energy')
            glcmcorrel[i, j] = greycoprops(glcm, 'correlation')
            glcmASM[i, j] = greycoprops(glcm, 'ASM')
            glcm = None
            glcm_window = None

    return glcmcontr, glcmdissim, glcmhomog, glcmenergy, glcmcorrel, glcmASM


def get_nicely_shaped_img(img, overlap):
    # get the new shape of the image
    new_cols = round_down(img.img.shape[1], overlap)
    new_rows = round_down(img.img.shape[0], overlap)

    new_img = img.img[0:new_rows, 0:new_cols]
    new_exts = img.ij2xy((new_rows, new_cols))

    if img.dy < 0:
        new_ext = [img.xmin, new_exts[0], new_exts[1], img.ymax]
    else:
        new_ext = [img.xmin, new_exts[0], img.ymin, new_exts[1]]

    return img.copy(new_raster=new_img, new_extent=new_ext)


def splitter(img, nblocks, overlap=0):
    blocks = []

    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    new_width = int(img.shape[1] / nblocks[1])
    new_height = int(img.shape[0] / nblocks[0])

    for j in range(nblocks[1]):
        for i in range(nblocks[0]):
            lind = max(0, j*new_width - overlap)
            rind = min(img.shape[1], (j+1)*new_width + overlap)

            tind = max(0, i*new_height - overlap)
            bind = min(img.shape[0], (i+1)*new_height + overlap)

            blocks.append(img[tind:bind, lind:rind])

    return blocks


def stitcher(outputs, nblocks, overlap=0):
    stitched_arrays = []  # create an empty list to put our stitched results in
    outarrays = []  # same

    # check if nblocks is a scalar or not.
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])

    for out in range(len(outputs[0])):
        stitched_arrays.append([])
        outarrays.append([])
        for j in range(nblocks[0]):
            stitched_arrays[out].append(outputs[j*nblocks[0]][out][overlap:-overlap, overlap:-overlap])
            for i in range(1, nblocks[1]):
                outind = i + j*nblocks[0]
                stitched_arrays[out][j] = np.vstack((stitched_arrays[out][j],
                                                     outputs[outind][out][overlap:-overlap, overlap:-overlap]))
        stitched_arrays[out] = np.hstack(stitched_arrays[out])
        outarrays[out] = np.zeros((stitched_arrays[out].shape[0]+2*overlap, stitched_arrays[out].shape[1]+2*overlap))
        outarrays[out][overlap:-overlap, overlap:-overlap] = stitched_arrays[out]

    return outarrays


def parallel_img_proc(fun, funargs, nproc=2):
    if nproc > mp.cpu_count():
        print("{} cores specified, but I can only find \
               {} cores on this machine. I'll use those.".format(nproc, mp.cpu_count()))

    pool = mp.Pool(nproc)
    outputs = pool.map(fun, funargs)
    pool.close()

    return outputs


def reshape_geoimg(fname, xr, yr, rescale=True):
    ds = gdal.Warp('', fname, xRes=xr, yRes=yr, format='VRT', resampleAlg=gdal.GRA_Lanczos)
    resamp = GeoImg(ds)
    if rescale:
        resamp.img = (resamp.img / 256).astype(np.uint8)
    return resamp


def match_hist(img, reference):
    img_eq = match_histograms(img, reference)
    img_eq[img == 0] = 0
    return img_eq.astype(np.uint8)
