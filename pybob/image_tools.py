import gdal
import os
import ogr
import numpy as np
import multiprocessing as mp
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage.feature import greycomatrix, greycoprops
from pybob.bob_tools import parse_lsat_scene, round_down
from pybob.GeoImg import GeoImg


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
    out.SetProjection(band1.proj)

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


def create_mask_from_shapefile(geoimg, shapefile):
    # load shapefile, rasterize to raster's extent
    masksource = ogr.Open(shapefile)
    masklayer = masksource.GetLayer()
    # masksrs = masklayer.GetSpatialRef()

    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    masktarget.SetProjection(geoimg.proj)
    masktarget.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(masktarget, [1], masklayer)
    mask = masktarget.GetRasterBand(1).ReadAsArray()
    mask[mask != 0] = 1

    return mask == 1


def rasterize_polygons(geoimg, shapefile, burn_handle=None, dtype=gdal.GDT_Int16):
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
    target.SetProjection(geoimg.proj)
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
        print "{} cores specified, but I can only find \
               {} cores on this machine. I'll use those.".format(nproc, mp.cpu_count())

    pool = mp.Pool(nproc)
    outputs = pool.map(fun, funargs)
    pool.close()

    return outputs
