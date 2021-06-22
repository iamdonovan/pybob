"""
pybob.image_tools is a collection of tools related to working with images.
"""
import os
import numpy as np
import cv2
from osgeo import gdal, ogr, osr
import multiprocessing as mp
from shapely.geometry import Point
from numba import jit
from scipy.interpolate import RectBivariateSpline as RBS
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max
from skimage.exposure import match_histograms
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import rank
from skimage.morphology import binary_dilation, disk
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
    @jit(nopython=True)
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


def make_template(img, pt, half_size):
    nrows, ncols = img.shape
    row, col = np.round(pt).astype(int)
    left_col = max(col - half_size, 0)
    right_col = min(col + half_size, ncols)
    top_row = max(row - half_size, 0)
    bot_row = min(row + half_size, nrows)
    row_inds = [row - top_row, bot_row - row]
    col_inds = [col - left_col, right_col - col]
    template = img[top_row:bot_row+1, left_col:right_col+1].copy()
    return template, row_inds, col_inds


def sliding_window_filter(img_shape, pts_df, winsize, stepsize=None, mindist=2000, how='z_corr', sort_asc=True):
    if stepsize is None:
        stepsize = winsize / 2

    out_inds = []
    out_pts = []

    for x_ind in np.arange(stepsize, img_shape[1], winsize):
        for y_ind in np.arange(stepsize, img_shape[0], winsize):
            min_x = x_ind - winsize / 2
            max_x = x_ind + winsize / 2
            min_y = y_ind - winsize / 2
            max_y = y_ind + winsize / 2
            samp_ = pts_df.loc[np.logical_and.reduce([pts_df.match_j > min_x,
                                                      pts_df.match_j < max_x,
                                                      pts_df.match_i > min_y,
                                                      pts_df.match_i < max_y])].copy()
            if samp_.shape[0] == 0:
                continue
            samp_.sort_values(how, ascending=sort_asc, inplace=True)
            if len(out_inds) == 0:
                best_ind = samp_.index[0]
                best_pt = Point(samp_.loc[best_ind, ['match_j', 'match_i']].values)

                out_inds.append(best_ind)
                out_pts.append(best_pt)
            else:
                for ind, row in samp_.iterrows():
                    this_pt = Point(row[['match_j', 'match_i']].values)
                    this_min_dist = np.array([this_pt.distance(pt) for pt in out_pts]).min()
                    if this_min_dist > mindist:
                        out_inds.append(ind)
                        out_pts.append(this_pt)

    return np.array(out_inds)


def highpass_filter(img):
    v = img.copy()
    v[np.isnan(img)] = 0
    vv = ndimage.gaussian_filter(v, 3)

    w = 0 * img.copy() + 1
    w[np.isnan(img)] = 0
    ww = ndimage.gaussian_filter(w, 3)

    tmplow = vv / ww
    tmphi = img - tmplow
    return tmphi


def crippen_filter(img, dtype=np.uint8, add_val=128, scan_axis=0):
    """
    Run the de-striping filter proposed by Crippen (Photogramm Eng Remote Sens 55, 1989) on an image.

    :param img: image to de-stripe.
    :param dtype: original datatype of image
    :param add_val: constant value to add to keep image within the original bit-depth.
    :param scan_axis: array axis along which the image is scanned. For Landsat (in the original image geometry),
     this is along the 0 (row) axis. For scanned historical photos, this is most likely the 1 (column) axis.

    :returns: filt_img: the filtered image.
    """
    assert scan_axis in [0, 1], "scan_axis corresponds to image axis of scan direction [0, 1]"
    if scan_axis == 0:
        k1 = np.ones((1, 101))
        k2 = np.ones((33, 1))
        k3 = np.ones((1, 31))
    else:
        k1 = np.ones((101, 1))
        k2 = np.ones((1, 33))
        k3 = np.ones((31, 1))

    F1 = rank.mean(img, selem=k1)
    F2 = F1 - rank.mean(F1, selem=k2) + add_val
    F3 = rank.mean(F2, selem=k3)

    outimg = img.astype(np.float32) - F3 + add_val
    outimg[outimg > np.iinfo(dtype).max] = np.iinfo(dtype).max
    outimg[outimg < np.iinfo(dtype).min] = np.iinfo(dtype).min

    return outimg.astype(dtype)


def find_match(img, template, method=cv2.TM_CCORR_NORMED):
    res = cv2.matchTemplate(img, template, method)
    i_off = (img.shape[0] - res.shape[0]) / 2
    j_off = (img.shape[1] - res.shape[1]) / 2
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    maxj, maxi = maxloc
    sp_delx, sp_dely = get_subpixel(res, how='max')

    return res, maxi + i_off + sp_dely, maxj + j_off + sp_delx


def get_subpixel(res, how='min'):
    assert how in ['min', 'max'], "have to choose min or max"

    mgx, mgy = np.meshgrid(np.arange(-1, 1.01, 0.1), np.arange(-1, 1.01, 0.1), indexing='xy')  # sub-pixel mesh

    if how == 'min':
        peakval, _, peakloc, _ = cv2.minMaxLoc(res)
        mml_ind = 2
    else:
        _, peakval, _, peakloc = cv2.minMaxLoc(res)
        mml_ind = 3

    rbs_halfsize = 3  # size of peak area used for spline for subpixel peak loc
    rbs_order = 4    # polynomial order for subpixel rbs interpolation of peak location

    if((np.array([n-rbs_halfsize for n in peakloc]) >= np.array([0, 0])).all()
                & (np.array([(n+rbs_halfsize) for n in peakloc]) < np.array(list(res.shape))).all()):
        rbs_p = RBS(range(-rbs_halfsize, rbs_halfsize+1), range(-rbs_halfsize, rbs_halfsize+1),
                    res[(peakloc[1]-rbs_halfsize):(peakloc[1]+rbs_halfsize+1),
                        (peakloc[0]-rbs_halfsize):(peakloc[0]+rbs_halfsize+1)],
                    kx=rbs_order, ky=rbs_order)

        b = rbs_p.ev(mgx.flatten(), mgy.flatten())
        mml = cv2.minMaxLoc(b.reshape(21, 21))
        # mgx,mgy: meshgrid x,y of common area
        # sp_delx,sp_dely: subpixel delx,dely
        sp_delx = mgx[mml[mml_ind][0], mml[mml_ind][1]]
        sp_dely = mgy[mml[mml_ind][0], mml[mml_ind][1]]
    else:
        sp_delx = 0.0
        sp_dely = 0.0
    return sp_delx, sp_dely


def do_match(prim, second, mask, pt, tmpl_size, search_size, highpass):
    _i, _j = pt
    submask, _, _ = make_template(mask, pt, tmpl_size)
    if np.count_nonzero(submask) / submask.size < 0.05:
        return (-1, -1), -1, -1
    # testchip, _, _ = imtools.make_template(rough_tfm, (pt[1], pt[0]), 40)
    # dst_chip, _, _ = imtools.make_template(prim.img, (pt[1], pt[0]), 200)
    testchip, _, _ = make_template(prim, pt, tmpl_size)
    if search_size is None:
        dst_chip = second
    else:
        dst_chip, rows, cols = make_template(second, pt, search_size)

    if np.count_nonzero(dst_chip) / dst_chip.size < 0.1:
        return (-1, -1), -1, -1

    testchip[np.isnan(testchip)] = 0
    dst_chip[np.isnan(dst_chip)] = 0
    if highpass:
        test = highpass_filter(testchip)
        dest = highpass_filter(dst_chip)
    else:
        test = testchip
        dest = dst_chip

    testmask = binary_dilation(testchip == 0, selem=disk(8))
    destmask = binary_dilation(dst_chip == 0, selem=disk(8))

    test[testmask] = np.random.rand(test.shape[0], test.shape[1])[testmask]
    dest[destmask] = np.random.rand(dest.shape[0], dest.shape[1])[destmask]

    corr_res, this_i, this_j = find_match(dest.astype(np.float32), test.astype(np.float32))
    peak_corr = cv2.minMaxLoc(corr_res)[1]

    pks = peak_local_max(corr_res, min_distance=5, num_peaks=2)
    this_z_corrs = []
    for pk in pks:
        max_ = corr_res[pk[0], pk[1]]
        this_z_corrs.append((max_ - corr_res.mean()) / corr_res.std())
    dz_corr = max(this_z_corrs) / min(this_z_corrs)
    z_corr = max(this_z_corrs)

    # if the correlation peak is very high, or very unique, add it as a match
    # out_i, out_j = this_i - 200 + pt[1], this_j - 200 + pt[0]
    if search_size is not None:
        out_i, out_j = this_i - min(rows) + _i, this_j - min(cols) + _j
    else:
        out_i, out_j = this_i, this_j

    return (out_j, out_i), z_corr, peak_corr


def gridded_matching(prim, second, mask, spacing, tmpl_size=25, search_size=None, highpass=False):
    # for each of these pairs (src, dst), find the precise subpixel match (or not...)

    # if search_size is None:
    jj = np.arange(tmpl_size, spacing * np.ceil((prim.shape[1]-tmpl_size) / spacing) + 1, spacing).astype(int)
    ii = np.arange(tmpl_size, spacing * np.ceil((prim.shape[0]-tmpl_size) / spacing) + 1, spacing).astype(int)
    # else:
    #     jj = np.arange(search_size, search_size * np.ceil((prim.shape[1]-search_size)/search_size)+1, spacing)
    #     ii = np.arange(search_size, search_size * np.ceil((prim.shape[0]-search_size)/search_size)+1, spacing)
    J, I = np.meshgrid(jj.astype(int), ii.astype(int))

    search_pts = np.concatenate([I.reshape(-1, 1), J.reshape(-1, 1)], axis=1)
    match_pts = -1 * np.ones(search_pts.shape)
    z_corrs = -1 * np.ones((search_pts.shape[0], 1))
    peak_corrs = -1 * np.ones((search_pts.shape[0], 1))

    for ind, pt in enumerate(search_pts):
        try:
            (out_j, out_i), z_corr, peak_corr = do_match(prim, second, mask, pt,
                                                         tmpl_size, search_size, highpass)
            z_corrs[ind] = z_corr
            peak_corrs[ind] = peak_corr
            match_pts[ind] = np.array([out_j, out_i])
        except:
            continue

    return search_pts, np.array(match_pts), np.array(peak_corrs), np.array(z_corrs)


def stretch_image(img, scale=(0,1), mult=255, outtype=np.uint8):
    maxval = np.nanquantile(img, max(scale))
    minval = np.nanquantile(img, min(scale))

    img[img > maxval] = maxval
    img[img < minval] = minval

    return (mult * (img - minval) / (maxval - minval)).astype(outtype)

