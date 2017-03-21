import gdal, sys, os, matplotlib, fiona, copy, random, shapely, ogr, osr, scipy, cv2
import numpy as np, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt, multiprocessing as mp
import scipy.ndimage as ndimage, scipy.ndimage.filters as filters, datetime as dt
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import griddata
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fiona.crs import from_epsg
from shapely.geometry import mapping, Point, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial
from skimage.feature import greycomatrix, greycoprops

def standard_landsat(instring):
    strsplit = instring.split('_')
    if len(strsplit) < 3: # if we only have 1 (or fewer) underscores, it's already fine
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
        elif sensor == '8': # chances are, we won't ever see this one.
            outstring += 'C8'
        outstring += strsplit[0][2:] # this is the path/row information
        # now it gets fun: getting the date right.
        year = strsplit[1][3:7]
        month = strsplit[1][7:9]
        day = strsplit[1][9:]
        doy = str(dt.datetime(int(year), int(month), int(day)).timetuple().tm_yday).zfill(3) # make sure we have 3 digits here
        outstring += year + doy
        # now, spoof the last bits so it's got the right size:    
        outstring += 'XXX01'
        return outstring
    
def doy2mmdd(year, doy, string_out=True, outform='%Y/%m/%d'):
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

def genpanchrome(scenename, outscenename=None, outfolder='.', interactive=False):
    if outscenename == None:
        outscenename = scenename
    
    outfilename = outscenename + "_B8.TIF"    

    B5 = GeoImg( scenename + '_B5.TIF' )
    B4 = GeoImg( scenename + '_B4.TIF' )
    B3 = GeoImg( scenename + '_B3.TIF' )
    B2 = GeoImg( scenename + '_B2.TIF' )

    B8sim = 0.45 * B4.img + 0.2 * B3.img + 0.25 * B2.img + 0.1 * B5.img
    B8 = B4.copy( new_raster=B8sim )
    B8.write(outfolder + os.path.sep + outfilename)

    if interactive:
        return B8

def write_composite_raster(band1name, band2name, band3name, outfilename, out_dir='.', in_dir='.', driver='GTiff'):
    band1 = GeoImg(band1name, in_dir=in_dir)
    band2 = GeoImg(band2name, in_dir=in_dir)
    band3 = GeoImg(band3name, in_dir=in_dir)

    driver = gdal.GetDriverByName(driver)

    ncols = band1.npix_x
    nrows = band1.npix_y
    nband = 3
    datatype = band1.gd.GetRasterBand(1).DataType

    out = driver.Create(out_dir + os.path.sep + outfilename, ncols, nrows, nband, datatype)

    setgeo = out.SetGeoTransform(band1.gt)
    setproj = out.SetProjection(band1.proj)

    write = out.GetRasterBand(1).WriteArray(band1.gd.ReadAsArray())
    write = out.GetRasterBand(2).WriteArray(band2.gd.ReadAsArray())
    write = out.GetRasterBand(3).WriteArray(band3.gd.ReadAsArray())

    for i in range(3):
        out.GetRasterBand(i+1).FlushCache()

def write_rgb(scenename, outfilename, out_dir='.', in_dir='.', driver='GTiff'):
    # takes the scenename, finds bands 1-3 (tm, etm), or 2-4 (oli), then writes outfilename.TIF, the RGB composite.
    
    parsed = parse_lsat_scene(scenename)
    sensor = parsed[0]

    if ( sensor == 'LC8' or sensor == 'LO8' ):
        band1name = scenename + '_B4.TIF'
        band2name = scenename + '_B3.TIF'
        band3name = scenename + '_B2.TIF'
    else:
        band1name = scenename + '_B3.TIF'
        band2name = scenename + '_B2.TIF'
        band3name = scenename + '_B1.TIF'    

    write_composite_raster(band1name, band2name, band3name, outfilename, out_dir, in_dir, driver)

def bin_data(bins, data2bin, bindata):
    digitized = np.digitize(bindata, bins)
    binned = [data2bin[np.logical_and(np.isfinite(bindata), digitized==i)].mean() for i in range(len(bins))]
    return binned

#def write_shapefile(filename, geometry, features, data, fields, EPSG):
#    schema = {'properties': [fields], 'geometry': geometry}

#    output = fiona.open()

#    for i, tmp in enumerate(features):
#        output.write({'properties':}}))

def random_points_in_polygon(poly, npts=1):
    xmin, ymin, xmax, ymax = poly.bounds
    rand_points = []

    for pt in range(npts):
        thisptin = False
        while not thisptin:        
            rand_point = Point(xmin + (random.random() * (xmax-xmin)), ymin + (random.random() * (ymax-ymin)) )
            thisptin = poly.contains(rand_point)
        rand_points.append(rand_point)

    return rand_points

def find_peaks(image, neighborhood_size=5, threshold=1500):
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    ij = np.array(ndimage.center_of_mass(image, labeled, range(1, num_objects+1)))

    return ij

def reproject_layer(inLayer, targetSRS):
    srcSRS = inLayer.GetSpatialRef()
    coordTrans = osr.CoordinateTransformation(srcSRS, targetSRS)

    outLayer = ogr.GetDriverByName('Memory').CreateDataSource('').CreateLayer(inLayer.GetName(), geom_type=inLayer.GetGeomType())
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

def create_mask_from_shapefile(geoimg, shapefile):
    # load shapefile, rasterize to raster's extent
    masksource = ogr.Open(shapefile)
    masklayer = masksource.GetLayer()
    masksrs = masklayer.GetSpatialRef()

    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    setGT = masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    setPR = masktarget.SetProjection(geoimg.proj)
    initBand = masktarget.GetRasterBand(1).Fill(0)
    rast = gdal.RasterizeLayer(masktarget, [1], masklayer)
    mask = masktarget.GetRasterBand(1).ReadAsArray()
    mask[mask != 0] = 1
        
    return mask == 1

def get_common_xy(bbox, img1, img2):

    if (bbox is None):
        #     find common x,y (using original UTM or other geo cartesian coordinates) box between images
        max_x = min([img1.xmax, img2.xmax])
        max_y = min([img1.ymax, img2.ymax])
        min_x = max([img1.xmin, img2.xmin])
        min_y = max([img1.ymin, img2.ymin])
    else: # read area from command line (still UTM etc.)
        max_x = bbox[2]
        max_y = bbox[3]
        min_x = bbox[0]
        min_y = bbox[1]
    
    return max_x, max_y, min_x, min_y

def calc_glcm_params(img):
    glcmcontr  = np.zeros(img.shape)
    glcmdissim = np.zeros(img.shape)
    glcmhomog  = np.zeros(img.shape)
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

            glcmcontr[i,j] = greycoprops(glcm, 'contrast')
            glcmdissim[i,j] = greycoprops(glcm, 'dissimilarity')
            glcmhomog[i,j] = greycoprops(glcm, 'homogeneity')
            glcmenergy[i,j] = greycoprops(glcm, 'energy')
            glcmcorrel[i,j] = greycoprops(glcm, 'correlation')
            glcmASM[i,j] = greycoprops(glcm, 'ASM')
            glcm = None
            glcm_window = None

    return glcmcontr, glcmdissim, glcmhomog, glcmenergy, glcmcorrel, glcmASM

def round_down(num, divisor):
    return num - (num%divisor)

def get_nicely_shaped_img(img, overlap):
    # get the new shape of the image
    new_cols = round_down(img.img.shape[1], overlap)
    new_rows = round_down(img.img.shape[0], overlap)

    new_img  = img.img[0:new_rows, 0:new_cols]
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
    new_width  = int(img.shape[1] / nblocks[1])
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
    stitched_arrays = [] # create an empty list to put our stitched results in 
    outarrays = [] # same

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
                stitched_arrays[out][j] = np.vstack((stitched_arrays[out][j], outputs[outind][out][overlap:-overlap, overlap:-overlap]))
        stitched_arrays[out] = np.hstack(stitched_arrays[out])
        outarrays[out] = np.zeros((stitched_arrays[out].shape[0]+2*overlap, stitched_arrays[out].shape[1]+2*overlap))
        outarrays[out][overlap:-overlap, overlap:-overlap] = stitched_arrays[out]

    return outarrays

def parallel_img_proc(fun, funargs, nproc=2):
    if nproc > mp.cpu_count():
        print "{} cores specified, but I can only find {} cores on this machine. I'll use those.".format( nproc, mp.cpu_count() )
    
    pool = mp.Pool(nproc)
    outputs = pool.map(fun, funargs)
    pool.close()

    return outputs


