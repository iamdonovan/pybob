"""
pybob.hexagon_tools is a collection of tools for working with KH-9 Hexagon imagery.
"""
from __future__ import print_function, division
import os
from glob import glob
import cv2
from skimage.morphology import disk
from skimage.filters import rank
from skimage import exposure
from scipy.interpolate import RectBivariateSpline as RBS
from scipy import ndimage
import numpy as np
import gdal
from shapely.ops import cascaded_union
import geopandas as gpd
import pyvips
from llc import jit_filter_function
import pandas as pd
import lxml.etree as etree
import lxml.builder as builder
from pybob.image_tools import match_hist, reshape_geoimg, create_mask_from_shapefile


######################################################################################################################
# MicMac interfaces - write xml files for MicMac to read
######################################################################################################################
def get_gcp_meas(im_name, meas_name, in_dir, E, nodist=None, gcp_name='GCP'):
    im = gdal.Open(os.path.sep.join([in_dir, im_name]))
    maxj = im.RasterXSize
    maxi = im.RasterYSize

    impts = pd.read_csv(os.path.join(in_dir, meas_name), sep=' ', names=['j', 'i'])
    if nodist is not None:
        impts_nodist = pd.read_csv(os.path.join(in_dir, nodist), sep=' ', names=['j', 'i'])

    this_im_mes = E.MesureAppuiFlottant1Im(E.NameIm(im_name))
    for ind, row in impts.iterrows():
        in_im = 0 < row.j < maxj and 0 < row.i < maxi
        if nodist is not None:
            in_nd = -200 < impts_nodist.j[ind]+200 < maxj and -200 < impts_nodist.i[ind] < maxi+200
            in_im = in_im and in_nd
        if in_im:
            this_mes = E.OneMesureAF1I(
                                E.NamePt('{}{}'.format(gcp_name, ind+1)),
                                E.PtIm('{} {}'.format(row['j'], row['i']))
                            )
            this_im_mes.append(this_mes)
    return this_im_mes


def get_im_meas(gcps, E):
    pt_els = []
    for ind, row in gcps.iterrows():
        this_mes = E.OneMesureAF1I(
                        E.NamePt(row['gcp']),
                        E.PtIm('{} {}'.format(row['im_col'], row['im_row']))
                        )
        pt_els.append(this_mes)
    return pt_els


def generate_measures_files(joined=False):
    i_list = np.arange(22, -1, -1)
    if not joined:
        j_list = np.arange(0, 24)
    else:
        j_list = np.arange(0, 47)

    J, I = np.meshgrid(np.arange(0, j_list.size), np.arange(0, i_list.size))
    gcp_names = list(zip(I[0, :], J[0, :]))
    for i in range(1, i_list.size):
        gcp_names.extend(list(zip(I[i, :], J[i, :])))

    JJ, II = np.meshgrid(np.round(j_list).astype(int), np.round(i_list).astype(int))
    ij = list(zip(II[0, :], JJ[0, :]))
    for i in np.arange(1, i_list.size):
        ij.extend(list(zip(II[i, :], JJ[i, :])))
    ij = 10 * np.array(ij)

    E = builder.ElementMaker()
    ImMes = E.MesureAppuiFlottant1Im(E.NameIm('Glob'))
    SpGlob = E.SetPointGlob()

    gcp_df = pd.DataFrame()
    with open('id_fiducial.txt', 'w') as f:
        for i, ind in enumerate(gcp_names):
            row, col = ind
            gcp_name = 'GCP_{}_{}'.format(row, col)
            gcp_df.loc[i, 'gcp'] = gcp_name
            gcp_df.loc[i, 'im_row'] = ij[i, 0]
            gcp_df.loc[i, 'im_col'] = ij[i, 1]

            pt_glob = E.PointGlob(E.Type('eNSM_Pts'),
                                  E.Name(gcp_name),
                                  E.LargeurFlou('0'),
                                  E.NumAuto('0'),
                                  E.SzRech('-1'))
            SpGlob.append(pt_glob)
            print(gcp_name, file=f)

    tree = etree.ElementTree(SpGlob)
    tree.write('Tmp-SL-Glob.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")
    pt_els = get_im_meas(gcp_df, E)
    for p in pt_els:
        ImMes.append(p)

    outxml = E.SetOfMesureAppuisFlottants(ImMes)
    tree = etree.ElementTree(outxml)
    tree.write('MeasuresCamera.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")

    with open('id_fiducial.txt', 'w') as f:
        for gcp in gcp_names:
            row, col = gcp
            print('GCP_{}_{}'.format(row, col), file=f)


######################################################################################################################
# image filtering tools
######################################################################################################################
@jit_filter_function
def nanstd(a):
    return np.nanstd(a)


def cross_template(shape, width=3):
    if isinstance(shape, int):
        rows = shape
        cols = shape
    else:
        rows, cols = shape
    half_r = int((rows-1)/2)
    half_c = int((cols-1)/2)
    half_w = int((width-1)/2)

    cross = np.zeros((rows, cols))
    cross[half_r-half_w-1:half_r+half_w+2:width+1, :] = 2
    cross[:, half_c-half_w-1:half_c+half_w+2:width+1] = 2

    cross[half_r-half_w:half_r+half_w+1, :] = 1
    cross[:, half_c-half_w:half_c+half_w+1] = 1
    return cross


def cross_filter(img, cross):
    cross_edge = cross == 2 
    cross_cent = cross == 1 
    edge_std = ndimage.filters.generic_filter(highpass_filter(img), nanstd, footprint=cross_edge) 
    cent_std = ndimage.filters.generic_filter(highpass_filter(img), nanstd, footprint=cross_cent) 
    return np.where(np.logical_and(edge_std != 0, cent_std != 0), cent_std / edge_std, 2) 


def make_template(img, pt, half_size):
    nrows, ncols = img.shape
    row, col = np.round(pt).astype(int)
    left_col = max(col - half_size, 0)
    right_col = min(col + half_size, ncols)
    top_row = max(row - half_size, 0)
    bot_row = min(row + half_size, nrows)
    row_inds = [row - top_row, bot_row - row]
    col_inds = [col - left_col, right_col - col]
    return img[top_row:bot_row+1, left_col:right_col+1], row_inds, col_inds


def find_match(img, template):
    img_eq = rank.equalize(img, selem=disk(100))
    # res = cross_filter(img_eq, template)
    res = cv2.matchTemplate(img_eq, template, cv2.TM_CCORR_NORMED)
    i_off = (img.shape[0] - res.shape[0])/2
    j_off = (img.shape[1] - res.shape[1])/2
    minval, _, minloc, _ = cv2.minMaxLoc(res)
    # maxj, maxi = maxloc
    minj, mini = minloc
    sp_delx, sp_dely = get_subpixel(res)
    # sp_delx, sp_dely = 0, 0
    return res, mini + i_off + sp_dely, minj + j_off + sp_delx


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


######################################################################################################################
# GCP matching tools
######################################################################################################################
def get_footprint_mask(shpfile, geoimg, filelist, fprint_out=False):
    imlist = [im.split('OIS-Reech_')[-1].split('.tif')[0] for im in filelist]
    footprints_shp = gpd.read_file(shpfile)
    fp = footprints_shp[footprints_shp.ID.isin(imlist)]
    fprint = cascaded_union(fp.to_crs(epsg=geoimg.epsg).geometry.values[1:-1]).minimum_rotated_rectangle
    tmp_gdf = gpd.GeoDataFrame(columns=['geometry'])
    tmp_gdf.loc[0, 'geometry'] = fprint
    tmp_gdf.crs = {'init': 'epsg:{}'.format(geoimg.epsg)}
    tmp_gdf.to_file('tmp_fprint.shp')

    maskout = create_mask_from_shapefile(geoimg, 'tmp_fprint.shp')

    for f in glob('tmp_fprint.*'):
        os.remove(f)
    if fprint_out:
        return maskout, fprint
    else:
        return maskout


def get_initial_transformation(img1, img2, landmask=None, footmask=None):
    im2_lowres = reshape_geoimg(img2, 800, 800)

    im2_eq = match_hist(im2_lowres.img, np.array(img1))
    im1_mask = 255 * np.ones(img1.shape, dtype=np.uint8)
    im1_mask[img1 == 0] = 0  # nodata from ortho

    im2_mask = 255 * np.ones(im2_eq.shape, dtype=np.uint8)
    im2_mask[im2_eq == 0] = 0
    if landmask is not None:
        mask_ = create_mask_from_shapefile(im2_lowres, landmask)
        im2_mask[~mask_] = 0
    if footmask is not None:
        mask_ = get_footprint_mask(footmask, im2_lowres, glob('OIS*.tif'))
        im2_mask[~mask_] = 0

    kp, des, matches = get_matches(img1, im2_eq, mask1=im1_mask, mask2=im2_mask)
    src_pts = np.float32([kp[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    aff_matrix, good_mask = cv2.estimateAffine2D(src_pts, dst_pts, ransacReprojThreshold=25)
    # check that the transformation was successful by correlating the two images.
    im1_tfm = cv2.warpAffine(img1, aff_matrix, (im2_lowres.img.shape[1], im2_lowres.img.shape[0]))
    im1_pad = np.zeros(np.array(im1_tfm.shape)+2, dtype=np.uint8)
    im1_pad[1:-1, 1:-1] = im1_tfm
    res = cv2.matchTemplate(im2_eq, im1_pad, cv2.TM_CCORR_NORMED)
    print(res[1,1])
    success = res[1, 1] > 0.6

    return aff_matrix, success, im2_eq.shape


def get_matches(img1, img2, mask1=None, mask2=None):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1.astype(np.uint8), mask=mask1)
    kp2, des2 = orb.detectAndCompute(img2.astype(np.uint8), mask=mask2)

    flann_idx = 6
    index_params = dict(algorithm=flann_idx, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)
    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append(m[0])

    return (kp1, kp2), (des1, des2), matches


def find_gcp_match(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    i_off = (img.shape[0] - res.shape[0]) / 2
    j_off = (img.shape[1] - res.shape[1]) / 2
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    maxj, maxi = maxloc
    sp_delx, sp_dely = get_subpixel(res, how='max')

    return res, maxi + i_off + sp_dely, maxj + j_off + sp_delx


######################################################################################################################
# image writing
######################################################################################################################
def join_halves(img, overlap, indir='.', outdir='.', color_balance=True):
    """
    Join scanned halves of KH-9 image into one, given a common overlap point.
    
    :param img: KH-9 image name (i.e., DZB1215-500454L001001) to join. The function will look for open image halves
        img_a.tif and img_b.tif, assuming 'a' is the left-hand image and 'b' is the right-hand image.
    :param overlap: Image coordinates for a common overlap point, in the form [x1, y1, x2, y2]. Best results tend to be
        overlaps toward the middle of the y range. YMMV.
    :param indir: Directory containing images to join ['.']
    :param outdir: Directory to write joined image to ['.']
    :param color_balance: Attempt to color balance the two image halves before joining [True].

    :type img: str
    :type overlap: array-like
    :type indir: str
    :type outdir: str
    :type color_balance: bool
    """

    left = pyvips.Image.new_from_file(os.path.sep.join([indir, '{}_a.tif'.format(img)]), memory=True)
    right = pyvips.Image.new_from_file(os.path.sep.join([indir, '{}_b.tif'.format(img)]), memory=True)
    outfile = os.path.sep.join([outdir, '{}.tif'.format(img)])

    if len(overlap) < 4:
        x1, y1 = overlap
        if x1 < 0:
            join = left.merge(right, 'horizontal', x1, y1)
        else:
            join = right.merge(left, 'horizontal', x1, y1)

        join.write_to_file(outfile)
    else:
        x1, y1, x2, y2 = overlap

        join = left.mosaic(right, 'horizontal', x1, y1, x2, y2, mblend=0)
        if color_balance:
            balance = join.globalbalance(int_output=True)
            balance.write_to_file(outfile)
        else:
            join.write_to_file(outfile)

    return
