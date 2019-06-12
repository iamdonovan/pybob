"""
pybob.hexagon_tools is a collection of tools for working with KH-9 Hexagon imagery.
"""
from __future__ import print_function, division
import os
# import errno
# import argparse
import cv2
# import multiprocessing as mp
# from functools import partial
# from scipy.ndimage.filters import median_filter
# from skimage.io import imsave
# from skimage.morphology import disk
# from skimage.filters import rank
from scipy.interpolate import RectBivariateSpline as RBS
# import skimage.transform as tf
from scipy import ndimage
import numpy as np
# import matplotlib.pyplot as plt
import gdal
import pyvips
# from numba import jit
from llc import jit_filter_function
import pandas as pd
# import lxml.etree as etree
# import lxml.builder as builder


######################################################################################################################
# MicMac interfaces - write xml files for MicMac to read
######################################################################################################################
def get_gcp_meas(im_name, in_dir, E):
    im = gdal.Open(os.path.sep.join([in_dir, im_name]))
    maxJ = im.RasterXSize
    maxI = im.RasterYSize
    
    impts = pd.read_csv(os.path.sep.join([in_dir, 'GCPs_{}.txt'.format(os.path.splitext(im_name)[0])]),
                        sep = ' ', names = ['j', 'i'])
    
    this_im_mes = E.MesureAppuiFlottant1Im(E.NameIm(im_name))
    for ind, row in impts.iterrows():
        if row['j'] < maxJ and row['i'] < maxI:
            this_mes = E.OneMesureAF1I(
                                E.NamePt('GCP{}'.format(ind)),
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


######################################################################################################################
# image filtering tools
######################################################################################################################
@jit_filter_function
def nanstd(a):
    return np.nanstd(a)


def highpass_filter(img):
    tmplow = ndimage.gaussian_filter(img, 3)  # 3x3 gaussian filter
    return img - tmplow


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


def find_match(img, template, half_size):
# def find_match(chip, template, method=cv2.TM_CCOEFF):
    # res = cv2.matchTemplate(np.float32(img), np.float32(template), cv2.TM_CCORR_NORMED)
    # img_eq = rank.equalize(img, selem=disk(40))
    res = cross_filter(img, template)
    i_off = int((img.shape[0] - res.shape[0])/2)
    j_off = int((img.shape[1] - res.shape[1])/2)
    minval, _, minloc, _ = cv2.minMaxLoc(res)
    # maxj, maxi = maxloc
    minj, mini = minloc
    # sp_delx, sp_dely = get_subpixel(res)
    sp_delx, sp_dely = 0, 0
    return res, mini + i_off + sp_dely, minj + j_off + sp_delx


def get_subpixel(res):
    mgx, mgy = np.meshgrid(np.arange(-1, 1.01, 0.1), np.arange(-1, 1.01, 0.1), indexing='xy')  # sub-pixel mesh

    minval, _, minloc, _ = cv2.minMaxLoc(res)
    rbs_halfsize = 3  # size of peak area used for spline for subpixel peak loc
    rbs_order = 4    # polynomial order for subpixel rbs interpolation of peak location

    if((np.array([n-rbs_halfsize for n in minloc]) >= np.array([0, 0])).all()
                & (np.array([(n+rbs_halfsize) for n in minloc]) < np.array(list(res.shape))).all()):
        rbs_p = RBS(range(-rbs_halfsize, rbs_halfsize+1), range(-rbs_halfsize, rbs_halfsize+1),
                    res[(minloc[1]-rbs_halfsize):(minloc[1]+rbs_halfsize+1),
                        (minloc[0]-rbs_halfsize):(minloc[0]+rbs_halfsize+1)],
                    kx=rbs_order, ky=rbs_order)

        b = rbs_p.ev(mgx.flatten(), mgy.flatten())
        mml = cv2.minMaxLoc(b.reshape(21, 21))
        # mgx,mgy: meshgrid x,y of common area
        # sp_delx,sp_dely: subpixel delx,dely
        sp_delx = mgx[mml[3][0], mml[3][1]]
        sp_dely = mgy[mml[3][0], mml[3][1]]
    else:
        sp_delx = 0.0
        sp_dely = 0.0
    return sp_delx, sp_dely


######################################################################################################################

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
    x1, y1, x2, y2 = overlap
    left = pyvips.Image.new_from_file(os.path.sep.join([indir, '{}_a.tif'.format(img)]), memory=True)
    right = pyvips.Image.new_from_file(os.path.sep.join([indir, '{}_b.tif'.format(img)]), memory=True)

    join = left.mosaic(right, 'horizontal', x1, y1, x2, y2, mblend=0)
    if color_balance:
        balance = join.globalbalance(int_output=True)
        balance.write_to_file('{}.tif'.format(img))
    else:
        join.write_to_file('{}.tif'.format(img))

    return
