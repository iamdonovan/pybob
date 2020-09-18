from glob import glob
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage import transform, exposure
from skimage.measure import ransac
from skimage.feature import corner_harris, corner_peaks, peak_local_max
from skimage import filters
from skimage.morphology import disk, binary_closing
from sklearn.linear_model import LinearRegression
from PIL import Image
import matplotlib.pyplot as plt
import lxml.etree as etree
import lxml.builder as builder
from pybob.bob_tools import mkdir_p
from pybob.ddem_tools import nmad
import sPyMicMac.image_tools as imtools
import sPyMicMac.micmac_tools as mmt
from shapely.geometry.linestring import LineString


def downsample_image(img, fact=4):
    _img = Image.fromarray(img)
    return np.array(_img.resize((np.array(_img.size) / fact).astype(int), Image.LANCZOS))


def get_fit(xdata, ydata):
    lr = LinearRegression()
    xdata = xdata[np.isfinite(ydata)]
    ydata = ydata[np.isfinite(ydata)]

    lr.fit(xdata.reshape(-1, 1), ydata.reshape(-1, 1))
    return lr.coef_[0][0], lr.intercept_[0]


def get_best_fit(xdata, ydata, offset):
    slope, intercept = get_fit(xdata, ydata)
    for i in range(3):
        _xdata = xdata[np.isfinite(ydata)]
        _ydata = ydata[np.isfinite(ydata)]
        diff = _ydata - (_xdata * slope + intercept)
        good = np.abs(diff - np.nanmedian(diff)) < 2 * nmad(diff)
        slope, intercept = get_fit(_xdata[good], _ydata[good])

    return offset + intercept + slope * xdata


def get_corners_midpts(cols, rows, lft, rgt, top, bot):
    _lft = LineString([(lft[0], rows[0]), (lft[-1], rows[-1])])
    _rgt = LineString([(rgt[0], rows[0]), (rgt[-1], rows[-1])])

    _top = LineString([(cols[0], top[0]), (cols[-1], top[-1])])
    _bot = LineString([(cols[0], bot[0]), (cols[-1], bot[-1])])
    ul = _lft.intersection(_top)
    ur = _rgt.intersection(_top)
    ll = _lft.intersection(_bot)
    lr = _rgt.intersection(_bot)

    _new_top = LineString([ul, ur])
    _new_bot = LineString([ll, lr])
    _new_lft = LineString([ul, ll])
    _new_rgt = LineString([ur, lr])
    # print('Left:   {:.2f}'.format(_new_lft.length * .025))
    # print('Right:  {:.2f}'.format(_new_rgt.length * .025))
    # print('Top:    {:.2f}'.format(_new_top.length * .025))
    # print('Bottom: {:.2f}'.format(_new_bot.length * .025))
    md_top = _new_top.interpolate(0.5, normalized=True)
    md_bot = _new_bot.interpolate(0.5, normalized=True)
    md_lft = _new_lft.interpolate(0.5, normalized=True)
    md_rgt = _new_rgt.interpolate(0.5, normalized=True)

    return (ul, ur, lr, ll), (md_lft, md_rgt, md_top, md_bot)


def get_rough_frame(img):
    img_lowres = downsample_image(img, fact=10)
    img_seg = np.zeros(img_lowres.shape)
    img_seg[img_lowres > filters.threshold_local(img_lowres, 101)] = 1
    img_seg = binary_closing(img_seg, selem=disk(1))

    v_sob = filters.sobel_v(img_seg)**2
    h_sob = filters.sobel_h(img_seg)**2

    vert = np.count_nonzero(v_sob > 0.5, axis=0)
    hori = np.count_nonzero(h_sob > 0.5, axis=1)

    xmin = 10 * peak_local_max(vert[:200], num_peaks=2, min_distance=10, threshold_rel=0.3, exclude_border=10).max()
    xmax = 10 * (peak_local_max(vert[-200:], num_peaks=2, min_distance=10,
                                threshold_rel=0.3, exclude_border=10).min() + img_lowres.shape[1] - 200)

    ymin = 10 * (peak_local_max(hori[10:50], num_peaks=1, min_distance=10,
                                threshold_rel=0.3, exclude_border=5).max() + 10)
    ymax = 10 * (peak_local_max(hori[-50:-10], num_peaks=1, min_distance=10, threshold_rel=0.3,
                                exclude_border=5).min() + img_lowres.shape[0] - 50)
    return xmin, xmax, ymin, ymax


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image 
        center = (int(w/2), int(h/2)) 
    if radius is None: # use the smallest distance between the center and image walls 
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w] 
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2) 
    mask = dist_from_center <= radius
    return mask    

# create a 201x201 pixel template with a white square in the upper right quadrant.
template = np.zeros((201, 201), dtype=np.uint8)
template[:101, 101:] = 255

# create a template to approximate the circular corners of the images.
corner = create_circular_mask(340, 340, center=(340, 340), radius=170)

# flip/transpose the templates to match each of the corners/fiducial marks.
# pattern is: middle left, middle right, middle upper, middle lower,
# upper left corner, lower right corner, upper right corner, lower left corner.
templates = [template, np.fliplr(template), template.T, np.fliplr(template).T,
             corner, np.flipud(np.fliplr(corner)), np.fliplr(corner), np.flipud(corner)]

# get a list of images
imlist = glob('AR*.tif')

for i, im in enumerate(imlist):
    print('{} ({}/{})'.format(im, i+1, len(imlist)))

    # try:
    img = imread(im)
    # find the frame in a downsampled image, get the approximate x, y pixels of the edges.
    xmin, xmax, ymin, ymax = get_rough_frame(img)

    # segment the image into bright/dark regions based on the local threshold function from skimage.
    img_seg = np.zeros(img.shape)
    img_seg[img > filters.threshold_local(img, 25)] = 1
    img_seg = binary_closing(img_seg, selem=disk(3))

    # run a vertical/horizontal sobel filter on the segmented image to find vertical/horizontal lines
    v_sob = filters.sobel_v(img_seg)**2
    h_sob = filters.sobel_h(img_seg)**2

    left = []
    right = []
    top = []
    bot = []

    # using the approximate image frame, find a more precise match for the left and right frame edges
    for i in np.arange(50, img.shape[0], 50):
        this_vert = np.count_nonzero(v_sob[i-50:i+1, :] > 0.5, axis=0)
        try:
            lt = peak_local_max(this_vert[xmin-100:xmin+100+1], num_peaks=1).min()
        except ValueError as e:
            lt = np.nan
        try:
            rt = peak_local_max(this_vert[xmax-100:xmax+100+1], num_peaks=1).min()
        except ValueError as e:
            rt = np.nan
        left.append(lt)
        right.append(rt)

    # same as above, but for the top and bottom edges
    for i in np.arange(50, img.shape[1], 50):
        this_hori = np.count_nonzero(h_sob[:, i-50:i+1] > 0.5, axis=1)
        try:
            tp = peak_local_max(this_hori[ymin-100:ymin+100+1], num_peaks=1).min()
        except ValueError as e:
            tp = np.nan

        try:
            bt = peak_local_max(this_hori[ymax-100:ymax+100+1], num_peaks=1).min()
        except ValueError as e:
            bt = np.nan
        top.append(tp)
        bot.append(bt)

    left = np.array(left)
    right = np.array(right)
    top = np.array(top)
    bot = np.array(bot)

    rows = np.arange(50, img.shape[0], 50)
    cols = np.arange(50, img.shape[1], 50)

    # least-squares linear fit to each edge
    top_line = get_best_fit(cols, top-100, ymin)
    bot_line = get_best_fit(cols, bot-100, ymax)

    lft_line = get_best_fit(rows, left-100, xmin)
    rgt_line = get_best_fit(rows, right-100, xmax)

    # plt.imshow(img[::2, ::2], cmap='gray', extent=[0, img.shape[1], img.shape[0], 0])
    # plt.imshow(img, cmap='gray')
    # plt.plot(cols, top_line, 'r')
    # plt.plot(cols, bot_line, 'r')
    # plt.plot(lft_line, rows, 'r')
    # plt.plot(rgt_line, rows, 'r')
    # find the approximate corners and midpoints to do the more precise template matching.
    corners, midpts = get_corners_midpts(cols, rows, lft_line, rgt_line, top_line, bot_line)
    ul, ur, lr, ll = corners
    md_lft, md_rgt, md_top, md_bot = midpts

    search_pts = [(md_lft.y, md_lft.x+40),
                  (md_rgt.y, md_rgt.x-40),
                  (md_top.y+40, md_top.x),
                  (md_bot.y-40, md_bot.x),
                  (ul.y, ul.x),
                  (lr.y, lr.x),
                  (ur.y, ur.x),
                  (ll.y, ll.x)]
    sizes = [125, 125, 125, 125, 250, 250, 250, 250]
    match_pts = []

    for templ, pt, sz in zip(templates, search_pts, sizes):
        subimg, rows, cols = imtools.make_template(img, pt, sz)

        # i've found that this works best by running a high-pass filter (subtract a gaussian blur) over the
        # sub-image and the template
        res, this_i, this_j = imtools.find_gcp_match(imtools.highpass_filter(subimg).astype(np.float32),
                                                     imtools.highpass_filter(templ).astype(np.float32))
        z_corr = res.max() / res.std()

        if res.max() > 0.1:
            match_pts.append((this_i - rows[0] + pt[0],
                              this_j - cols[0] + pt[1]))
        else:
            match_pts.append((np.nan, np.nan))

    match_pts = np.array(match_pts).reshape(-1, 2)

    gcps = pd.DataFrame()
    gcps['gcp'] = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    # these are the coordinates (in mm) of each of the fiducial marks
    gcps['x'] = [7, 231, 119, 119, 6, 232, 232, 6]
    gcps['y'] = [119, 119, 7, 231, 6, 232, 6, 232]
    gcps['im_col'] = match_pts[:, 1]
    gcps['im_row'] = match_pts[:, 0]

    # try to fit an affine transformation model to the matched points, in order to warp the image and
    # try a more refined search.
    init_model = transform.AffineTransform()
    est = init_model.estimate(gcps.loc[np.isfinite(match_pts[:,0]), ['im_col', 'im_row']].values,
                              gcps.loc[np.isfinite(match_pts[:,0]), ['x', 'y']].values / 0.025)
    resids = init_model.residuals(gcps[['im_col', 'im_row']].values, gcps[['x', 'y']].values / 0.025) # divide the x,y
    # by the mm/pixel resolution of the images.
    inliers = resids < 9  # count the number of points that fit within 9 pixels of the predicted location,
    # based on the transformation
    print('{} matches used in transformation'.format(np.count_nonzero(inliers)))
    print('RMS Resid.: {:.4f}'.format(np.sqrt(np.nanmean(resids**2))))

    # warp the image to the new geometry. the size is based on leaving a uniform border around the edge
    img_tfm = transform.warp(img, init_model.inverse, output_shape=[9520, 9520], preserve_range=True, order=5)

    # re-segment the transformed image
    img_seg = np.zeros(img_tfm.shape)
    img_seg[img_tfm > filters.threshold_local(img_tfm, 51)] = 1
    img_seg = binary_closing(img_seg, selem=disk(1))

    # search from the predicted locations in the transformed image, but with a smaller window.
    new_search = gcps[['y', 'x']].values / .025
    sizes = [110, 110, 110, 110, 220, 220, 220, 220]
    new_match = []
    corrs = []

    for templ, pt, sz in zip(templates, new_search, sizes):
        subimg, rows, cols = imtools.make_template(img_seg, pt, sz)
        res, this_i, this_j = imtools.find_gcp_match(imtools.highpass_filter(subimg).astype(np.float32),
                                                     imtools.highpass_filter(templ).astype(np.float32))
        corrs.append(res.max())
        new_match.append((this_i - rows[0] + pt[0],
                          this_j - cols[0] + pt[1]))

    new_match = np.array(new_match).reshape(-1, 2)
    orig_match = init_model.inverse((new_match[:, ::-1]))  # run the back-transformation
    # to find the points in the original image geometry.
    gcps['im_col'] = orig_match[:, 0]
    gcps['im_row'] = orig_match[:, 1]
    gcps['corr'] = corrs

    # the next block takes the first 4 fiducial marks and writes them to an xml file for MicMac to read.
    E = builder.ElementMaker()
    ImMes = E.MesureAppuiFlottant1Im(E.NameIm(im))

    pt_els = mmt.get_im_meas(gcps.loc[:3], E)
    for p in pt_els:
        ImMes.append(p)
    mkdir_p('Ori-InterneScan')

    outxml = E.SetOfMesureAppuisFlottants(ImMes)
    tree = etree.ElementTree(outxml)
    tree.write('Ori-InterneScan/MeasuresIm-' + im + '.xml', pretty_print=True,
               xml_declaration=True, encoding="utf-8")

    # optional: uncomment the following lines to plot a figure that plots each of the fiducial marks.
    # plt.imshow(img[::4, ::4], cmap='gray', extent=[0, img.shape[1], img.shape[0], 0])
    # plt.plot(gcps.im_col, gcps.im_row, 'w*', ms=4)
    # plt.plot(match_pts[:, 1], match_pts[:, 0], 'r.', ms=2)
    # plt.savefig('fid_imgs/{}_fid.png'.format(im.split('.')[0]), bbox_inches='tight', dpi=200)
    # plt.close()

    # except:
    #     print('error in {}'.format(im))
    #     continue
