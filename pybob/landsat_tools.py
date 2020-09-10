import os
import shutil
import yaml
import numpy as np
import osr
import gdal
import pandas as pd
import geopandas as gpd
from glob import glob
from scipy.interpolate import griddata
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString
from skimage.transform import EuclideanTransform, AffineTransform, warp
from skimage.morphology import binary_closing, disk
from skimage.measure import ransac
from pybob.GeoImg import GeoImg
from pybob.bob_tools import mkdir_p
from pybob.ddem_tools import nmad
from pybob.coreg_tools import RMSE
import pybob.image_tools as imtools

#################################
# metadata tools
#################################
def get_gcp_file(path, row):
    pass


def parse_ang_file(fn_ang):
    with open(fn_ang, 'r') as f:
        raw_ang = [l.strip() for l in f.readlines()]

    out_dict = dict()
    for l in raw_ang:
        if l == 'END':
            continue
        lsplit = l.split(' = ')
        if lsplit[0] == 'GROUP':
            this_group = lsplit[1]
            this_dict = dict()
        else:
            if lsplit[0] == 'END_GROUP':
                out_dict[this_group] = this_dict
                continue
            if '=' in l:
                this_name = lsplit[0]
                this_dict[this_name] = lsplit[1]
            else:
                this_dict[this_name] = this_dict[this_name] + ' ' + l

    # now clean the results
    for key in out_dict.keys():
        this_dict = out_dict[key]
        for _k in this_dict.keys():
            if ',' in this_dict[_k]:
                out_dict[key][_k] = np.array([yaml.load(a, Loader=yaml.SafeLoader) for a in this_dict[_k].strip('(').strip(')').split(',')])
            else:
                out_dict[key][_k] = yaml.load(this_dict[_k], Loader=yaml.SafeLoader)

    return out_dict


def get_nominal_altitude(fname):
    sensor_name = fname.split('_')[0]

    alt_dict = {'LM01': 900000,
                'LM02': 900000,
                'LM03': 900000,
                'LM04': 705000,
                'LT04': 705000,
                'LM04': 705000,
                'LT05': 705000,
                'LE07': 705000,
                'LC08': 705000}
    return alt_dict[sensor_name]


def is_descending(row):
    if row <= 122 or row > 246:
        return True
    else:
        return False


#################################
# parameter tools
#################################
def get_earth_radius(latitude):
    mysrs = osr.SpatialReference()
    mysrs.ImportFromEPSG(4326)  # get wgs84 srs to get the earth_radius

    r_major = mysrs.GetSemiMajor()
    r_minor = mysrs.GetSemiMinor()

    num = (r_major**2 * np.cos(np.deg2rad(latitude)))**2 + (r_minor**2 * np.sin(np.deg2rad(latitude)))**2
    den = (r_major * np.cos(np.deg2rad(latitude)))**2 + (r_minor * np.sin(np.deg2rad(latitude)))**2

    return np.sqrt(num / den)


#################################
# rpc tools
#################################
def calculate_rpc_transform(gcps, img, dem):
    mean_j, mean_i = np.array(img.shape) / 2
    J, I = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))

    mean_z = np.nanmean(dem.img[img > 0])

    normJ = (J - mean_j) / mean_j
    normI = (I - mean_i) / mean_i

    s_coeffs = rpc_coeffs_from_gcps(gcps,
                                    ((gcps.match_j.values - mean_j) / mean_j).reshape(-1, 1),
                                    mean_i, mean_j, mean_z)

    l_coeffs = rpc_coeffs_from_gcps(gcps,
                                    ((gcps.match_i.values - mean_i) / mean_i).reshape(-1, 1),
                                    mean_i, mean_j, mean_z)

    s_nom = s_coeffs[0] + s_coeffs[1] * normI + s_coeffs[2] * normJ + s_coeffs[3] * 0
    s_den = 1 + s_coeffs[4] * normI + s_coeffs[5] * normJ + s_coeffs[6] * 0

    samps = (s_nom / s_den) * mean_j + mean_j

    l_nom = l_coeffs[0] + l_coeffs[1] * normI + l_coeffs[2] * normJ + l_coeffs[3] * 0
    l_den = 1 + l_coeffs[4] * normI + l_coeffs[5] * normJ + l_coeffs[6] * 0

    lines = (l_nom / l_den) * mean_i + mean_i

    return samps, lines


def rpc_coeffs_from_gcps(gcps, Y, mean_i, mean_j, mean_z):
    L = ((gcps.src_i.values - mean_i) / mean_i).reshape(-1, 1)
    P = ((gcps.src_j.values - mean_j) / mean_j).reshape(-1, 1)
    H = ((gcps.elevation.values - mean_z) / mean_z).reshape(-1, 1)

    M = np.concatenate([np.ones(Y.shape), L, P, H, -Y * L, -Y * P, -Y * H], axis=1)

    coeffs = np.linalg.inv(M.T.dot(M)).dot(M.T.dot(Y))

    return np.array(coeffs)


def apply_angle_rpc(numerator, denominator, l1t_line, l1t_samp, l1r_line, l1r_samp, height=0):
    eqn_num = numerator[0] + numerator[1] * l1t_line \
                           + numerator[2] * l1t_samp \
                           + numerator[3] * height \
                           + numerator[4] * l1r_line \
                           + numerator[5] * l1t_line * l1t_line \
                           + numerator[6] * l1t_samp * l1t_line \
                           + numerator[7] * l1t_samp * l1t_samp \
                           + numerator[8] * l1r_samp * l1r_line * l1r_line \
                           + numerator[9] * l1r_line * l1r_line * l1r_line

    eqn_den = 1 + denominator[0] * l1t_line \
                + denominator[1] * l1t_samp \
                + denominator[2] * height \
                + denominator[3] * l1r_line \
                + denominator[4] * l1t_line * l1t_line \
                + denominator[5] * l1t_line * l1t_samp \
                + denominator[6] * l1t_samp * l1t_samp \
                + denominator[7] * l1r_samp * l1r_line * l1r_line \
                + denominator[8] * l1r_line * l1r_line * l1r_line

    return eqn_num / eqn_den


def apply_l1r_rpc(numerator, denominator, lines, samps, height=0):
    eqn_num = numerator[0] + numerator[1] * lines \
              + numerator[2] * samps \
              + numerator[3] * height \
              + numerator[4] * lines * samps

    eqn_den = 1 + denominator[0] * lines \
              + denominator[1] * samps \
              + denominator[2] * height \
              + denominator[3] * lines * samps

    return eqn_num / eqn_den


def reverse_l1r_rpc(numerator, denominator, lines, samps, l1r, height=0):
    lines_num = l1r * (1 + denominator[1] + denominator[2] * height) \
                - numerator[0] \
                - numerator[2] * samps \
                - numerator[3] * height

    lines_den = numerator[1] + numerator[4] * samps \
                - l1r * denominator[0] \
                - l1r * denominator[3] * samps

    samps_num = l1r * (1 + denominator[0] * lines + denominator[1] * height) \
                - numerator[0] \
                - numerator[1] * lines \
                - numerator[3] * height

    samps_den = numerator[2] + numerator[4] * lines \
                - l1r * denominator[1] \
                - l1r * denominator[3] * samps

    return (lines_num / lines_den), (samps_num / samps_den)


#################################
# grid tools
#################################
def generate_l1r_dir_grids(metadata, band, direction, dem=None):
    l1t_line, l1t_samp = generate_l1t_grids(metadata, band, direction=direction)
    if dem is not None:
        dem = dem - metadata['BAND{:02d}_DIR{:02d}_MEAN_HEIGHT'.format(band, direction)]
    else:
        dem = 0

    line_rpc = apply_l1r_rpc(metadata['BAND{:02d}_DIR{:02d}_LINE_NUM_COEF'.format(band, direction)],
                             metadata['BAND{:02d}_DIR{:02d}_LINE_DEN_COEF'.format(band, direction)],
                             l1t_line,
                             l1t_samp,
                             height=dem)
    l1r_line = metadata['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][0] + line_rpc

    samp_rpc = apply_l1r_rpc(metadata['BAND{:02d}_DIR{:02d}_SAMP_NUM_COEF'.format(band, direction)],
                             metadata['BAND{:02d}_DIR{:02d}_SAMP_DEN_COEF'.format(band, direction)],
                             l1t_line,
                             l1t_samp,
                             height=dem)
    l1r_samp = metadata['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][1] + samp_rpc

    return l1r_line, l1r_samp


def get_l1r_grids(metadata, band, dem=None):
    l1r_samp = np.zeros((metadata['BAND{:02d}_NUM_L1T_LINES'.format(band)],
                         metadata['BAND{:02d}_NUM_L1T_SAMPS'.format(band)]))
    l1r_line = np.zeros((metadata['BAND{:02d}_NUM_L1T_LINES'.format(band)],
                         metadata['BAND{:02d}_NUM_L1T_SAMPS'.format(band)]))

    for d in range(metadata['BAND{:02d}_NUMBER_OF_DIRECTIONS'.format(band)]):
        lines, samps = generate_l1r_dir_grids(metadata, band, d, dem=dem)

        scan_number = (lines / metadata['BAND{:02d}_LINES_PER_SCAN'.format(band)]).astype(int)
        l1r_line[scan_number % 2 == d] = lines[scan_number % 2 == d]
        l1r_samp[scan_number % 2 == d] = samps[scan_number % 2 == d]

    return l1r_line, l1r_samp


def generate_l1t_grids(metadata, band, offset=True, direction=None):
    img_I = np.arange(0, metadata['BAND{:02d}_NUM_L1T_LINES'.format(band)])
    img_J = np.arange(0, metadata['BAND{:02d}_NUM_L1T_SAMPS'.format(band)])

    JJ, II = np.meshgrid(img_J, img_I)

    if offset:
        if direction is not None:
            l1t_line = II - metadata['BAND{:02d}_DIR{:02d}_MEAN_L1T_LINE_SAMP'.format(band, direction)][0]
            l1t_samp = JJ - metadata['BAND{:02d}_DIR{:02d}_MEAN_L1T_LINE_SAMP'.format(band, direction)][1]
        else:
            l1t_line = II - metadata['BAND{:02d}_MEAN_L1T_LINE_SAMP'.format(band)][0]
            l1t_samp = JJ - metadata['BAND{:02d}_MEAN_L1T_LINE_SAMP'.format(band)][1]
    else:
        l1t_line = II
        l1t_samp = JJ
    return l1t_line, l1t_samp


def get_l1t_grids(img):
    img_I = np.arange(0, img.shape[0])
    img_J = np.arange(0, img.shape[1])

    JJ, II = np.meshgrid(img_J, img_I)

    return II, JJ


def smooth_holes(img):
    smoothed = imtools.nanmedian_filter(img, footprint=disk(3), mode='nearest')
    diff = img - smoothed
    smoothed[np.abs(diff) < 1] = img[np.abs(diff) < 1]
    return smoothed


#################################
# satellite tools
#################################
def get_sat_look_vector(metadata, band, axis, dem=None):
    this_meta = metadata['RPC_BAND{:02d}'.format(band)]

    l1t_line, l1t_samp = generate_l1t_grids(this_meta, band)

    l1r_line, l1r_samp = get_l1r_grids(this_meta, band, dem=dem)

    l1r_line = l1r_line - this_meta['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][0]
    l1r_samp = l1r_samp - this_meta['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][1]

    if dem is None:
        dem = dem - this_meta['BAND{:02d}_MEAN_HEIGHT'.format(band)]
    else:
        dem = 0

    look_vector = apply_angle_rpc(this_meta['BAND{:02d}_SAT_{}_NUM_COEF'.format(band, axis)],
                                  this_meta['BAND{:02d}_SAT_{}_DEN_COEF'.format(band, axis)],
                                  l1t_line,
                                  l1t_samp,
                                  l1r_line,
                                  l1r_samp,
                                  height=dem)
    return look_vector


def find_nadir_line(samps):
    left = np.where(samps[:,0] > 0)[0].min()
    right = np.where(samps[:,-1] > 0)[0].min()

    a = (right - left) / samps.shape[1]
    b = left

    return a, b


def get_sample_direction(samps):
    m, b = find_nadir_line(samps)

    endpt = np.array([samps.shape[1], m * samps.shape[1]])
    a = endpt / np.linalg.norm(endpt)
    return np.array([-a[1], a[0]])


def get_center_latlon(img):
    xmid = np.mean([img.xmin, img.xmax])
    ymid = np.mean([img.ymin, img.ymax])

    source = osr.SpatialReference()
    source.ImportFromWkt(img.proj_wkt)

    # The target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    # Create the transform - this can be used repeatedly
    transform = osr.CoordinateTransformation(source, target)

    # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
    lon, lat, _ = transform.TransformPoint(xmid, ymid)
    return lat


def find_nadir_track(img, imgname):
    left, top, right, bottom = find_img_corners(img)

    row = int(imgname.split('_')[2][3:])
    if is_descending(row):
        _left = LineString([bottom, left])
        _right = LineString([right, top])
    else:
        _left = LineString([left, top])
        _right = LineString([bottom, right])

    nadir_track = LineString([_right.interpolate(0.5, normalized=True),
                             _left.interpolate(0.5, normalized=True)])

    midpoint = nadir_track.interpolate(0.5, normalized=True)

    diff_track = np.diff(nadir_track.xy)
    m = diff_track[1] / diff_track[0]
    b = nadir_track.xy[1][1] - m * nadir_track.xy[0][1]

    a = diff_track / np.linalg.norm(diff_track)
    perp = np.array([-a[1], a[0]])

    return m[0], b[0], (midpoint.y, midpoint.x), perp


def get_track_distance(img, imgname):
    slope, intercept, midpt, perp = find_nadir_track(img, imgname)

    II, JJ = get_l1t_grids(img)

    track_dist = (II + -slope * JJ - intercept) / np.sqrt(1 + slope ** 2)
    return track_dist.astype(np.float32), perp.astype(np.float32)


def find_img_corners(img, nodata=0):
    if np.isnan(nodata):
        img_closing = binary_closing(np.isfinite(img), selem=disk(3))
    else:
        img_closing = binary_closing(img > 0, selem=disk(3))

    x_nonzero = np.where(np.sum(img_closing.astype(np.uint8), axis=0) > 0)
    y_nonzero = np.where(np.sum(img_closing.astype(np.uint8), axis=1) > 0)

    left_x, right_x = x_nonzero[0][0], x_nonzero[0][-1]
    top_y, bot_y = y_nonzero[0][0], y_nonzero[0][-1]

    left_y = np.where(img_closing[:, left_x])[0].mean()
    right_y = np.where(img_closing[:, right_x])[0].mean()

    top_x = np.where(img_closing[top_y, :])[0].mean()
    bot_x = np.where(img_closing[bot_y, :])[0].mean()

    return [(left_x-1, left_y), (top_x, top_y-1), (right_x+1, right_y), (bot_x, bot_y+1)]


def get_pixel_displacements(img, dem, metadata=None, sat_altitude=None):

    r_earth = get_earth_radius(get_center_latlon(img))

    # assert any([metadata is not None, sat_altitude is not None]), \
    #    'Have to specify LSAT Metadata file or Satellite Altitude'
    if metadata is not None:
        ecef_x = metadata['EPHEMERIS']['EPHEMERIS_ECEF_X']
        ecef_y = metadata['EPHEMERIS']['EPHEMERIS_ECEF_Y']
        ecef_z = metadata['EPHEMERIS']['EPHEMERIS_ECEF_Z']

        dist = np.sqrt(ecef_x ** 2 + ecef_y ** 2 + ecef_z ** 2)
        sat_alt = dist.mean() - r_earth
    elif sat_altitude is not None:
        sat_alt = sat_altitude
    # l1r_line, l1r_samp = get_l1r_grids(metadata['RPC_BAND{:02d}'.format(band)], band, dem=dem.img)
    # l1r_samp = l1r_samp - metadata['RPC_BAND{:02d}'.format(band)]['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][1]
    l1r_samp, perp = get_track_distance(img.img, img.filename)

    theta = l1r_samp / r_earth

    sin_phi = r_earth * np.sin(theta) / np.sqrt(r_earth**2 + (r_earth + sat_alt)**2
                                                - 2*r_earth*(r_earth+sat_alt)*np.cos(theta))
    phi = np.arcsin(sin_phi)

    tan_delphi = ((r_earth + sat_alt) * sin_phi * (1 - (r_earth + dem.img)/r_earth)) / \
                 ((r_earth + dem.img) * np.sqrt(1 - (r_earth + sat_alt)**2 * sin_phi**2 / r_earth**2) \
                  - (r_earth + sat_alt) * np.cos(phi))
    delphi = np.arctan(tan_delphi)

    sin_alpha = (r_earth + sat_alt) * np.sin(phi + delphi) / r_earth
    alpha = np.arcsin(sin_alpha)

    delta_x = r_earth * (alpha - theta - (phi + delphi))

    return delta_x.astype(np.float32), perp.astype(np.float32)


def get_l1t_displacement_map(img, dem, metadata=None, sat_altitude=None):
    # this_meta = metadata['RPC_BAND{:02d}'.format(band)]

    pixel_shifts, perp = get_pixel_displacements(img, dem, metadata=metadata, sat_altitude=sat_altitude)

    l1t_x_shift = perp[0] * pixel_shifts
    l1t_y_shift = perp[1] * pixel_shifts

    return l1t_x_shift.astype(np.float32), l1t_y_shift.astype(np.float32)


#################################
# orthorectification tools
#################################
def orthorectify_registered(imgname, fn_dem, sat_altitude=None):
    img = GeoImg(imgname)
    band = int(os.path.splitext(imgname)[0].split('_')[-1].strip('B'))

    img.img[np.isnan(img.img)] = 0
    img.img[img.img < 0] = 0

    if sat_altitude is None:
        try:
            metadata = parse_ang_file(imgname.replace('_B{}.TIF'.format(band), '_ANG.txt'))
        except FileNotFoundError:
            print('No Metadata file found. Using nominal altitude for satellite.')
            sat_altitude = get_nominal_altitude(imgname)
            metadata = None
    else:
        metadata = None

    # gcps = pd.read_csv(fn_gcps, delimiter=' ', names=['x', 'y', 'z', 'j', 'i'])

    # orig_ij = np.array([img.xy2ij(pt) for pt in gcps[['x', 'y']].values])
    # diff_j = orig_ij[:,1] - gcps['j']
    # diff_i = orig_ij[:,0] - gcps['i']

    # img.shift(img.dx * diff_j.median(), img.dy * diff_i.median())
    # img.write('tmp.tif', dtype=np.uint8)

    dem = GeoImg(fn_dem).reproject(img)

    dem.img[np.isnan(dem.img)] = np.nanmin(dem.img)

    # l1t_x_shift, l1t_y_shift = get_l1t_displacement_map(ang_data, band, dem)
    x_shift, y_shift = get_l1t_displacement_map(img, dem, metadata=metadata, sat_altitude=sat_altitude)
    l1t_line, l1t_samp = get_l1t_grids(img.img)

    # l1t_line, l1t_samp = generate_l1t_grids(ang_data['RPC_BAND{:02d}'.format(band)], band, offset=False)
    new_line = l1t_line - y_shift
    new_samp = l1t_samp - x_shift

    outimg = warp(img.img, np.array([new_line, new_samp]), order=5, preserve_range=True)

    out = img.copy(new_raster=outimg)
    out.write('Ortho.' + imgname, dtype=img.dtype)


def remove_orthorectification(imgname, fn_dem, sat_altitude=None, dtype=np.uint8):
    img = GeoImg(imgname, dtype=dtype)

    imname = os.path.basename(imgname)
    imdir = os.path.dirname(imgname)
    band = int(os.path.splitext(imname)[0].split('_')[-1].strip('B'))

    if sat_altitude is None:
        try:
            metadata = parse_ang_file(imgname.replace('_B{}.TIF'.format(band), '_ANG.txt'))
        except FileNotFoundError:
            print('No Metadata file found. Using nominal altitude for satellite.')
            sat_altitude = get_nominal_altitude(imgname)
    else:
        metadata = None

    dem = GeoImg(fn_dem).reproject(img)

    dem.img[np.isnan(dem.img)] = np.nanmin(dem.img)

    x_shift, y_shift = get_l1t_displacement_map(img, dem, metadata=metadata, sat_altitude=sat_altitude)
    l1t_line, l1t_samp = get_l1t_grids(img.img)

    new_line = l1t_line + y_shift
    new_samp = l1t_samp + x_shift

    outimg = warp(img.img, np.array([new_line, new_samp]), order=5, preserve_range=True)

    out = img.copy(new_raster=outimg)
    out.write(os.path.join(imdir, 'NoOrtho.' + imname), dtype=dtype)


#################################
# transform tools
#################################
def get_back_transform(img, metadata, band):
    l1r_shape = (metadata['BAND{:02d}_NUM_L1R_LINES'.format(band)],
                 metadata['BAND{:02d}_NUM_L1R_SAMPS'.format(band)])

    l1r_line, l1r_samp = get_l1r_grids(metadata, 4)

    l1r_line = smooth_holes(l1r_line)
    l1r_samp = smooth_holes(l1r_samp)

    l1t_line, l1t_samp = np.where(img.img > 0)

    _line = l1r_line[img.img > 0]
    _samp = l1r_samp[img.img > 0]

    l1t = np.concatenate([l1t_samp.reshape(-1, 1), l1t_line.reshape(-1, 1)], axis=1)
    l1r = np.concatenate([_samp.reshape(-1, 1), _line.reshape(-1, 1)], axis=1)

    samp = np.zeros(l1r_shape)
    line = np.zeros(l1r_shape)

    for i, pt in enumerate(l1r):
        samp[int(pt[1]), int(pt[0])] = l1t[i, 0]
        line[int(pt[1]), int(pt[0])] = l1t[i, 1]

    samp = smooth_holes(samp)
    line = smooth_holes(line)

    return line, samp


def back_rotate_image(img, metadata, band):
    line, samp = get_back_transform(img, metadata, band)
    return warp(img.img, np.array([line, samp]), order=5, preserve_range=True)


def retransform_image(prim, second, gcps, angfile, band, quality_mask=None):
    if quality_mask is not None:
        qmask = GeoImg(quality_mask)
    else:
        qmask = None

    metadata = parse_ang_file(angfile)
    line, samp = get_back_transform(second, metadata['RPC_BAND{:02d}'.format(band)], band)

    scaled_match = gcps[['match_i', 'match_j']].values * 400 / 30  # TODO: fix this
    xy = [(row.geometry.x, row.geometry.y) for i, row in gcps.iterrows()]
    scaled_src = np.array([prim.xy2ij(pt) for pt in xy])

    img_tfm = warp(second.img, np.array([line, samp]), order=1, preserve_range=True)
    if qmask is not None:
        q_tfm = warp(qmask.img, np.array([line, samp]), order=0, preserve_range=True)
    else:
        q_tfm = None

    l1r_line, l1r_samp = get_l1r_grids(metadata['RPC_BAND{:02d}'.format(band)], band)
    samps = l1r_samp[scaled_match[:, 0].astype(int), scaled_match[:, 1].astype(int)]
    lines = l1r_line[scaled_match[:, 0].astype(int), scaled_match[:, 1].astype(int)]

    model = AffineTransform()
    model.estimate(np.array([samps, lines]).T, scaled_src[:, ::-1])

    re_tfm = warp(img_tfm, model.inverse, order=1, output_shape=prim.img.shape, preserve_range=True)

    if q_tfm is not None:
        q_out = warp(q_tfm, model.inverse, order=0, output_shape=prim.img.shape, preserve_range=True)
    else:
        q_out = None

    outimg = prim.copy(new_raster=re_tfm)
    return outimg, q_out, (samps, lines)


def lowres_back_transform(prim, second, landmask, quality_mask):
    second_name = os.path.basename(second).split('_B')[0]
    second_dir = os.path.dirname(os.path.abspath(second))
    second_band = int(os.path.basename(second).split('_B')[1].split('.')[0].strip('B'))

    prim_fullres = GeoImg(prim)
    second_fullres = GeoImg(second)
    prim_fullres = prim_fullres.reproject(second_fullres)

    Minit, inliers_init, lowres_gcps = get_rough_geotransformation(prim_fullres, second_fullres, landmask=landmask)

    prim_lowres = prim_fullres.resample(400)
    xy = np.array([prim_lowres.ij2xy(pt) for pt in
                   lowres_gcps.loc[inliers_init, ['src_i', 'src_j']].values]).reshape(-1, 2)
    lowres_gcps.loc[inliers_init, 'x'] = xy[:, 0]
    lowres_gcps.loc[inliers_init, 'y'] = xy[:, 1]

    prim_fullres = GeoImg(prim)  # reload primary, then re-project
    second_fullres, qmask, init_tfm = retransform_image(prim_fullres, second_fullres, lowres_gcps,
                                                        os.path.join(second_dir, second_name + '_ANG.txt'),
                                                        second_band, quality_mask)
    return second_fullres, qmask, init_tfm


def lowres_transform(prim, second, fn_dem, landmask=None, quality_mask=None):
    prim_fullres = GeoImg(prim)
    second_fullres = GeoImg(second)
    prim_fullres = prim_fullres.reproject(second_fullres)

    second_lowres = second_fullres.resample(400)
    dem = GeoImg(fn_dem)
    dem_lowres = dem.reproject(second_lowres)
    dem = dem.reproject(second_fullres)

    Minit, inliers_init, lowres_gcps = get_rough_geotransformation(prim_fullres, second_fullres,
                                                                   landmask=landmask, dem=dem_lowres)

    lowres_gcps = lowres_gcps[lowres_gcps.residuals < 3 * nmad(lowres_gcps.residuals)]

    lowres_gcps['match_j'] *= second_lowres.dx / second_fullres.dx
    lowres_gcps['match_i'] *= second_lowres.dx / second_fullres.dx

    lowres_gcps['src_j'] *= second_lowres.dx / prim_fullres.dx
    lowres_gcps['src_i'] *= second_lowres.dx / prim_fullres.dx

    init_samps, init_lines = calculate_rpc_transform(lowres_gcps, second_fullres.img, dem)

    tfm_img = warp(second_fullres.img, np.array([init_lines, init_samps]), order=1, preserve_range=True)
    outimg = second_fullres.copy(new_raster=tfm_img.astype(second_fullres.dtype))

    if quality_mask is not None:
        qmask_geo = GeoImg(quality_mask, update=True)

        tfm_qmask = warp(qmask_geo.img, np.array([init_lines, init_samps]), order=1, preserve_range=True)
        qmask = tfm_qmask.astype(qmask_geo.dtype)
    else:
        qmask = None

    return outimg, qmask, (init_samps, init_lines)


def interp_coords(pt, coords_j, coords_i):
    jj = np.arange(int(pt[1])-5, int(pt[1])+6)
    ii = np.arange(int(pt[0])-5, int(pt[0])+6)

    J, I = np.meshgrid(jj, ii)

    interp_j = griddata((I.reshape(-1), J.reshape(-1)),
                        coords_j[int(pt[0])-5:int(pt[0])+6, int(pt[1])-5:int(pt[1])+6].reshape(-1), pt)
    interp_i = griddata((I.reshape(-1), J.reshape(-1)),
                        coords_i[int(pt[0])-5:int(pt[0])+6, int(pt[1])-5:int(pt[1])+6].reshape(-1), pt)

    return interp_j.flatten()[0], interp_i.flatten()[0]


#################################
# transform tools
#################################
def get_gcps(prim, second, mask, spacing, t_size=200, s_size=240, highpass=True, dem=None):
    search_pts, match_pts, peak_corr, z_corr = imtools.gridded_matching(prim.img,
                                                                        second.img,
                                                                        mask,
                                                                        spacing=spacing,
                                                                        tmpl_size=t_size,
                                                                        search_size=s_size,
                                                                        highpass=highpass)

    xy = np.array([prim.ij2xy(pt) for pt in search_pts]).reshape(-1, 2)

    gcps = gpd.GeoDataFrame()
    gcps['geometry'] = [Point(pt) for pt in xy]
    gcps['pk_corr'] = peak_corr
    gcps['z_corr'] = z_corr
    gcps['match_j'] = match_pts[:, 0]
    gcps['match_i'] = match_pts[:, 1]
    gcps['src_j'] = search_pts[:, 1]  # remember that search_pts is i, j, not j, i
    gcps['src_i'] = search_pts[:, 0]
    gcps['dj'] = gcps['src_j'] - gcps['match_j']
    gcps['di'] = gcps['src_i'] - gcps['match_i']
    gcps['elevation'] = 0
    gcps.crs = prim.proj4

    gcps.loc[gcps.z_corr == -1, 'z_corr'] = np.nan
    gcps.dropna(inplace=True)

    gcps = gcps[np.logical_and(gcps.z_corr > gcps.z_corr.quantile(0.25),
                               gcps.pk_corr > gcps.pk_corr.quantile(0.25))]

    if dem is not None:
        for i, row in gcps.to_crs(crs=dem.proj4).iterrows():
            gcps.loc[i, 'elevation'] = dem.raster_points([(row.geometry.x, row.geometry.y)], nsize=9, mode='cubic')
        gcps.dropna(inplace=True)

    return gcps


#################################
# registration tools
#################################
def get_rough_geotransformation(prim, second, landmask=None, dem=None):
    # prim_lowres = prim.resample(400, method=gdal.GRA_NearestNeighbour)
    second_lowres = second.resample(400)
    prim_lowres = prim.resample(400)

    prim_lowres = prim_lowres.reproject(second_lowres)

    second_lowres.img[np.isnan(second_lowres.img)] = 0
    prim_lowres.img[np.isnan(prim_lowres.img)] = 0

    if landmask is not None:
        lmask = imtools.create_mask_from_shapefile(prim_lowres, landmask)

    prim_rescale = imtools.stretch_image(prim_lowres.img, (0.05, 0.95))

    _mask = 255 * np.ones(prim_lowres.img.shape, dtype=np.uint8)

    if landmask is not None:
        _mask[np.logical_or(prim_rescale == 0, ~lmask)] = 0
    else:
        _mask[np.logical_or(np.isnan(prim_rescale), prim_rescale == 0)] = 0

    lowres_gcps = get_gcps(prim_lowres, second_lowres, _mask,
                           spacing=40, t_size=40, s_size=None, highpass=True, dem=dem)

    # best_lowres = lowres_gcps[lowres_gcps.z_corr > lowres_gcps.z_corr.quantile(0.5)].copy()
    init_filter = np.logical_and(np.abs(lowres_gcps.dj - lowres_gcps.dj.median()) < 2 * max(nmad(lowres_gcps.dj), 0.1),
                                 np.abs(lowres_gcps.di - lowres_gcps.di.median()) < 2 * max(nmad(lowres_gcps.di), 0.1))
    lowres_gcps = lowres_gcps.loc[init_filter]

    dst_pts = lowres_gcps[['match_j', 'match_i']].values
    src_pts = lowres_gcps[['src_j', 'src_i']].values

    # dst_scale = (second_lowres.dx / second.dx)
    # src_scale = (prim_lowres.dx / prim.dx)

    # lowres_gcps['dj'] *= dst_scale
    # lowres_gcps['di'] *= dst_scale

    Minit, inliers = ransac((dst_pts, src_pts), EuclideanTransform, min_samples=5,
                            residual_threshold=4, max_trials=5000)
    print('{} points used to find initial transformation'.format(np.count_nonzero(inliers)))
    lowres_gcps['residuals'] = Minit.residuals(lowres_gcps[['match_j', 'match_i']].values,
                                               lowres_gcps[['src_j', 'src_i']].values)
    return Minit, inliers, lowres_gcps


def register_landsat(fn_prim, fn_second, fn_dem, fn_qmask=None, spacing=400, fn_glacmask=None,
                     fn_landmask=None, no_lowres=False, all_bands=False, back_tfm=False):
    # second_name = os.path.basename(fn_second).split('_B')[0]
    second_dir = os.path.dirname(os.path.abspath(fn_second))
    # second_band = int(os.path.basename(fn_second).split('_B')[1].split('.')[0].strip('B'))
    second_fullres = GeoImg(fn_second)
    orig_dtype = second_fullres.dtype
    dem = GeoImg(fn_dem)

    if back_tfm:
        second_fullres, qmask, init_tfm = lowres_back_transform(fn_prim, fn_second, fn_landmask, fn_qmask)
    elif not no_lowres:
        second_fullres, qmask, init_tfm = lowres_transform(fn_prim, fn_second, fn_dem, fn_landmask, fn_qmask)
    else:
        print('skipping low-res transformation')
        second_fullres = GeoImg(fn_second)
        init_tfm = None

        if fn_qmask is not None:
            qmask_geo = GeoImg(fn_qmask)
            qmask_geo = qmask_geo.reproject(second_fullres, method=gdal.GRA_NearestNeighbour)
            qmask = qmask_geo.img
        else:
            qmask = None

    prim_fullres = GeoImg(fn_prim)
    prim_fullres = prim_fullres.reproject(second_fullres)
    prim_fullres.img[np.isnan(prim_fullres.img)] = 0

    mask = get_mask(prim_fullres, fn_landmask, fn_glacmask, qmask)
    mask[np.logical_and(prim_fullres.img == 0, np.isnan(prim_fullres.img))] = 0  # TODO: fix to work w/ masked images

    gcps = get_gcps(prim_fullres, second_fullres, mask, spacing=spacing, dem=dem)

    Mfin, inliers_fin = ransac((gcps[['match_j', 'match_i']].values, gcps[['src_j', 'src_i']].values),
                               AffineTransform, min_samples=10, residual_threshold=16, max_trials=5000)
    print('{} points used to find final transformation'.format(np.count_nonzero(inliers_fin)))
    gcps['residuals'] = Mfin.residuals(gcps[['match_j', 'match_i']].values, gcps[['src_j', 'src_i']].values)

    inliers = gcps.residuals < 9

    dem = dem.reproject(prim_fullres)

    if init_tfm is not None:
        for i, row in gcps.iterrows():
            interp_j, interp_i = interp_coords((row.match_i, row.match_j), init_tfm[0], init_tfm[1])
            gcps.loc[i, 'match_j'] = interp_j
            gcps.loc[i, 'match_i'] = interp_i

    samps, lines = calculate_rpc_transform(gcps.loc[inliers], second_fullres.img, dem)

    print('warping image to new geometry')
    second_fullres = GeoImg(fn_second)
    img_tfm = warp(second_fullres.img, np.array([lines, samps]), order=1, preserve_range=True)

    mkdir_p('warped')
    outimg = prim_fullres.copy(new_raster=img_tfm)
    outimg.write(os.path.join('warped', os.path.basename(fn_second)), dtype=orig_dtype)

    if all_bands:
        cwdir = os.getcwd()
        os.chdir(second_dir)

        flist = glob('*' + os.path.splitext(fn_second)[-1])
        flist.remove(os.path.basename(fn_second))

        for im in flist:
            print(im)
            this_img = GeoImg(im)
            if 'BQA' not in im:
                tfm_img = warp(this_img.img, np.array([lines, samps]), output_shape=second_fullres.img.shape,
                               preserve_range=True, order=1)
            else:
                tfm_img = warp(this_img.img, np.array([lines, samps]), output_shape=second_fullres.img.shape,
                               preserve_range=True, order=0)

            outimg = prim_fullres.copy(new_raster=tfm_img)
            outimg.write(os.path.join('warped', os.path.basename(im)), dtype=this_img.dtype)

        os.chdir(cwdir)


#################################
# mask tools
#################################
def tm_qa_mask(img):
    return np.logical_not(np.logical_or.reduce((img == 672, img == 676, img == 680, img == 684,
                                                img == 1696, img == 1700, img == 1704, img == 1708)))


def get_mask(prim, fn_landmask=None, fn_glacmask=None, qmask=None):
    mask = 255 * np.ones(prim.img.shape, dtype=np.uint8)

    if qmask is not None:
        quality_mask = tm_qa_mask(qmask)
        mask[quality_mask] = 0

    if fn_landmask is not None:
        lm = imtools.create_mask_from_shapefile(prim, fn_landmask)
        mask[lm == 0] = 0

    if fn_glacmask is not None:
        gm = imtools.create_mask_from_shapefile(prim, fn_glacmask)
        mask[gm > 0] = 0

    return mask