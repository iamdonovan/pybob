import os
import yaml
import numpy as np
import osr
import pandas as pd
from shapely.geometry.linestring import LineString
from skimage.transform import warp
from skimage.morphology import binary_closing, disk
from pybob.GeoImg import GeoImg
from pymmaster.mmaster_tools import rmse


####################### parameter tools
def get_earth_radius(latitude):
    mysrs = osr.SpatialReference()
    mysrs.ImportFromEPSG(4326) # get wgs84 srs to get the earth_radius

    r_major = mysrs.GetSemiMajor()
    r_minor = mysrs.GetSemiMinor()

    num = (r_major**2 * np.cos(np.deg2rad(latitude)))**2 + (r_minor**2 * np.sin(np.deg2rad(latitude)))**2
    den = (r_major * np.cos(np.deg2rad(latitude)))**2 + (r_minor * np.sin(np.deg2rad(latitude)))**2

    return np.sqrt(num / den)



###################### metadata tools
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


###################### rpc tools
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


##################################
# def generate_l1t_grids(metadata, band, offset=True, direction=None):
def generate_l1t_grids(img):
    # img_I = np.arange(0, metadata['BAND{:02d}_NUM_L1T_LINES'.format(band)])
    # img_J = np.arange(0, metadata['BAND{:02d}_NUM_L1T_SAMPS'.format(band)])
    img_I = np.arange(0, img.shape[0])
    img_J = np.arange(0, img.shape[1])

    JJ, II = np.meshgrid(img_J, img_I)

    # if offset:
    #     if direction is not None:
    #         l1t_line = II - metadata['BAND{:02d}_DIR{:02d}_MEAN_L1T_LINE_SAMP'.format(band, direction)][0]
    #         l1t_samp = JJ - metadata['BAND{:02d}_DIR{:02d}_MEAN_L1T_LINE_SAMP'.format(band, direction)][1]
    #     else:
    #         l1t_line = II - metadata['BAND{:02d}_MEAN_L1T_LINE_SAMP'.format(band)][0]
    #         l1t_samp = JJ - metadata['BAND{:02d}_MEAN_L1T_LINE_SAMP'.format(band)][1]
    # else:
    #     l1t_line = II
    #     l1t_samp = JJ
    return II, JJ

def get_pixel_displacements(img, dem, metadata=None, sat_altitude=None):

    r_earth = get_earth_radius(get_center_latlon(img))

    # assert any([metadata is not None, sat_altitude is not None]), \
    #    'Have to specify LSAT Metadata file or Satellite Altitude'

    if metadata is not None:
        sat_alt = metadata['EPHEMERIS']['EPHEMERIS_ECEF_Z'].mean() - r_earth
    elif sat_altitude is not None:
        sat_alt = sat_altitude
    # l1r_line, l1r_samp = get_l1r_grids(metadata['RPC_BAND{:02d}'.format(band)], band, dem=dem.img)
    # l1r_samp = l1r_samp - metadata['RPC_BAND{:02d}'.format(band)]['BAND{:02d}_MEAN_L1R_LINE_SAMP'.format(band)][1]
    l1r_samp, perp = get_track_distance(img.img, img.filename)

    theta = l1r_samp / r_earth

    sin_phi = r_earth * np.sin(theta) / \
              np.sqrt(r_earth**2 + (r_earth + sat_alt)**2 - 2*r_earth*(r_earth+sat_alt)*np.cos(theta))
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
    l1t_line, l1t_samp = generate_l1t_grids(img.img)

    # l1t_line, l1t_samp = generate_l1t_grids(ang_data['RPC_BAND{:02d}'.format(band)], band, offset=False)
    new_line = l1t_line - y_shift
    new_samp = l1t_samp - x_shift

    outimg = warp(img.img, np.array([new_line, new_samp]), order=5, preserve_range=True)

    out = img.copy(new_raster=outimg)
    out.write('Ortho.' + imgname, dtype=np.uint8)


def is_descending(row):
    if row <= 122 or row > 246:
        return True
    else:
        return False


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

def get_track_distance(img, imgname):
    slope, intercept, midpt, perp = find_nadir_track(img, imgname)

    II, JJ = generate_l1t_grids(img)

    track_dist = (II + -slope * JJ - intercept) / np.sqrt(1 + slope ** 2)
    return track_dist.astype(np.float32), perp.astype(np.float32)


def remove_orthorectification(imgname, fn_dem, sat_altitude=None, dtype=np.uint8):
    img = GeoImg(imgname, dtype=dtype)
    band = int(os.path.splitext(imgname)[0].split('_')[-1].strip('B'))

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
    l1t_line, l1t_samp = generate_l1t_grids(img.img)

    new_line = l1t_line + y_shift
    new_samp = l1t_samp + x_shift

    outimg = warp(img.img, np.array([new_line, new_samp]), order=5, preserve_range=True)

    out = img.copy(new_raster=outimg)
    out.write('NoOrtho.' + imgname, dtype=np.uint8)
