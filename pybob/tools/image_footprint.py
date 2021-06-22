#!/usr/bin/env python
import argparse
import os
import fiona
import pyproj
from functools import partial
from numpy import argmax, array
from fiona import crs
from shapely.ops import transform
from shapely.geometry import mapping, LineString
from shapely.geometry.polygon import Polygon, orient
from pybob.GeoImg import GeoImg


def reproject_geometry(src_data, src_crs, dst_crs):
    # unfortunately this requires pyproj>1.95, temporary fix to avoid shambling dependencies in mmaster_environment
    src_proj = pyproj.Proj(src_crs)
    dst_proj = pyproj.Proj(dst_crs)

    project = partial(pyproj.transform, src_proj, dst_proj)
    return transform(project, src_data)


def orient_footprint(fprint):
    # orient the footprint coordinates so that they are clockwise
    fprint = orient(fprint, sign=-1)
    x, y = fprint.boundary.coords.xy
    x = x[:-1] # drop the last coordinate, which is a duplicate of the first
    y = y[:-1]
    # as long as the footprints are coming from the .met file, the upper left corner 
    # will be the maximum y value.
    upper_left = argmax(y)
    new_inds = range(upper_left, len(x)) + range(0, upper_left)
    return Polygon(list(zip(array(x)[new_inds], array(y)[new_inds])))
    

def lhand_chop(footprint, chop):
    # have to project chop m in from the right (east) side,
    # along the upper and lower boundaries. it's lhand_chop because
    # in the satellite's view, this is the left-hand of the image.
    footprint = orient_footprint(footprint)
    coords = footprint.exterior.coords
    # geoimg.xycorners gives corners as: UL, UR, LR, LL, so take 1:0 and 2:3
    upper = LineString([coords[1], coords[0]])
    lower = LineString([coords[2], coords[3]])
    new_ur = upper.interpolate(chop)
    new_lr = lower.interpolate(chop)

    new_coords = [coords[0], (new_ur.x, new_ur.y), (new_lr.x, new_lr.y), coords[3]]
    return Polygon(new_coords)


def _argparser():
    parser = argparse.ArgumentParser(description="Create footprint of valid image area for one (or more) images.")
    parser.add_argument('image', action='store', type=str, nargs='+', help="Image(s) to read in")
    parser.add_argument('-o', '--outshape', action='store', type=str, default='Footprints.shp',
                        help="Shapefile to be written. [default: Footprints.shp]")
    parser.add_argument('-b', '--buffer', action='store', type=float, default=0, help="buffer size to use [0]")
    parser.add_argument('--chop', action='store', type=float, nargs='?', const=1000.0,
                        help="Amount of image to crop in m [default 1000]")
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    img1 = GeoImg(args.image[0])
    schema = {'properties': [('filename', 'str'), ('path', 'str')], 'geometry': 'Polygon'}

    outshape = fiona.open(args.outshape, 'w', crs=crs.from_epsg(img1.epsg),
                          driver='ESRI Shapefile', schema=schema)

    for img in args.image:
        imgpath = os.path.abspath(img)
        dirpath = os.path.dirname(imgpath)

        geo = GeoImg(img)
        if os.path.sep in img:
            img = img.split(os.path.sep)[-1]
        xycorners = geo.find_corners(mode='xy')
        footprint = Polygon(xycorners)
        if args.chop is not None:
            x = [p[0] for p in xycorners]
            y = [p[1] for p in xycorners]
            sfact = min(max(x) - min(x), max(y) - min(y)) / 2
            simple_print = footprint.simplify(sfact)
            footprint = lhand_chop(simple_print, args.chop)

        footprint = footprint.buffer(args.buffer)
        if geo.epsg != img1.epsg:
            outprint = reproject_geometry(footprint, geo.proj4, img1.proj4)
        else:
            outprint = footprint
        outshape.write({'properties': {'filename': img, 'path': dirpath}, 'geometry': mapping(outprint)})
    outshape.close()


if __name__ == "__main__":
    main()
