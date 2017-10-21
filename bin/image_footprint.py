#!/usr/bin/env python
import argparse
import os
import fiona
from fiona import crs
from shapely.geometry import mapping, LineString
from shapely.geometry.polygon import Polygon
from pybob.GeoImg import GeoImg


def lhand_chop(footprint, chop):
    # have to project chop m in from the right (east) side,
    # along the upper and lower boundaries. it's lhand_chop because
    # in the satellite's view, this is the left-hand of the image.
    coords = footprint.exterior.coords
    # geoimg.xycorners gives corners as: UL, UR, LR, LL, so take 1:0 and 2:3
    upper = LineString([coords[1], coords[0]])
    lower = LineString([coords[2], coords[3]])
    new_ur = upper.interpolate(chop)
    new_lr = lower.interpolate(chop)

    new_coords = [coords[0], (new_ur.x, new_ur.y), (new_lr.x, new_lr.y), coords[3]]
    return Polygon(new_coords)


def main():
    parser = argparse.ArgumentParser(description="Create footprint of valid image area for one (or more) images.")
    parser.add_argument('image', action='store', type=str, nargs='+', help="Image(s) to read in")
    parser.add_argument('-o', '--outshape', action='store', type=str, default='Footprints.shp',
                        help="Shapefile to be written. [default: Footprints.shp]")
    parser.add_argument('-b', '--buffer', action='store', type=float, default=0, help="buffer size to use [0]")
    parser.add_argument('--chop', action='store', type=float, nargs='?', const=1000.0,
                        help="Amount of image to crop in m [default 1000]")
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
            footprint = lhand_chop(footprint, args.chop)
        # footprint = footprint.buffer(args.buffer)
        outshape.write({'properties': {'filename': img, 'path': dirpath}, 'geometry': mapping(footprint)})
    outshape.close()


if __name__ == "__main__":
    main()
