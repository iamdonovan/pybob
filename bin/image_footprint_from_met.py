#!/usr/bin/env python
import os
import argparse
import glob
import fiona
import fiona.crs
from shapely.geometry.polygon import Polygon
from shapely.geometry import mapping


def main():
    parser = argparse.ArgumentParser(description="Create footprint of valid image area for one (or more) images.")
    # parser.add_argument('--image', action='store', type=str, nargs='+', help="Image(s) to read in")
    parser.add_argument('-o', '--outshape', action='store', type=str, default='Footprints.shp',
                        help="Shapefile to be written. [default: Footprints.shp]")
    # parser.add_argument('-b', '--buffer', action='store', type=float, default=0, help="buffer size to use [0]")
    # parser.add_argument('--chop', action='store', type=float, nargs='?', const=1000.0,
    #                    help="Amount of image to crop in m [default 1000]")
    parser.add_argument('-p', '--processed', action='store_true',
                        help="Use if images have been processed (i.e., there are folders with met filesin them.)")
    args = parser.parse_args()

    schema = {'properties': [('filename', 'str'), ('path', 'str')], 'geometry': 'Polygon'}

    outshape = fiona.open(args.outshape, 'w', crs=fiona.crs.from_epsg(4326),
                          driver='ESRI Shapefile', schema=schema)

    if args.processed:
        flist = glob.glob('AST*/*.zip.met')
    else:
        flist = glob.glob('*.zip.met')

    for mfile in flist:
        imgpath = os.path.abspath(mfile)
        dirpath = os.path.dirname(imgpath)

        clean = [line.strip() for line in open(mfile).read().split('\n')]

        if os.path.sep in mfile:
            mfile = mfile.split(os.path.sep)[-1]

        latinds = [i for i, line in enumerate(clean) if 'GRingPointLatitude' in line]
        loninds = [i for i, line in enumerate(clean) if 'GRingPointLongitude' in line]

        latlines = clean[latinds[0]:latinds[1]+1]
        lonlines = clean[loninds[0]:loninds[1]+1]

        lonvalstr = lonlines[2]
        latvalstr = latlines[2]

        lats = [float(val) for val in latvalstr.strip('VALUE =()').split(',')]
        lons = [float(val) for val in lonvalstr.strip('VALUE =()').split(',')]

        coords = zip(lons, lats)
        footprint = Polygon(coords)

        outshape.write({'properties': {'filename': mfile, 'path': dirpath}, 'geometry': mapping(footprint)})

    outshape.close()


if __name__ == "__main__":
    main()
