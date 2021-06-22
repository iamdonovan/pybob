#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import glob
import fiona
import fiona.crs
import pyproj
from functools import partial
from shapely.geometry.polygon import Polygon
from shapely.geometry import mapping
from shapely.ops import transform


def reproject_geometry(src_data, src_epsg, dst_epsg):
    src_proj = pyproj.Proj(init='epsg:{}'.format(src_epsg))
    dst_proj = pyproj.Proj(init='epsg:{}'.format(dst_epsg))
    project = partial(pyproj.transform, src_proj, dst_proj)
    return transform(project, src_data)
    

def _argparser():
    parser = argparse.ArgumentParser(description="Create footprint of valid image area for one (or more) images using \
                                                 the .zip.met file downloaded from earthdata.nasa.gov")
    # parser.add_argument('--image', action='store', type=str, nargs='+', help="Image(s) to read in")
    parser.add_argument('-o', '--outshape', action='store', type=str, default='Footprints.shp',
                        help="Shapefile to be written. [default: Footprints.shp]")
    parser.add_argument('-b', '--buffer', action='store', type=float, default=0, help="buffer size to use [0]")
    # parser.add_argument('--chop', action='store', type=float, nargs='?', const=1000.0,
    #                    help="Amount of image to crop in m [default 1000]")
    parser.add_argument('-p', '--processed', action='store_true',
                        help="Use if images have been processed (i.e., there are folders with met filesin them.)")
    parser.add_argument('-t_srs', '--out_srs', action='store', type=str, default=None,
                        help="EPSG code for output spatial reference system [defaults to WGS84 Lat/Lon]")
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    schema = {'properties': [('filename', 'str'), ('path', 'str')], 'geometry': 'Polygon'}

    if args.out_srs is not None:
        out_crs = fiona.crs.from_epsg(args.out_srs)
    else:
        out_crs = fiona.crs.from_epsg(4326)

    outshape = fiona.open(args.outshape, 'w', crs=out_crs,
                          driver='ESRI Shapefile', schema=schema)

    if args.processed:
        flist = glob.glob('AST*/*.zip.met') + glob.glob('AST*/zips/*.zip.met')
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
        if args.out_srs is not None:
            footprint = reproject_geometry(footprint, 4326, args.out_srs)
        
        footprint = footprint.buffer(args.buffer)
        outshape.write({'properties': {'filename': mfile, 'path': dirpath}, 'geometry': mapping(footprint)})

    outshape.close()


if __name__ == "__main__":
    main()
