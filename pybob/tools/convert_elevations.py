import argparse
import os
from pybob.GeoImg import GeoImg


def _argparser():
    parser = argparse.ArgumentParser(description="""Convert DEM from geoid to ellipsoid heights (or vice-versa)""")
    parser.add_argument('dem', type=str, help='DEM to convert')
    parser.add_argument('geoid', type=str, help='geoid file')
    parser.add_argument('-g', '--to_geoid', action='store_true', default=False,
                        help="Convert to geoid heights (i.e., subtract geoid model from dem) [False]")
    parser.add_argument('-o', '--out_folder', type=str, default=None,
                        help='Folder to write converted DEM to. [same as input]')
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    if args.to_geoid:
        suff = 'geod'
    else:
        suff = 'ell'

    if args.out_folder is None:
        args.out_folder = os.getcwd()

    dem = GeoImg(args.dem)
    geoid = GeoImg(args.geoid)

    geoid = geoid.reproject(dem)

    if args.to_geoid:
        out_dem = dem.copy(new_raster=(dem.img - geoid.img))
    else:
        out_dem = dem.copy(new_raster=(dem.img + geoid.img))

    out_dem.write(os.path.join(args.out_folder,
                               os.path.splitext(dem.filename)[0] + '_ell' + os.path.splitext(dem.filename)[1]))


if __name__ == "__main__":
    main()

