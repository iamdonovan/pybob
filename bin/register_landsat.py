#!/usr/bin/env python
import argparse
from pybob.landsat_tools import register_landsat


def _argparser():
    parser = argparse.ArgumentParser(description="Automatically register two Landsat images",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('primary', action='store', type=str, help='non-referenced orthophoto mosaic')
    parser.add_argument('secondary', action='store', type=str, help='georeferenced satellite image')
    parser.add_argument('dem', action='store', type=str, default=None,
                        help='digital elevation model')
    parser.add_argument('-q', '--quality_mask', action='store', type=str, default=None,
                        help='BQA band for the L1GS image (recommended)')
    parser.add_argument('-s', '--spacing', action='store', type=int, default=400,
                        help='grid spacing to search for GCPs')
    parser.add_argument('-glacmask', action='store', type=str, default=None,
                        help='path to shapefile of glacier outlines')
    parser.add_argument('-landmask', action='store', type=str, default=None,
                        help='path to shapefile of land outlines')
    # parser.add_argument('-no_lowres', action='store_true', default=False,
    #                     help="Don't do an initial low-res transformation")
    parser.add_argument('-a', '--all_bands', action='store_true', default=False,
                        help="Register all bands based on computed transformation.")
    parser.add_argument('-b', '--back_tfm', action='store_true', default=False,
                        help="Before doing dense matching, use RPC model to back-transform image.")

    return parser


def main():
    parser_ = _argparser()
    args = parser_.parse_args()

    register_landsat(args.primary,
                     args.secondary,
                     args.dem,
                     fn_qmask=args.quality_mask,
                     spacing=args.spacing,
                     fn_glacmask=args.glacmask,
                     fn_landmask=args.landmask,
                     # no_lowres=args.no_lowres,
                     all_bands=args.all_bands,
                     back_transform=args.back_tfm)


if __name__ == "__main__":
    main()
