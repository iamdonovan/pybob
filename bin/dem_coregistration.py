#!/usr/bin/env python
import argparse
import numpy as np
from pybob.coreg_tools import dem_coregistration


def _argparser():
    parser = argparse.ArgumentParser(description="""Iteratively calculate co-registration parameters for two DEMs, as seen in `Nuth and Kääb (2011)`_.
                                                                                                   
                                     .. _Nuth and Kääb (2011): https://www.the-cryosphere.net/5/271/2011/tc-5-271-2011.html""")
    parser.add_argument('primarydem', type=str, help='path to primary DEM to be used for co-registration')
    parser.add_argument('secondarydem', type=str, help='path to secondary DEM to be co-registered')
    parser.add_argument('-a', '--mask1', type=str, default=None,
                        help='Glacier mask. Areas inside of this shapefile will not be used for coregistration [None]')
    parser.add_argument('-b', '--mask2', type=str, default=None,
                        help='Land mask. Areas outside of this mask (i.e., water) \
                             will not be used for coregistration. [None]')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='Directory to output files to (creates if not already present). [.]')
    parser.add_argument('-i', '--icesat', action='store_true', default=False,
                        help="Process assuming that primary DEM is ICESat data [False].")
    parser.add_argument('-f', '--full_ext', action='store_true', default=False,
                        help="Write full extent of primary DEM and shifted secondary DEM. [False].")
    parser.add_argument('-g', '--alg', type=str, default='Horn',
                        help="Algorithm to calculate slope, aspect. One of 'ZevenbergenThorne' or 'Horn'. [Horn]")
    return parser


def main():
    np.seterr(all='ignore')
    # add primary, secondary, masks to argparse
    # can also add output directory
    parser = _argparser()

    args = parser.parse_args()
    dem_coregistration(args.primarydem, args.secondarydem,
                       glaciermask=args.mask1, landmask=args.mask2,
                       outdir=args.outdir, pts=args.icesat,
                       full_ext=args.full_ext, return_var=False)


if __name__ == "__main__":
    main()
