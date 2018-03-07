#!/usr/bin/env python
import argparse
import numpy as np
from pybob.coreg_tools import dem_coregistration


def main():
    np.seterr(all='ignore')
    # add master, slave, masks to argparse
    # can also add output directory
    parser = argparse.ArgumentParser(description="Iteratively calculate co-registration \
                                     parameters for two DEMs, as seen in Nuth and Kaeaeb (2011).")
    parser.add_argument('masterdem', type=str, help='path to master DEM to be used for co-registration')
    parser.add_argument('slavedem', type=str, help='path to slave DEM to be co-registered')
    parser.add_argument('-a', '--mask1', type=str, default=None,
                        help='Glacier mask. Areas inside of this shapefile will not be used for coregistration [None]')
    parser.add_argument('-b', '--mask2', type=str, default=None,
                        help='Land mask. Areas outside of this mask (i.e., water) \
                             will not be used for coregistration. [None]')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='Directory to output files to (creates if not already present). [.]')
    parser.add_argument('-i', '--icesat', action='store_true', default=False,
                        help="Process assuming that master DEM is ICESat data [False].")
    args = parser.parse_args()

    master, coreg_slave = dem_coregistration(args.masterdem, args.slavedem,
                                             glaciermask=args.mask1, landmask=args.mask2,
                                             outdir=args.outdir, pts=args.icesat)


if __name__ == "__main__":
    main()
