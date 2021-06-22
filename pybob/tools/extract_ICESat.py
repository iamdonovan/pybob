#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:03:11 2018

@author: christnu
"""
import argparse
import numpy as np
from pybob.ICESat import extract_ICESat


def _argparser():
    parser = argparse.ArgumentParser(description="Extracts ICESat data to the extent of a DEM or Image (GeoImg)")
    parser.add_argument('DEM', type=str, help='path to DEM from which the extent is used to extract ICESat data')
    parser.add_argument('-d', '--workdir', type=str,
                        help='Current working directory. [default=./]')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output filename. [default=ICESat_DEM.h5]')
    return parser


def main():
    np.seterr(all='ignore')
    # add primary, secondary, masks to argparse
    # can also add output directory
    parser = _argparser()
    args = parser.parse_args()
        
    extract_ICESat(args.DEM, workdir=args.workdir, outfile=args.outfile)


if __name__ == "__main__":
    main()