#!/usr/bin/env python
import gdal, sys, os, argparse, numpy as np
from pybob.GeoImg import GeoImg
from pybob.image_tools import generate_panchrome

def main():
    parser = argparse.ArgumentParser(description = "Generate simulated panchromatic image for Landsat TM data.")
    parser.add_argument("inputscene", type=str, help="Base Landsat scene name (do not specify bands) to be read in.")
    parser.add_argument("-o", "--outputscene", type=str, help="Output scene name (if unspecified, defaults to inputscenename_B8.TIF)")
    args = parser.parse_args()

    if args.outputscene is None:
	    args.outputscene = args.inputscene

    outfilename = args.outputscene + "_B8.TIF"

    # first, read in the bands (4, 3, 2)
    B4 = GeoImg( args.inputscene + "_B4.TIF" )
    B3 = GeoImg( args.inputscene + "_B3.TIF" )
    B2 = GeoImg( args.inputscene + "_B2.TIF" )

    # now, make a new band
    B8sim = 0.5 * B4.img + 0.2 * B3.img + 0.3 * B2.img

    B8 = B4.copy(new_raster=B8sim)

    B8.write(outfilename)

if __name__ == "__main__":
    main()
