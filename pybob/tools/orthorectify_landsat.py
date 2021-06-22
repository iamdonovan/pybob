#!/usr/bin/env python
import argparse
import os
import pybob.landsat_tools as lt


def _argparser():
    parser = argparse.ArgumentParser(description="Orthorectify a Landsat L1GS image",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('img', action='store', type=str, help='name of non-referenced landsat image')
    parser.add_argument('dem', action='store', type=str, help='digital elevation model')

    return parser


def main():
    parser_ = _argparser()
    args = parser_.parse_args()

    print(args.img)

    fn_gcp = args.img.replace('.TIF', '_gcps.txt')
    band = int(os.path.splitext(args.img)[0].split('_')[-1].strip('B'))

    lt.orthorectify_registered(args.img, args.dem)


if __name__ == "__main__":
    main()
