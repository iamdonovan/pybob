#!/usr/bin/env python
import argparse
import datetime as dt
import numpy as np
import geopandas as gpd
from shapely.strtree import STRtree


def mmdd2dec(year, month, day):
    yearstart = dt.datetime(year, 1, 1).toordinal()
    nextyearstart = dt.datetime(year+1, 1, 1).toordinal()
    datestruct = dt.datetime(year, month, day)
    datefrac = float(datestruct.toordinal() + 0.5 - yearstart) / (nextyearstart - yearstart)
    return year + datefrac


def parse_aster_date(img, imgdata, args):
    name = imgdata[args.imagename][imgdata['geometry'] == img].values[0]
    datestr = name[11:19]
    yy = int(datestr[4:])
    mm = int(datestr[0:2])
    dd = int(datestr[2:4])
    date = dt.datetime(yy, mm, dd)

    return date, name


def main():
    parser = argparse.ArgumentParser(description="Find ASTER dDEM pairs based on area overlap, time separation.")
    parser.add_argument('footprints', action='store', type=str, help="Shapefile of image footprints to read in.")
    parser.add_argument('--overlap', action='store', type=float, default=0.75,
                        help="Amount of fractional area that should overlap to use as candidate pair. [default: 0.75]")
    parser.add_argument('--tmin_sep', action='store', type=float, default=2,
                        help="Minimum amount of time (in years) to separate images [default: 2 years]")
    parser.add_argument('--tmax_sep', action='store', type=float, default=1000,
                        help="Maximum amount of time (in years) to separate images [default: 1000 years]")
    parser.add_argument('--imagename', action='store', type=str, default='filename',
                        help="Name of the shapefile field that contains the DEM filenames")
    parser.add_argument('--datefield', action='store', type='int', default=None,
                        help="Name of the shapefield field that contains the date information [None]")
    args = parser.parse_args()

    footprint_data = gpd.read_file(args.footprints)
    footprints = [f for f in footprint_data['geometry'].values]

    for img in footprints:
        s = STRtree(footprints)
        if args.datefield is None:
            this_date, this_name = parse_aster_date(img, footprint_data, args)
        else:
            this_row = footprint_data[footprint_data['geometry'] == img]
            this_name = this_row[args.imagename].values[0]
            date_data = this_row[args.datefield].values[0]
            yy, mm, dd = [int(i) for i in date_data.split('-')]
            this_date = mmdd2dec(yy, mm, dd)
        candidates = s.query(img)
        candidates.remove(img)  # query returns itself
        overlaps = [img.intersection(c).area / img.area for c in candidates]

        cull1 = [c for i, c in enumerate(candidates) if overlaps[i] > args.overlap]
        for c in cull1:
            cdate, cname = parse_aster_date(c, footprint_data, args)
            tsep = np.abs((cdate - this_date).days / 365.2425)  # this is seriously close enough.
            if tsep >= args.tmin_sep and tsep <= args.tmax_sep:
                print cname, this_name
        footprints.remove(img)


if __name__ == "__main__":
    main()
