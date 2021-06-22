#!/usr/bin/env python
import geopandas as gpd                                                                                                                                                                               


def main():
    singles = gpd.read_file('Footprints.shp')

    with open('zone_list.txt', 'w') as f:
        for i, row in singles.iterrows():
            print('{} {}'.format(row.filename, row.best_zone), file=f)


if __name__ == "__main__":
    main()