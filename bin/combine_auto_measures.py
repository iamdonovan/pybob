#!/usr/bin/env python
import argparse
from pybob.hexagon_tools import get_gcp_meas
import lxml.etree as etree
import lxml.builder as builder


def _argparser():
    parser = argparse.ArgumentParser(description="Combine outputs of XYZ2Im into single xml file for further processing",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('ij_files', action='store', type=str, nargs='+', help='txt files containing image i,j points')
    parser.add_argument('-o', '--out_file', action='store', type=str, default='MeasuresAuto.xml',
                        help='output filename [AutoMeasures.xml]')
    parser.add_argument('-n', '--no_distortion', action='store_true',
                        help='Use gcp locations computed assuming no distortion to filter GCPs from images.')
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    E = builder.ElementMaker()
    MesureSet = E.SetOfMesureAppuisFlottants()
    for ij_file in args.ij_files:
        imname = ij_file.split('Auto-')[1].split('.txt')[0]
        if args.no_distortion:
            nodist_file = ij_file.replace('Auto', 'NoDist')
        else:
            nodist_file = None

        print(imname)
        this_meas = get_gcp_meas(imname, ij_file, '.', E, nodist=nodist_file)
        MesureSet.append(this_meas)

    tree = etree.ElementTree(MesureSet)
    tree.write(args.out_file, pretty_print=True, xml_declaration=True, encoding="utf-8")


if __name__ == "__main__":
    main()
