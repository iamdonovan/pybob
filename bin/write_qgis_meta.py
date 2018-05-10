#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import xml.etree.ElementTree as ET


def write_xml_meta(layer, outdir):
    doc = ET.Element("metadata")

    # a list of the fields that have only one value
    meta_text = ['shortname', 'title', 'abstract', 'datasource']
    # a list of the fields that have multiple values
    meta_list = ['keywordList', 'extent']
    # try to fetch and write the fields
    for textfield in meta_text:
        try:
            metadata = layer.find(textfield).text
            ET.SubElement(doc, textfield).text = metadata
        except:
            print('No field {} found.'.format(textfield))
            ET.SubElement(doc, textfield).text = None

    for listfield in meta_list:
        this_el = ET.SubElement(doc, listfield)
        try:
            mlist = layer.find(listfield).getchildren()
            for el in mlist:
                ET.SubElement(this_el, el.tag).text = el.text
        except:
            print('No field {} found.'.format(listfield))
            ET.SubElement(doc, listfield).text = None

    srs_list = layer.find('srs').find('spatialrefsys').getchildren()
    this_el = ET.SubElement(doc, 'srs')
    for el in srs_list:
        ET.SubElement(this_el, el.tag).text = el.text

    filename = layer.find('datasource').text
    if outdir is None:
        outdir = os.path.basename(filename)

    out_filename = os.path.sep.join([outdir, os.path.basename(filename) + '.xml'])
    tree = ET.ElementTree(doc)
    tree.write(out_filename)


# fields to write:
# shortname, title, abstract, keywordList, srs (more later?), extent
def main():
    parser = argparse.ArgumentParser(description="Parse a QGIS document and write metadata \
             as an XML file for given layer(s).", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('xmlfile', action='store', type=str, help='XML file to open and parse')
    parser.add_argument('--layer', action='store', type=str, nargs='+',
                        help='layer(s) to write metadata for. If left empty, writes metadata for each layer.')
    parser.add_argument('--outdir', action='store', type=str,
                        help='directory to write metadata to. If left empty, writes metadata \
                              to same directory as original layer.')
    args = parser.parse_args()

    tree = ET.parse(args.xmlfile)
    root = tree.getroot()
    layers = root.findall('projectlayers')[0].findall('maplayer')

    layer_names = [l.findall('layername')[0].text for l in layers]
    layer_dict = dict(zip(layer_names, layers))

    if args.layer is None:
        write_names = layer_names
    else:
        write_names = args.layer

    for name in write_names:
        print('Writing metadata for layer {}'.format(name))
        write_xml_meta(layer_dict[name], args.outdir)


if __name__ == "__main__":
    main()
