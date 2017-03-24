#!/usr/bin/env python
import glob, pandas as pd, argparse

def main():
	# set up command line arguments
	parser = argparse.ArgumentParser(description="Read a folder of browse images downloaded from NASA reverb, return the granule name for each scene.", formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('csvfile', action='store', type=str, default='.', help='name of reverb csv file to read in')
	parser.add_argument('--gid', action='store_true', default=False, help='print Granule ID instead of Granule UR [False].')
	args = parser.parse_args()

	mydata = pd.read_csv(args.csvfile)

	browselist = glob.glob('*.jpg')

	for img in browselist:
		if args.gid:
			print mydata[mydata['Browse URLs'].str.contains(img)]['Producer Granule ID'].values[0]
		else:
			print mydata[mydata['Browse URLs'].str.contains(img)]['Granule UR'].values[0]			

if __name__ == "__main__":
    main()
