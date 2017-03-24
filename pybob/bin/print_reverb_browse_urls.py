#!/usr/bin/env python
import pandas as pd, argparse, numpy as np

def main():
	# set up command line arguments
	parser = argparse.ArgumentParser(description="Read a Table of NASA Reverb Data, return the URLs of the browse images it points to.", formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('csvfile', action='store', type=str, default='.', help='name of csv file to read in')
	parser.add_argument('--max_cloud', action='store', type=float, default=100.0, help='maximum percent cloud cover for images [100]')
	args = parser.parse_args()

	mydata = pd.read_csv(args.csvfile)

	for index, row in mydata.iterrows():
		try:
			if float(row['Cloud Cover']) <= args.max_cloud:
				try:
					print row['Browse URLs'].split(',')[0]
				except AttributeError:
					continue
			elif np.isnan(float(row['Cloud Cover'])):
				try:
					print row['Browse URLs'].split(',')[0]
				except AttributeError:
					continue
		except ValueError:
			continue

if __name__ == "__main__":
    main()
