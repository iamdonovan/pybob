# pybob
A collection of geospatial and other tools that I find useful.

## Installing Pybob

```sh
# Install the required libraries
conda install h5py numba descartes gdal
pip install llc

# Clone the repository (using git ssh)
git clone git@github.com:iamdonovan/pybob.git

# install the development verion in editing mode
pip install -e [path2folder/pybob]
```

## Basic usage

```python
from pybob import GeoImg

# Open a geotiff:
test = GeoImg.GeoImg('myraster.tif')

# plot raster to screen
test.diplay()

# crop raster to extent
test.crop_to_extent([xmin, xmax, ymin, ymax], band)

```
