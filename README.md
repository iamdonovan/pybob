# pybob

## Installing Pybob

1. Install the required libraries

```sh
conda install h5py numba descartes gdal
pip install llc
```

2. Clone the git repository
3. pip install [path2folder]

## Basic usage

```python
from pybob import GeoImg

# Open a geotiff:
test = GeoImg.GeoImg('myraster.tif')

# plot raster to screen
test.diplay()

# crop raster to extent
test.crop_to_extent([xmin, xmax, ymin, ymax])

```

