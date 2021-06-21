from setuptools import setup

setup(name='pybob',
      version='0.25.1',
      description='Collection of geospatial and other tools I find useful.',
      url='http://github.com/iamdonovan/pybob',
      author='Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      license='MIT',
      packages=['pybob'],
      install_requires=[
                        'numpy', 'scipy', 'matplotlib', 'fiona', 'pyvips',
                        'shapely', 'opencv-python', 'pandas', 'geopandas',
                        'scikit-image>=0.18', 'gdal', 'h5py', 'pyproj', 'llc', 'numba', 'descartes',
                        'sphinx-argparse'],
      scripts=['bin/dem_coregistration.py', 'bin/generate_panchromatic.py',
               'bin/find_aster_dem_pairs.py', 'bin/image_footprint.py',
               'bin/write_qgis_meta.py', 'bin/dem_difference.py',
               'bin/calculate_mmaster_dh_curves.py', 'bin/image_footprint_from_met.py',
               'bin/dem_coregistration_grid.py','bin/extract_ICESat.py',
               'bin/write_utm_zones.py', 'bin/register_landsat.py',
               'bin/orthorectify_landsat.py', 'bin/convert_elevations.py'],
      zip_safe=False)
