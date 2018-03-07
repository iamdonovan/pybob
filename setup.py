from setuptools import setup

setup(name='pybob',
      version='0.15',
      description='Collection of geospatial and other tools I find useful.',
      url='http://github.com/iamdonovan/pybob',
      author='Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      license='MIT',
      packages=['pybob'],
      install_requires=[
                        'numpy', 'scipy', 'matplotlib', 'fiona',
                        'shapely', 'opencv-python', 'pandas', 'geopandas',
                        'scikit-image', 'gdal', 'h5py', 'pyproj'
                         ],
      scripts=['bin/dem_coregistration.py', 'bin/generate_panchromatic.py',
               'bin/print_reverb_browse_urls.py', 'bin/print_reverb_granule_names.py',
               'bin/find_aster_dem_pairs.py', 'bin/image_footprint.py',
               'bin/write_qgis_meta.py'],
      zip_safe=False)
