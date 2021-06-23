from pathlib import Path
from setuptools import find_packages, setup

readme = Path(__file__).parent / 'README.md'

setup(name='pybob',
      version='0.26.1',
      description='Collection of geospatial and other tools I find useful.',
      long_description=readme.read_text(),
      long_description_content_type='text/markdown',
      url='https://github.com/iamdonovan/pybob',
      doc_url='https://pybob.readthedocs.io/',
      author='Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      maintainer='iamdonovan',
      license='MIT',
      license_file='LICENSE',
      include_package_data=True,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      packages=['pybob'],
      python_requires='>=3.7',
      install_requires=[
                        'numpy', 'scipy', 'matplotlib', 'fiona', 'pyvips',
                        'shapely', 'opencv-python', 'pandas', 'geopandas',
                        'scikit-image>=0.18', 'gdal', 'h5py', 'pyproj', 'numba', 'descartes',
                        'sphinx-argparse'],
      entrypoints={
          'console_scripts': [
              'calculate_mmaster_dh_curves = pybob.tools.calculate_mmaster_dh_curves:main',
              'convert_elevations = pybob.tools.convert_elevations:main',
              'dem_coregistration = pybob.tools.dem_coregistration:main',
              'dem_coregistration_grid = pybob.tools.dem_coregistration_grid:main',
              'dem_difference = pybob.tools.dem_difference:main',
              'extract_ICESat = pybob.tools.extract_ICESat:main',
              'find_aster_dem_pairs = pybob.tools.find_aster_dem_pairs:main',
              'generate_panchromatic = pybob.tools.generate_panchronmatic:main',
              'image_footprint = pybob.tools.image_footprint:main',
              'image_footprint_from_met = pybob.tools.image_footprint_from_met:main',
              'orthorectify_landsat = pybob.tools.orthorectify_landsat:main',
              'register_landsat = pybob.tools.register_landsat:main',
              'write_utm_zones = pybob.tools.write_utm_zones:main',
          ]
      },
      zip_safe=False)
