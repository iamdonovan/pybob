from setuptools import setup

setup(name='pybob',
      version='0.11.4',
      description='Collection of geospatial and other tools I find useful.',
      url='http://github.com/iamdonovan/pybob',
      author='Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      license='MIT',
      packages=['pybob'],
      install_requires=[
                        'numpy', 'scipy', 'matplotlib', 'fiona',
                        'shapely', 'opencv-python', 'pandas', 'geopandas',
                        'scikit-image',
                         ],
      scripts=['bin/dem_coregistration.py', 'bin/generate_panchromatic.py',
               'bin/print_reverb_browse_urls.py', 'bin/print_reverb_granule_names.py'],
      zip_safe=False)
