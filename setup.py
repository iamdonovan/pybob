from setuptools import setup

setup(name='pybob',
    version='0.11',
    description='Collection of geospatial and other tools I find useful.',
    url='http://github.com/iamdonovan/pybob',
    author='Bob McNabb',
    author_email='robertmcnabb@gmail.com',
    license='MIT',
    packages=['pybob'],
    install_requires = [
        'numpy', 'scipy', 'matplotlib', 'fiona', 
        'shapely', 'opencv-python', 'pandas', 'geopandas',
        'scikit-image', 
    ],
    zip_safe=False)
