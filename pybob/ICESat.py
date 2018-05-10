from __future__ import print_function
from future_builtins import zip
import os
import h5py
import numpy as np
import pyproj
import matplotlib.pyplot as plt
#import numpy as np


def get_file_info(in_filestring):
    in_filename = os.path.basename(in_filestring)
    in_dir = os.path.dirname(in_filestring)
    if in_dir == '':
        in_dir = '.'
    return in_filename, in_dir


def find_keyname(keys, subkey, mode='first'):
    out_keys = [(i, k) for i, k in enumerate(keys) if subkey in k]
    if mode == 'first':
        return out_keys[0]
    elif mode == 'last':
        return out_keys[-1]
    else:
        return out_keys


class ICESat(object):
    """Create an ICESat dataset from an HDF5 file containing ICESat data.

    Parameters
    ----------
    in_filename : string
        Filename (optionally with path) of HDF5 file to be read in.
    in_dir : string, optional
        Directory where in_filename is located. If not given, the directory
        will be determined from the input filename.
    cols : list of strings, optional
        Columns to read in and set as attributes in the ICESat object.
        Default values are ['lat', 'lon', 'd_elev', 'd_UTCTime'], and will
        be added if not specified.
    ellipse_hts : bool, optional
        Convert elevations to ellipsoid heights. Default is True.

    Examples
    --------
    >>> ICESat('donjek_icesat.h5', ellipse_hts=True)
    """
    def __init__(self, in_filename, in_dir=None, cols=['lat', 'lon', 'd_elev', 'd_UTCTime'], ellipse_hts=True):
        if in_dir is None:
            in_filename, in_dir = get_file_info(in_filename)
        self.filename = in_filename
        self.in_dir_path = in_dir
        self.in_dir_abs_path = os.path.abspath(in_dir)

        h5f = h5py.File(os.path.join(self.in_dir_path, self.filename))
        h5data = h5f['ICESatData']  # if data come from Anne's ICESat scripts, should be the only data group
        data_names = h5data.attrs.keys()
        # make sure that our default attributes are included.
        for c in ['lat', 'lon', 'd_elev', 'd_UTCTime']:
            if c not in cols:
                cols.append(c)
                
        for c in cols:
            ind, _ = find_keyname(data_names, c)
            # set the attribute, removing d_ from the attribute name if it exists
            setattr(self, c.split('d_', 1)[-1], h5data[ind, :])
            if c == 'lon':
                # return longitudes ranging from -180 to 180 (rather than 0 to 360)
                self.lon[self.lon > 180] = self.lon[self.lon > 180] - 360

        self.data_names = data_names
        self.column_names = cols
        self.h5data = h5data
        self.ellipse_hts = False
        self.proj = pyproj.Proj(init='epsg:4326')
        self.x = self.lon
        self.y = self.lat
        self.xy = list(zip(self.x, self.y))

        if ellipse_hts:
            self.to_ellipse()

    def to_ellipse(self):
        """ Convert ICESat elevations to ellipsoid heights, based on the data read in."""
        # fgdh = find_keyname(self.data_names, 'd_gdHt')
        fde, _ = find_keyname(self.data_names, 'd_deltaEllip')
        de = self.h5data[fde, :]
        self.ellipse_hts = True
        self.elev = self.elev + de

    def from_ellipse(self):
        """ Convert ICESat elevations from ellipsoid heights, based on the data read in."""
        fde, _ = find_keyname(self.data_names, 'd_deltaEllip')
        de = self.h5data[fde, :]
        self.ellipse_hts = False
        self.elev = self.elev - de

    def to_shp(self, out_filename):
        """ Write ICESat data to shapefile (NOT IMPLEMENTED YET) """
        pass

    def clean(self, el_limit=-500):
        """ Remove all elevation points below a given elevation.
        """
        mykeep = self.elev > el_limit
        self.x = self.x[mykeep]
        self.y = self.y[mykeep]
        self.elev = self.elev[mykeep]
        self.xy = list(zip(self.x,self.y))
        pass
        
    def clip(self, bounds):
        """ Remove ICEsat data that falls outside of a given bounding box.
        
        Parameters
        ----------
        bounds: array-like
            bounding box to clip to, given as xmin, xmax, ymin, ymax
        """
        xmin, xmax, ymin, ymax = bounds
        mykeep_x = np.logical_and(self.x > xmin, self.x < xmax)
        mykeep_y = np.logical_and(self.y > ymin, self.y < ymax)
        mykeep = np.logical_and(mykeep_x, mykeep_y)
        
        self.x=self.x[mykeep]
        self.y=self.y[mykeep]
        self.elev=self.elev[mykeep]
        self.xy = list(zip(self.x,self.y))
        pass
    
    def get_bounds(self, geo=False):
        """Return bounding coordinates of the dataset.

        Parameters
        ----------
        geo : bool, optional
            Return geographic (lat/lon) coordinates (default is False)
        
        Example
        -------
        >>> xmin, xmax, ymin, ymax = my_icesat.get_bounds()
        """
        if not geo:
            return self.x.min(), self.x.max(), self.y.min(), self.y.max()
        else:
            return self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()

    def project(self, dest_proj):
        """Project the ICESat dataset to a given coordinate system, using pyproj.transform.
        ICESat.project does not overwrite the lat/lon coordinates, so calling
        ICESat.project will only update self.x, self.y for the dataset, as well as self.proj.

        Parameters
        ----------
        dest_proj : str or pyproj.Proj
            Coordinate system to project the dataset into. If dest_proj is a string,
            ICESat.project() will create a pyproj.Proj instance with it.

        Examples
        --------
        Project icesat_data to Alaska Albers (NAD83) using epsg code:
        >>> icesat_data.project('epsg:3338')
        """
        if isinstance(dest_proj, str):
            dest_proj = pyproj.Proj(init=dest_proj)
        wgs84 = pyproj.Proj(init='epsg:4326')
        self.x, self.y = pyproj.transform(wgs84, dest_proj, self.lon, self.lat)
        self.xy = list(zip(self.x, self.y))
        self.proj = dest_proj

    def display(self, fig=None, extent=None, sfact=1, showfig=True, geo=False, **kwargs):
        """
        Plot ICESat tracks in a figure.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot tracks in. If not set, creates a new figure.
        extent : array-like, optional
            Spatial extent to limit the figure to, given as xmin, xmax, ymin, ymax.
        sfact : int, optional
            Factor by which to reduce the number of points plotted.
            Default is 1 (i.e., all points are plotted).
        showfig : bool, optional
            Open the figure window. Default is True.
        geo : bool, optional
            Plot tracks in lat/lon coordinates, rather than projected coordinates.
            Default is False.
        **kwargs :
            Optional keyword arguments to pass to matplotlib.pyplot.plot
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle pointing to the matplotlib Figure created (or passed to display).
        """
        if fig is None:
            fig = plt.figure(facecolor='w')
            # fig.hold(True)
        # else:
            # fig.hold(True)

        if extent is None:
            extent = self.get_bounds(geo=geo)
        else:
            xmin, xmax, ymin, ymax = extent

        # if we aren't told which marker to use, pick one.
        if 'marker' not in kwargs:
            this_marker = '.'
        else:
            this_marker = kwargs['marker']
            kwargs.pop('marker')

        if geo:
            plt.plot(self.lon[::sfact], self.lat[::sfact], marker=this_marker, ls='None', **kwargs)
        else:
            plt.plot(self.x[::sfact], self.y[::sfact], marker=this_marker, ls='None', **kwargs)

        ax = fig.gca()  # get current axes
        ax.set_aspect('equal')    # set equal aspect
        ax.autoscale(tight=True)  # set axes tight
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        
        if showfig:
            fig.show()  # don't forget this one!

        return fig