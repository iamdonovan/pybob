import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


class VelField(object):

    def __init__(self, in_filename, in_dir='.'):
        self.filename = in_filename
        self.in_dir_path = in_dir
        self.in_dir_abs_path = os.path.abspath(in_dir)

        infile = gpd.GeoDataFrame.from_file(in_dir + os.path.sep + in_filename)

        self.x = np.empty(0)
        self.y = np.empty(0)

        for pt in infile['geometry']:
            self.x = np.append(self.x, pt.x)
            self.y = np.append(self.y, pt.y)

        self.ux = infile['ux'].as_matrix()
        self.uy = infile['uy'].as_matrix()
        self.corr_max = None
        self.dcorr = None
        self.z = None
        # self.date = infile['centerdate'].as_matrix()
        if 'corr_max' in infile.keys():
            self.corr_max = infile['corr_max'].as_matrix()
        if 'dcorr' in infile.keys():
            self.dcorr = infile['dcorr'].as_matrix()
        if 'z' in infile.keys():
            self.z = infile['z'].as_matrix()
        # self.img1 = infile['img1'].as_matrix()

    def display_vector_field(self, fig=None, cmap='spring', clim=None):

        if fig is None:
            fig = plt.figure()
        else:
            fig.hold(True)  # make sure we don't do away with anything

        vv = np.sqrt(self.ux**2 + self.uy**2)
        qplt = plt.quiver(self.x, self.y, self.ux, self.uy, vv, cmap=cmap)

        if clim is not None:
            qplt.set_clim(vmin=clim[0], vmax=clim[1])

        ax = fig.gca()  # get current axes
        ax.set_aspect('equal')  # set equal aspect
        ax.autoscale(tight=True)  # set axes tight

        fig.show()

        return fig
