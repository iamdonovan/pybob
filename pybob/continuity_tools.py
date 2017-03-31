import os
from scipy.interpolate import griddata, interp1d
from scipy.integrate import odeint
from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from pybob.GeoImg import GeoImg
from pybob.VelField import VelField
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd


# Cell class defines a cell over which we are evaluating flux divergence,
# and thus calculating thickness. Includes normal vectors at upstream and
# downstream boundaries, as well as midpoints of upstream and downstream
# boundaries (eventually, these are x,y coordinates tied to thickness values)
class Cell(Polygon):

    def __init__(self, Flow1, Flow2):
        Coords = Flow1 + list(reversed(Flow2))
        Polygon.__init__(self, Coords)

        self.x = self.centroid.x
        self.y = self.centroid.y
        self.n_up = norm_vector(np.array(Flow1[0]) - np.array(Flow2[0]))
        self.n_dn = norm_vector(np.array(Flow2[-1]) - np.array(Flow1[-1]))

        self.x_up = (Flow2[0][0] + Flow1[0][0])/2
        self.x_dn = (Flow2[-1][0] + Flow1[-1][0])/2
        self.y_up = (Flow2[0][1] + Flow1[0][1])/2
        self.y_dn = (Flow2[-1][1] + Flow1[-1][1])/2

    def display(self, fig=None):

        if fig is None:
            fig = plt.figure()
        else:
            fig.hold(True)  # make sure we don't do away with anything

        # first, plot the boundary of the cell
        plt.plot(self.boundary.xy[0], self.boundary.xy[1], color='b', linewidth=1.5)
        plt.quiver(self.x_up, self.y_up, self.n_up[0], self.n_up[1], color='k')
        plt.quiver(self.x_dn, self.y_dn, self.n_dn[0], self.n_dn[1], color='r')
        plt.plot(self.x, self.y, 'r*', markersize=12)


# ditto for 'flowline'
class FlowLine(LineString):

    def __init__(self, Coords):
        LineString.__init__(self, Coords)
        self.x, self.y = self.xy
        self.linedist = [self.project(Point(p)) for p in self.coords]


# load spatially defined data such as a DEM, dh/dt map/profile, SMB map/profile.
# extensions recognized are .shp, .tif, .csv
# data are assumed to be f(x,y) - if not, set ftype='z' (or anything other than xy)
# dataName is assumed to be 'z' (this is the field it will look for in .shp, .csv files)
# units are assumed to be meters (or m/yr) - if they aren't,
# be sure to specify this with unitConv flag, i.e. unitConv=365.25 for m/d
class SpatialData():

    def __init__(self, in_filename, in_dir='.', unitConv=1, ftype='xy', dataName='z'):
        FileExt = in_filename.split('.')[-1]
        if FileExt.lower() == 'tif' or FileExt.lower() == 'tiff':
            tmp = GeoImg(in_filename, in_dir=in_dir)
            ndv = tmp.gd.GetRasterBand(1).GetNoDataValue()
            X, Y = tmp.xy()
            self.x = X.reshape(-1)
            self.y = Y.reshape(-1)
            self.c, self.r = tmp.img.shape
            self.data = tmp.img.reshape(-1) * unitConv
            self.data[self.data == ndv] = np.nan
            self.img = True
        elif FileExt.lower() == 'shp':
            tmp = gpd.GeoDataFrame.from_file(in_dir + os.path.sep + in_filename)
            self.x = np.empty(0)
            self.y = np.empty(0)
            for pt in tmp['geometry']:
                self.x = np.append(self.x, pt.x)
                self.y = np.append(self.y, pt.y)
            # not sure how people would call these things
            # just assume that the default is going to be 'z'
            self.data = tmp[dataName] * unitConv
            self.img = False

        elif FileExt.lower() == 'csv':
            tmp = pd.read_csv(in_dir + os.path.sep + in_filename, sep=',|;', engine='python')
            if ftype == 'xy':
                self.x = tmp['x']
                self.y = tmp['y']
            else:
                self.x = tmp['z']
                self.y = None
            self.data = tmp[dataName] * unitConv
            self.img = False

        self.xy = ftype == 'xy'

    def interpolate(self, pts):
        if self.xy:
            interpData = griddata((self.x, self.y), self.data, (pts[0], pts[1]))
        else:
            f = interp1d(self.x, self.data)
            interpData = f(pts)
        return interpData

    def display(self, fig=None, clim=None, extent=None, cmap='RdBu'):

        if fig is None:
            fig = plt.figure()
        else:
            fig.hold(True)  # make sure we don't do away with anything

        if self.xy:
            if self.img:
                if extent is None:
                    extent = [min(self.x), max(self.x), min(self.y), max(self.y)]

                imshw = plt.imshow(self.data.reshape(self.c, self.r), extent=extent, cmap=cmap)
                if clim is not None:
                    imshw.set_clim(vmin=clim[0], vmax=clim[1])
                plt.colorbar()
                ax = fig.gca()  # get current axes
                ax.set_aspect('equal')    # set equal aspect
                ax.autoscale(tight=True)  # set axes tight

                fig.show()  # don't forget this one!
                return fig
        else:
            return


class VelData():

    def __init__(self, in_filename, in_dir='.', unitConv=1):
        FileExt = in_filename.split('.')[-1]
        if FileExt == 'shp':
            tmp = VelField(in_filename, in_dir)
            self.x = tmp.x
            self.y = tmp.y
            self.ux = tmp.ux * unitConv
            self.uy = tmp.uy * unitConv
            self.u = np.sqrt(self.ux**2 + self.uy**2)
            self.unitConv = unitConv
        elif FileExt == 'csv':
            tmp = pd.read_csv(in_dir + os.path.sep + in_filename, sep=',|;', engine='python')
            self.x = tmp['x']
            self.y = tmp['y']
            self.ux = tmp['ux'] * unitConv
            self.uy = tmp['uy'] * unitConv
            self.u = np.sqrt(self.ux**2 + self.uy**2)
            self.unitConv = unitConv

    def display(self, fig=None, cmap='spring', clim=None):

        if fig is None:
            fig = plt.figure()
        else:
            fig.hold(True)  # make sure we don't do away with anything

        qplt = plt.quiver(self.x, self.y, self.ux, self.uy, self.u, cmap=cmap)

        if clim is not None:
            qplt.set_clim(vmin=clim[0], vmax=clim[1])

        ax = fig.gca()  # get current axes
        ax.set_aspect('equal')    # set equal aspect
        ax.autoscale(tight=True)  # set axes tight

        fig.show()

        return fig

    def reverse_flow(self):
        self.ux = -self.ux
        self.uy = -self.uy


# okay, this is the big one: where we take start (end), and walk through the flowbands until we get to the end
def calculate_over_fluxgate(start, vdata, surfdata, bdot, dhdt, beddata, end=None, offset=100, gamma=0.9, plotfig=None):

    startgate = LineString(start)  # just to make sure...
    startpt = startgate.coords[0]
    flowLine1 = get_flowline(startpt, vdata, end=end)
    fluxDist = 0
    gates = []
    flowcellnumber = []
    bed = np.empty(0)

    while (startgate.length - fluxDist) > offset:
        startpt = startgate.interpolate(fluxDist+offset).coords[0]
        flowLine2 = get_flowline(startpt, vdata, end=end)

        if plotfig is not None:
            plt.plot(flowLine1.x, flowLine1.y, 'b')
            plt.plot(flowLine2.x, flowLine2.y, 'b')

        thesegates, thisbed = calculate_bed_elevs(flowLine1, flowLine2, vdata, surfdata,
                                                  bdot, dhdt, beddata, gamma, offset, plotfig)

        for j, x in enumerate(thesegates[0]):
            gates.append(Point(x, thesegates[1][j]))
            flowcellnumber.append(j)

        bed = np.append(bed, thisbed)

        flowLine1 = flowLine2
        fluxDist += offset

    return gates, bed, flowcellnumber


# this is the one where the magic happens
def calculate_bed_elevs(flowline1, flowline2, vdata, surfdata, bdot, dhdt, beddata, gamma, offset, plotfig=None):

    minlength = min(len(flowline1.x), len(flowline2.x))
    # make sure we agree on lengths of flowlines
    flow1 = FlowLine(zip(flowline1.x[0:minlength], flowline1.y[0:minlength]))
    flow2 = FlowLine(zip(flowline2.x[0:minlength], flowline2.y[0:minlength]))

    # create each cell in this flowband
    cells, widths, centers, gates = get_flow_cells(flow1, flow2, offset)

    if plotfig is not None:
        for c in cells:
            c.display(plotfig)

    # get z, dhdt, bdot for centroids
    # centersurf = griddata((surfdata.x, surfdata.y), surfdata.data, centers)
    gatesurf = surfdata.interpolate(gates)
    centersurf = surfdata.interpolate(centers)
    # if dhdt, bdot are f(z), can't just run griddata on them.
    if bdot.xy:
        centerbdot = bdot.interpolate(centers)
    else:
        centerbdot = bdot.interpolate(centersurf)
    # centerdhdt = griddata((dhdt.x, dhdt.y), dhdt.data, centers)
    # centerbdot = griddata((bdot.x, bdot.y), bdot.data, centers)
    if dhdt.xy:
        centerdhdt = dhdt.interpolate(centers)
    else:
        centerdhdt = dhdt.interpolate(centersurf)

    # need to get x,y components of velocity, and dot with the normal vector of each cell
    ux = griddata((vdata.x, vdata.y), vdata.ux, gates)
    uy = griddata((vdata.x, vdata.y), vdata.uy, gates)

    bed = np.array([beddata.interpolate((gates[0][0], gates[1][0]))])

    thickness = np.array(gatesurf[0] - bed[0])
    # the flux in is the flux at the upstream gate
    # qup = np.dot(np.array(ux[0],uy[0]), cells[0].n_up) * thickness * widths[0] * gamma
    qup = np.sqrt(ux[0]**2 + uy[0]**2) * thickness * widths[0] * gamma

    for j, c in enumerate(cells):
        # the thickness at the downstream gate is (qin - rhs)/(v_down * width_down * gamma)
        rhs = (centerbdot[j] - centerdhdt[j]) * c.area  # a somewhat crude approximation, maybe a better option?
        thickness = np.append(thickness, (qup - rhs) /
                              (np.dot(np.array(ux[j+1], uy[j+1]), c.n_dn) * widths[j+1] * gamma))
        # thickness = np.append(thickness, (qup - rhs) / (np.sqrt(ux[j+1]**2 + uy[j+1]**2) * widths[j+1] * gamma))
        bed = np.append(bed, gatesurf[j+1] - thickness[-1])  # surface at downstream boundary
        #                                                      minus calculated thicknes (at downstream bndry)

        # j+1 is the downstream index (j starts at 0, corresponds to first upstream gate)
        qup = thickness[-1] * widths[j+1] * np.dot(np.array(ux[j+1], uy[j+1]), -c.n_dn)
        # qup = thickness[-1] * widths[j+1] * np.sqrt(ux[j+1]**2 + uy[j+1]**2)

    return gates, bed


# get a flowline from the current location
def get_flowline(start, vdata, maxt=5000, end=None):
    # should figure out what the realtionship between t, ground distance is...
    t = np.arange(0, maxt / vdata.unitConv, 10 / vdata.unitConv)
    sol = odeint(velocityvect, start, t, args=(vdata,))

    this_flowline = cut_flowline(FlowLine(sol[np.isfinite(sol[:, 0]), :]), end)
    return this_flowline


# interpolate the current location to the velocity
def velocityvect(xy, t, vdata):
    uxy = np.empty(2)
    uxy[0] = griddata((vdata.x, vdata.y), vdata.ux, (xy[0], xy[1]))
    uxy[1] = griddata((vdata.x, vdata.y), vdata.uy, (xy[0], xy[1]))
    return uxy


# calculate the distance between two points
def distance(pt1, pt2):  # use shapely!
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return np.sqrt(dx*dx + dy*dy)


# returns the distance along a line described by x,y
def linedistance(x, y):
    tmp = LineString(zip(x, y))
    return [tmp.project(Point(p)) for p in zip(x, y)]


# find the width of each flowband at each vertex
def flowbandwidth(flowline1, flowline2, offsets):
    widths = np.empty(0)
    for i, pt in enumerate(flowline1.x):
        widths = np.append(widths, Point(flowline1.x[i],
                           flowline1.y[i]).distance(Point(flowline2.x[i], flowline2.y[i])))
    return widths


# get the center positions and areas for each of our cells on which to calculate continuity
# can add surf information back in later if it makes sense (?)
def get_flow_cells(flowline1, flowline2, offset):
    flowcells = []
    centerx = []
    centery = []
    widths = []
    thisdist = 0
    thisind1 = 0
    thisind2 = 0

    maxdist = min(flowline1.length, flowline2.length)
    gatex = [(flowline1.x[0]+flowline2.x[0])/2]
    gatey = [(flowline1.y[0]+flowline2.y[0])/2]

    lastpt1 = Point(flowline1.coords[0])
    lastpt2 = Point(flowline2.coords[0])
    widths.append(lastpt1.distance(lastpt2))

    while thisdist < maxdist:
        thispt1 = flowline1.interpolate(thisdist+widths[-1])
        # now, get the perp to flow at this point
        nextind1 = np.searchsorted(flowline1.linedist, thisdist+widths[-1])
        if nextind1 == 0:
            flowvect = np.array(flowline1.coords[1]) - np.array(flowline1.coords[0])
        elif nextind1 == len(flowline1.x):
            break
        else:
            flowvect = np.array(flowline1.coords[nextind1]) - np.array(flowline1.coords[nextind1-1])

        perp = norm_vector(flowvect)
        f2dist = thispt1.distance(flowline2)
        flowperp = LineString([(thispt1.x - perp[0]*1.5*f2dist, thispt1.y - perp[1]*1.5*f2dist),
                               (thispt1.x + perp[0]*1.5*f2dist, thispt1.y + perp[1]*1.5*f2dist)])
        thispt2 = flowline2.intersection(flowperp)
        if not flowline2.intersects(flowperp):
            break
        nextind2 = np.searchsorted(flowline2.linedist, flowline2.project(thispt2))

        # there might be a less convoluted way to do this...
        f1x = np.concatenate([np.asarray([lastpt1.x]), flowline1.x[thisind1:nextind1], np.asarray([thispt1.x])])
        f1y = np.concatenate([np.asarray([lastpt1.y]), flowline1.y[thisind1:nextind1], np.asarray([thispt1.y])])

        f2x = np.concatenate([np.asarray([lastpt2.x]), flowline2.x[thisind2:nextind2], np.asarray([thispt2.x])])
        f2y = np.concatenate([np.asarray([lastpt2.y]), flowline2.y[thisind2:nextind2], np.asarray([thispt2.y])])

        flow1coords = list(zip(f1x, f1y))
        flow2coords = list(zip(f2x, f2y))

        newcell = Cell(flow1coords, flow2coords)
        flowcells.append(newcell)

        centerx.append(newcell.x)
        centery.append(newcell.y)

        gatex.append(newcell.x_dn)
        gatey.append(newcell.y_dn)

        thisind1 = nextind1
        thisind2 = nextind2

        lastpt1 = thispt1
        lastpt2 = thispt2
        widths.append(lastpt1.distance(lastpt2))

        thisdist += offset

    return flowcells, widths, (centerx, centery), (gatex, gatey)


# clip flowlines to another line
def cut_flowline(thisflowline, cutline):
    if cutline is not None and thisflowline.intersects(cutline):
        IntPt = thisflowline.intersection(cutline)
        FlowDst = thisflowline.project(IntPt)
        IntInd = np.searchsorted(thisflowline.linedist, FlowDst)
        CutCoords = thisflowline.coords[0:IntInd+1]  # add one to include IntInd
        if thisflowline.linedist[IntInd] < FlowDst:
            CutCoords.append((IntPt.x, IntPt.y))
        thisflowline = FlowLine(CutCoords)
    return thisflowline


# normal vector!
def norm_vector(a):
    b = np.array([a[1], -a[0]])
    b = b / np.linalg.norm(a)
    return b
