#!/usr/bin/env python
import os
import argparse
import errno
import glob
import shutil
import gdal
import numpy as np
import multiprocessing as mp
from pybob.GeoImg import GeoImg
from pybob.coreg_tools import dem_coregistration
from scipy.interpolate import griddata


def batch_wrapper(arg_dict):
    return (dem_coregistration(**arg_dict))


def _argparser():
    parser = argparse.ArgumentParser(description="""Iteratively calculate co-registration parameters for sub-grids of two DEMs, as seen in `Nuth and Kääb (2011)`_.

                .. _Nuth and Kääb (2011): https://www.the-cryosphere.net/5/271/2011/tc-5-271-2011.html""")
    parser.add_argument('primarydem', type=str, help='path to primary DEM to be used for co-registration')
    parser.add_argument('secondarydem', type=str, help='path to secondary DEM to be co-registered')
    parser.add_argument('-a', '--mask1', type=str, default=None,
                        help='Glacier mask. Areas inside of this shapefile will not be used for coregistration [None]')
    parser.add_argument('-b', '--mask2', type=str, default=None,
                        help='Land mask. Areas outside of this mask (i.e., water) \
                             will not be used for coregistration. [None]')
    parser.add_argument('-s', '--mysize', type=int, default=30000,
                        help='determines size of individual blocks in coordinate system units \
                             [30000]')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='Directory to output files to (creates if not already present). [.]')
    parser.add_argument('-i', '--icesat', action='store_true', default=False,
                        help="Process assuming that primary DEM is ICESat data [False].")
    parser.add_argument('-f', '--full_ext', action='store_true', default=False,
                        help="Write full extent of primary DEM and shifted secondary DEM. [False].")
    return parser


def main():
    np.seterr(all='ignore')
    # add primary, secondary, masks to argparse
    # can also add output directory
    parser = _argparser()
    args = parser.parse_args()

    def gen_outdir_name(tDEM):
        xname=np.array2string(np.asarray(np.floor_divide(tDEM.xmin + tDEM.xmax,2),dtype=np.int32))
        yname=np.array2string(np.asarray(np.floor_divide(tDEM.ymin + tDEM.ymax,2),dtype=np.int32))
        return "cr_" + xname + "_" + yname
    
    def collect_subimages(demname,mysize):
        # import data
        myDEM = GeoImg(demname)
#        mybounds = myDEM.find_valid_bbox()
        
#        tx = np.asarray(np.floor_divide(mybounds[1]-mybounds[0],mysize),dtype=np.int32)
#        ty = np.asarray(np.floor_divide(mybounds[3]-mybounds[2],mysize),dtype=np.int32)
        tx = np.asarray(np.floor_divide(myDEM.xmax-myDEM.xmin,mysize),dtype=np.int32)
        ty = np.asarray(np.floor_divide(myDEM.ymax-myDEM.ymin,mysize),dtype=np.int32)
        
        # Divide the DEM into subimages for co-registration
        myDEMs = myDEM.subimages(tx,Ny=ty)
        
        # clear the list for empty DEMs
        myDEMs = [tDEM for tDEM in myDEMs if (np.sum(~np.isnan(tDEM.img)) > 1000)]  
    
        return myDEMs

    def read_shift_params(filename):
        with open(filename) as f:
            lines = f.readlines()      
        return np.fromstring(lines[-2],dtype=np.float,sep="\t")
    
    def read_stats_file(filename):
        with open(filename) as f:
            lines = f.readlines()      
        return np.fromstring(lines[-1][1:-2],dtype=np.float,sep=",")

    def gather_results(outdir):
        # Gather the result
        data_out = []
        myrow=0    
        for cfolder in os.listdir(outdir):
            if os.path.isdir(os.path.join(outdir, cfolder)):
                filename = os.path.join(outdir,cfolder,'coreg_params.txt')
                sfilename = os.path.join(outdir,cfolder,'stats.txt')
                if os.path.getsize(os.path.join(outdir,cfolder,'CoRegistration_Results.pdf')) <  7E4:
                    shutil.rmtree(os.path.join(outdir,cfolder)) #CLEANING
#                    print('YEA')
                else:
                    shift_params = read_shift_params(filename)
                    xy = np.asarray(cfolder.split("_")[1:],dtype=np.int)
                    if os.path.getsize(sfilename)>0:
                        stats = read_stats_file(sfilename)
                        data_out.append(np.concatenate((xy,shift_params,stats)))
                    else:
                        temp=np.empty(5)
                        temp[:]=np.nan
                        data_out.append(np.concatenate((xy,shift_params,temp)))
                    myrow += 1
            
#        dd = np.asarray(data_out)
#        plt.plot(dd[:,0],dd[:,1],'s')
        return np.asarray(data_out)

    def write_geotiff(mymat,myproj,myextent,myres,filename):
        driver = "GTIFF"
        datatype=gdal.GDT_Float32
        drv = gdal.GetDriverByName(driver)
        
        newGdal = drv.Create(filename, mymat.shape[1], mymat.shape[0], 1, datatype)
        newgt = (myextent[0], myres, 0, myextent[3], 0, -myres)
        wa = newGdal.GetRasterBand(1).WriteArray(np.flipud(mymat))
        sg = newGdal.SetGeoTransform(newgt)
        sp = newGdal.SetProjection(myproj)
        newGdal.FlushCache()
        del newGdal
        pass 

    # Divide secondary DEM into grid for co-registration
    myDEMs = collect_subimages(args.secondarydem,args.mysize)
       
    # if the output directory does not exist, create it.
    outdir = os.path.abspath(args.outdir)
    try:
        os.makedirs(outdir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(outdir):
            pass
        else:
            raise
    # generate list of output directories 
    mydirs = [os.path.join(args.outdir,gen_outdir_name(tDEM)) for tDEM in myDEMs]
    mynames = [os.path.join(args.outdir,gen_outdir_name(tDEM)+".tif") for tDEM in myDEMs]
    
    # write out temporary versions of the DEM subimages
    for ix in np.arange(0,len(mynames)):
        myDEMs[ix].write(mynames[ix])
    # get projection information
    myproj = myDEMs[0].proj
    
    # DIDNT WORK ON WINDOWS; TRY AGAIN ON LINUX
    # myDEMs=myDEMs[:15]
    # mydirs=mydirs[:15]

    # get a dictionary of arguments for each of the different DEMs,
    # starting with the common arguments (primary dem, glacier mask, etc.)
    # dem_coregistration(primaryDEM, secondaryDEM, glaciermask=None,
    # landmask=None, outdir='.', pts=False, full_ext=False, return_var=True):
    arg_dict = {'primaryDEM': args.primarydem,
                'glaciermask': args.mask1, 
                'landmask': args.mask2, 
                'pts': args.icesat, 
                'full_ext': args.full_ext, 
                'return_var': False}
    u_args = [{'outdir': mydirs[ix], 'secondaryDEM': mynames[ix]} for ix in np.arange(0,len(myDEMs))]
    for d in u_args:
        d.update(arg_dict)

    # Iterate lists using parallel processors
    pool = mp.Pool(processes=20)
    pool.map(batch_wrapper, u_args)
    pool.close()

    # remove all temporary tif files
    for ff in glob.glob(os.path.join(outdir,"**","*.tif"), recursive=True):
        os.remove(ff)
    

    # def gather_results():
    # fdsa
    dd = gather_results(outdir)
    np.savetxt(os.path.sep.join([outdir, 'Gridded_Coreg_Parameters.txt']),dd,fmt='%f',delimiter=',')
    
    # define the grid, was complicated due to different sizes of the subimages. 
    myxsize = np.unique(np.diff(np.unique(dd[:,0])))
    myysize = np.unique(np.diff(np.unique(dd[:,1])))
    mysize = np.nanmean([myxsize, myysize])
    myextent = [np.min(dd[:,0])-mysize/2, np.max(dd[:,0])+mysize/2,
                np.min(dd[:,1])-mysize/2, np.max(dd[:,1])+mysize/2]
    xgrid = np.arange(np.min(dd[:,0]),np.max(dd[:,0])+1,mysize)
    ygrid = np.arange(np.min(dd[:,1]),np.max(dd[:,1])+1,mysize)
    
    X, Y = np.meshgrid(xgrid,ygrid)
    
    # Interpolate    
    dx = griddata((dd[:,:2]),dd[:,2],(X,Y))
    dy = griddata((dd[:,:2]),dd[:,3],(X,Y))
    dz = griddata((dd[:,:2]),dd[:,4],(X,Y))
    dm = griddata((dd[:,:2]),np.sqrt(dd[:,2]**2+dd[:,3]**2),(X,Y))
    drmse = griddata((dd[:,:2]),dd[:,8],(X,Y))
    dcount = griddata((dd[:,:2]),dd[:,9],(X,Y))

    # find all pixels that are not under points, and set to nan. 
    mymask = np.ones(X.size)
    for ix in np.arange(0,dd.shape[0]):
        mydist = np.sqrt(np.square(X.flatten()-dd[ix,0])+np.square(Y.flatten()-dd[ix,1]))
        if np.min(mydist)<np.sqrt(np.square(mysize)*2):
            mix = np.argmin(mydist)
            mymask[mix] = 0
    mymask=mymask.reshape(X.shape)
    
    dx[mymask==1]=np.nan
    dy[mymask==1]=np.nan
    dz[mymask==1]=np.nan
    dm[mymask==1]=np.nan
    drmse[mymask==1]=np.nan
    dcount[mymask==1]=np.nan
    
    write_geotiff(dz, myproj, myextent, mysize, os.path.sep.join([outdir, 'VerticalShift.tif']))
    write_geotiff(dx, myproj, myextent, mysize, os.path.sep.join([outdir, 'XShift.tif']))
    write_geotiff(dy, myproj, myextent, mysize, os.path.sep.join([outdir, 'YShift.tif']))
    write_geotiff(dm, myproj, myextent, mysize, os.path.sep.join([outdir, 'ShiftMagnitude.tif']))
    write_geotiff(drmse, myproj, myextent, mysize, os.path.sep.join([outdir, 'RMSE.tif']))
    write_geotiff(dcount, myproj, myextent, mysize, os.path.sep.join([outdir, 'NumberSamples.tif']))


if __name__ == "__main__":
    main()
    #    u_args = main()

#    # Iterate lists using parallel processors
#    pool = mp.Pool(processes=10)
#    pool.map(batch_wrapper, u_args)
#    results_objects = []
#    for proc in np.arange(0,len(u_args)):
#        pool.apply_async(batch_wrapper, args=(u_args[proc],))
#        time.sleep(5)
#    pool.close()
