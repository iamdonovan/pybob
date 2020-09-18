import os
from glob import glob
from skimage.io import imread, imsave
from pybob.image_tools import crippen_filter
from pybob.bob_tools import mkdir_p


mkdir_p('filt')
for im in glob('*.tif'):
    print(im)
    img = imread(im)
    img_filt = crippen_filter(img, scan_axis=1)

    imsave(os.path.join('filt', im), img_filt)