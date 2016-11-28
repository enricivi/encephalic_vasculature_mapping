import argparse
import re
import os
import warnings
import numpy as np
import tifffile as tf
#from scipy.misc import imsave, imread

def order_file(a, b):
    pattern= r"[^0-9]"
    a= re.sub(pattern, '', a)
    b= re.sub(pattern, '', b)
    if ( long(a) > long(b) ):
        return 1
    elif ( long(a) < long(b) ):
        return -1
    return 0

def main(input, output, z):
    print "opening images and creating stacks..."
    file = sorted( os.listdir(input), cmp=order_file )
    im_stack = None
    for i in xrange( len(file) ):
        f = file[i]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if im_stack == None:
                mono_shape = imread(input + f).shape
                im_stack = np.empty( (len(file), mono_shape[0], mono_shape[1]) )
        im_stack[i] = tf.imread(input + f)
        #im_stack[i] = imread(input + f)

    print "overlapping image and saving..."
    tmp_dx= None
    tmp_sx= None
    for i in xrange( len(im_stack) ):
        if (i >= z) and (i < (len(im_stack)-z)):
            tmp_dx = im_stack[i:i+z+1,:,:]
            tmp_sx = im_stack[i-z:i,:,:]
        else:
            if (i >= z):    #not (i < (len(out_stack)-z-1))
                tmp_dx = im_stack[i:,:,:]
                tmp_sx = im_stack[i-z:i,:,:]
            else:   #not (i >= z)
                tmp_dx = im_stack[i+1:i+z+1,:,:]
                tmp_sx = im_stack[:i+1,:,:]
        tmp_dx= np.amax(tmp_dx, axis=0)
        tmp_sx= np.amax(tmp_sx, axis=0)
        tmp= np.empty((2, tmp_dx.shape[0], tmp_dx.shape[1]))
        tmp[0]= tmp_dx
        tmp[1]= tmp_sx
        tf.imsave(output+str(i)+".tif", np.amax(tmp, axis=0))
        #imsave(output+str(i)+".jpg", np.amax(tmp, axis=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type= str, help="path to input images (TIF)")
    parser.add_argument("output", type= str, help= "where the output images (TIF) will be saved")
    parser.add_argument("z", type= int, help="how many images the script used to overlap (+ -), must be >= 1")

    args = parser.parse_args()
    assert args.z >= 1, "z must be >= 1"
    main(args.input, args.output, args.z)
