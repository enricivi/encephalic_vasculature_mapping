import argparse
import re
import os
import warnings
import numpy as np
import tifffile as tf

def main(input, output, z):
    print "opening images..."
    im_stack = tf.imread(input)

    print "overlapping image and saving..."
    tmp_out = np.empty(im_stack.shape) 
    tmp_dx = None
    tmp_sx = None
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
        tmp_out[i]= np.amax(tmp, axis=0)
    tf.imsave(output+"overlap.tif", tmp_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type= str, help="path to input image (TIF)")
    parser.add_argument("output", type= str, help= "where the output image (TIF) will be saved")
    parser.add_argument("z", type= int, help="how many images the script used to overlap (+ -), must be >= 1")

    args = parser.parse_args()
    assert args.z >= 1, "z must be >= 1"
    main(args.input, args.output, args.z)
