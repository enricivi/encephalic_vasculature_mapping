import argparse
import tifffile as tf
from scipy.signal import medfilt
from skimage.filters import threshold_otsu
import numpy as np

def main(input, output):
    print "open image..."
    im_stack= tf.imread(input)
    print "median filter..."
    im_stack = medfilt(im_stack)
    print "evaluating threshold..."
    for i in xrange( len(im_stack) ):
        im = im_stack[i]
        im = (im >= threshold_otsu(im)) * 255
        im_stack[i] = im
        '''
        maximum = np.max(im) + 1
        invert = np.ones(im.shape) * maximum
        im_stack[i] = invert - im - np.ones(im.shape)
        '''
    print "convert in np.uint8..."
    im_stack = im_stack.astype(np.uint8)
    print "save result..."
    tf.imsave(output + "prep_vasculature.tif", im_stack, compress=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="path to input image (TIF)")
    parser.add_argument("output", type=str, help="where the output image (TIF) will be saved")

    args = parser.parse_args()
    main(args.input, args.output)
