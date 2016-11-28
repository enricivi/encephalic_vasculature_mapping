import h5py
import argparse
import random
import numpy as np
from scipy import ndimage

def add_veil(image, white_div):
    avg = np.mean(image)
    veil = np.ones(image.shape, dtype=np.float64) * (avg / float(white_div))
    ret= veil + image
    return (ret / np.amax(ret))

def blur_img(image, mean, str_dev):
    sigma = abs(random.gauss(mean, str_dev))
    return ndimage.gaussian_filter(image, sigma)

def main(start, out, p, a):
    h5= h5py.File(start, 'r')
    x_h5= h5["X"]
    y_h5= h5["y"]
    
    assert (len(x_h5) != 0) and (x_h5.shape != (0, 0, 0)), "empty file"

    taken= []
    excluded= ([], [], [])  #blur, white veil, blur + white
    print "choosing elements to take unchanged..."
    for i in range( len(x_h5) ):
        if ( random.random() < p ):
            taken.append(i)
        else:
            j= random.randrange(0, len(excluded))
            excluded[j].append(i)
    print "unchanged: " + str(len(taken))
    print "changed: " + str( len(x_h5) - len(taken) )
    print ""

    finalX= []
    finalY= []
    print "'blurring' image (tot: " + str( len(excluded[0]) ) + ")"
    for i in excluded[0]:   #blur
        finalX.append( blur_img(x_h5[i], 10, 1.6) )
        finalY.append(y_h5[i])

    print "adding 'white veil' to image (tot: " + str(len(excluded[1])) + ")"
    for i in excluded[1]:   #white veil
        finalX.append( add_veil(x_h5[i], 0.55) )
        finalY.append(y_h5[i])

    print "adding 'white veil' and 'blurring' image (tot: " + str(len(excluded[2])) + ")"
    for i in excluded[2]: #white veil + blur
        im= add_veil(x_h5[i], 0.75)
        finalX.append( blur_img(im, 6.5, 1.2) )
        finalY.append(y_h5[i])

    print "adding unchanged elements to set..."
    for i in taken:
        finalX.append(x_h5[i])
        finalY.append(y_h5[i])

    print "additional file to add: " + str(a)
    if (a != None):
        h5 = h5py.File(a, 'r')
        assert (h5["X"][0].shape == x_h5[0].shape), "image dimensions must be equal"
        for i in range( len(h5["X"]) ):
            finalX.append(h5["X"][i])
            finalY.append(h5["y"][i])

    finalX= np.array(finalX)
    finalY= np.array(finalY)
    h5file = h5py.File(out + "mod_" + str(p) + "_" + str(len(excluded)) + "_" + start.split('/')[-1], "w")
    h5file.create_dataset('X', finalX.shape, h5py.h5t.NATIVE_FLOAT, data= finalX)
    h5file.create_dataset('y', finalY.shape, h5py.h5t.NATIVE_FLOAT, data= finalY)
    print "done..."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("initial_training", type= str, help="initial training file (HDF5)")
    parser.add_argument("output", type= str, help= "where the output file will be saved")
    parser.add_argument("-p", type= float,
                        help= "the chance to take an unchanged element from initial file",
                        choices= np.linspace(0, 1, endpoint= True, num= 21), default= 0.4)
    parser.add_argument("-a", help="files (hdf5) that must be added at the generated set",
                        default= None)
    args = parser.parse_args()
    main(args.initial_training, args.output, args.p, args.a)
