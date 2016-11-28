from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.core import Layer, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
import tensorflow as tf

import argparse
import h5py
import re
import numpy as np
import os
import time
import tifffile as tf
import scipy.misc as sm
'''
def order_file(a, b):
    pattern= r"[^0-9]"
    a= re.sub(pattern, '', a)
    b= re.sub(pattern, '', b)
    if ( long(a) > long(b) ):
        return 1
    elif ( long(a) < long(b) ):
        return -1
    return 0

class Vascular:
    def __init__(self, input, p_shape):
        self.p_shape= p_shape
        self.X= []
        file = sorted(os.listdir(input), cmp=order_file)
        for f in file:
            f= sm.imread(input + f)
            self.X.append( f / float(f.max()) )
        self.X= np.array(self.X)
'''
def max_minibatch_size(n_samples, n_patches, patch_size, gpu_mem=10): #GB
    mem_patches = 4.*n_patches*patch_size**2/2**30
    if mem_patches > gpu_mem:
        return 1, min(n_patches, np.ceil(n_patches/(mem_patches/gpu_mem)).astype(np.int32))
    else:
        return min(n_samples, np.ceil(n_samples/((mem_patches*n_samples)/gpu_mem)).astype(np.int32)), n_patches
    
def get_input_shape(model):
    import re
    for l in model.to_yaml().split('\n'):
        m = re.search(r'.*input_shape',l)
        if m:
            return [int(x) for x in l.split('[')[1].split(']')[0].split(',')[2:]]
    return None

def get_output_shape(model):
    import re
    for l in model.to_yaml().split('\n'):
        m = re.search(r'.*output_shape',l)
        if m:
            return [int(x) for x in l.split('[')[1].split(']')[0].split(',')[1:]]
    return None

def segment(samples, model, gpu_mem=10, save_file=None, stride=None):
    all_pred = np.zeros(samples.X.shape)
    """
    if save_file:
        h5file = h5py.File(save_file, 'w')
        pred_h5 = h5file.create_dataset('preds', samples.X.shape, h5py.h5t.NATIVE_FLOAT)
    """
    #predictions = np.zeros_like(samples)
    patch_size = samples.p_shape[0]
    if stride == None:
        stride = patch_size//2
    i_indices = range(0, samples.X.shape[1], stride) #L/2
    j_indices = range(0, samples.X.shape[2], stride)

    n_patches = len(i_indices)*len(j_indices)
 
    z_mbsize, patch_mbsize = max_minibatch_size(samples.X.shape[0], n_patches, patch_size, gpu_mem=gpu_mem)
   
    n_z_batches = np.ceil(1.*samples.X.shape[0]/z_mbsize).astype(np.int32)
    n_patch_batches = np.ceil(1.*n_patches/patch_mbsize).astype(np.int32)
 
    start = time.time()
    for z in range(n_z_batches):
        left_z  = min(z*z_mbsize, samples.X.shape[0]-1)
        right_z = min((z+1)*z_mbsize, samples.X.shape[0])
        print left_z, right_z
    
        print 'Depth from {} to {} (total: {}). Filling in {} patches'.format(left_z, right_z, samples.X.shape[0], n_patches)
        patches = np.zeros(((right_z-left_z), n_patches, 1, patch_size, patch_size))
        pidx = 0
        for i in i_indices:
            for j in j_indices:
                aux_samples = samples.X[left_z:right_z, i:i+patch_size,j:j+patch_size]
                ps_x = aux_samples.shape[1]
                ps_y = aux_samples.shape[2]
                patches[:,pidx,0,:ps_x,:ps_y] = aux_samples
                pidx += 1
        patches = patches.reshape(((right_z-left_z)*n_patches, 1, patch_size, patch_size))     

        print 'Predicting...'
        pred_patches = np.zeros_like(patches)
        batchsize = z_mbsize*patch_mbsize
        for b in range(n_z_batches*n_patch_batches):
            if (b % 1000) == 0 and b > 0:
                #TODO: n_batches
                print '{}/{}'.format(b,n_patches)
            first = min(b*batchsize, pred_patches.shape[0]-1)
            last  = min((b+1)*batchsize, pred_patches.shape[0])
            pred_patches[first:last,:,:,:] = model.predict([patches[first:last,:,:,:]])[:,:,:patch_size,:patch_size]

        print 'Assembling...'
        pred_X = np.zeros(((right_z-left_z), samples.X.shape[1], samples.X.shape[2]))
        n_pred_X = np.zeros_like(pred_X)
        pidx = 0
        pred_patches = pred_patches.reshape(((right_z-left_z),n_patches, 1, patch_size, patch_size))
        for i in i_indices:
            for j in j_indices:
                ps_x, ps_y = pred_X[left_z:right_z, i:i+patch_size,j:j+patch_size].shape[1:]
                pred_X[left_z:right_z, i:i+ps_x, j:j+ps_y] += pred_patches[left_z:right_z, pidx, 0, :ps_x, :ps_y]
                n_pred_X[left_z:right_z, i:i+ps_x, j:j+ps_y] += 1
                pidx += 1
                if (pidx % 10000) == 0:
                    print '{}/{}'.format(pidx,n_patches)
    
        pred_X = pred_X / np.maximum(n_pred_X,1)
        """
        if save_file:
            pred_h5[left_z:right_z,:,:] = pred_X.astype(np.float32)
        """
        all_pred[left_z:right_z,:,:] = pred_X.astype(np.float32)
        
    end = time.time() - start
    print end

    if (save_file != None):
        tf.imsave(save_file + 'preds' + ".tif", all_pred*255.0)

    return all_pred   

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_file', type=str, help="Model (yaml) file")
    parser.add_argument('weights_file', type=str, help="Weights file")
    parser.add_argument("output_file", type= str, help="directory to save the predicted image")
    #parser.add_argument('jpg_file', type=str, help="path to jpg file to predict")
    parser.add_argument('h5_file', type=str, help="path to h5 file to predict")
    parser.add_argument('h5_set', type=str, help="choose the set inside the h5 file",
                        choices=['train', 'test', 'all_test', 'all_train'])
    parser.add_argument('-s', '--stride', type=int, help="Stride dimension", default=None)

    args = parser.parse_args()

    assert os.path.isfile(args.h5_file), "h5 file not found"

    model_file = ''.join(open(args.model_file).readlines())
    weights_file = args.weights_file
    stride = args.stride
    model = model_from_yaml(model_file)
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    model.load_weights(weights_file)

    patch_size = get_input_shape(model)

    #vascular= Vascular(args.jpg_file, patch_size)
    from vas_data import VascularData
    vascular = VascularData(
        args.h5_file,
        p_shape=patch_size,
        which_set=args.h5_set,
        batchsize=128
    )

    prefix = args.output_file
    if len(prefix) == 0:
        prefix = '.'
    if stride is None:
        str_stride = str(patch_size[0]//2)
    else:
        str_stride = str(stride)
    """
    prediction_file = '{}/predictions_{}_stride_{}.h5'.format(prefix,
        os.path.basename(args.weights_file).split('_')[1].split('.')[0],
        str_stride )
    """
    predictions = segment(vascular, model, gpu_mem=10, save_file=prefix, stride=stride)

if __name__ == '__main__':
    main()
