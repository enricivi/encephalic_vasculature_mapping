from keras.models import Sequential
from keras.layers.core import Layer, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from progressbar import ProgressBar, Percentage, Bar, ETA
import argparse
import numpy as np
import os
import time
from vas_data import VascularData
import tensorflow as tf
import keras.backend as K
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB
    '''
    if K._BACKEND != 'tensorflow':
        return

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def custom_objective(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred[:,:,0:36,0:36], y_true), axis=-1)

def create_autoencoder(model_folder, patch_size):
    total_patches = 1
    for x in patch_size:
        total_patches *= x
    if len(patch_size) == 2:
        encoder = [
            Convolution2D(32, 7, 7, input_shape = [1] + patch_size, border_mode='valid', init='he_normal'), PReLU(),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(32, 5, 5, border_mode='valid', init='he_normal'), PReLU(),
            MaxPooling2D(pool_size=(2, 2)),
        ]
    
        decoder = [
            Flatten(),
            Dense(total_patches, init='he_normal'), PReLU(),
            Reshape([1] + patch_size),
            Activation('sigmoid')
        ]
    if len(patch_size) == 3:
        encoder = [
                Convolution3D(32, 7, 7, 7, input_shape = [1] + patch_size, border_mode='same', init='he_normal'), PReLU(),
                MaxPooling3D(pool_size=(2, 2, 2)),
                Convolution3D(32, 5, 5, 5, border_mode='same', init='he_normal'), PReLU(),
                MaxPooling3D(pool_size=(2, 2, 2)),
            ]
        decoder = [
                Convolution3D(8, 5, 5, 5, border_mode='same', init='he_normal'), PReLU(),
                UpSampling3D(size=(2, 2, 2)),
                Convolution3D(8, 7, 7, 7, border_mode='same', init='he_normal'), PReLU(),
                UpSampling3D(size=(2, 2, 2)),
                Convolution3D(1, 7, 7, 7, border_mode='same', init='he_normal'), PReLU(),
                Activation('sigmoid')
            ]

    model = Sequential()
    for l in encoder:
        model.add(l)
        print(l.output_shape)
    for l in decoder:
        model.add(l)
        print(l.output_shape)

    with open('{}/model.yaml'.format(model_folder), 'w') as modelfile:
        modelfile.write(model.to_yaml())
    try:
        model.compile(loss='binary_crossentropy', optimizer='adadelta')
        return model
    except:
        return None

def check_equal(lst):
    return not lst or [lst[0]]*len(lst) == lst

def get_int_from_string_tuple(tuple_str):
    patch_size = []
    for x in ''.join(tuple_str).split(','):
        try:
            p = int(x)
            if p%2:
               p += 1
            if (p//2)%2:
               p += 2
            patch_size.append(p)
        except:
            return None
    return patch_size

def train(model, model_folder, data, epochs=1000, pint=1000, nb_minibatches=25000):
    '''pint: print interval (in minibatches)
    '''
    patch_size = data.p_shape
    widgets=[Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=nb_minibatches*epochs)
    pbar.start()
    for e in range(epochs):
        losses = []
        for i, X_batch, y_batch in data.flow(count=nb_minibatches, fixed_random=False):
            X_batch = X_batch.reshape([X_batch.shape[0], 1] + patch_size)
            y_batch = y_batch.reshape([y_batch.shape[0], 1] + patch_size)
            loss = model.train_on_batch(X_batch, y_batch)
            losses.append(loss)
            pbar.update((e*nb_minibatches) + i)
            if (i > 0) and (i % pint) == 0:
                L = np.array(losses)
                timestr = time.strftime("%b %d %H:%M:%S", time.localtime())
                print '{:4d}/{:5d} - {:.4f} {:.4f} - {}'.format((e+1), (i+1), L.mean(), L[-pint:].mean(), timestr)
            L = np.array(losses)
        print 'End of epoch {:4d} - {:.4f}'.format((e+1), L.mean())
        if ( (e < 10) or ((e % 10) == 0) or (e == (epochs-1)) ):
            model.save_weights('{}/model_{}.h5'.format(model_folder, (e+1)))
    pbar.finish()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_prefix', type=str, help='Model folder')
    parser.add_argument('-h5', '--h5_file', type= str, help="path to h5 file used for training", required= True)
    parser.add_argument('-s', '--h5_set', type= str, help="choose the set inside the h5 file",
                        choices=['train', 'test', 'all_test', 'all_train'], required= True)
    parser.add_argument('-p', '--patch_size', type=tuple, help='Size of the patch. It can be either a single integer or a tuple.', required=True)
    parser.add_argument('-d', '--conv_dim', type=int, help='Convolution dimension', choices=[2,3], default=2)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1000)
    parser.add_argument('-n', '--nb_minibatches', type=int, help='Number of minibatch per epoch', default=25000)
    args = parser.parse_args()   

    model_prefix = args.model_prefix
    patch_size = get_int_from_string_tuple(args.patch_size)
    conv_dim = args.conv_dim
    epochs = args.epochs
    nb_minibatches = args.nb_minibatches

    if not patch_size:
        print 'Type of patch_size not understood'
        return None

    if len(patch_size) == 1:
        patch_size = patch_size*args.conv_dim
     

    if len(patch_size) != conv_dim:
        print 'Patch size must be either 1-dimension or %d-dimension' % conv_dim
        return None


    if check_equal(patch_size):
        str_patch_size = str(patch_size[0])
    else:
        str_patch_size = '_'.join([str(x) for x in patch_size])

    model_folder = model_prefix + '/model_conv_%d_%s_nfc_%s' % (conv_dim, str_patch_size, time.strftime("%b-%d-%H-%M-%S", time.localtime()))
    while(os.path.isdir(model_folder)):
        model_folder = model_prefix + '/model_%d_%s_nfc_%s' % (conv_dim, str_patch_size, time.strftime("%b-%d-%H-%M-%S", time.localtime()))
       
    try:
        os.makedirs(model_folder)
    except:
        print 'Folder %s can not be created. Check permission rights.' % model_prefix
        return None

    KTF.set_session(get_session(gpu_fraction=0.8))
    data = VascularData(  args.h5_file,
                          p_shape= patch_size,
                          which_set= args.h5_set,
                          batchsize= 128 )

    model = create_autoencoder(model_folder, patch_size)
    train(model, model_folder, data, epochs=epochs, nb_minibatches=nb_minibatches)
   
if __name__ == '__main__':
    main()
