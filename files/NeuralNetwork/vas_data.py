import h5py
import numpy as np
np.set_printoptions(linewidth=240, edgeitems=32, precision=1)
from progressbar import ProgressBar, Percentage, Bar, ETA


sets = {
    2:
    {
        'train': 
        [{
            'starts': [0, 0, 0],
            'ends': [10, None, None],
        }],
        'test':
        [{
            'starts':  [10, 0, 0],
            'ends': [None, None, None]
        }],
        'all_test':
        [{
            'starts': [0, 0, 0],
            'ends': [None, None, None]
        }],
        'all_train':
        [{
            'starts': [0, 0, 0],
            'ends': [None, None, None]
        }]
    },

    3:
    {
        'train':
        [{
            'starts': [0, 0, 0],
            'ends': [None, 1024, 1024],
        }],
        'test':
        [ 
            { 
                'starts':  [0, 0, 1024],
                'ends': [None, 1024, None]
            },
            { 
                'starts':  [0, 1024, 0],
                'ends': [None, None, 1024]
            },
            { 
                'starts':  [0, 1024, 1024],
                'ends': [None, None, None]
            }
        ]
    }
}

class VascularData(object):
    def __init__(self, filename, p_shape, which_set=None, range_=None, batchsize=128):

        #{'starts': [z0, x0,y0], 'ends':[z1,x1,y1]}
        assert filename is not None 
        assert (which_set in ['train', 'test', 'all_test', 'all_train'] or range_ is not None)
        assert (type(p_shape) == list or type(p_shape) == tuple)


        self.X = None
        self.y = None
       
        # range_ must be list where size_1 and size_2 are the same
  
        if range_ is not None:
            final_set = range_ 
        else:
            final_set = sets[len(p_shape)][which_set]

        h5file= h5py.File(filename, 'r')
        for r in final_set:
            zs, xs, ys = r['starts']
            ze, xe, ye = r['ends']
            if self.X is None:
                self.X = h5file['X'][zs:ze, xs:xe, ys:ye]
                if (which_set != "all_test"):
                    self.y = h5file['y'][zs:ze, xs:xe, ys:ye]
                if self.y is None:
                    self.y = self.X
                #else:
                # self.y = h5file['y'][zs:ze, xs:xe, ys:ye]
            else:
                self.X = np.vstack((self.X, h5file['X'][zs:ze, xs:xe, ys:ye]))
                if (which_set != "all_test"):
                    self.y = np.vstack((self.y, h5file["y"][zs:ze, xs:xe, ys:ye]))
                if self.y is None:
                    self.y = self.X
                #else:
                #    self.y = np.vstack((self.y, h5file['y'][zs:ze, xs:xe, ys:ye]))

        assert self.X.shape == self.y.shape
        self.batchsize = batchsize
        self.nz, self.nx, self.ny = self.X.shape
        self.p_shape = p_shape

        if len(p_shape) == 2:
            self.ext_p_shape = [1] + list(p_shape)
        if len(p_shape) == 3:
            self.ext_p_shape = p_shape

    def flow(self, count=1024, fixed_random=False):
        if fixed_random:
            np.random.rand(4)
        p_shape = self.ext_p_shape 
        for i in range(count):
            z_s = np.random.random_integers(0,self.nz-p_shape[0],self.batchsize)
            x_s = np.random.random_integers(0,self.nx-p_shape[1],self.batchsize)
            y_s = np.random.random_integers(0,self.ny-p_shape[2],self.batchsize)

            patch_X = np.array([self.X[z:z+p_shape[0],x:x+p_shape[1],y:y+p_shape[2]] for x,y,z in zip(x_s,y_s,z_s)])
            patch_y = np.array([self.y[z:z+p_shape[0],x:x+p_shape[1],y:y+p_shape[2]] for x,y,z in zip(x_s,y_s,z_s)])
            yield i, patch_X, patch_y


def make_training_set(filename):
    patch_size = 36
    nb_epoch = 4
    nb_minibatches = 22000
    batchsize = 128
    n = nb_epoch * nb_minibatches * batchsize
    h5file = h5py.File(filename, 'w')
    X = h5file.create_dataset("X", (n,1,patch_size,patch_size), h5py.h5t.NATIVE_FLOAT)
    y = h5file.create_dataset("y", (n,1,patch_size,patch_size), h5py.h5t.NATIVE_FLOAT)

    vascular = VascularData('vasculature.h5',
                            batchsize=batchsize,
                            start=0,
                            end=10,
                            p_shape=(patch_size,patch_size))

    widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=n)
    pbar.start()

    idx = 0
    for e in range(nb_epoch):
        for i, X_batch, y_batch in vascular.flow(count=nb_minibatches, fixed_random=False):
            X_batch = X_batch.reshape(X_batch.shape[0], 1, patch_size, patch_size)
            y_batch = y_batch.reshape(y_batch.shape[0], 1, patch_size, patch_size)
            for k in range(X_batch.shape[0]):
                X[idx,:,:,:] = X_batch[k]
                y[idx,:,:,:] = y_batch[k]
                pbar.update(idx + 1)
                idx += 1
    pbar.finish()
    h5file.close()


def main():
    v = VascularData('vasculature.h5', (16,16), which_set='train', batchsize=10)
    print v.X.shape
    for i, x,y in v.flow(count=5):
        print x.shape
        print y.shape
        print '-======================='

if __name__ == '__main__':
    main()
