# cython: language_level=3

import numpy as np
cimport numpy as np

from tqdm import tqdm


DEF DEF_LEARNING_RATE = 0.02


cdef inline np_shape(np.ndarray a):
    return (<object>a).shape


cdef inline (Py_ssize_t,Py_ssize_t,Py_ssize_t) img_shape(np.ndarray img):
    IF TESTING: assert img.ndim == 3, "images must be 3 dimensional"
    return img.shape[0], img.shape[1], img.shape[2]


cdef class Layer:

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backprop(self, *args, **kwargs):
        raise NotImplementedError


cdef class ConvolutionalLayer(Layer):
    cdef:
        readonly Py_ssize_t n_filters
        readonly Py_ssize_t depth
        readonly Py_ssize_t dim
        readonly np.ndarray filters


    def __init__(
        self, Py_ssize_t n_filters, Py_ssize_t depth, Py_ssize_t dim,
    ):
        self.n_filters = n_filters
        self.depth = depth
        self.dim = dim

        assert dim % 2, "Conv dim must be odd"

        self.filters = np.random.randn(n_filters, depth, dim, dim)
        self.filters /= depth * dim**2  # normalize individual filters


    cdef np.ndarray pad(self, np.ndarray image):
        cdef Py_ssize_t p, i, j
        cdef np.ndarray out

        p = self.dim // 2
        i = image.shape[1]
        j = image.shape[2]
        out = np.zeros((
            image.shape[0],
            i+2*p,
            j+2*p,
        ))

        out[:, p:i+p, p:j+p] = image

        return out


    def forward(self, np.ndarray image):
        cdef np.ndarray out, m
        cdef Py_ssize_t xdim, ydim, i, j, p

        nchannels = image.shape[0]
        xdim = image.shape[1]
        ydim = image.shape[2]
        assert nchannels == self.depth

        out = np.zeros((self.n_filters, xdim, ydim))
        p = self.dim // 2  # padding
        image = self.pad(image)

        for i in range(p, xdim):
            for j in range(p, ydim):
                m = self.filters * image[:, i-p:i+p+1, j-p:j+p+1]
                out[:, i-p, j-p] = np.sum(m, axis=(1,2,3))

        return out


cdef class MaxPoolingLayer(Layer):
    cdef:
        readonly Py_ssize_t dim


    def __init__(self, Py_ssize_t dim):
        self.dim = dim


    def forward(self, np.ndarray image):
        cdef Py_ssize_t nchan, xdim, ydim, d, x, y, i, j
        cdef np.ndarray out

        nchan, xdim, ydim = img_shape(image)

        d = self.dim
        newx = xdim // d
        newy = ydim // d
        out = np.empty((nchan, newx, newy))

        for x in range(newx):
            i = x * d
            for y in range(newy):
                j = y * d
                out[:,x,y] = np.max(image[:, i:i+d, j:j+d], axis=(1,2))

        return out


cdef class DenseSoftmaxLayer(Layer):
    cdef readonly:
        np.ndarray w
        np.ndarray b
        Py_ssize_t insize
        Py_ssize_t outsize

        np.ndarray last_image
        np.ndarray last_fc  # last state of fully-connected layer output


    def __init__(self, insize, outsize=10):
        self.insize = insize
        self.outsize = outsize

        self.w = np.random.randn(insize, outsize) / insize
        self.b = np.random.randn(outsize) / outsize


    def forward(self, np.ndarray image):
        self.last_image = image
        image = image.flatten()

        # fully-connected/matmul phase
        fc = np.dot(image, self.w)
        #fc = image @ self.w
        fc += self.b
        self.last_fc = fc

        # softmax phase
        result = np.exp(fc)
        result /= np.sum(result, axis=0)

        return result

    def backprop(self, np.ndarray loss_grad, double lr):
        gradient = self.backprop_softmax(loss_grad, lr)
        return self.backprop_dense(gradient, lr)

    def backprop_softmax(self, np.ndarray loss_grad, lr=None):
        # only one non-zero element in cross entropy loss gradient
        assert 0 <= np.count_nonzero(loss_grad) <= 1

        # exp of dense output prior to softmax
        exp_fc = np.exp(self.last_fc)
        exp_fc_sm = exp_fc.sum()

        # softmax gradient with respect to input from fc layer
        c = loss_grad.argmax()  # class index
        # true for all but grad_fc[c]
        grad_fc = -exp_fc[c] * exp_fc / exp_fc_sm**2
        # grad_fc[c] depends on exp_fc[c] differently
        grad_fc[c] = exp_fc[c] * (exp_fc_sm - exp_fc[c]) / exp_fc_sm**2

        # loss gradient wrt input layer
        return loss_grad * grad_fc  # zeroed out except for sole contibutor

    def backprop_dense(self, np.ndarray loss_grad, double lr):
        # output gradients wrt input, biases, weights
        ograd_input = self.w
        ograd_biases = 1
        ograd_weights = self.last_image.flatten()

        #print(
        #    np.isnan(ograd_input).any(),
        #    np.isnan(loss_grad).any(),
        #    np.isnan(self.w).any(),
        #    self.w.mean(),
        #    self.w.min(),
        #    self.w.max(),
        #)
        #if np.isnan(self.w).any():
        #    import sys
        #    sys.exit(1)

        # loss gradients wrt input, biases, weights
        lgrad_input = ograd_input @ loss_grad
        lgrad_biases = ograd_biases * loss_grad
        #lgrad_weights = ograd_weights[np.newaxis].T @ loss_grad[np.newaxis]
        lgrad_weights = ograd_weights[:,np.newaxis] @ loss_grad[np.newaxis]

        # update layer
        self.w += lr * lgrad_weights
        self.b += lr * lgrad_biases
        return lgrad_input.reshape(np_shape(self.last_image))


cdef class CNN:
    cdef readonly:
        np.ndarray layers
        np.ndarray out
        int size
        double lr


    def __init__(
        self,
        layers,
        double lr = DEF_LEARNING_RATE,
    ):
        self.layers = np.array(layers)
        self.size = len(layers)
        self.lr = lr


    def forward(self, np.ndarray image, Py_ssize_t label = -1):
        cdef np.ndarray out

        out = image
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out

        result = np.argmax(out)
        if label < 0:
            return result, out

        correct = float(result == label)
        loss = -np.log(out[label])

        return result, out, correct, loss

    def learn(
        self,
        np.ndarray image,
        int label,
    ):
        res, out, correct, loss = self.forward(image, label)

        # cross entropy gradient
        grad = np.zeros(10)
        grad[label] = - 1 / out[label]

        for layer in self.layers[::-1]:
            grad = layer.backprop(grad, self.lr)

        return loss, correct

    def train(self, np.ndarray images, np.ndarray labels):
        cdef:
            double l, loss = 0
            double c, ncorrect = 0
            int n = len(images)
            Py_ssize_t i

        assert len(images) == len(labels)
        for i in tqdm(range(n)):
            l, c = self.learn(images[i], labels[i])
            loss += l
            ncorrect += c

        return loss / n, ncorrect / n

    def train_epochs(self, int n, np.ndarray images, np.ndarray labels):
        for i in range(n):
            loss, accuracy = self.train(images, labels)
            print(f"Epoch {i}: {loss:.2f} loss, {100*accuracy:.2f}% accurate")

