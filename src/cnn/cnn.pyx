# cython: language_level=3

import numpy as np
cimport numpy as np


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


    def pad(self, np.ndarray image):
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
        np.ndarray last_out


    def __init__(self, insize, outsize=10):
        self.insize = insize
        self.outsize = outsize

        self.w = np.random.randn(insize, outsize) / insize
        self.b = np.random.randn(outsize) / outsize


    def forward(self, image):
        self.last_image = image
        image = image.flatten()

        # fully-connected/matmul phase
        out = image @ self.w
        out += self.b
        self.last_out = out

        # softmax phase
        result = np.exp(out)
        result /= np.sum(result, axis=0)

        return result

    def backprop(self, output_gradient):
        # 
        results = self.last_out
        exp_res = np.exp(res)
        out = exp_res / np.sum(exp_res)



cdef class CNN:
    cdef readonly:
        list layers
        np.ndarray out
        dict classes
        int size


    def __init__(self, layers, classes = (*range(10),)):
        self.layers = layers
        self.size = len(layers)

        if type(classes) != dict:
            classes = {i:cls for i,cls in enumerate(classes)}

        self.classes = classes
        self.out = np.zeros(len(classes))

        print(classes)


    def forward(self, np.ndarray image, Py_ssize_t label):
        cdef np.ndarray out

        out = image
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out

        result = self.classes[np.argmax(out)]

        correct = float(result == label)
        cost = -np.log(out[label])

        return result, out, correct, cost
