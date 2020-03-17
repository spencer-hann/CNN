# cython: language_level=3

import numpy as np
cimport numpy as np


cdef inline (Py_ssize_t,Py_ssize_t,Py_ssize_t) img_shape(np.ndarray img):
    IF TESTING: assert img.ndim == 3, "images must be 3 dimensional"
    return img.shape[0], img.shape[1], img.shape[2]


cdef class ConvolutionalLayer:
    cdef:
        Py_ssize_t n_filters
        Py_ssize_t depth
        Py_ssize_t dim
        np.ndarray filters


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


cdef class MaxPoolingLayer:
    cdef:
        Py_ssize_t dim


    def __init__(self, Py_ssize_t dim):
        self.dim = dim


    def forward(self, image):
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

