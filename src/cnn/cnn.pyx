import numpy as np
cimport numpy as np


cdef class ConvolutionalLayer:
    # Convention: (depth, xdim, ydim)

    def __init__(self, int n_filters, int depth, int dim):
        self.n_filters = n_filters
        self.depth = depth
        self.dim = dim

        assert dim % 2, "Conv dim must be odd"

        self.filters = np.random.randn((depth, dim, dim))
        self.filters /= dim**2 * depth

    cdef np.ndarray pad_input(self, np.ndarray data):
        pwidth = self.dim // 2

    def forward(self, np.ndarray image):
        cdef np.ndarray out, m
        cdef Py_ssize_t nchannels, xdim, ydim, i, j, p

        nchannels, xdim, ydim = image.shape
        assert nchannels == self.depth

        out = np.array_like(image)
        p = self.dim // 2  # padding
        image = np.pad(data, (0, p, p))

        for i in range(p, xdim-p):
            for j in range(p, ydim-p):
                m = self.filters * image[:,i-p:i+p,j-p:j+p]
                out[:,i-p,j-p] = np.sum(m, axis=(1,2))

        return out

