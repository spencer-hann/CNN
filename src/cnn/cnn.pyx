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


cdef class ReLULayer(Layer):
    cdef np.ndarray active_neurons

    def forward(self, np.ndarray tensor):
        self.active_neurons = tensor > 0
        return tensor * self.active_neurons

    def backprop(self, np.ndarray loss_grad, lr=None):
        return loss_grad * self.active_neurons


cdef class ConvolutionalLayer(Layer):
    cdef readonly:
        Py_ssize_t n_filters
        Py_ssize_t depth
        Py_ssize_t dim
        np.ndarray filters

        np.ndarray last_input
        np.ndarray last_grad
        double momentum


    def __init__(
        self,
        Py_ssize_t depth,
        Py_ssize_t n_filters,
        Py_ssize_t dim,
        double momentum = 0.0,
    ):
        self.n_filters = n_filters
        self.depth = depth
        self.dim = dim

        self.momentum = momentum

        assert dim % 2, "Conv dim must be odd"

        self.filters = np.random.randn(n_filters, depth, dim, dim)
        self.filters /= depth * dim**2  # normalize individual filters

        self.last_grad = np.zeros(np_shape(self.filters))

    def __repr__(self):
        return (
            f"<{type(self).__name__}: " \
            + f"({self.depth}, {self.n_filters}, {self.dim})"
            + f", momentum={self.momentum}"
            + ">"
        )


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
        cdef np.ndarray out
        self.last_input = image
        out = self.convolve(image, self.filters)
        return out

    def convolve(self, np.ndarray image, np.ndarray filters):
        cdef np.ndarray out, m
        cdef Py_ssize_t xdim, ydim, i, j, d

        nchannels = image.shape[0]
        xdim = image.shape[1]
        ydim = image.shape[2]
        assert nchannels == self.depth

        out = np.zeros((self.n_filters, xdim, ydim))
        d = self.dim
        image = self.pad(image)

        for i in range(xdim - d + 1):  # (self.dim // 2)):
            for j in range(ydim - d + 1):  # (self.dim // 2)):
                for f in range(self.n_filters):
                    m = filters[f] * image[:, i:i+d, j:j+d]
                    out[f, i, j] = np.sum(m)

        return out

    def backprop(self, np.ndarray loss_grad, double lr):
        cdef np.ndarray grad_filters, grad_input, img
        cdef Py_ssize_t d, p, xdim, ydim

        grad_filters = np.zeros(np_shape(self.filters))
        grad_input = np.zeros(np_shape(self.last_input))
        d = self.dim
        p = d // 2
        xdim = self.last_input.shape[1]
        ydim = self.last_input.shape[2]

        img = self.pad(self.last_input)

        for i in range(xdim - d + 1):
            for j in range(ydim - d + 1):
                for f in range(self.n_filters):
                    grad_filters[f] += \
                        loss_grad[f, i, j] * img[:, i:i+d, j:j+d]
                    grad_input[:, i:i+d, j:j+d] += \
                            loss_grad[f, i, j] * self.filters[f]

        self.last_grad += grad_filters
        self.filters += lr * self.last_grad
        self.last_grad *= self.momentum

        return grad_input


cdef inline tuple ndargmax(np.ndarray a):
    return np.unravel_index(a.argmax(), (<object>a).shape)


cdef class MaxPoolingLayer(Layer):
    cdef readonly:
        Py_ssize_t dim
        np.ndarray last_input

    def __init__(self, Py_ssize_t dim):
        self.dim = dim

    def __repr__(self):
        return f"{type(self).__name__}: {self.dim}"

    def forward(self, np.ndarray image):
        cdef Py_ssize_t nchan, xdim, ydim, d, x, y, i, j
        cdef np.ndarray out

        self.last_input = image

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

    def backprop(self, np.ndarray loss_grad, lr=None):
        cdef Py_ssize_t nchan, xdim, ydim, i, j, d
        cdef tuple mxindex
        cdef np.ndarray grad_input

        nchan, xdim, ydim = img_shape(loss_grad)
        d = self.dim
        image = self.last_input
        grad_input = np.zeros(np_shape(image))

        for x in range(xdim):
            i = x * d
            for y in range(ydim):
                j = y * d
                for c in range(nchan):
                    mxindex = ndargmax(image[c, i:i+d, j:j+d])
                    grad_input[mxindex] = loss_grad[c, x, y]

        return grad_input





    #def backprop(self, np.ndarray loss_grad):
    #    cdef Py_ssize_t nchan, xdim, ydim, d, x, y, i, j
    #    cdef np.ndarray out

    #    nchan, xdim, ydim = img_shape(image)

    #    d = self.dim
    #    newx = xdim // d
    #    newy = ydim // d
    #    out = np.empty((nchan, newx, newy))

    #    for x in range(newx):
    #        i = x * d
    #        for y in range(newy):
    #            j = y * d
    #            for z in range(nchan):
    #                if self.last_input[z,x,y] == self.out[z,x,y]:
    #                    out[:,x,y] = np.max(image[:, i:i+d, j:j+d], axis=(1,2))

    #    return out


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


    def __repr__(self):
        return f"{type(self).__name__}: ({self.insize}, {self.outsize})"


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
        return self._forward(image, label)

    cdef tuple _forward(self, np.ndarray image, Py_ssize_t label):
        cdef np.ndarray out
        cdef double loss, correct

        out = image
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out

        if label < 0:
            return out, -1, -1

        correct = <double>(np.argmax(out) == label)
        loss = -np.log(out[label])

        return out, loss, correct

    def learn(
        self,
        np.ndarray image,
        int label,
    ):
        cdef np.ndarray out
        cdef double loss, correct

        out, loss, correct = self._forward(image, label)

        if correct: return loss, correct

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
            img = images[i].reshape((28,28))[np.newaxis,...]
            l, c = self.learn(img, labels[i])
            loss += l
            ncorrect += c

        return loss / n, ncorrect / n

    def train_epochs(self, n, np.ndarray images, np.ndarray labels):
        for i in range(n):
            images, labels = parallel_shuffle(images, labels)
            loss, accuracy = self.train(images, labels)
            print(f"Epoch {i}/{n}: {loss:.2f} loss, {100*accuracy:.2f}% accurate")

    def test(self, np.ndarray images, np.ndarray labels, display = True):
        cdef:
            double l, loss = 0
            double c, correct = 0
            int n = len(images)
            Py_ssize_t i

        assert n == len(labels)
        for i in tqdm(range(n)):
            img = images[i].reshape((28,28))[np.newaxis,...]
            _, l, c = self._forward(img, labels[i])
            loss += l
            correct += c

        loss, correct = loss / n, correct / n
        if display:
            print(f"Test: {loss:.2f} loss, {100*correct:.2f}% accurate")
        return loss, correct

    def __repr__(self):
        layers = ", ".join((str(l) for l in self.layers))
        return f"<{type(self).__name__}: lr={self.lr}, layers=({layers})>"


cpdef tuple parallel_shuffle(np.ndarray a, np.ndarray b):
    cdef np.ndarray indices
    assert len(a) == len(b)
    print("shuffling...")
    indices = np.arange(len(a))
    np.random.shuffle(indices)
    return a[indices], b[indices]

