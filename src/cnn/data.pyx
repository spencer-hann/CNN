import numpy as np


def preprocess_data(filename, max_rows=None, shuffle=True):
    data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)

    if shuffle:
        np.random.shuffle(data)
    targets = data[:,0].astype(np.intc)
    data[:,1:] /= 255 # all values into [0,1] range, except bias column
    data[:,0] = 1 # adds bias(=1) column

    return data, targets

