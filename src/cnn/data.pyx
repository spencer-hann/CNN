import numpy as np


def preprocess_data(
    filename, max_rows=None, shuffle=True, zero_mean=False,
):
    data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)

    if shuffle:
        np.random.shuffle(data)

    targets = data[:,0].astype(np.intc)

    # black is 0, white is 1
    data /= 255 # all values into [0,1] range

    if zero_mean:  # now in range [-1,1]
        data *= 2
        data -= 1

    data[:,0] = 1 # adds bias(=1) column

    return data, targets

