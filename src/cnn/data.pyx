# cython: language_level=3
import numpy as np

from pathlib import Path


def preprocess_data(
    filename, max_rows=None, shuffle=True, zero_mean=False,
):
    print(f"loading {filename}...", end=' ', flush=True)
    data = np.genfromtxt(filename, delimiter=',', max_rows=max_rows)
    print("done.", data.shape, flush=True)

    if shuffle: np.random.shuffle(data)

    targets = data[:,0].astype(np.intc)
    data = np.ascontiguousarray(data[:,1:])

    # black is 0, white is 1
    data /= data.max() # all values into [0,1] range

    if zero_mean:  # now in range [-1,1]
        #data -= data.mean(axis=0)
        data -= .5

    data.flags.writeable = False
    targets.flags.writeable = False

    return data, targets

def _load_data(str name, str csv):
    datapath = Path.cwd() / "data"
    p = datapath / (name + ".npy")
    if p.exists():
        return np.load(p)
