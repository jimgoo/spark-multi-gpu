import numpy as np


def make_batches(size, batch_size, include_partial=True):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
        include_partial: Boolean, include last partial batch.
    # Returns
        A list of tuples of array indices.
    """
    if include_partial:
        func = np.ceil
    else:
        func = np.floor
    num_batches = int(func(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def load_test_data():
    # load data
    x_train = np.load('/mnt/data/inv-12H-x.npy').astype(np.float32)
    y_train = np.load('/mnt/data/inv-12H-y.npy').astype(np.float32)
    assert x_train.shape[0] == y_train.shape[0]
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)
    return x_train, y_train