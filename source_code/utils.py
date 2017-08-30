import numpy as np
import scipy.sparse as sc


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sc.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
