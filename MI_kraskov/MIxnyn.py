from cffi import FFI
import numpy as np

from MI_kraskov._MIxnyn import lib


ffi = FFI()


def MIxnyn(x, y, k, addnoise=None):
    if addnoise is None:
        addnoise = -1.
    if x.ndim == 1:
        x = x[None]
    if y.ndim == 1:
        y = y[None]
    assert x.shape[1] == y.shape[1], (x.shape[1], y.shape[1])
    xy = np.asarray([x, y], dtype=np.float64)
    ptr = ffi.cast('double *', ffi.from_buffer(xy))
    # Warning! MIxnyn modifies the value!
    return lib.MIxnyn(ptr, x.shape[0], y.shape[0], x.shape[1], k, addnoise)
