from cffi import FFI
import numpy as np

from MI_kraskov._MIxnyn import lib


ffi = FFI()


def MIxnyn(x, y, k, addnoise=None):
    if addnoise is None:
        addnoise = -1.
    assert x.ndim == 1, x.ndim
    assert y.ndim == 1, y.ndim
    assert len(x) == len(y), (len(x), len(y))
    xy = np.asarray([x, y], dtype=np.float64)
    ptr = ffi.cast('double *', ffi.from_buffer(xy))
    # Warning! MIxnyn modifies the value!
    return lib.MIxnyn(ptr, 1, 1, len(x), k, addnoise)


if __name__ == '__main__':
    x = np.random.randn(2, 2)
    y = np.random.randn(2, 2)
    print(MIxnyn(x, y, 4))
