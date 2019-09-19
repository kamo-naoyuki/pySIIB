import os

from cffi import FFI

ffi = FFI()

base = os.path.abspath(os.path.dirname(__file__))
ffi.set_source(os.path.basename(base) + "._MIxnyn", "",
               sources=[os.path.join(base, j) for j in ["MIxnyn.C", "miutils.C"]],
               include_dirs=[base],
               libraries=["c"])

ffi.cdef("""
double MIxnyn(double *_x, int dimx, int dimy, int N, int K, double addnoise);
""")

if __name__ == "__main__":
    ffi.compile()
