import os

from cffi import FFI

ffi = FFI()

base = os.path.abspath(os.path.dirname(__file__))
source = ""
for j in ["MIxnyn.C", "miutils.C"]:
    with open(os.path.join(base, j)) as f:
        source += f.read()
ffi.set_source(os.path.basename(base) + "._MIxnyn", source,
               include_dirs=[base])

ffi.cdef("""
double MIxnyn(double *_x, int dimx, int dimy, int N, int K, double addnoise);
""")

if __name__ == "__main__":
    ffi.compile()
