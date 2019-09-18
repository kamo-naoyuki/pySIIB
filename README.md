# pySIIB: A python implementation of speech intelligibility in bits (SIIB)

[![Build Status](https://travis-ci.org/kamo-naoyuki/pySIIB.svg?branch=master)](https://travis-ci.org/kamo-naoyuki/pySIIB)
[![codecov](https://codecov.io/gh/kamo-naoyuki/pySIIB/branch/master/graph/badge.svg)](https://codecov.io/gh/kamo-naoyuki/pySIIB)

SIIB is an intrusive instrumental intelligibility metric.
This Python implementation of SIIB is ported from the author's matlab codes: https://stevenvankuyk.com/matlab_code/

## Install

```bash
pip install git+https://github.com/kamo-naoyuki/pySIIB.git
```


## Usage

```python
from pysiib import SIIB
from scipy.io import wavefile

fs, x = wavfile.read("reference.wav")
fs, y = wavfile.read("enhance.wav")

# SIIB with MI function in C-implementation (this is used in [1],[2])
SIIB(x, y, fs)
# SIIB with MI function in python implementation
SIIB(x, y, fs, use_MI_Kraskov=False)
# SIIB^gauss
SIIB(x, y, fs, gauss=True)
```

The first version is proposed in [1] and  SIIB^Gauss is more simple implmentation and faster.
The speed comparison is SIIB^gauss > C-SIIB > python-SIIB.

## Reference

- [1] S. Van Kuyk, W. B. Kleijn, and R. C. Hendriks, ‘An instrumental intelligibility metric based on information theory’, IEEE Signal Processing Letters, 2018.
- [2] S. Van Kuyk, W. B. Kleijn, and R. C. Hendriks, ‘An evaluation of intrusive instrumental intelligibility metrics’, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2018.
