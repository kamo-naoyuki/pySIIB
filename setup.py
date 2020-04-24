#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages


base = os.path.abspath(os.path.dirname(__file__))

setup(
    name="pysiib",
    version="0.0",
    description="A python implementation of speech intelligibility in bits (SIIB)",
    long_description="",
    url="https://github.com/kamo-naoyuki/pySIIB",
    author="Naoyuki Kamo",
    author_email="",
    classifiers=[
        "Programming Language :: Python",
    ],
    packages=["MI_kraskov"],
    py_modules=["pysiib"],
    install_requires=["cffi>=1.0.0", "numpy", "scipy"],
    setup_requires=["cffi>=1.0.0", 'pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    cffi_modules=[
        os.path.join(base, "MI_kraskov", "build_MIxnyn.py:ffi"),
    ],
)
