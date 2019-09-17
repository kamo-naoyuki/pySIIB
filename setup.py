#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

os.chdir(os.path.dirname(sys.argv[0]) or ".")

setup(
    name="pysiib",
    version="0.0",
    description="A python implementation of Speech intelligibility in bits (SIIB)",
    long_description="",
    url="",
    author="",
    author_email="",
    classifiers=[
        "Programming Language :: Python",
    ],
    packages=["", "MI_kraskov"],
    install_requires=["cffi>=1.0.0", "numpy", "scipy"],
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=[
        "./MI_kraskov/build_MIxnyn.py:ffi",
    ],
)
