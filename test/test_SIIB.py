import os
from subprocess import Popen, PIPE
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile
import shutil

from pysiib import SIIB


def run_octave(cmd):
    p = Popen(['octave', '--no-gui'], stdin=PIPE, stdout=PIPE)
    stdin, stderr = p.communicate(input=cmd.encode())
    if p.returncode != 0:
        raise RuntimeError('Failed to execute octave')
    return stdin.decode()


@pytest.fixture
def wavs():
    os.chdir(os.path.dirname(__file__))

    d = tempfile.mkdtemp()
    xp = os.path.join(d, 'x.wav')
    yp = os.path.join(d, 'y.wav')

    fs = 16000
    x = np.random.randint(-1000, 1000, fs * 10, dtype=np.int16)
    y = np.random.randint(-1000, 1000, fs * 10, dtype=np.int16)
    wavfile.write(xp, fs, x)
    wavfile.write(yp, fs, y)
    yield fs, x, y, xp, yp
    shutil.rmtree(d)


def test_SIIB_C(wavs):
    execpath = os.path.join(os.path.dirname(__file__), 'MI_kraskov', 'MIxnyn')

    fs, x, y, xp, yp = wavs
    cmd = '''
pkg load signal;
pkg load specfun;
[x, fs] = audioread("{xp}");
[y, fs] = audioread("{yp}");
I =  SIIB(x, y, fs, false, '{execpath}');
disp(I)
'''.format(xp=xp, yp=yp, execpath=execpath)
    s = run_octave(cmd)
    s = float(s)
    t = SIIB(x, y, fs)
    np.testing.assert_allclose(s, t, rtol=1e-02)


def test_SIIB_python(wavs):
    fs, x, y, xp, yp = wavs
    cmd = '''
pkg load signal;
pkg load specfun;
[x, fs] = audioread("{xp}");
[y, fs] = audioread("{yp}");
I =  SIIB(x, y, fs, true);
disp(I)
'''.format(xp=xp, yp=yp)
    s = run_octave(cmd)
    s = float(s)
    t = SIIB(x, y, fs, use_MI_Kraskov=False)
    np.testing.assert_allclose(s, t, rtol=1e-02)


def test_SIIB_gauss(wavs):
    fs, x, y, xp, yp = wavs
    cmd = '''
pkg load signal;
pkg load specfun;
[x, fs] = audioread("{xp}");
[y, fs] = audioread("{yp}");
I =  SIIB_Gauss(x, y, fs);
disp(I)
'''.format(xp=xp, yp=yp)
    s = run_octave(cmd)
    s = float(s)
    t = SIIB(x, y, fs, gauss=True)
    np.testing.assert_allclose(s, t, rtol=1e-02)
