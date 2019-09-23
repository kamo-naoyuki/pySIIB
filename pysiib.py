from __future__ import division

import math
import warnings

import numpy as np
from scipy.fftpack import fft
from scipy.signal import get_window
from scipy.signal import resample_poly
from scipy.special import psi

from MI_kraskov.MIxnyn import MIxnyn


EPS = np.finfo(np.float64).eps


def SIIB(x, y, fs_signal, gauss=False, use_MI_Kraskov=True,
         window_length=400, window_shift=200, window='hanning', delta_dB=40):
    """Speech intelligibility in bits (SIIB)
    and with Gaussian capacity (SIIB^Gauss)

    Python implementation is ported from
    https://stevenvankuyk.com/matlab_code/

    Args:
        x (np.ndarray): Clean signal
        y (np.ndarray): Distorted signal
        fs_signal (float): The sample frequency of input signal.
        gauss (bool): Use SIIB^Gauss.
        use_MI_Kraskov (bool): Use C-implementation for SIIB calculation.
            This is not valid for SIIB^Gauss mode.
        window_length (float):
        window_shift (float):
        window (str):
        delta_dB (float)): Decide VAD threshold

    --------------------------------------------------------------------------
     Copyright 2018: Steven Van Kuyk.
     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
    --------------------------------------------------------------------------

    References:
      [1] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An instrumental
          intelligibility metric based on information theory', 2018
      [2] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An evaluation of
          intrusive instrumental intelligibility metrics', 2018

    [1] proposed the first version of SIIB, which uses a non-parametric
    mutual information estimator. In [2] it was shown that the mutual
    information estimator could be replaced with the capacity of a Gaussian
    channel. The resulting algorithm is called SIIB^Gauss. SIIB^Gauss has
    similar performance to SIIB, but takes less time to compute.

    IMPORTANT: SIIB assumes that x any y are time-aligned.

    IMPORTANT: SIIB may not be reliable for stimuli with short durations
    (e.g., less than 20 seconds). Note that longer stimuli can be created by
    concatenating short stimuli together.
    """
    if x.ndim != 1:
        raise RuntimeError('x must have 1-dim shape')
    if y.ndim != 1:
        raise RuntimeError('y must have 1-dim shape')
    if x.shape != y.shape:
        raise RuntimeError('x and y should have the same length')

    # initialization
    fs = 16000                             # sample rate of acoustic signals
    R = 1 / window_shift * fs              # frames/second
    std = max(np.std(x), EPS)
    x = x / std                            # clean speech
    y = y / std                            # received speech

    # resample signals to fs
    if fs_signal != fs:
        x = resample_oct(x, fs, fs_signal)
        y = resample_oct(y, fs, fs_signal)

    # get |STFT|**2
    x_hat = stft(x, window_length, window_shift, window).T
    y_hat = stft(y, window_length, window_shift, window).T
    x_hat = x_hat.real ** 2 + x_hat.imag ** 2
    y_hat = y_hat.real ** 2 + y_hat.imag ** 2

    # VAD
    vad_index_x = get_vad(x, window_length, window_shift, window, delta_dB)
    x_hat = x_hat[:, vad_index_x]
    y_hat = y_hat[:, vad_index_x]

    # check that the duration (after removing silence) is at least 20 s
    if x_hat.shape[1] / R < 20:
        warnings.warn('stimuli must have at least 20 seconds of speech')

    # ERB gammatone filterbank
    mn = 100   # minimum center frequency
    mx = 6500  # maximum center frequency

    # J: number of filters
    J = int(round(21.4 * np.log10(1 + 0.00437 * mx) -
                  21.4 * np.log10(1 + 0.00437 * mn)))
    G = gammatone(fs, window_length, J, mn, mx)
    X = np.log(np.matmul(G ** 2, x_hat + EPS))  # equation (2) in [1]
    Y = np.log(np.matmul(G ** 2, y_hat + EPS))

    # forward temporal masking (see Rhebergen et al., 2006)
    Tf = int(np.floor(0.2 * R))  # 200 ms
    # 'hearing threshold' replacement (dB)
    E_tf = X.min(axis=1, keepdims=True)
    # initialize forward masking function
    Xd = np.full(X.shape, -np.inf)
    Yd = np.full(X.shape, -np.inf)

    T0 = 1
    ind = np.log(np.arange(T0, X.shape[1]) / T0) / np.log(Tf / T0)
    ii_ = np.minimum(np.arange(X.shape[1] + Tf), X.shape[1] - 1)
    for i in range(X.shape[1]):
        # frame indices
        ii = ii_[i:i + Tf]

        f = X[:, i, None]
        # forward masking function [Rhebergen et al., 2006]
        frame = f - ind[None, :Tf] * (f - E_tf)
        # max between clean signal and masking function
        Xd[:, ii] = np.maximum(Xd[:, ii], frame)

        f = Y[:, i, None]
        frame = f - ind[None, :Tf] * (f - E_tf)
        Yd[:, ii] = np.maximum(Yd[:, ii], frame)
    X = Xd
    Y = Yd

    # remove mean (for KLT)
    X = X - np.mean(X, 1, keepdims=True)
    Y = Y - np.mean(Y, 1, keepdims=True)

    # stack spectra
    K = 15  # number of stacked vectors
    temp = np.ravel(X, order='F')
    X = temp[np.arange(0, J * K)[:, None] +
             np.arange(0, len(temp) - J * K, J)[None, :]]
    temp = np.ravel(Y, order='F')
    Y = temp[np.arange(0, J * K)[:, None] +
             np.arange(0, len(temp) - J * K, J)[None, :]]

    # KLT
    _, U = np.linalg.eigh(np.cov(X))
    X = np.matmul(U.T, X)
    Y = np.matmul(U.T, Y)

    if not gauss:
        # estimate MI (assuming no time-freq dependencies)
        g = 150
        # number of nearest neighbours (Kraskov recommends k=2-6
        # but really it depends on the amount of data
        # available and bias vs variance tradeoff)
        k = max(2, int(np.ceil(X.shape[1] / g)))
        I_channels = []
        for j in range(X.shape[0]):
            I_channels.append(I_kras(X[j], Y[j], k,
                                     use_MI_Kraskov=use_MI_Kraskov))
        I_channels = np.array(I_channels)

        # speech production channel
        rho_p = 0.75
        Imx = -0.5 * np.log2(1 - rho_p ** 2)

        # compute SIIB
        retval = (R / K) * np.sum(np.minimum(Imx, I_channels))  # bit/s
    else:
        # Estimate mutual information using the capacity of a Gaussian channel
        X = X.T
        Y = Y.T
        # production noise correlation coefficient
        rho_p_squared = 0.75 ** 2
        # long-time squared correlation coefficient
        # for the environmental channel
        rho_squared = np.mean(X * Y, 0) ** 2 / \
            (np.mean(X ** 2, 0) * np.mean(Y ** 2, 0))
        # Gaussian capacity (bits/s) (equation (1) in [2])
        retval = - 0.5 * R / K * \
            np.sum(np.log2(1 - rho_p_squared * rho_squared), 0)

    retval = np.maximum(0, retval)
    return retval


def I_kras(x, y, k, use_MI_Kraskov=True):
    """
    this function estimates the mutual information (in bits) of
    x and y using a non-parametric
    nearest neighbour estimator
    ['Estimating Mutual Information", Kraskov et al., 2004]
    """
    # make sure the sequences are scaled 'reasonably'
    x = x - np.mean(x)
    y = y - np.mean(y)
    x = x / max(np.std(x), EPS)
    y = y / max(np.std(y), EPS)

    # small amount of noise to prevent 'singularities'
    x = x + 1e-10 * np.random.randn(*x.shape)
    y = y + 1e-10 * np.random.randn(*y.shape)

    # use Python implementation (not used in [1] or [2])
    if not use_MI_Kraskov:
        # x and y must be [1xn] or [nx1]
        # (this implementation assumes univariate data)
        N = len(x)

        nx = []
        ny = []

        for i in range(N):
            dx = np.abs(x[i] - x)  # distance from x(i) to x(j) where i/=j
            dy = np.abs(y[i] - y)
            dx = np.delete(dx, i)
            dy = np.delete(dy, i)

            dz = np.maximum(dx, dy)
            # distance to the k'th nearest neighbour
            e = np.partition(dz, k - 1)[k - 1]

            # number of x(j) points with distance from x(i) less than e(i)
            nx.append(np.sum(dx < e))
            ny.append(np.sum(dy < e))
        nx = np.array(nx)
        ny = np.array(ny)
        # info in nats (Eq. 8 in Kraskov)
        retval = psi(k) - np.mean(psi(nx + 1) + psi(ny + 1)) + psi(N)

    # use Kraskov et al. implementation (requires C-code)
    else:
        retval = MIxnyn(x, y, k)

    retval = retval / np.log(2)  # nats to bits
    return retval


def gammatone(fs, N_fft, numBands, cf_min, cf_max):
    """gammatone filterbank"""
    # convert to erbs
    erbminmax = 21.4 * np.log10(4.37 * (np.array([cf_min, cf_max]) / 1000) + 1)
    # linspace M filters on ERB-scale
    cf_erb = np.linspace(erbminmax[0], erbminmax[1], numBands)
    # obtain center frequency in Hz
    cf = (10 ** (cf_erb / 21.4) - 1) / 4.37 * 1000

    order = 4
    # Normalization factor that ensures the gammatone
    # filter has the correct ERB [Holdsworth & Patterson 1988].
    a = math.factorial(order - 1) ** 2 / \
        (np.pi * math.factorial(2 * order - 2) * 2 ** -(2 * order - 2))
    # bandwidth
    b = a * 24.7 * (4.37 * cf / 1000 + 1)

    # frequency vector (Hz)
    f = np.linspace(0, fs, N_fft + 1)
    f = f[:N_fft // 2 + 1]

    # filter bank
    A = []
    for i in range(numBands):
        # gammatone magnitude response
        temp = 1 / (b[i] ** 2 + (f - cf[i]) ** 2) ** (order / 2)
        # normalise the maximum value
        A.append(temp / temp.max())
    A = np.array(A)
    A[A < 0.001] = 0
    return A


def framing(x, window_length, window_shift, window):
    """
    Args:
        x: (Samples,)
        window_length:
        window_shift:
        window:
    Returns:
        y: (num_frame, window_length)
    """
    slen = x.shape[-1]
    if slen < window_length + 1:
        z = [(0, 0) for _ in range(x.ndim - 1)]
        x = np.pad(x, z + [(0, window_length + 1 - slen)], mode='constant')
    shape = x.shape[:-1] + (x.shape[-1] - window_length, window_length)
    strides = x.strides + (x.strides[-1],)
    y = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides)[..., ::window_shift, :]
    w = get_window(window, window_length)[None, :]
    return y * w


def stft(x, window_length, window_shift, window):
    frames = framing(x, window_length, window_shift, window=window)
    return fft(frames, n=window_length, axis=-1)[:, :window_length // 2 + 1]


def get_vad(x, window_length, window_shift, window, delta_db):
    """
    Args:
        x: Time domain (Sample,)
    """
    # returns the indices of voice active frames
    x_frame = framing(x, window_length, window_shift, window)
    # compute the power (dB) of each frame
    x_dB = 10 * np.log10((x_frame ** 2).mean(axis=1) + EPS)

    # find the 99.9 percentile
    ind = int(round(len(x_dB) * 0.999) - 1)
    max_x = np.partition(x_dB, ind)[ind]
    return x_dB > (max_x - delta_db)


def resample_oct(x, p, q):
    """Resampler that is compatible with Octave

    This function is copied from https://github.com/mpariente/pystoi
    """
    h = _resample_window_oct(p, q)
    window = h / np.sum(h)
    return resample_poly(x, p, q, window=window)


def _resample_window_oct(p, q):
    """Port of Octave code to Python"""

    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd

    # Properties of the antialiasing filter
    log10_rejection = -3.0
    stopband_cutoff_f = 1. / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10

    # Determine filter length
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))

    # Ideal sinc filter
    t = np.arange(-L, L + 1)
    ideal_filter = \
        2 * p * stopband_cutoff_f * np.sinc(2 * stopband_cutoff_f * t)

    # Determine parameter of Kaiser window
    if (rejection_dB >= 21) and (rejection_dB <= 50):
        beta = 0.5842 * (rejection_dB - 21)**0.4 + \
            0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0

    # Apodize ideal filter response
    h = np.kaiser(2 * L + 1, beta) * ideal_filter

    return h


if __name__ == '__main__':
    from scipy.io import wavfile
    import time
    fs, x = wavfile.read('demo/clean.wav')
    fs, y = wavfile.read('demo/noise.wav')
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    y = x + y[:len(x)]
    t = time.perf_counter()
    print(SIIB(x, y, fs, gauss=False))
    print(time.perf_counter() - t)
