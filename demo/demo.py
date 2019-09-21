import numpy as np
from scipy.io import wavfile
from scipy import optimize
import matplotlib

from pysiib import SIIB

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# This script demonstrates how SIIB can be related to intelligibility
# scores.
#
# --------------------------------------------------------------------------
# Copyright 2018: Steven Van Kuyk.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------
#
# Contact: steven.jvk@gmail.com
#
# References:
#   [1] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An instrumental
#       intelligibility metric based on information theory', 2018
#   [2] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An evaluation of
#       intrusive instrumental intelligibility metrics', 2018
#
# The listening test data used in this demo is described in
# [3] Kjems et al., 2009, 'Role of mask pattern in intelligibility of ideal
#     binary-masked noisy speech.'
# The full data set is available at
# http://web.cse.ohio-state.edu/pnl/corpus/Kjems-jasa09/README.html


def db2pow(x):
    return 10 ** (x / 10)


def main(gauss=False):
    np.random.seed(0)

    fs, x = wavfile.read('clean.wav')
    fs, n = wavfile.read('noise.wav')
    x = x.astype(np.float64)
    n = n.astype(np.float64)

    # listening test data
    S50 = 15.1 / 100  # parameters for speech-shaped noise (see [3])
    L50 = -7.3

    # percentage of words correct
    intelligibility = np.array([0.01, 0.1, 1, 10, 20,
                                30, 40, 50, 60, 70, 80, 90,
                                95, 98, 99])

    # invert Kjems psychometric curve to find the required SNR [3]
    SNRdB = L50 - np.log((100 - intelligibility) / intelligibility) / (4 * S50)

    # compute SIIB^Gauss for different stimuli
    siib = []
    for i in range(len(SNRdB)):
        # clean speech
        s = np.sqrt(db2pow(SNRdB[i]))
        x = s * x / np.std(x)

        # randomise noise segment
        start = np.random.randint(0, len(n) - len(x))
        n_seg = n[start:start + len(x)]
        n_seg = n_seg / np.std(n_seg)
        y = x + n_seg

        siib.append(SIIB(x, y, fs, gauss=gauss))
    siib = np.array(siib)

    def logistic(x, a, b):
        return 100. / (1 + np.exp(a * (x - b)))

    popt, pcov = optimize.curve_fit(logistic, siib, intelligibility, p0=(0, 0))

    plt.clf()
    plt.ylabel('Intelligibility [%]')
    plt.scatter(siib, intelligibility)
    x = np.arange(0, max(siib) + 5, 5)
    plt.plot(x, logistic(x, *popt),
             color='red',
             label='Fitting by logistic function')
    plt.legend()
    if gauss:
        plt.xlabel('SIIB^Gauss [b/s]')
        plt.savefig('SIIB_Gauss.png')
    else:
        plt.xlabel('SIIB [b/s]')
        plt.savefig('SIIB.png')


if __name__ == '__main__':
    main(True)
    main(False)
