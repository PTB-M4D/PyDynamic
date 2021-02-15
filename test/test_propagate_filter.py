"""Perform test for uncertainty.propagate_filter"""
import itertools

import numpy as np
import pytest
import scipy
from PyDynamic.misc.tools import make_semiposdef, trimOrPad
from PyDynamic.uncertainty.propagate_filter import FIRuncFilter, _fir_filter
from PyDynamic.uncertainty.propagate_MonteCarlo import MC
from scipy.linalg import toeplitz
from scipy.signal import lfilter, lfilter_zi


def random_array(length):
    array = np.random.randn(length)
    return array


def random_nonnegative_array(length):
    array = np.random.random(length)
    return array


def random_rightsided_autocorrelation(length):
    array = random_array(length)
    acf = scipy.signal.correlate(array, array, mode="full")
    return acf[len(acf) // 2 :]


def random_covariance_matrix(length):
    """construct a valid (but random) covariance matrix with good condition number"""

    # because np.cov estimates the mean from data, the returned covariance matrix
    # has one eigenvalue close to numerical zero (rank n-1).
    # This leads to a singular matrix, which is badly suited to be used as valid
    # covariance matrix. To circumvent this:

    ## draw random (n+1, n+1) matrix
    cov = np.cov(np.random.random((length + 1, length + 1)))

    ## calculate SVD
    u, s, vh = np.linalg.svd(cov, full_matrices=False, hermitian=True)

    ## reassemble a covariance of size (n, n) by discarding the smallest singular value
    cov_adjusted = (u[:-1, :-1] * s[:-1]) @ vh[:-1, :-1]

    return cov_adjusted


def valid_filters():
    N = np.random.randint(2, 100)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N)

    valid_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
        {"theta": theta, "Utheta": random_covariance_matrix(N)},
    ]

    return valid_filters


def valid_signals():
    N = np.random.randint(100, 1000)
    signal = random_array(N)

    valid_signals = [
        {"y": signal, "sigma_noise": np.random.randn(), "kind": "float"},
        {"y": signal, "sigma_noise": random_nonnegative_array(N), "kind": "diag"},
        {
            "y": signal,
            "sigma_noise": random_rightsided_autocorrelation(N // 2),
            "kind": "corr",
        },
    ]

    return valid_signals


def valid_lows():
    N = np.random.randint(2, 10)  # scipy.linalg.companion requires N >= 2
    blow = random_array(N)

    valid_lows = [
        {"blow": None},
        {"blow": blow},
    ]

    return valid_lows


@pytest.fixture
def equal_filters():
    # Create two filters with assumed identical FIRuncFilter() output to test
    # equality of the more efficient with the standard implementation.

    N = np.random.randint(2, 100)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N)

    equal_filters = [
        {"theta": theta, "Utheta": None},
        {"theta": theta, "Utheta": np.zeros((N, N))},
    ]

    return equal_filters


@pytest.fixture
def equal_signals():
    # Create three signals with assumed identical FIRuncFilter() output to test
    # equality of the different cases of input parameter 'kind'.

    # some shortcuts
    N = np.random.randint(100, 1000)
    signal = random_array(N)
    s = np.random.randn()
    acf = np.array([s ** 2] + [0] * (N - 1))

    equal_signals = [
        {"y": signal, "sigma_noise": s, "kind": "float"},
        {"y": signal, "sigma_noise": np.full(N, s), "kind": "diag"},
        {"y": signal, "sigma_noise": acf, "kind": "corr"},
    ]

    return equal_signals


@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals())
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter(filters, signals, lowpasses):
    # Check expected output for thinkable permutations of input parameters.
    y, Uy = FIRuncFilter(**filters, **signals, **lowpasses)
    assert len(y) == len(signals["y"])
    assert len(Uy) == len(signals["y"])

    # note: a direct comparison against scipy.signal.lfilter is not needed,
    #       as y is already computed using this method


def legacy_FIRuncFilter(
    y, sigma_noise, theta, Utheta=None, shift=0, blow=None, kind="corr"
):
    """Uncertainty propagation for signal y and uncertain FIR filter theta

    A preceding FIR low-pass filter with coefficients `blow` can be provided optionally.

    Parameters
    ----------
        y: np.ndarray
            filter input signal
        sigma_noise: float or np.ndarray
            float:    standard deviation of white noise in y
            1D-array: interpretation depends on kind
        theta: np.ndarray
            FIR filter coefficients
        Utheta: np.ndarray, optional
            covariance matrix associated with theta
            if the filter is fully certain, use `Utheta = None` (default) to make use of more efficient calculations.
            see also the comparison given in <examples\Digital filtering\FIRuncFilter_runtime_comparison.py>
        shift: int, optional
            time delay of filter output signal (in samples) (defaults to 0)
        blow: np.ndarray, optional
            optional FIR low-pass filter
        kind: string
            only meaningful in combination with sigma_noise a 1D numpy array
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/correlated)
            noise (default)

    Returns
    -------
        x: np.ndarray
            FIR filter output signal
        ux: np.ndarray
            point-wise standard uncertainties associated with x


    References
    ----------
        * Elster and Link 2008 [Elster2008]_

    .. seealso:: :mod:`PyDynamic.deconvolution.fit_filter`

    """

    Ntheta = len(theta)  # FIR filter size

    # check which case of sigma_noise is necessary
    if isinstance(sigma_noise, float):
        sigma2 = sigma_noise ** 2

    elif isinstance(sigma_noise, np.ndarray) and len(sigma_noise.shape) == 1:
        if kind == "diag":
            sigma2 = sigma_noise ** 2
        elif kind == "corr":
            sigma2 = sigma_noise
        else:
            raise ValueError("unknown kind of sigma_noise")

    else:
        raise ValueError(
            f"FIRuncFilter: Uncertainty sigma_noise associated "
            f"with input signal is expected to be either a float or a 1D array but "
            f"is of shape {sigma_noise.shape}. Please check documentation for input "
            f"parameters sigma_noise and kind for more information."
        )

    if isinstance(
        blow, np.ndarray
    ):  # calculate low-pass filtered signal and propagate noise

        if isinstance(sigma2, float):
            Bcorr = np.correlate(blow, blow, "full")  # len(Bcorr) == 2*Ntheta - 1
            ycorr = (
                sigma2 * Bcorr[len(blow) - 1 :]
            )  # only the upper half of the correlation is needed

            # trim / pad to length Ntheta
            ycorr = trimOrPad(ycorr, Ntheta)
            Ulow = toeplitz(ycorr)

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # [Leeuw1994](Covariance matrix of ARMA errors in closed form) can be used, to derive this formula
                # The given "blow" corresponds to a MA(q)-process.
                # Going through the calculations of Leeuw, but assuming
                # that E(vv^T) is a diagonal matrix with non-identical elements,
                # the covariance matrix V becomes (see Leeuw:corollary1)
                # V = N * SP * N^T + M * S * M^T
                # N, M are defined as in the paper
                # and SP is the covariance of input-noise prior to the observed time-interval
                # (SP needs be available len(blow)-timesteps into the past. Here it is
                # assumed, that SP is constant with the first value of sigma2)

                # V needs to be extended to cover Ntheta-1 timesteps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                N = toeplitz(blow[1:][::-1], np.zeros_like(sigma2_extended)).T
                M = toeplitz(
                    trimOrPad(blow, len(sigma2_extended)),
                    np.zeros_like(sigma2_extended),
                )
                SP = np.diag(sigma2[0] * np.ones_like(blow[1:]))
                S = np.diag(sigma2_extended)

                # Ulow is to be sliced from V, see below
                V = N.dot(SP).dot(N.T) + M.dot(S).dot(M.T)

            elif kind == "corr":

                # adjust the lengths sigma2 to fit blow and theta
                # this either crops (unused) information or appends zero-information
                # note1: this is the reason, why Ulow will have dimension (Ntheta x Ntheta) without further ado

                # calculate Bcorr
                Bcorr = np.correlate(blow, blow, "full")

                # pad or crop length of sigma2, then reflect some part to the left and invert the order
                # [0 1 2 3 4 5 6 7] --> [0 0 0 7 6 5 4 3 2 1 0 1 2 3]
                sigma2 = trimOrPad(sigma2, len(blow) + Ntheta - 1)
                sigma2_reflect = np.pad(sigma2, (len(blow) - 1, 0), mode="reflect")

                ycorr = np.correlate(
                    sigma2_reflect, Bcorr, mode="valid"
                )  # used convolve in a earlier version, should make no difference as Bcorr is symmetric
                Ulow = toeplitz(ycorr)

        xlow, _ = lfilter(blow, 1.0, y, zi=y[0] * lfilter_zi(blow, 1.0))

    else:  # if blow is not provided
        if isinstance(sigma2, float):
            Ulow = np.eye(Ntheta) * sigma2

        elif isinstance(sigma2, np.ndarray):

            if kind == "diag":
                # V needs to be extended to cover Ntheta timesteps more into the past
                sigma2_extended = np.append(sigma2[0] * np.ones((Ntheta - 1)), sigma2)

                # Ulow is to be sliced from V, see below
                V = np.diag(
                    sigma2_extended
                )  #  this is not Ulow, same thing as in the case of a provided blow (see above)

            elif kind == "corr":
                Ulow = toeplitz(trimOrPad(sigma2, Ntheta))

        xlow = y

    # apply FIR filter to calculate best estimate in accordance with GUM
    x, _ = lfilter(theta, 1.0, xlow, zi=xlow[0] * lfilter_zi(theta, 1.0))
    x = np.roll(x, -int(shift))

    # add dimension to theta, otherwise transpose won't work
    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    # NOTE: In the code below whereever `theta` or `Utheta` get used, they need to be flipped.
    #       This is necessary to take the time-order of both variables into account. (Which is descending
    #       for `theta` and `Utheta` but ascending for `Ulow`.)
    #
    #       Further details and illustrations showing the effect of not-flipping
    #       can be found at https://github.com/PTB-PSt1/PyDynamic/issues/183

    # handle diag-case, where Ulow needs to be sliced from V
    if kind == "diag":
        # UncCov needs to be calculated inside in its own for-loop
        # V has dimension (len(sigma2) + Ntheta) * (len(sigma2) + Ntheta) --> slice a fitting Ulow of dimension (Ntheta x Ntheta)
        UncCov = np.zeros((len(sigma2)))

        if isinstance(Utheta, np.ndarray):
            for k in range(len(sigma2)):
                Ulow = V[k : k + Ntheta, k : k + Ntheta]
                UncCov[k] = np.squeeze(
                    np.flip(theta).T.dot(Ulow.dot(np.flip(theta)))
                    + np.abs(np.trace(Ulow.dot(np.flip(Utheta))))
                )  # static part of uncertainty
        else:
            for k in range(len(sigma2)):
                Ulow = V[k : k + Ntheta, k : k + Ntheta]
                UncCov[k] = np.squeeze(
                    np.flip(theta).T.dot(Ulow.dot(np.flip(theta)))
                )  # static part of uncertainty

    else:
        if isinstance(Utheta, np.ndarray):
            UncCov = np.flip(theta).T.dot(Ulow.dot(np.flip(theta))) + np.abs(
                np.trace(Ulow.dot(np.flip(Utheta)))
            )  # static part of uncertainty
        else:
            UncCov = np.flip(theta).T.dot(
                Ulow.dot(np.flip(theta))
            )  # static part of uncertainty

    if isinstance(Utheta, np.ndarray):
        unc = np.empty_like(y)

        # use extended signal to match assumption of stationary signal prior to first entry
        xlow_extended = np.append(np.full(Ntheta - 1, xlow[0]), xlow)

        for m in range(len(xlow)):
            # extract necessary part from input signal
            XL = xlow_extended[m : m + Ntheta, np.newaxis]
            unc[m] = XL.T.dot(np.flip(Utheta).dot(XL))  # apply formula from paper
    else:
        unc = np.zeros_like(y)

    ux = np.sqrt(np.abs(UncCov + unc))
    ux = np.roll(ux, -int(shift))  # correct for delay

    return x, ux.flatten()  # flatten in case that we still have 2D array


def test_FIRuncFilter_equality(equal_filters, equal_signals):
    # Check expected output for being identical across different equivalent input
    # parameter cases.
    all_y = []
    all_uy = []

    # run all combinations of filter and signals
    for (f, s) in itertools.product(equal_filters, equal_signals):
        y, uy = FIRuncFilter(**f, **s)
        all_y.append(y)
        all_uy.append(uy)

    # check that all have the same output, as they are supposed to represent equal cases
    for a, b in itertools.combinations(all_y, 2):
        assert np.allclose(a, b)

    for a, b in itertools.combinations(all_uy, 2):
        assert np.allclose(a, b)


# in the following test, we exclude the case of a valid signal with uncertainty given as
# the right-sided auto-covariance (acf). This is done, because we currently do not ensure, that
# the random-drawn acf generates a positive-semidefinite Toeplitz-matrix. Therefore we cannot
# construct a valid and equivalent input for the Monte-Carlo method in that case.
@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals()[:2])  # exclude kind="corr"
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter_MC_uncertainty_comparison(filters, signals, lowpasses):
    # Check output for thinkable permutations of input parameters against a Monte Carlo approach.

    # run method
    y_fir, Uy_fir = FIRuncFilter(
        **filters, **signals, **lowpasses, return_full_covariance=True
    )

    # run Monte Carlo simulation of an FIR
    ## adjust input to match conventions of MC
    x = signals["y"]
    ux = signals["sigma_noise"]

    b = filters["theta"]
    a = [1.0]
    if isinstance(filters["Utheta"], np.ndarray):
        Uab = filters["Utheta"]
    else:  # Utheta == None
        Uab = np.zeros((len(b), len(b)))  # MC-method cant deal with Utheta = None

    blow = lowpasses["blow"]
    if isinstance(blow, np.ndarray):
        n_blow = len(blow)
    else:
        n_blow = 0

    ## run FIR with MC and extract diagonal of returned covariance
    y_mc, Uy_mc = MC(x, ux, b, a, Uab, blow=blow, runs=2000)

    # HACK for visualization during debugging
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].plot(y_fir, label="fir")
    # ax[0].plot(y_mc, label="mc")
    # ax[0].set_title("filter: {0}, signal: {1}".format(len(b), len(x)))
    # ax[0].legend()
    # ax[1].imshow(Uy_fir)
    # ax[1].set_title("FIR")
    # ax[2].imshow(Uy_mc)
    # ax[2].set_title("MC")
    # plt.show()
    # /HACK

    # check basic properties
    assert np.all(np.diag(Uy_fir) >= 0)
    assert np.all(np.diag(Uy_mc) >= 0)
    assert Uy_fir.shape == Uy_mc.shape

    # approximative comparison after swing-in of MC-result
    # (which is after the combined length of blow and b)
    assert np.allclose(
        Uy_fir[len(b) + n_blow :, len(b) + n_blow :],
        Uy_mc[len(b) + n_blow :, len(b) + n_blow :],
        atol=2e-1*Uy_fir.max(),  # very broad check, increase runs for better fit
        rtol=1e-1,
    )


@pytest.mark.parametrize("filters", valid_filters())
@pytest.mark.parametrize("signals", valid_signals())
@pytest.mark.parametrize("lowpasses", valid_lows())
def test_FIRuncFilter_legacy_comparison(filters, signals, lowpasses):
    # Compare output of both functions for thinkable permutations of input parameters.
    y, Uy = legacy_FIRuncFilter(**filters, **signals, **lowpasses)
    y2, Uy2 = FIRuncFilter(**filters, **signals, **lowpasses)

    # check output dimensions
    assert len(y2) == len(signals["y"])
    assert Uy2.shape == (len(signals["y"]),)

    # check value identity
    assert np.allclose(y, y2)
    assert np.allclose(Uy, Uy2)


def test_fir_filter_MC_comparison():
    N_signal = np.random.randint(20, 25)
    x = random_array(N_signal)
    Ux = random_covariance_matrix(N_signal)

    N_theta = np.random.randint(2, 5)  # scipy.linalg.companion requires N >= 2
    theta = random_array(N_theta)  # scipy.signal.firwin(N_theta, 0.1)
    Utheta = random_covariance_matrix(N_theta)

    # run method
    y_fir, Uy_fir = _fir_filter(x, theta, Ux, Utheta, initial_conditions="zero")

    ## run FIR with MC and extract diagonal of returned covariance
    y_mc, Uy_mc = MC(x, Ux, theta, [1.0], Utheta, blow=None, runs=10000)

    # HACK: for visualization during debugging
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].plot(y_fir, label="fir")
    # ax[0].plot(y_mc, label="mc")
    # ax[0].set_title("filter: {0}, signal: {1}".format(len(theta), len(x)))
    # ax[0].legend()
    # ax[1].imshow(Uy_fir)
    # ax[1].set_title("FIR")
    # ax[2].imshow(Uy_mc)
    # ax[2].set_title("MC")
    # plt.show()
    # /HACK

    # approximate comparison
    assert np.all(np.diag(Uy_fir) >= 0)
    assert np.all(np.diag(Uy_mc) >= 0)
    assert Uy_fir.shape == Uy_mc.shape
    assert np.allclose(Uy_fir, Uy_mc, atol=1e-1, rtol=1e-1)


def test_IIRuncFilter():
    pass
