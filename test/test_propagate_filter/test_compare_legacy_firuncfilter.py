import numpy as np
import pytest
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz
from scipy.signal import lfilter, lfilter_zi

from PyDynamic.misc.tools import trimOrPad

# noinspection PyProtectedMember
from PyDynamic.uncertainty.propagate_filter import (
    FIRuncFilter,
)
from ..conftest import _print_during_test_to_avoid_timeout, FIRuncFilter_input


@given(FIRuncFilter_input())
@settings(
    deadline=None,
    suppress_health_check=[
        *settings.default.suppress_health_check,
        HealthCheck.function_scoped_fixture,
    ],
)
@pytest.mark.slow
def test_FIRuncFilter_legacy_comparison(capsys, fir_unc_filter_input):
    def legacy_FIRuncFilter(
        y, sigma_noise, theta, Utheta=None, shift=0, blow=None, kind="corr"
    ):
        """Uncertainty propagation for signal y and uncertain FIR filter theta

        A preceding FIR low-pass filter with coefficients `blow` can be provided
        optionally.

        Parameters
        ----------
        y : np.ndarray
            filter input signal
        sigma_noise : float or np.ndarray
            float: standard deviation of white noise in y
            1D-array: interpretation depends on kind
        theta : np.ndarray
            FIR filter coefficients
        Utheta : np.ndarray, optional
            covariance matrix associated with theta
            if the filter is fully certain, use `Utheta = None` (default) to make use of
            more efficient calculations. See also the comparison given in
            <examples/Digital filtering/FIRuncFilter_runtime_comparison.py>
        shift : int, optional
            time delay of filter output signal (in samples) (defaults to 0)
        blow : np.ndarray, optional
            optional FIR low-pass filter
        kind : string
            only meaningful in combination with sigma_noise a 1D numpy array
            "diag": point-wise standard uncertainties of non-stationary white noise
            "corr": single sided autocovariance of stationary (colored/correlated)
            noise (default)

        Returns
        -------
        x : np.ndarray
            FIR filter output signal
        ux : np.ndarray
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
                f"is of shape {sigma_noise.shape}. Please check documentation for "
                f"input "
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
                    # [Leeuw1994](Covariance matrix of ARMA errors in closed form)
                    # can be
                    # used, to derive this formula
                    # The given "blow" corresponds to a MA(q)-process.
                    # Going through the calculations of Leeuw, but assuming
                    # that E(vv^T) is a diagonal matrix with non-identical elements,
                    # the covariance matrix V becomes (see Leeuw:corollary1)
                    # V = N * SP * N^T + M * S * M^T
                    # N, M are defined as in the paper
                    # and SP is the covariance of input-noise prior to the observed
                    # time-interval (SP needs be available len(blow)-time steps into the
                    # past. Here it is assumed, that SP is constant with the first value
                    # of sigma2)

                    # V needs to be extended to cover Ntheta-1 time steps more into
                    # the past
                    sigma2_extended = np.append(
                        sigma2[0] * np.ones((Ntheta - 1)), sigma2
                    )

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
                    # note1: this is the reason, why Ulow will have dimension
                    # (Ntheta x Ntheta) without further ado

                    # calculate Bcorr
                    Bcorr = np.correlate(blow, blow, "full")

                    # pad or crop length of sigma2, then reflect some part to the
                    # left and
                    # invert the order
                    # [0 1 2 3 4 5 6 7] --> [0 0 0 7 6 5 4 3 2 1 0 1 2 3]
                    sigma2 = trimOrPad(sigma2, len(blow) + Ntheta - 1)
                    sigma2_reflect = np.pad(sigma2, (len(blow) - 1, 0), mode="reflect")

                    ycorr = np.correlate(
                        sigma2_reflect, Bcorr, mode="valid"
                    )  # used convolve in a earlier version, should make no
                    # difference as
                    # Bcorr is symmetric
                    Ulow = toeplitz(ycorr)

            xlow, _ = lfilter(blow, 1.0, y, zi=y[0] * lfilter_zi(blow, 1.0))

        else:  # if blow is not provided
            if isinstance(sigma2, float):
                Ulow = np.eye(Ntheta) * sigma2

            elif isinstance(sigma2, np.ndarray):

                if kind == "diag":
                    # V needs to be extended to cover Ntheta time steps more into the
                    # past
                    sigma2_extended = np.append(
                        sigma2[0] * np.ones((Ntheta - 1)), sigma2
                    )

                    # Ulow is to be sliced from V, see below
                    V = np.diag(
                        sigma2_extended
                    )  # this is not Ulow, same thing as in the case of a provided blow
                    # (see above)

                elif kind == "corr":
                    Ulow = toeplitz(trimOrPad(sigma2, Ntheta))

            xlow = y

        # apply FIR filter to calculate best estimate in accordance with GUM
        x, _ = lfilter(theta, 1.0, xlow, zi=xlow[0] * lfilter_zi(theta, 1.0))
        x = np.roll(x, -int(shift))

        # add dimension to theta, otherwise transpose won't work
        if len(theta.shape) == 1:
            theta = theta[:, np.newaxis]

        # NOTE: In the code below wherever `theta` or `Utheta` get used, they need to be
        # flipped. This is necessary to take the time-order of both variables into
        # account. (Which is descending for `theta` and `Utheta` but ascending for
        # `Ulow`.)
        #
        # Further details and illustrations showing the effect of not-flipping
        # can be found at https://github.com/PTB-M4D/PyDynamic/issues/183

        # handle diag-case, where Ulow needs to be sliced from V
        if kind == "diag":
            # UncCov needs to be calculated inside in its own for-loop
            # V has dimension (len(sigma2) + Ntheta) * (len(sigma2) + Ntheta) -->
            # slice a
            # fitting Ulow of dimension (Ntheta x Ntheta)
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

            # use extended signal to match assumption of stationary signal prior to
            # first
            # entry
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

    legacy_y, legacy_Uy = legacy_FIRuncFilter(**fir_unc_filter_input)
    current_y, current_Uy = FIRuncFilter(**fir_unc_filter_input)
    _print_during_test_to_avoid_timeout(capsys)
    assert_allclose(
        legacy_y,
        current_y,
        atol=1e-15,
    )
    assert_allclose(
        legacy_Uy,
        current_Uy,
        atol=np.max((np.max(current_Uy) * 1e-7, 1e-7)),
        rtol=3e-6,
    )
