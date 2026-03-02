"""
channel_estimator.py — Pilot-based channel estimation for Rayleigh fading.

PilotChannelEstimator inserts known pilot symbols at regular intervals,
estimates the channel at pilot positions via least-squares, and
interpolates to obtain a full-frame channel estimate.

Supported interpolation methods:  'linear', 'spline'.
"""

import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Pilot channel estimator
# ---------------------------------------------------------------------------

class PilotChannelEstimator:
    """
    Pilot-aided LS channel estimator for flat Rayleigh fading.

    Frame structure:  [pilot | data data … data | pilot | data …]
    Every ``pilot_spacing``-th symbol is a known pilot.

    Parameters
    ----------
    pilot_spacing : int
        Number of data symbols between consecutive pilots (must be ≥ 1).
        A pilot is inserted at every ``pilot_spacing + 1``-th position.
    pilot_value : complex
        Known pilot symbol value (default +1).
    interpolation : str
        Method for interpolating channel estimates between pilots:
        ``'linear'`` or ``'spline'``.
    """

    def __init__(
        self,
        pilot_spacing: int = 8,
        pilot_value: complex = 1.0 + 0j,
        interpolation: str = 'linear',
    ):
        if pilot_spacing < 1:
            raise ValueError("pilot_spacing must be >= 1.")
        if interpolation not in ('linear', 'spline'):
            raise ValueError("interpolation must be 'linear' or 'spline'.")
        self.pilot_spacing = pilot_spacing
        self.pilot_value = complex(pilot_value)
        self.interpolation = interpolation
        # Period: 1 pilot + pilot_spacing data symbols
        self._period = pilot_spacing + 1

    # ------------------------------------------------------------------
    # Frame assembly / disassembly
    # ------------------------------------------------------------------

    def insert_pilots(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Interleave pilot symbols into the data stream.

        The returned frame has length:
        ``len(data_symbols) + ceil(len(data_symbols) / pilot_spacing)``

        Parameters
        ----------
        data_symbols : ndarray of complex, shape (N_data,)

        Returns
        -------
        frame : ndarray of complex, shape (N_frame,)
        """
        data = np.asarray(data_symbols, dtype=complex)
        n_data = len(data)
        n_pilots = int(np.ceil(n_data / self.pilot_spacing))
        n_frame = n_data + n_pilots
        frame = np.zeros(n_frame, dtype=complex)
        pilot_positions = self._pilot_positions(n_frame)
        frame[pilot_positions] = self.pilot_value
        data_positions = np.setdiff1d(np.arange(n_frame), pilot_positions)
        frame[data_positions[:n_data]] = data
        return frame

    def extract_pilots(self, rx_frame: np.ndarray):
        """
        Extract received pilot positions and their values.

        Parameters
        ----------
        rx_frame : ndarray of complex, shape (N_frame,)

        Returns
        -------
        pilot_pos : ndarray of int
            Sample indices of pilots in the frame.
        pilot_vals : ndarray of complex
            Received pilot values at those positions.
        """
        rx_frame = np.asarray(rx_frame, dtype=complex)
        pilot_pos = self._pilot_positions(len(rx_frame))
        return pilot_pos, rx_frame[pilot_pos]

    def extract_data(self, rx_frame: np.ndarray) -> np.ndarray:
        """
        Extract non-pilot (data) positions from the received frame.

        Parameters
        ----------
        rx_frame : ndarray of complex, shape (N_frame,)

        Returns
        -------
        data_symbols : ndarray of complex
        """
        rx_frame = np.asarray(rx_frame, dtype=complex)
        n_frame = len(rx_frame)
        pilot_pos = self._pilot_positions(n_frame)
        data_pos = np.setdiff1d(np.arange(n_frame), pilot_pos)
        return rx_frame[data_pos]

    # ------------------------------------------------------------------
    # Channel estimation
    # ------------------------------------------------------------------

    def estimate_channel(self, rx_frame: np.ndarray) -> np.ndarray:
        """
        Estimate the channel at every sample in the frame.

        Algorithm
        ---------
        1. LS estimate at pilot positions:  ĥ[k] = y[k] / x_p
        2. Interpolation between pilot estimates.

        Parameters
        ----------
        rx_frame : ndarray of complex, shape (N_frame,)

        Returns
        -------
        h_est : ndarray of complex, shape (N_frame,)
            Channel estimate at every sample position.
        """
        rx_frame = np.asarray(rx_frame, dtype=complex)
        n_frame = len(rx_frame)
        pilot_pos, pilot_rx = self.extract_pilots(rx_frame)

        # LS: ĥ = y_pilot / x_pilot
        h_at_pilots = pilot_rx / self.pilot_value

        return self._interpolate(pilot_pos, h_at_pilots, n_frame)

    def _interpolate(self, x_known: np.ndarray, y_known: np.ndarray,
                     n: int) -> np.ndarray:
        """
        Interpolate complex channel estimates from pilot positions to all n samples.
        """
        x_all = np.arange(n, dtype=float)

        if self.interpolation == 'linear' or len(x_known) < 3:
            # Real and imaginary parts separately
            h_real = np.interp(x_all, x_known.astype(float),
                               y_known.real,
                               left=y_known.real[0],
                               right=y_known.real[-1])
            h_imag = np.interp(x_all, x_known.astype(float),
                               y_known.imag,
                               left=y_known.imag[0],
                               right=y_known.imag[-1])
        else:
            cs_real = CubicSpline(x_known.astype(float), y_known.real,
                                  extrapolate=True)
            cs_imag = CubicSpline(x_known.astype(float), y_known.imag,
                                  extrapolate=True)
            h_real = cs_real(x_all)
            h_imag = cs_imag(x_all)

        return h_real + 1j * h_imag

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pilot_positions(self, n_frame: int) -> np.ndarray:
        """Return pilot sample indices within a frame of length n_frame."""
        return np.arange(0, n_frame, self._period)

    def pilot_overhead(self) -> float:
        """Fractional overhead due to pilots (0…1)."""
        return 1.0 / self._period

    def __repr__(self) -> str:
        return (
            f"PilotChannelEstimator("
            f"pilot_spacing={self.pilot_spacing}, "
            f"pilot_value={self.pilot_value}, "
            f"interpolation='{self.interpolation}')"
        )
