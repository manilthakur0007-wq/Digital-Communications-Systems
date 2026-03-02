"""
pulse_shaping.py — Root Raised Cosine (RRC) pulse shaping.

The RRC filter is designed for zero ISI at optimum sampling instants
when transmit- and receive-RRC filters are cascaded (forming a full
raised-cosine response).

Classes
-------
RRCFilter
    Designs and applies the matched RRC filter pair.

Functions
---------
rrc_impulse_response
    Compute the RRC impulse response coefficients analytically.
eye_diagram_data
    Extract the eye-diagram matrix from an oversampled signal.
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# RRC impulse response (closed-form)
# ---------------------------------------------------------------------------

def rrc_impulse_response(alpha: float, span: int, sps: int) -> np.ndarray:
    """
    Compute the Root Raised Cosine filter impulse response.

    Parameters
    ----------
    alpha : float
        Roll-off factor, 0 < alpha ≤ 1.
    span  : int
        Filter span in symbol periods (one-sided).  Total taps = 2*span*sps + 1.
    sps   : int
        Samples per symbol (oversampling factor).

    Returns
    -------
    h : ndarray of float, shape (2*span*sps + 1,)
        Normalised so that h @ h = 1.0 (unit energy per tap-length).

    Notes
    -----
    Uses the closed-form expression from Proakis & Salehi (5th ed.),
    handling the special cases t = 0 and t = ±Ts/(4α) separately to
    avoid 0/0 division.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Roll-off factor alpha must be in (0, 1].")

    n_taps = 2 * span * sps + 1
    t = np.arange(-(span * sps), span * sps + 1) / sps  # in symbol periods

    h = np.zeros(n_taps)
    Ts = 1.0  # normalised symbol period

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1 / Ts) * (1 - alpha + (4 * alpha / np.pi))
        elif abs(ti) == Ts / (4 * alpha):
            h[i] = ((alpha / (Ts * np.sqrt(2))) *
                    ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                     (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))))
        else:
            num = (np.sin(np.pi * ti / Ts * (1 - alpha)) +
                   4 * alpha * ti / Ts *
                   np.cos(np.pi * ti / Ts * (1 + alpha)))
            denom = (np.pi * ti / Ts *
                     (1 - (4 * alpha * ti / Ts) ** 2))
            h[i] = num / (denom * Ts)

    # Normalise to unit energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ---------------------------------------------------------------------------
# Eye-diagram extractor
# ---------------------------------------------------------------------------

def eye_diagram_data(signal: np.ndarray, sps: int,
                     n_traces: int = 100) -> np.ndarray:
    """
    Extract the eye-diagram matrix from an oversampled signal.

    Parameters
    ----------
    signal   : ndarray of float or complex, shape (N,)
        The baseband waveform sampled at ``sps`` samples per symbol.
    sps      : int
        Samples per symbol.
    n_traces : int
        Maximum number of symbol traces to include.

    Returns
    -------
    eye : ndarray, shape (n_traces, 2*sps)
        Each row is one 2-symbol-period segment of the signal
        (real part only).
    """
    signal = np.real(np.asarray(signal, dtype=complex))
    n_cols = 2 * sps
    # Number of complete 2-symbol windows
    n_avail = (len(signal) - n_cols) // sps
    n_traces = min(n_traces, n_avail)
    eye = np.zeros((n_traces, n_cols))
    for i in range(n_traces):
        start = i * sps
        eye[i] = signal[start: start + n_cols]
    return eye


# ---------------------------------------------------------------------------
# RRC filter class
# ---------------------------------------------------------------------------

class RRCFilter:
    """
    Root Raised Cosine matched filter pair.

    Parameters
    ----------
    alpha : float
        Roll-off factor (0, 1].
    span  : int
        One-sided filter span in symbol periods.
    sps   : int
        Samples per symbol (oversampling factor).
    """

    def __init__(self, alpha: float = 0.5, span: int = 8, sps: int = 8):
        self.alpha = alpha
        self.span = span
        self.sps = sps
        self._h = rrc_impulse_response(alpha, span, sps)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coefficients(self) -> np.ndarray:
        """RRC filter tap coefficients (copy)."""
        return self._h.copy()

    @property
    def n_taps(self) -> int:
        return len(self._h)

    @property
    def group_delay(self) -> int:
        """Group delay in samples (half the filter length)."""
        return (self.n_taps - 1) // 2

    # ------------------------------------------------------------------
    # Transmit path
    # ------------------------------------------------------------------

    def upsample(self, symbols: np.ndarray) -> np.ndarray:
        """
        Upsample a symbol sequence by ``sps`` (insert zero samples).

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        upsampled : ndarray of complex, shape (N * sps,)
        """
        symbols = np.asarray(symbols, dtype=complex)
        out = np.zeros(len(symbols) * self.sps, dtype=complex)
        out[::self.sps] = symbols
        return out

    def transmit(self, symbols: np.ndarray) -> np.ndarray:
        """
        Upsample symbols and apply the transmit RRC filter.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        tx_waveform : ndarray of complex, shape (N * sps,)
            The waveform is trimmed to N*sps samples (causal).
        """
        upsampled = self.upsample(symbols)
        filtered = fftconvolve(upsampled, self._h)
        return filtered[:len(upsampled)]

    # ------------------------------------------------------------------
    # Receive path
    # ------------------------------------------------------------------

    def receive(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply the receive (matched) RRC filter and downsample.

        The RRC filter is its own matched filter, so the same
        coefficients are used.  The combined response is a raised-cosine
        with zero ISI at the optimum sampling instants.

        Parameters
        ----------
        waveform : ndarray of complex, shape (N * sps,)

        Returns
        -------
        symbols : ndarray of complex, shape (N,)
            Downsample by ``sps`` at the peak filter output.
        """
        waveform = np.asarray(waveform, dtype=complex)
        filtered = fftconvolve(waveform, self._h)
        # Account for group delay of both TX and RX filters
        delay = 2 * self.group_delay
        if delay < len(filtered):
            aligned = filtered[delay:]
        else:
            aligned = filtered
        # Downsample
        n_sym = len(waveform) // self.sps
        return aligned[:n_sym * self.sps:self.sps]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def power_spectral_density(self, n_fft: int = 1024):
        """
        Return the normalised PSD of the RRC filter.

        Parameters
        ----------
        n_fft : int

        Returns
        -------
        (freqs, psd) : tuple of ndarray
            ``freqs`` in normalised units (cycles per sample),
            ``psd``   in dB.
        """
        H = np.fft.fftshift(np.fft.fft(self._h, n=n_fft))
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))
        psd_dB = 20 * np.log10(np.abs(H) + 1e-20)
        return freqs, psd_dB

    def eye_diagram(self, symbols: np.ndarray,
                    n_traces: int = 100) -> np.ndarray:
        """
        Generate an eye-diagram matrix for the given symbol sequence.

        Symbols are transmitted through the RRC filter and the resulting
        waveform is sliced into overlapping 2-symbol windows.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)
        n_traces : int

        Returns
        -------
        eye : ndarray, shape (n_traces, 2*sps)
        """
        tx_wave = self.transmit(symbols)
        return eye_diagram_data(tx_wave, self.sps, n_traces)
