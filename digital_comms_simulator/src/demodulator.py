"""
demodulator.py — Coherent receivers, matched filter, and equalisers.

Classes
-------
BPSKDemodulator
    Hard-decision and soft (LLR) demodulation for BPSK.
QPSKDemodulator
    Hard-decision demodulation for QPSK.
QAM16Demodulator
    Minimum-distance demodulation for 16-QAM.
MatchedFilter
    Correlator-based pulse matched filter.
ZeroForcingEqualizer
    Flat-fading and frequency-domain (block) ZF equalisers.
"""

import numpy as np
from scipy.signal import fftconvolve

from modulator import ConstellationMapper


# ---------------------------------------------------------------------------
# BPSK demodulator
# ---------------------------------------------------------------------------

class BPSKDemodulator:
    """
    Coherent BPSK demodulator.

    Uses the sign of the real component as the decision statistic —
    equivalent to ML detection for AWGN.
    """

    def __init__(self):
        self._mapper = ConstellationMapper('bpsk')

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision demodulation.

        Parameters
        ----------
        symbols : ndarray of complex or real, shape (N,)

        Returns
        -------
        bits : ndarray of int, shape (N,)
            0 for symbol −1, 1 for symbol +1.
        """
        return (np.real(symbols) >= 0).astype(np.int32)

    def compute_llr(self, symbols: np.ndarray,
                    noise_var: float) -> np.ndarray:
        """
        Log-likelihood ratio for BPSK:

            LLR(y) = log P(y|bit=1) / P(y|bit=0)
                   = 2 Re(y) / σ²

        Positive LLR → bit 1 is more likely.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)
        noise_var : float
            Noise variance per real dimension (σ²).

        Returns
        -------
        llr : ndarray of float, shape (N,)
        """
        return 2 * np.real(symbols) / noise_var


# ---------------------------------------------------------------------------
# QPSK demodulator
# ---------------------------------------------------------------------------

class QPSKDemodulator:
    """
    Coherent QPSK demodulator.

    Decision regions are the four quadrants of the complex plane.
    """

    def __init__(self):
        self._mapper = ConstellationMapper('qpsk')

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision QPSK demodulation.

        The constellation is (see modulator.py):
            index 0 (bits 00): (+1+1j)/√2   – real>0, imag>0
            index 1 (bits 01): (+1−1j)/√2   – real>0, imag<0
            index 2 (bits 10): (−1+1j)/√2   – real<0, imag>0
            index 3 (bits 11): (−1−1j)/√2   – real<0, imag<0

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        bits : ndarray of int, shape (2N,)
        """
        symbols = np.asarray(symbols, dtype=complex)
        i_bits = (np.real(symbols) < 0).astype(np.uint8)   # 0=+I, 1=−I
        q_bits = (np.imag(symbols) < 0).astype(np.uint8)   # 0=+Q, 1=−Q
        return np.column_stack([i_bits, q_bits]).flatten().astype(np.int32)


# ---------------------------------------------------------------------------
# 16-QAM demodulator
# ---------------------------------------------------------------------------

class QAM16Demodulator:
    """
    Coherent 16-QAM demodulator (minimum Euclidean distance).

    Delegates nearest-neighbour search to ConstellationMapper.demap_symbols.
    """

    def __init__(self):
        self._mapper = ConstellationMapper('qam16')

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision 16-QAM demodulation.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        bits : ndarray of int, shape (4N,)
        """
        return self._mapper.demap_symbols(symbols)


# ---------------------------------------------------------------------------
# Generic demodulator factory
# ---------------------------------------------------------------------------

def make_demodulator(modulation: str):
    """
    Return the appropriate demodulator instance for a modulation string.

    Parameters
    ----------
    modulation : str
        One of ``'bpsk'``, ``'qpsk'``, ``'qam16'``.

    Returns
    -------
    demodulator : BPSKDemodulator | QPSKDemodulator | QAM16Demodulator
    """
    mod = modulation.lower().replace('-', '').replace('_', '')
    if mod in ('16qam',):
        mod = 'qam16'
    if mod == 'bpsk':
        return BPSKDemodulator()
    elif mod == 'qpsk':
        return QPSKDemodulator()
    elif mod == 'qam16':
        return QAM16Demodulator()
    raise ValueError(f"Unknown modulation '{modulation}'.")


# ---------------------------------------------------------------------------
# Matched filter
# ---------------------------------------------------------------------------

class MatchedFilter:
    """
    Correlator-based pulse matched filter.

    At the transmitter, symbols are upsampled and filtered with a pulse
    p(t).  The matched filter at the receiver is the time-reversed
    conjugate of p(t): h_MF(t) = p*(−t).

    Parameters
    ----------
    pulse_shape : array_like of float or complex
        The transmit pulse coefficients p[n].
    """

    def __init__(self, pulse_shape: np.ndarray):
        self.pulse = np.asarray(pulse_shape, dtype=complex)
        # MF kernel = time-reversed conjugate
        self._kernel = np.conj(self.pulse[::-1])

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the matched filter to a received signal.

        Uses overlap-add via FFT for efficiency.  The output is aligned
        so that the peak response occurs at sample index
        ``len(pulse) − 1`` (causal processing).

        Parameters
        ----------
        signal : ndarray of complex, shape (N,)

        Returns
        -------
        filtered : ndarray of complex, shape (N,)
        """
        signal = np.asarray(signal, dtype=complex)
        full = fftconvolve(signal, self._kernel)
        # Trim to original length (causal)
        return full[:len(signal)]

    def correlate(self, signal: np.ndarray,
                  reference: np.ndarray) -> np.ndarray:
        """
        Cross-correlate ``signal`` with ``reference`` (MF operation).

        Parameters
        ----------
        signal : ndarray
        reference : ndarray

        Returns
        -------
        corr : ndarray
        """
        kernel = np.conj(reference[::-1])
        full = fftconvolve(signal, kernel)
        return full[:len(signal)]


# ---------------------------------------------------------------------------
# Zero-forcing equaliser
# ---------------------------------------------------------------------------

class ZeroForcingEqualizer:
    """
    Zero-forcing channel equaliser.

    Supports two modes:

    1.  **Flat fading** (scalar per-symbol channel):
        ``equalize_flat(received, h_estimate)``
        Divides each received sample by its channel coefficient.

    2.  **Frequency-domain block ZF** (for ISI channels):
        ``equalize_frequency_domain(received, h_estimate)``
        Transforms to frequency domain, applies 1/H(f), transforms back.

    Parameters
    ----------
    regularization : float
        Ridge term added to |H(f)|² to stabilise the inversion
        (prevents divide-by-zero in deep notches).  Also called
        MMSE-ZF when applied per frequency bin.
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization

    def equalize_flat(self, received: np.ndarray,
                      h_estimate: np.ndarray) -> np.ndarray:
        """
        Flat-fading ZF equalisation: ŷ = y / h.

        Parameters
        ----------
        received : ndarray of complex, shape (N,)
        h_estimate : ndarray of complex, shape (N,)
            Per-symbol channel estimates.

        Returns
        -------
        equalized : ndarray of complex, shape (N,)
        """
        received = np.asarray(received, dtype=complex)
        h = np.asarray(h_estimate, dtype=complex)
        return received / (h + self.regularization * (np.abs(h) < 1e-10))

    def equalize_frequency_domain(self, received: np.ndarray,
                                  h_estimate: np.ndarray) -> np.ndarray:
        """
        Block frequency-domain ZF equalisation for ISI channels.

        Assumes a cyclic-prefix (or sufficiently long block) model.
        Uses MMSE-ZF regularisation to handle spectral nulls.

        Parameters
        ----------
        received : ndarray of complex, shape (N,)
        h_estimate : array_like of complex
            CIR estimate (may be shorter than N; zero-padded to N).

        Returns
        -------
        equalized : ndarray of complex, shape (N,)
        """
        received = np.asarray(received, dtype=complex)
        N = len(received)
        H = np.fft.fft(np.asarray(h_estimate, dtype=complex), n=N)
        Y = np.fft.fft(received)
        # Wiener-ZF: W = H* / (|H|² + λ)
        W = np.conj(H) / (np.abs(H) ** 2 + self.regularization)
        X_eq = W * Y
        return np.fft.ifft(X_eq)
