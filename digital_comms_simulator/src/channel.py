"""
channel.py — Channel models for digital communications simulation.

Classes
-------
AWGNChannel
    Adds complex Gaussian noise at a specified Eb/N0 (dB).
RayleighChannel
    Flat Rayleigh fading with optional Doppler spread plus AWGN.
ISIChannel
    FIR inter-symbol-interference channel with AWGN.

All channels assume unit average symbol energy (Es = 1) at the input.

Noise power derivation
----------------------
    Es  = 1  (normalised symbols)
    Eb  = Es / k          where k = bits_per_symbol
    N0  = Eb / EbN0_lin   = 1 / (k · EbN0_lin)
    σ²  = N0 / 2          noise variance per real dimension (I or Q)

Complex noise sample: n = σ·(nI + j·nQ),  nI,nQ ~ N(0,1)
so E[|n|²] = 2σ² = N0.
"""

import numpy as np


# ---------------------------------------------------------------------------
# AWGN channel
# ---------------------------------------------------------------------------

class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    Parameters
    ----------
    EbN0_dB : float
        Signal-to-noise ratio per bit in dB.
    bits_per_symbol : int
        Number of bits encoded per complex symbol (k).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, EbN0_dB: float, bits_per_symbol: int = 1,
                 seed=None):
        self.EbN0_dB = EbN0_dB
        self.bits_per_symbol = bits_per_symbol
        self._rng = np.random.default_rng(seed)
        self._update_sigma()

    def _update_sigma(self):
        EbN0_lin = 10 ** (self.EbN0_dB / 10)
        # N0 = 1 / (k * EbN0_lin);  σ = sqrt(N0/2)
        self._sigma = np.sqrt(1.0 / (2 * self.bits_per_symbol * EbN0_lin))

    @property
    def noise_std(self) -> float:
        """Standard deviation of the noise per real dimension."""
        return self._sigma

    @property
    def noise_variance(self) -> float:
        """Noise variance per real dimension (σ²)."""
        return self._sigma ** 2

    def corrupt(self, symbols: np.ndarray) -> np.ndarray:
        """
        Add complex Gaussian noise to a symbol array.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        noisy : ndarray of complex, shape (N,)
        """
        symbols = np.asarray(symbols, dtype=complex)
        noise = self._sigma * (
            self._rng.standard_normal(symbols.shape) +
            1j * self._rng.standard_normal(symbols.shape)
        )
        return symbols + noise

    def set_EbN0(self, EbN0_dB: float):
        """Update Eb/N0 without replacing the RNG state."""
        self.EbN0_dB = EbN0_dB
        self._update_sigma()


# ---------------------------------------------------------------------------
# Rayleigh flat-fading channel
# ---------------------------------------------------------------------------

class RayleighChannel:
    """
    Flat Rayleigh fading channel (one complex gain per symbol) plus AWGN.

    The channel coefficient h ~ CN(0, 1), i.e.
        h = (nI + j·nQ) / √2,   nI,nQ ~ N(0,1)
    so E[|h|²] = 1 (unit average power).

    After applying the channel:  y = h · x + n

    Parameters
    ----------
    EbN0_dB : float
        Eb/N0 in dB (referred to the transmitted symbol energy).
    bits_per_symbol : int
        k for the modulation in use.
    doppler_spread : float
        Normalised Doppler spread fd·Ts (0 ⟹ quasi-static).
        Currently used only for metadata; Clarke's Doppler shaping is
        not applied (slow-fading approximation).
    seed : int or None
    """

    def __init__(self, EbN0_dB: float, bits_per_symbol: int = 1,
                 doppler_spread: float = 0.0, seed=None):
        self.EbN0_dB = EbN0_dB
        self.bits_per_symbol = bits_per_symbol
        self.doppler_spread = doppler_spread
        self._rng = np.random.default_rng(seed)
        self._awgn = AWGNChannel(EbN0_dB, bits_per_symbol, seed=None)
        # Share the same RNG so sequences are deterministic when seeded
        self._awgn._rng = self._rng

    def corrupt(self, symbols: np.ndarray):
        """
        Apply flat Rayleigh fading and add AWGN.

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        received : ndarray of complex, shape (N,)
            y = h · x + n
        h : ndarray of complex, shape (N,)
            Per-symbol complex channel coefficients.
        """
        symbols = np.asarray(symbols, dtype=complex)
        h = (
            self._rng.standard_normal(symbols.shape) +
            1j * self._rng.standard_normal(symbols.shape)
        ) / np.sqrt(2)
        received = h * symbols
        noisy = self._awgn.corrupt(received)
        return noisy, h

    def set_EbN0(self, EbN0_dB: float):
        self.EbN0_dB = EbN0_dB
        self._awgn.set_EbN0(EbN0_dB)


# ---------------------------------------------------------------------------
# ISI channel
# ---------------------------------------------------------------------------

class ISIChannel:
    """
    FIR inter-symbol-interference channel with AWGN.

    The transmitted signal is convolved with a finite impulse response h
    before noise is added.

    Parameters
    ----------
    impulse_response : array_like of complex
        FIR coefficients h[0], h[1], …, h[L-1].
    EbN0_dB : float
        Eb/N0 in dB.
    bits_per_symbol : int
    seed : int or None
    """

    def __init__(self, impulse_response, EbN0_dB: float,
                 bits_per_symbol: int = 1, seed=None):
        self.h = np.asarray(impulse_response, dtype=complex)
        self.EbN0_dB = EbN0_dB
        self.bits_per_symbol = bits_per_symbol
        self._awgn = AWGNChannel(EbN0_dB, bits_per_symbol, seed)

    @property
    def channel_length(self) -> int:
        return len(self.h)

    def corrupt(self, symbols: np.ndarray) -> np.ndarray:
        """
        Apply ISI via FIR convolution, then add AWGN.

        The convolution output is truncated to ``len(symbols)`` samples
        (causal, no look-ahead).

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)

        Returns
        -------
        noisy : ndarray of complex, shape (N,)
        """
        symbols = np.asarray(symbols, dtype=complex)
        # Full convolution; keep first N samples (causal truncation)
        distorted = np.convolve(symbols, self.h)[:len(symbols)]
        return self._awgn.corrupt(distorted)

    def frequency_response(self, n_fft: int = 512):
        """Return normalised frequency and |H(f)| for the channel."""
        H = np.fft.fft(self.h, n=n_fft)
        freqs = np.fft.fftfreq(n_fft)
        idx = np.argsort(freqs)
        return freqs[idx], np.abs(H[idx])

    def set_EbN0(self, EbN0_dB: float):
        self.EbN0_dB = EbN0_dB
        self._awgn.set_EbN0(EbN0_dB)
