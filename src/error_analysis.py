"""
error_analysis.py — Monte Carlo BER/SER/FER engine.

MonteCarloSimulator runs configurable numbers of trials per Eb/N0 point,
counts bit errors, symbol errors, and frame errors, and returns 95%
Wilson-score confidence intervals on the BER.

All inner loops are fully NumPy-vectorised — no Python loops per symbol.
"""

import numpy as np
from typing import Optional

from modulator import ConstellationMapper
from channel import AWGNChannel, RayleighChannel, ISIChannel
from demodulator import ZeroForcingEqualizer


# ---------------------------------------------------------------------------
# Wilson-score confidence interval helper
# ---------------------------------------------------------------------------

def wilson_ci(n_errors: int, n_total: int, z: float = 1.96):
    """
    Wilson-score 95% confidence interval for a Bernoulli proportion.

    More accurate than the normal approximation when BER is very small
    or when n_errors = 0.

    Parameters
    ----------
    n_errors : int
    n_total  : int
    z        : float  (1.96 for 95% CI)

    Returns
    -------
    (lower, upper) : tuple of float
    """
    if n_total == 0:
        return 0.0, 1.0
    p = n_errors / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2))
                  / denom)
    return max(0.0, centre - half_width), centre + half_width


# ---------------------------------------------------------------------------
# Monte Carlo simulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Monte Carlo BER / SER / FER simulator.

    Parameters
    ----------
    modulation : str
        ``'bpsk'``, ``'qpsk'``, or ``'qam16'``.
    channel_type : str
        ``'awgn'``, ``'rayleigh'``, or ``'isi'``.
    n_bits : int
        Number of bits per Eb/N0 point (minimum recommended: 100 000).
    seed : int or None
        Random seed for full reproducibility.
    frame_length : int
        Frame size in bits for FER computation.
    isi_response : array_like or None
        FIR coefficients for the ISI channel (required if
        ``channel_type='isi'``).
    """

    def __init__(
        self,
        modulation: str = 'bpsk',
        channel_type: str = 'awgn',
        n_bits: int = 100_000,
        seed: Optional[int] = None,
        frame_length: int = 1_000,
        isi_response=None,
    ):
        if n_bits < 10_000:
            raise ValueError("n_bits should be >= 10 000 for reliable BER estimates.")
        self.modulation = modulation
        self.channel_type = channel_type.lower()
        self.n_bits = n_bits
        self.seed = seed
        self.frame_length = frame_length
        self.isi_response = (np.asarray(isi_response, dtype=complex)
                             if isi_response is not None
                             else np.array([1.0 + 0j]))

        self.mapper = ConstellationMapper(modulation)
        self._zf = ZeroForcingEqualizer()

    # ------------------------------------------------------------------
    # Single-point simulation
    # ------------------------------------------------------------------

    def run_single(self, EbN0_dB: float) -> dict:
        """
        Run a Monte Carlo trial at one Eb/N0 operating point.

        Parameters
        ----------
        EbN0_dB : float

        Returns
        -------
        result : dict with keys:
            'EbN0_dB', 'BER', 'SER', 'FER',
            'CI_lower', 'CI_upper',
            'n_bit_errors', 'n_sym_errors', 'n_frame_errors',
            'n_bits', 'n_symbols', 'n_frames'
        """
        rng = np.random.default_rng(self.seed)
        k = self.mapper.bits_per_symbol

        # Align n_bits to a whole number of symbols
        n_sym = self.n_bits // k
        n_bits_used = n_sym * k

        # ---- Transmit ----
        tx_bits = rng.integers(0, 2, size=n_bits_used, dtype=np.int32)
        tx_symbols = self.mapper.map_bits(tx_bits)

        # ---- Channel ----
        if self.channel_type == 'awgn':
            ch = AWGNChannel(EbN0_dB, k, seed=None)
            ch._rng = rng
            rx_symbols = ch.corrupt(tx_symbols)
            noise_var = ch.noise_variance

        elif self.channel_type == 'rayleigh':
            ch = RayleighChannel(EbN0_dB, k, seed=None)
            ch._rng = rng
            ch._awgn._rng = rng
            rx_raw, h_coeff = ch.corrupt(tx_symbols)
            # Zero-forcing equalisation with perfect CSI
            rx_symbols = self._zf.equalize_flat(rx_raw, h_coeff)
            noise_var = ch._awgn.noise_variance

        elif self.channel_type == 'isi':
            ch = ISIChannel(self.isi_response, EbN0_dB, k, seed=None)
            ch._awgn._rng = rng
            rx_raw = ch.corrupt(tx_symbols)
            # Frequency-domain ZF equalisation
            rx_symbols = self._zf.equalize_frequency_domain(
                rx_raw, self.isi_response
            )
            noise_var = ch._awgn.noise_variance

        else:
            raise ValueError(f"Unknown channel '{self.channel_type}'.")

        # ---- Demodulate ----
        rx_bits = self.mapper.demap_symbols(rx_symbols)

        # ---- Error counting ----
        n_bit_errors = int(np.sum(tx_bits[:len(rx_bits)] != rx_bits))

        # Symbol errors
        tx_idx = self.mapper.bits_to_indices(tx_bits)
        rx_idx = self.mapper.bits_to_indices(rx_bits)
        n_sym_errors = int(np.sum(tx_idx != rx_idx))

        # Frame errors (vectorised)
        n_frames = n_bits_used // self.frame_length
        if n_frames > 0:
            b_tx = tx_bits[:n_frames * self.frame_length].reshape(n_frames, self.frame_length)
            b_rx = rx_bits[:n_frames * self.frame_length].reshape(n_frames, self.frame_length)
            n_frame_errors = int(np.sum(np.any(b_tx != b_rx, axis=1)))
        else:
            n_frame_errors = 0
            n_frames = 1  # avoid division by zero

        BER = n_bit_errors / n_bits_used
        SER = n_sym_errors / n_sym
        FER = n_frame_errors / n_frames
        ci_lo, ci_hi = wilson_ci(n_bit_errors, n_bits_used)

        return {
            'EbN0_dB': EbN0_dB,
            'BER': BER,
            'SER': SER,
            'FER': FER,
            'CI_lower': ci_lo,
            'CI_upper': ci_hi,
            'n_bit_errors': n_bit_errors,
            'n_sym_errors': n_sym_errors,
            'n_frame_errors': n_frame_errors,
            'n_bits': n_bits_used,
            'n_symbols': n_sym,
            'n_frames': n_frames,
            'noise_var': noise_var,
        }

    # ------------------------------------------------------------------
    # SNR sweep
    # ------------------------------------------------------------------

    def run_sweep(self, EbN0_range) -> dict:
        """
        Run Monte Carlo over an array of Eb/N0 values.

        Parameters
        ----------
        EbN0_range : array_like of float
            Eb/N0 values in dB.

        Returns
        -------
        results : dict
            Keys are the same as ``run_single`` output but each value is
            a list aligned with ``EbN0_range``.
        """
        EbN0_range = np.asarray(EbN0_range, dtype=float)
        aggregated: dict = {k: [] for k in (
            'EbN0_dB', 'BER', 'SER', 'FER',
            'CI_lower', 'CI_upper',
            'n_bit_errors', 'n_sym_errors', 'n_frame_errors',
            'n_bits', 'n_symbols', 'n_frames', 'noise_var'
        )}
        for snr in EbN0_range:
            r = self.run_single(float(snr))
            for key in aggregated:
                aggregated[key].append(r[key])
        # Convert lists to numpy arrays for convenience
        return {k: np.array(v) for k, v in aggregated.items()}

    # ------------------------------------------------------------------
    # Convenience: count errors only (no overhead of dict creation)
    # ------------------------------------------------------------------

    def count_errors(self, tx_bits: np.ndarray,
                     rx_bits: np.ndarray) -> tuple:
        """
        Count bit and symbol errors between two bit arrays.

        Parameters
        ----------
        tx_bits : ndarray of int
        rx_bits : ndarray of int

        Returns
        -------
        (n_bit_errors, n_sym_errors) : tuple of int
        """
        n = min(len(tx_bits), len(rx_bits))
        n_bit = int(np.sum(tx_bits[:n] != rx_bits[:n]))
        k = self.mapper.bits_per_symbol
        tx_i = self.mapper.bits_to_indices(tx_bits[:n])
        rx_i = self.mapper.bits_to_indices(rx_bits[:n])
        n_sym = int(np.sum(tx_i != rx_i))
        return n_bit, n_sym
