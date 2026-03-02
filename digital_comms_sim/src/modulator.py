"""
modulator.py — Digital modulation: BPSK, QPSK, 16-QAM.

ConstellationMapper maps bit streams to complex baseband symbols with
Gray coding for QPSK and 16-QAM.  All constellations are normalized
to unit average symbol energy (Es = 1).

Gray coding ensures that adjacent symbols in the constellation differ
by exactly one bit, minimising BER at moderate-to-high SNR.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gray-code helpers
# ---------------------------------------------------------------------------

def binary_to_gray(n: int) -> int:
    """Convert a non-negative integer from binary to Gray code."""
    return n ^ (n >> 1)


def gray_to_binary(g: int) -> int:
    """Convert a Gray-coded integer back to standard binary."""
    b = g
    mask = g >> 1
    while mask:
        b ^= mask
        mask >>= 1
    return b


# ---------------------------------------------------------------------------
# Constellation mapper
# ---------------------------------------------------------------------------

class ConstellationMapper:
    """
    Maps groups of bits to complex constellation points (modulation) and
    maps received complex symbols back to bits (hard-decision demodulation).

    Parameters
    ----------
    modulation : str
        One of ``'bpsk'``, ``'qpsk'``, ``'qam16'``.
    """

    _SUPPORTED = ('bpsk', 'qpsk', 'qam16')

    def __init__(self, modulation: str = 'bpsk'):
        mod = modulation.lower().replace('-', '').replace('_', '')
        # Accept '16qam' as alias
        if mod == '16qam':
            mod = 'qam16'
        if mod not in self._SUPPORTED:
            raise ValueError(
                f"Unsupported modulation '{modulation}'. "
                f"Choose from {self._SUPPORTED}."
            )
        self.modulation = mod
        self._build_constellation()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_constellation(self):
        if self.modulation == 'bpsk':
            self._build_bpsk()
        elif self.modulation == 'qpsk':
            self._build_qpsk()
        elif self.modulation == 'qam16':
            self._build_16qam()

    def _build_bpsk(self):
        self.bits_per_symbol = 1
        # index 0 → bit 0 → symbol -1; index 1 → bit 1 → symbol +1
        self.constellation = np.array([-1.0 + 0j, 1.0 + 0j])

    def _build_qpsk(self):
        self.bits_per_symbol = 2
        # Gray-coded QPSK.
        # Two-bit index: MSB selects I sign, LSB selects Q sign.
        # 00 → (+1+1j), 01 → (+1−1j), 10 → (−1+1j), 11 → (−1−1j)
        # Verify Gray property: 00↔01, 00↔10, 01↔11, 10↔11 each differ by 1 bit ✓
        raw = np.array([
            1 + 1j,   # 00
            1 - 1j,   # 01
            -1 + 1j,  # 10
            -1 - 1j,  # 11
        ])
        self.constellation = raw / np.sqrt(2)  # normalise to Es = 1

    def _build_16qam(self):
        self.bits_per_symbol = 4
        # Gray-coded 16-QAM.
        # Four-bit index: top 2 bits (Gray) → I level, bottom 2 bits (Gray) → Q level.
        # Gray-code → PAM level: 00→−3, 01→−1, 11→+1, 10→+3
        gray_to_level = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
        points = np.zeros(16, dtype=complex)
        for idx in range(16):
            i_gray = (idx >> 2) & 0x3
            q_gray = idx & 0x3
            points[idx] = gray_to_level[i_gray] + 1j * gray_to_level[q_gray]
        # Average symbol energy = 10 for unnormalised {±1,±3}² constellation
        norm = np.sqrt(np.mean(np.abs(points) ** 2))
        self.constellation = points / norm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def M(self) -> int:
        """Constellation size (number of distinct symbols)."""
        return len(self.constellation)

    def map_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Map a flat bit array to complex symbols.

        Parameters
        ----------
        bits : array_like of int, shape (N * bits_per_symbol,)
            Bit stream; length must be a multiple of ``bits_per_symbol``.

        Returns
        -------
        symbols : ndarray of complex, shape (N,)
        """
        bits = np.asarray(bits, dtype=np.int32)
        k = self.bits_per_symbol
        n_sym = len(bits) // k
        # Reshape to (n_sym, k) and convert each row to an integer index
        bit_matrix = bits[: n_sym * k].reshape(n_sym, k)
        powers = (2 ** np.arange(k - 1, -1, -1)).astype(np.int32)
        indices = bit_matrix @ powers          # shape (n_sym,)
        return self.constellation[indices]

    def demap_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision ML demodulation (nearest-neighbour for AWGN).

        Parameters
        ----------
        symbols : ndarray of complex, shape (N,)
            Received baseband samples (after channel and equalisation).

        Returns
        -------
        bits : ndarray of int, shape (N * bits_per_symbol,)
        """
        symbols = np.asarray(symbols)
        k = self.bits_per_symbol
        # Vectorised squared Euclidean distance: (N, M)
        dist2 = np.abs(symbols[:, None] - self.constellation[None, :]) ** 2
        indices = np.argmin(dist2, axis=1).astype(np.uint8)  # (N,)
        # Unpack each index to bits_per_symbol bits (big-endian)
        all_bits = np.unpackbits(indices[:, None], axis=1, bitorder='big')  # (N, 8)
        return all_bits[:, 8 - k:].flatten()                                # (N*k,)

    def bits_to_indices(self, bits: np.ndarray) -> np.ndarray:
        """Convert a flat bit array to symbol indices (integers 0…M−1)."""
        bits = np.asarray(bits, dtype=np.int32)
        k = self.bits_per_symbol
        n = len(bits) // k
        matrix = bits[: n * k].reshape(n, k)
        powers = (2 ** np.arange(k - 1, -1, -1)).astype(np.int32)
        return matrix @ powers

    def indices_to_bits(self, indices: np.ndarray) -> np.ndarray:
        """Convert symbol indices to a flat bit array."""
        k = self.bits_per_symbol
        idx_u8 = np.asarray(indices, dtype=np.uint8)
        all_bits = np.unpackbits(idx_u8[:, None], axis=1, bitorder='big')
        return all_bits[:, 8 - k:].flatten()

    def get_constellation(self) -> np.ndarray:
        """Return the complex constellation points as a copy."""
        return self.constellation.copy()

    def __repr__(self) -> str:
        return (
            f"ConstellationMapper(modulation='{self.modulation}', "
            f"M={self.M}, bits_per_symbol={self.bits_per_symbol})"
        )
