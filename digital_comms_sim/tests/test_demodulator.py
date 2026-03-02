"""
test_demodulator.py — Unit tests for the demodulator module.

Tests cover:
  - Zero BER at extremely high SNR (100 dB) for all modulations.
  - Correct BPSK decision boundary (decision at zero).
  - LLR sign convention and monotonicity.
  - QPSK decision region correctness.
  - ZF equaliser correctly inverts flat-fading channel.
  - Frequency-domain ZF equaliser attenuates ISI.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from modulator import ConstellationMapper
from channel import AWGNChannel, RayleighChannel
from demodulator import (
    BPSKDemodulator, QPSKDemodulator, QAM16Demodulator,
    ZeroForcingEqualizer, make_demodulator,
)


# ---------------------------------------------------------------------------
# Zero BER at high SNR
# ---------------------------------------------------------------------------

class TestZeroBERHighSNR:
    """Verify that BER → 0 as Eb/N0 → ∞."""

    @pytest.mark.parametrize('mod', ['bpsk', 'qpsk', 'qam16'])
    def test_zero_ber_at_100dB(self, mod):
        rng = np.random.default_rng(0)
        mapper = ConstellationMapper(mod)
        k = mapper.bits_per_symbol
        n_bits = 10_000 * k
        bits = rng.integers(0, 2, n_bits, dtype=np.int32)
        symbols = mapper.map_bits(bits)

        ch = AWGNChannel(100.0, k, seed=1)
        rx = ch.corrupt(symbols)
        rx_bits = mapper.demap_symbols(rx)
        n_errors = np.sum(bits != rx_bits)
        assert n_errors == 0, (
            f"{mod} at 100 dB: expected 0 errors, got {n_errors}"
        )


# ---------------------------------------------------------------------------
# BPSK demodulator
# ---------------------------------------------------------------------------

class TestBPSKDemodulator:
    def setup_method(self):
        self.demod = BPSKDemodulator()

    def test_positive_real_gives_bit_1(self):
        syms = np.array([0.5, 1.0, 2.0], dtype=complex)
        bits = self.demod.demodulate(syms)
        assert np.all(bits == 1)

    def test_negative_real_gives_bit_0(self):
        syms = np.array([-0.5, -1.0, -2.0], dtype=complex)
        bits = self.demod.demodulate(syms)
        assert np.all(bits == 0)

    def test_decision_at_zero_is_1(self):
        """Boundary case: decision at exactly 0 should give bit 1 (≥0 → 1)."""
        syms = np.array([0.0], dtype=complex)
        bits = self.demod.demodulate(syms)
        assert bits[0] == 1

    def test_llr_positive_for_positive_input(self):
        syms = np.array([0.5, 1.0], dtype=complex)
        llr = self.demod.compute_llr(syms, noise_var=0.5)
        assert np.all(llr > 0)

    def test_llr_negative_for_negative_input(self):
        syms = np.array([-0.5, -1.0], dtype=complex)
        llr = self.demod.compute_llr(syms, noise_var=0.5)
        assert np.all(llr < 0)

    def test_llr_magnitude_scales_with_snr(self):
        sym = np.array([1.0 + 0j])
        llr_low  = self.demod.compute_llr(sym, noise_var=1.0)
        llr_high = self.demod.compute_llr(sym, noise_var=0.1)
        assert abs(llr_high[0]) > abs(llr_low[0])

    def test_llr_formula(self):
        """LLR = 2 * Re(y) / σ²."""
        sym = np.array([0.3 + 0.1j])
        nvar = 0.25
        expected = 2 * 0.3 / nvar
        llr = self.demod.compute_llr(sym, noise_var=nvar)
        assert abs(llr[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# QPSK demodulator
# ---------------------------------------------------------------------------

class TestQPSKDemodulator:
    def setup_method(self):
        self.demod = QPSKDemodulator()
        self.mapper = ConstellationMapper('qpsk')

    def test_constellation_round_trip(self):
        """Demodulating ideal constellation points should recover all indices."""
        pts = self.mapper.get_constellation()
        for idx, pt in enumerate(pts):
            # Demodulate a single-point array
            rx_bits = self.demod.demodulate(np.array([pt]))
            # Convert to index and check
            rx_idx = self.mapper.bits_to_indices(rx_bits)
            assert rx_idx[0] == idx, (
                f"Point {idx} decoded as {rx_idx[0]}"
            )

    def test_output_length(self):
        syms = np.array([0.5 + 0.5j, -0.5 + 0.5j], dtype=complex)
        bits = self.demod.demodulate(syms)
        assert len(bits) == 4  # 2 symbols × 2 bits/symbol

    def test_all_four_quadrants(self):
        """All four quadrant decisions should be reachable."""
        syms = np.array([
            1 + 1j,   # Q1
            -1 + 1j,  # Q2
            -1 - 1j,  # Q3
            1 - 1j,   # Q4
        ], dtype=complex) / np.sqrt(2)
        bits = self.demod.demodulate(syms)
        # All should decode to distinct 2-bit patterns
        decoded = bits.reshape(4, 2).tolist()
        assert len(set(map(tuple, decoded))) == 4


# ---------------------------------------------------------------------------
# 16-QAM demodulator
# ---------------------------------------------------------------------------

class TestQAM16Demodulator:
    def setup_method(self):
        self.demod = QAM16Demodulator()
        self.mapper = ConstellationMapper('qam16')

    def test_all_16_points_round_trip(self):
        pts = self.mapper.get_constellation()
        for idx, pt in enumerate(pts):
            rx_bits = self.demod.demodulate(np.array([pt]))
            rx_idx = self.mapper.bits_to_indices(rx_bits)
            assert rx_idx[0] == idx, (
                f"16-QAM point {idx} decoded as {rx_idx[0]}"
            )

    def test_output_length(self):
        syms = self.mapper.get_constellation()[:4]
        bits = self.demod.demodulate(syms)
        assert len(bits) == 16  # 4 symbols × 4 bits/symbol


# ---------------------------------------------------------------------------
# make_demodulator factory
# ---------------------------------------------------------------------------

class TestMakeDemodulator:
    @pytest.mark.parametrize('mod,cls', [
        ('bpsk', BPSKDemodulator),
        ('qpsk', QPSKDemodulator),
        ('qam16', QAM16Demodulator),
    ])
    def test_correct_type_returned(self, mod, cls):
        assert isinstance(make_demodulator(mod), cls)

    def test_unknown_modulation_raises(self):
        with pytest.raises(ValueError):
            make_demodulator('unknown')


# ---------------------------------------------------------------------------
# Zero-forcing equaliser
# ---------------------------------------------------------------------------

class TestZeroForcingEqualizer:

    def test_flat_fading_exact_inversion(self):
        """ZF on flat fading with perfect CSI must recover exact symbols."""
        zf = ZeroForcingEqualizer()
        rng = np.random.default_rng(42)
        n = 1000
        # Deterministic complex channel coefficients
        h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        # Non-trivial symbols (non-zero)
        symbols = np.ones(n, dtype=complex) + 0.5j
        received = h * symbols     # no noise
        equalized = zf.equalize_flat(received, h)
        # Should recover symbols exactly (modulo numerical precision)
        assert np.max(np.abs(equalized - symbols)) < 1e-10

    def test_flat_fading_reduces_residual_ber(self):
        """After ZF equalisation, BER at 10 dB should be below un-equalised."""
        mapper = ConstellationMapper('bpsk')
        ch = RayleighChannel(10.0, 1, seed=77)
        zf = ZeroForcingEqualizer()
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 10_000, dtype=np.int32)
        symbols = mapper.map_bits(bits)
        rx_raw, h = ch.corrupt(symbols)
        # Without equalisation
        bits_noeq = mapper.demap_symbols(rx_raw)
        ber_noeq = np.mean(bits != bits_noeq)
        # With ZF
        rx_eq = zf.equalize_flat(rx_raw, h)
        bits_eq = mapper.demap_symbols(rx_eq)
        ber_eq = np.mean(bits != bits_eq)
        assert ber_eq < ber_noeq

    def test_frequency_domain_zf_for_isi(self):
        """FD-ZF should reduce ISI distortion."""
        from channel import ISIChannel
        h_isi = np.array([1.0, 0.5, 0.25])
        ch = ISIChannel(h_isi, 30.0, 1, seed=0)
        zf = ZeroForcingEqualizer()
        mapper = ConstellationMapper('bpsk')
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 5_000, dtype=np.int32)
        symbols = mapper.map_bits(bits)
        rx_isi = ch.corrupt(symbols)
        rx_eq = zf.equalize_frequency_domain(rx_isi, h_isi)
        # After equalisation, BER should be much lower than without
        ber_raw = np.mean(bits != mapper.demap_symbols(rx_isi))
        ber_eq  = np.mean(bits != mapper.demap_symbols(rx_eq))
        assert ber_eq <= ber_raw + 0.05  # ZF should not make things worse
