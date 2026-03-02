"""
test_theoretical.py — Unit tests for analytical BER expressions.

Tests cover:
  - Q-function values at known input points.
  - BPSK BER formula at textbook-verified Eb/N0 values.
  - 16-QAM BER is higher than BPSK (for all realistic SNR).
  - Shannon limit curve is always below the achievable modulation rate.
  - BER → 0 as Eb/N0 → ∞ for all modulations.
  - BER → 0.5 as Eb/N0 → −∞.
  - Rayleigh BER decays slower (1/SNR) than AWGN.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from scipy.special import erfc

from theoretical import (
    q_function,
    bpsk_ber_awgn, qpsk_ber_awgn, qam16_ber_awgn,
    bpsk_ber_rayleigh,
    shannon_capacity, channel_capacity_vs_snr, ber_table,
)


# ---------------------------------------------------------------------------
# Q-function
# ---------------------------------------------------------------------------

class TestQFunction:
    def test_q_of_zero_is_half(self):
        assert abs(q_function(0) - 0.5) < 1e-12

    def test_q_of_positive_infinity(self):
        assert q_function(1e10) < 1e-100

    def test_q_of_negative_infinity(self):
        # Q(−∞) = 1
        assert abs(q_function(-1e10) - 1.0) < 1e-10

    def test_q_of_known_value(self):
        # Q(1.0) ≈ 0.15866
        assert abs(q_function(1.0) - 0.15866) < 1e-4

    def test_q_of_two(self):
        # Q(2.0) ≈ 0.02275
        assert abs(q_function(2.0) - 0.02275) < 1e-4

    def test_vectorised(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        qs = q_function(xs)
        assert qs.shape == (4,)
        assert np.all(qs > 0)
        assert np.all(np.diff(qs) < 0)   # strictly decreasing

    def test_symmetry(self):
        """Q(x) + Q(−x) = 1"""
        for x in [0.5, 1.0, 2.0, 3.5]:
            assert abs(q_function(x) + q_function(-x) - 1.0) < 1e-12

    def test_equivalence_to_erfc(self):
        xs = np.linspace(0, 5, 100)
        expected = 0.5 * erfc(xs / np.sqrt(2))
        np.testing.assert_allclose(q_function(xs), expected, atol=1e-15)


# ---------------------------------------------------------------------------
# BPSK BER (AWGN)
# ---------------------------------------------------------------------------

class TestBPSKBERAWGN:
    def test_ber_at_0dB(self):
        # Eb/N0 = 0 dB → lin = 1 → BER = Q(√2) ≈ 0.0786
        ber = bpsk_ber_awgn(0.0)
        assert abs(ber - q_function(np.sqrt(2))) < 1e-12

    def test_ber_at_10dB(self):
        # Eb/N0 = 10 dB → lin = 10 → BER = Q(√20) ≈ 3.87e-6
        ber = bpsk_ber_awgn(10.0)
        expected = q_function(np.sqrt(20.0))
        assert abs(ber - expected) / expected < 1e-9

    def test_ber_decreases_with_snr(self):
        snr = np.arange(-4, 15, 1, dtype=float)
        ber = bpsk_ber_awgn(snr)
        assert np.all(np.diff(ber) < 0)

    def test_ber_approaches_half_at_low_snr(self):
        ber = bpsk_ber_awgn(-100.0)
        assert abs(ber - 0.5) < 1e-3

    def test_ber_approaches_zero_at_high_snr(self):
        ber = bpsk_ber_awgn(30.0)
        assert ber < 1e-15

    def test_vectorised_input(self):
        snr = np.array([0.0, 5.0, 10.0])
        ber = bpsk_ber_awgn(snr)
        assert ber.shape == (3,)

    def test_textbook_value_at_8dB(self):
        # BER at 8 dB should be approximately 1.9e-4 (textbook)
        ber = float(bpsk_ber_awgn(8.0))
        assert 1e-5 < ber < 1e-3


# ---------------------------------------------------------------------------
# QPSK BER (AWGN) — same as BPSK per bit
# ---------------------------------------------------------------------------

class TestQPSKBERAWGN:
    def test_qpsk_equals_bpsk(self):
        snr = np.linspace(-4, 16, 50)
        np.testing.assert_allclose(
            qpsk_ber_awgn(snr), bpsk_ber_awgn(snr), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# 16-QAM BER (AWGN)
# ---------------------------------------------------------------------------

class TestQAM16BERAWGN:
    def test_qam16_ber_higher_than_bpsk_at_moderate_snr(self):
        snr = np.arange(0, 15, 1, dtype=float)
        assert np.all(qam16_ber_awgn(snr) >= bpsk_ber_awgn(snr))

    def test_ber_at_10dB(self):
        ber = float(qam16_ber_awgn(10.0))
        # Should be around 1e-3 at 10 dB
        assert 5e-4 < ber < 5e-3

    def test_ber_decreases_with_snr(self):
        snr = np.arange(0, 20, 1, dtype=float)
        ber = qam16_ber_awgn(snr)
        assert np.all(np.diff(ber) < 0)

    def test_ber_at_high_snr_is_tiny(self):
        ber = float(qam16_ber_awgn(20.0))
        assert ber < 1e-4

    def test_formula_value_at_known_point(self):
        # At Eb/N0 = 14 dB (≈25.1 linear):
        # BER = (3/8)*erfc(sqrt(2*25.1/5)) = (3/8)*erfc(sqrt(10.04))
        EbN0_lin = 10 ** (14 / 10)
        expected = (3 / 8) * erfc(np.sqrt(2 * EbN0_lin / 5))
        result = float(qam16_ber_awgn(14.0))
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Rayleigh fading BER
# ---------------------------------------------------------------------------

class TestRayleighBER:
    def test_rayleigh_higher_than_awgn(self):
        snr = np.arange(0, 20, 1, dtype=float)
        assert np.all(bpsk_ber_rayleigh(snr) >= bpsk_ber_awgn(snr))

    def test_at_zero_snr_approaches_half(self):
        ber = float(bpsk_ber_rayleigh(-100.0))
        assert abs(ber - 0.5) < 0.01

    def test_diversity_order_one_decay(self):
        """At high SNR, Rayleigh BER ≈ 1/(4·Eb/N0)."""
        snr_lin = np.array([100.0, 1000.0, 10000.0])
        snr_dB  = 10 * np.log10(snr_lin)
        ber     = bpsk_ber_rayleigh(snr_dB)
        asymptote = 1 / (4 * snr_lin)
        ratio = ber / asymptote
        # Ratio should be close to 1 at very high SNR
        assert np.all(np.abs(ratio - 1) < 0.1)


# ---------------------------------------------------------------------------
# Shannon capacity
# ---------------------------------------------------------------------------

class TestShannonCapacity:
    def test_capacity_locus_shape(self):
        ebno_dB, eta = shannon_capacity(None)
        assert len(ebno_dB) == len(eta)
        assert np.all(eta > 0)
        assert np.all(np.diff(ebno_dB))  # monotonically changing

    def test_shannon_limit_at_minus_1p59_dB(self):
        """As η → 0, required Eb/N0 → ln(2) ≈ −1.59 dB."""
        ebno_dB, eta = shannon_capacity(None)
        # At small η, Eb/N0 should approach −1.59 dB
        idx = np.argmin(np.abs(eta - 0.01))
        assert abs(ebno_dB[idx] - (-1.59)) < 0.1

    def test_min_required_ebno_increases_with_rate(self):
        ebno_dB, eta = shannon_capacity(None)
        # Eb/N0 requirement should increase monotonically with η
        assert np.all(np.diff(ebno_dB[eta > 0.1]) > 0)

    def test_capacity_vs_snr(self):
        snr_dB = np.array([0.0, 10.0, 20.0])
        C = channel_capacity_vs_snr(snr_dB)
        snr_lin = 10 ** (snr_dB / 10)
        expected = np.log2(1 + snr_lin)
        np.testing.assert_allclose(C, expected, rtol=1e-12)

    def test_capacity_positive_everywhere(self):
        C = channel_capacity_vs_snr(np.linspace(-10, 30, 1000))
        assert np.all(C > 0)

    def test_ber_table_structure(self):
        table = ber_table()
        assert isinstance(table, dict)
        assert ('bpsk', 10) in table
        assert 0 < table[('bpsk', 10)] < 1


# ---------------------------------------------------------------------------
# Cross-validation: simulated BER matches theory
# ---------------------------------------------------------------------------

class TestSimulatedVsTheoretical:
    """
    Quick sanity check: Monte Carlo BER at a single operating point
    must fall within the 95% confidence interval around the theoretical value.
    """

    @pytest.mark.parametrize('mod,EbN0_dB,theory_fn', [
        ('bpsk', 6.0, bpsk_ber_awgn),
        ('qpsk', 6.0, qpsk_ber_awgn),
    ])
    def test_sim_matches_theory(self, mod, EbN0_dB, theory_fn):
        from error_analysis import MonteCarloSimulator
        sim = MonteCarloSimulator(mod, 'awgn', n_bits=500_000, seed=0)
        result = sim.run_single(EbN0_dB)

        ber_sim    = result['BER']
        ber_theory = float(theory_fn(EbN0_dB))
        ci_lo      = result['CI_lower']
        ci_hi      = result['CI_upper']

        assert ci_lo <= ber_theory <= ci_hi, (
            f"{mod} at {EbN0_dB} dB: theory={ber_theory:.2e} "
            f"not in CI=[{ci_lo:.2e}, {ci_hi:.2e}], sim={ber_sim:.2e}"
        )
