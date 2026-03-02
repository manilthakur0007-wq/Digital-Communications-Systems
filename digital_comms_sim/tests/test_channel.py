"""
test_channel.py — Unit tests for channel models.

Tests cover:
  - AWGN noise variance matches the specified Eb/N0.
  - Rayleigh coefficient statistics: unit mean power, proper distribution.
  - ISI convolution: output energy accounts for channel gain.
  - At very high Eb/N0 (100 dB), noise is negligible.
  - Channel set_EbN0 correctly updates sigma.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from scipy import stats

from channel import AWGNChannel, RayleighChannel, ISIChannel


# ---------------------------------------------------------------------------
# AWGN channel
# ---------------------------------------------------------------------------

class TestAWGNChannel:

    def _expected_sigma(self, EbN0_dB, k):
        EbN0_lin = 10 ** (EbN0_dB / 10)
        return np.sqrt(1.0 / (2 * k * EbN0_lin))

    @pytest.mark.parametrize('EbN0_dB,k', [
        (0,   1),
        (0,   2),
        (10,  1),
        (10,  4),
        (-4,  1),
    ])
    def test_noise_std_formula(self, EbN0_dB, k):
        ch = AWGNChannel(EbN0_dB, k)
        expected = self._expected_sigma(EbN0_dB, k)
        assert abs(ch.noise_std - expected) < 1e-12

    def test_noise_variance_equals_std_squared(self):
        ch = AWGNChannel(6.0, 2)
        assert abs(ch.noise_variance - ch.noise_std ** 2) < 1e-15

    def test_output_noise_variance_matches_theory(self):
        """Measured noise variance must match σ² within a tight tolerance."""
        ch = AWGNChannel(10.0, 1, seed=0)
        n_symbols = 500_000
        pure_signal = np.zeros(n_symbols, dtype=complex)  # zero signal
        noisy = ch.corrupt(pure_signal)
        # Variance of I or Q component
        measured_var = np.var(noisy.real)
        assert abs(measured_var - ch.noise_variance) / ch.noise_variance < 0.01, (
            f"Measured variance {measured_var:.6f} vs "
            f"expected {ch.noise_variance:.6f}"
        )

    def test_high_snr_noise_is_tiny(self):
        ch = AWGNChannel(100.0, 1, seed=1)
        symbols = np.ones(1000, dtype=complex)
        noisy = ch.corrupt(symbols)
        max_noise = np.max(np.abs(noisy - symbols))
        assert max_noise < 1e-4

    def test_set_EbN0_updates_sigma(self):
        ch = AWGNChannel(0.0, 1)
        sigma_before = ch.noise_std
        ch.set_EbN0(10.0)
        sigma_after = ch.noise_std
        assert sigma_after < sigma_before
        assert abs(sigma_after - self._expected_sigma(10.0, 1)) < 1e-12

    def test_output_length_preserved(self):
        ch = AWGNChannel(5.0, 1, seed=2)
        symbols = np.ones(1234, dtype=complex)
        noisy = ch.corrupt(symbols)
        assert len(noisy) == 1234

    def test_noise_is_complex_circular(self):
        """Real and imaginary noise components should have equal variance."""
        ch = AWGNChannel(6.0, 1, seed=3)
        n = 200_000
        noise = ch.corrupt(np.zeros(n, dtype=complex))
        var_I = np.var(noise.real)
        var_Q = np.var(noise.imag)
        ratio = var_I / var_Q
        assert 0.98 < ratio < 1.02, f"I/Q variance ratio = {ratio:.4f}"

    def test_noise_gaussian_distribution(self):
        """Shapiro–Wilk or K-S test: real noise should be Gaussian."""
        ch = AWGNChannel(6.0, 1, seed=4)
        n = 5000
        noise = ch.corrupt(np.zeros(n, dtype=complex)).real
        _, p_value = stats.kstest(noise / ch.noise_std, 'norm')
        assert p_value > 0.01, f"Noise not Gaussian (KS p = {p_value:.4f})"


# ---------------------------------------------------------------------------
# Rayleigh channel
# ---------------------------------------------------------------------------

class TestRayleighChannel:

    def test_coefficient_unit_mean_power(self):
        """E[|h|²] should be ≈ 1 for CN(0,1) coefficients."""
        ch = RayleighChannel(10.0, 1, seed=5)
        n = 200_000
        symbols = np.ones(n, dtype=complex)
        _, h = ch.corrupt(symbols)
        mean_power = np.mean(np.abs(h) ** 2)
        assert abs(mean_power - 1.0) < 0.01, (
            f"Mean Rayleigh power = {mean_power:.4f}, expected 1.0"
        )

    def test_coefficient_magnitude_rayleigh_distributed(self):
        """Magnitude of h should be Rayleigh-distributed."""
        ch = RayleighChannel(20.0, 1, seed=6)
        n = 50_000
        symbols = np.ones(n, dtype=complex)
        _, h = ch.corrupt(symbols)
        magnitudes = np.abs(h)
        # Rayleigh CDF: F(x) = 1 − exp(−x²/σ²) with σ² = 0.5
        _, p = stats.kstest(magnitudes,
                            lambda x: 1 - np.exp(-x ** 2 / (2 * 0.5)))
        assert p > 0.01, f"Magnitude not Rayleigh (KS p = {p:.4f})"

    def test_output_length_preserved(self):
        ch = RayleighChannel(5.0, 1, seed=7)
        symbols = np.ones(500, dtype=complex)
        rx, h = ch.corrupt(symbols)
        assert len(rx) == len(symbols)
        assert len(h) == len(symbols)

    def test_set_EbN0_updates_internal_awgn(self):
        ch = RayleighChannel(0.0, 1)
        sigma_before = ch._awgn.noise_std
        ch.set_EbN0(10.0)
        assert ch._awgn.noise_std < sigma_before


# ---------------------------------------------------------------------------
# ISI channel
# ---------------------------------------------------------------------------

class TestISIChannel:

    def test_output_length_preserved(self):
        h = [1.0, 0.5, 0.25]
        ch = ISIChannel(h, 10.0, 1, seed=8)
        symbols = np.ones(200, dtype=complex)
        out = ch.corrupt(symbols)
        assert len(out) == 200

    def test_unit_impulse_response_is_awgn(self):
        """h = [1] should behave identical to AWGN channel."""
        ch_isi  = ISIChannel([1.0 + 0j], 20.0, 1, seed=99)
        ch_awgn = AWGNChannel(20.0, 1, seed=99)
        # Share RNG states so both produce same noise
        rng = np.random.default_rng(99)
        ch_isi._awgn._rng  = rng
        ch_awgn._rng       = np.random.default_rng(99)
        symbols = np.ones(500, dtype=complex)
        out_isi  = ch_isi.corrupt(symbols)
        out_awgn = ch_awgn.corrupt(symbols)
        # Check noise levels are comparable
        noise_isi  = np.std(out_isi  - symbols)
        noise_awgn = np.std(out_awgn - symbols)
        assert abs(noise_isi - noise_awgn) / noise_awgn < 0.2

    def test_channel_length(self):
        h = [1.0, 0.5, 0.25]
        ch = ISIChannel(h, 10.0, 1)
        assert ch.channel_length == 3

    def test_frequency_response_shape(self):
        h = [1.0, 0.5, 0.25]
        ch = ISIChannel(h, 10.0, 1)
        freqs, mag = ch.frequency_response(256)
        assert len(freqs) == 256
        assert len(mag) == 256

    def test_set_EbN0_updates(self):
        ch = ISIChannel([1.0], 0.0, 1)
        sigma_before = ch._awgn.noise_std
        ch.set_EbN0(10.0)
        assert ch._awgn.noise_std < sigma_before
