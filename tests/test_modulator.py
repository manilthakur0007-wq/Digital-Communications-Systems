"""
test_modulator.py — Unit tests for the ConstellationMapper.

Tests cover:
  - Correct symbol mapping for all modulations.
  - Gray-coding property (adjacent symbols differ by 1 bit).
  - Unit average energy normalisation.
  - Round-trip (modulate → demodulate at zero noise → zero BER).
  - bits_to_indices / indices_to_bits consistency.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from modulator import ConstellationMapper, binary_to_gray, gray_to_binary


# ---------------------------------------------------------------------------
# Gray-code helpers
# ---------------------------------------------------------------------------

class TestGrayHelpers:
    def test_binary_to_gray_zero(self):
        assert binary_to_gray(0) == 0

    def test_binary_to_gray_known(self):
        # Standard 4-bit Gray codes
        expected = {0: 0, 1: 1, 2: 3, 3: 2, 4: 6, 5: 7, 6: 5, 7: 4}
        for b, g in expected.items():
            assert binary_to_gray(b) == g

    def test_gray_to_binary_roundtrip(self):
        for n in range(64):
            assert gray_to_binary(binary_to_gray(n)) == n


# ---------------------------------------------------------------------------
# ConstellationMapper construction
# ---------------------------------------------------------------------------

class TestMapperConstruction:
    def test_unsupported_modulation_raises(self):
        with pytest.raises(ValueError):
            ConstellationMapper('fsk')

    def test_bpsk_bits_per_symbol(self):
        assert ConstellationMapper('bpsk').bits_per_symbol == 1

    def test_qpsk_bits_per_symbol(self):
        assert ConstellationMapper('qpsk').bits_per_symbol == 2

    def test_qam16_bits_per_symbol(self):
        assert ConstellationMapper('qam16').bits_per_symbol == 4

    def test_constellation_size(self):
        for mod, M in [('bpsk', 2), ('qpsk', 4), ('qam16', 16)]:
            assert ConstellationMapper(mod).M == M


# ---------------------------------------------------------------------------
# Energy normalisation
# ---------------------------------------------------------------------------

class TestEnergyNorm:
    @pytest.mark.parametrize('mod', ['bpsk', 'qpsk', 'qam16'])
    def test_unit_average_energy(self, mod):
        mapper = ConstellationMapper(mod)
        pts = mapper.get_constellation()
        energy = np.mean(np.abs(pts) ** 2)
        assert abs(energy - 1.0) < 1e-9, (
            f"{mod}: average energy = {energy}, expected 1.0"
        )

    def test_bpsk_symbols_on_unit_circle(self):
        pts = ConstellationMapper('bpsk').get_constellation()
        for p in pts:
            assert abs(abs(p) - 1.0) < 1e-9

    def test_qpsk_symbols_on_unit_circle(self):
        pts = ConstellationMapper('qpsk').get_constellation()
        for p in pts:
            assert abs(abs(p) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# BPSK mapping
# ---------------------------------------------------------------------------

class TestBPSKMapping:
    def setup_method(self):
        self.mapper = ConstellationMapper('bpsk')

    def test_bit_0_maps_to_negative_one(self):
        syms = self.mapper.map_bits(np.array([0]))
        assert np.isclose(syms[0].real, -1.0)

    def test_bit_1_maps_to_positive_one(self):
        syms = self.mapper.map_bits(np.array([1]))
        assert np.isclose(syms[0].real, 1.0)

    def test_length_preserved(self):
        bits = np.random.randint(0, 2, 100)
        syms = self.mapper.map_bits(bits)
        assert len(syms) == 100


# ---------------------------------------------------------------------------
# QPSK Gray coding
# ---------------------------------------------------------------------------

class TestQPSKGrayCoding:
    def setup_method(self):
        self.mapper = ConstellationMapper('qpsk')

    def _hamming(self, a, b):
        """Hamming distance between two 2-bit indices."""
        return bin(a ^ b).count('1')

    def test_all_four_symbols_distinct(self):
        pts = self.mapper.get_constellation()
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.isclose(pts[i], pts[j])

    def test_adjacent_qpsk_symbols_differ_by_one_bit(self):
        """
        For QPSK, 'adjacent' means 90° apart.
        Indices that differ by exactly 1 bit should be 90° apart.
        """
        pts = self.mapper.get_constellation()
        for i in range(4):
            for j in range(4):
                if self._hamming(i, j) == 1:
                    # They should be adjacent quadrants (90° apart)
                    angle_diff = abs(np.angle(pts[i]) - np.angle(pts[j]))
                    # Normalise to [0, π]
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    assert abs(angle_diff - np.pi / 2) < 0.1, (
                        f"Indices {i:02b} and {j:02b} differ by 1 bit but "
                        f"angle diff = {np.degrees(angle_diff):.1f}°"
                    )


# ---------------------------------------------------------------------------
# 16-QAM Gray coding
# ---------------------------------------------------------------------------

class TestQAM16GrayCoding:
    def setup_method(self):
        self.mapper = ConstellationMapper('qam16')

    def test_all_sixteen_symbols_distinct(self):
        pts = self.mapper.get_constellation()
        for i in range(16):
            for j in range(i + 1, 16):
                assert not np.isclose(pts[i], pts[j])

    def test_nearest_neighbors_differ_by_one_bit(self):
        """
        Gray-coding property: every nearest-neighbour pair in the
        constellation must differ by exactly one bit.

        Note: the converse (1-bit-apart => nearest-neighbour) does NOT
        hold for 16-QAM.  E.g. bits 0000 and 0010 differ by 1 bit
        (Q Gray-code 00<->10 spans -3 to +3) but are not nearest neighbours.
        """
        pts = self.mapper.get_constellation()
        # Global minimum distance between any two distinct points
        all_pairs = [
            np.abs(pts[i] - pts[j])
            for i in range(16) for j in range(i + 1, 16)
        ]
        min_dist = np.min(all_pairs)
        for i in range(16):
            for j in range(16):
                if i == j:
                    continue
                d = np.abs(pts[i] - pts[j])
                # Is j a nearest neighbour of i?
                if abs(d - min_dist) < 1e-9:
                    ham = bin(i ^ j).count('1')
                    assert ham == 1, (
                        f"Nearest-neighbour pair ({i:04b}, {j:04b}) "
                        f"has Hamming distance {ham}, expected 1. "
                        f"|d| = {d:.4f}, min_dist = {min_dist:.4f}"
                    )


# ---------------------------------------------------------------------------
# Round-trip (zero noise)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize('mod', ['bpsk', 'qpsk', 'qam16'])
    def test_zero_ber_at_zero_noise(self, mod):
        rng = np.random.default_rng(0)
        mapper = ConstellationMapper(mod)
        k = mapper.bits_per_symbol
        bits = rng.integers(0, 2, 10_000 * k, dtype=np.int32)
        symbols = mapper.map_bits(bits)
        rx_bits = mapper.demap_symbols(symbols)
        n_errors = np.sum(bits != rx_bits)
        assert n_errors == 0, f"{mod}: {n_errors} errors with zero noise"


# ---------------------------------------------------------------------------
# Index conversion helpers
# ---------------------------------------------------------------------------

class TestIndexConversion:
    @pytest.mark.parametrize('mod', ['bpsk', 'qpsk', 'qam16'])
    def test_bits_to_indices_roundtrip(self, mod):
        rng = np.random.default_rng(7)
        mapper = ConstellationMapper(mod)
        k = mapper.bits_per_symbol
        bits = rng.integers(0, 2, 1000 * k, dtype=np.int32)
        indices = mapper.bits_to_indices(bits)
        recovered = mapper.indices_to_bits(indices)
        assert np.all(bits == recovered)

    def test_bpsk_indices_are_0_and_1(self):
        mapper = ConstellationMapper('bpsk')
        bits = np.array([0, 1, 0, 0, 1, 1], dtype=np.int32)
        idx = mapper.bits_to_indices(bits)
        assert set(idx).issubset({0, 1})

    def test_qam16_indices_in_range_0_to_15(self):
        rng = np.random.default_rng(0)
        mapper = ConstellationMapper('qam16')
        bits = rng.integers(0, 2, 400, dtype=np.int32)
        idx = mapper.bits_to_indices(bits)
        assert np.all((idx >= 0) & (idx <= 15))
