"""
simulation.py — Master simulation runner.

SimulationConfig holds all parameters; SimulationRunner executes the
full Eb/N0 sweep and returns a pandas DataFrame.

The runner optionally saves results to CSV so they can be loaded and
re-plotted without re-running the simulation.
"""

import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from error_analysis import MonteCarloSimulator
from theoretical import (
    bpsk_ber_awgn, qpsk_ber_awgn, qam16_ber_awgn,
    bpsk_ber_rayleigh, shannon_capacity,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All parameters for a simulation run."""

    modulation: str = 'bpsk'
    """One of: 'bpsk', 'qpsk', 'qam16'."""

    channel: str = 'awgn'
    """One of: 'awgn', 'rayleigh', 'isi'."""

    n_bits: int = 100_000
    """Bits per Eb/N0 point."""

    EbN0_min: float = -4.0
    """Minimum Eb/N0 in dB."""

    EbN0_max: float = 16.0
    """Maximum Eb/N0 in dB."""

    EbN0_step: float = 1.0
    """Step size in dB."""

    seed: Optional[int] = None
    """Random seed (None → non-reproducible)."""

    frame_length: int = 1_000
    """Frame size (bits) for FER calculation."""

    isi_response: List[complex] = field(
        default_factory=lambda: [1.0, 0.5, 0.25]
    )
    """FIR coefficients for ISI channel (only used when channel='isi')."""

    output_dir: str = 'results'
    """Directory for CSV output and plots."""

    @property
    def EbN0_range(self) -> np.ndarray:
        n = round((self.EbN0_max - self.EbN0_min) / self.EbN0_step) + 1
        return np.linspace(self.EbN0_min, self.EbN0_max, n)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Execute a Monte Carlo BER sweep and collect results in a DataFrame.

    Parameters
    ----------
    config : SimulationConfig
    verbose : bool
        Print progress to stdout.
    """

    def __init__(self, config: SimulationConfig, verbose: bool = True):
        self.cfg = config
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run the full Eb/N0 sweep.

        Returns
        -------
        df : pd.DataFrame
            Columns: EbN0_dB, BER, SER, FER, CI_lower, CI_upper,
                     n_bit_errors, n_sym_errors, n_frame_errors,
                     n_bits, n_symbols, n_frames, noise_var,
                     BER_theoretical
        """
        cfg = self.cfg
        sim = MonteCarloSimulator(
            modulation=cfg.modulation,
            channel_type=cfg.channel,
            n_bits=cfg.n_bits,
            seed=cfg.seed,
            frame_length=cfg.frame_length,
            isi_response=cfg.isi_response,
        )

        EbN0_range = cfg.EbN0_range
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Simulation: {cfg.modulation.upper()} / {cfg.channel.upper()}")
            print(f"Eb/N0 range: {cfg.EbN0_min} to {cfg.EbN0_max} dB "
                  f"(step {cfg.EbN0_step} dB)")
            print(f"Bits per point: {cfg.n_bits:,}")
            if cfg.seed is not None:
                print(f"Seed: {cfg.seed}")
            print(f"{'='*60}")

        t0 = time.perf_counter()
        results = sim.run_sweep(EbN0_range)
        elapsed = time.perf_counter() - t0

        # Add theoretical BER column
        ber_theory = self._theoretical_ber(EbN0_range, cfg.modulation, cfg.channel)
        results['BER_theoretical'] = ber_theory
        results['modulation'] = cfg.modulation
        results['channel'] = cfg.channel

        df = pd.DataFrame(results)

        if self.verbose:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(df[['EbN0_dB', 'BER', 'BER_theoretical', 'CI_lower',
                       'CI_upper']].to_string(index=False))

        return df

    # ------------------------------------------------------------------
    # Theoretical BER helper
    # ------------------------------------------------------------------

    @staticmethod
    def _theoretical_ber(EbN0_range, modulation, channel):
        mod = modulation.lower()
        ch  = channel.lower()
        if ch == 'rayleigh':
            return bpsk_ber_rayleigh(EbN0_range)  # approx for all mods
        if mod == 'bpsk':
            return bpsk_ber_awgn(EbN0_range)
        elif mod == 'qpsk':
            return qpsk_ber_awgn(EbN0_range)
        elif mod == 'qam16':
            return qam16_ber_awgn(EbN0_range)
        return np.full_like(EbN0_range, np.nan)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the results DataFrame to CSV.

        Parameters
        ----------
        df : pd.DataFrame
        filename : str or None
            If None, a name is auto-generated from the config.

        Returns
        -------
        filepath : str
        """
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        if filename is None:
            filename = (f"{cfg.modulation}_{cfg.channel}"
                        f"_snr{cfg.EbN0_min}to{cfg.EbN0_max}"
                        f"_bits{cfg.n_bits}.csv")
        filepath = os.path.join(cfg.output_dir, filename)
        df.to_csv(filepath, index=False)
        if self.verbose:
            print(f"Results saved to: {filepath}")
        return filepath

    @staticmethod
    def load_results(filepath: str) -> pd.DataFrame:
        """Load a previously saved results CSV."""
        return pd.read_csv(filepath)
