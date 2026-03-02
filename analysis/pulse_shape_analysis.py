"""
pulse_shape_analysis.py — Effect of RRC pulse shaping.

Demonstrates:
  1. Eye diagrams for α = 0.25, 0.5, 0.99 (with RRC filtering).
  2. PSD with and without pulse shaping.
  3. ISI reduction (eye opening measure).

Run from the repository root:
    python analysis/pulse_shape_analysis.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from modulator import ConstellationMapper
from pulse_shaping import RRCFilter, eye_diagram_data
from visualizer import save_figure

SEED    = 42
OUT_DIR = os.path.join(_ROOT, 'results', 'plots')
N_SYM   = 1000
SPS     = 8


# ---------------------------------------------------------------------------
# Helper: generate BPSK symbols
# ---------------------------------------------------------------------------

def generate_symbols(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mapper = ConstellationMapper('bpsk')
    bits = rng.integers(0, 2, n)
    return mapper.map_bits(bits).astype(complex)


# ---------------------------------------------------------------------------
# Eye diagrams for different roll-off factors
# ---------------------------------------------------------------------------

def plot_eye_diagrams():
    alphas = [0.25, 0.50, 0.99]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    symbols = generate_symbols(N_SYM)

    for ax, alpha in zip(axes, alphas):
        rrc = RRCFilter(alpha=alpha, span=8, sps=SPS)
        tx_wave = rrc.transmit(symbols)
        # Apply receive filter too (matched filter)
        rx_wave = rrc.receive(tx_wave)
        # Build eye from receive-filtered waveform
        eye = eye_diagram_data(np.real(rx_wave), SPS, n_traces=150)

        t = np.linspace(-1, 1, eye.shape[1])
        for trace in eye:
            ax.plot(t, trace, color='royalblue', alpha=0.1, linewidth=0.5)

        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(f'α = {alpha}', fontsize=12)
        ax.set_xlabel('Time / Ts', fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('Amplitude', fontsize=10)
    fig.suptitle('Eye Diagrams: RRC Matched-Filter (BPSK)', fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PSD with / without pulse shaping
# ---------------------------------------------------------------------------

def plot_psd_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    symbols = generate_symbols(N_SYM)

    alphas = [0.25, 0.50, 0.99]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # ---- Left: No pulse shaping (rect) ----
    ax = axes[0]
    # Upsampled NRZ pulse (rectangular)
    up_rect = np.repeat(symbols.real, SPS)
    f_rect, P_rect = welch(up_rect, fs=SPS, nperseg=256, return_onesided=False)
    idx = np.argsort(f_rect)
    ax.plot(f_rect[idx], 10 * np.log10(np.abs(P_rect[idx]) + 1e-15),
            color='gray', linewidth=1.5, label='No shaping (Rect pulse)')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-50, 20)
    ax.set_xlabel('Normalised Frequency (cycles/sample)', fontsize=10)
    ax.set_ylabel('PSD (dB/Hz)', fontsize=10)
    ax.set_title('Without Pulse Shaping', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Right: RRC filtered ----
    ax2 = axes[1]
    for alpha, color in zip(alphas, colors):
        rrc = RRCFilter(alpha=alpha, span=8, sps=SPS)
        tx_wave = rrc.transmit(symbols)
        f, P = welch(tx_wave, fs=SPS, nperseg=256, return_onesided=False)
        idx = np.argsort(f)
        ax2.plot(f[idx], 10 * np.log10(np.abs(P[idx]) + 1e-15),
                 color=color, linewidth=1.5, label=f'RRC α={alpha}')
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-50, 20)
    ax2.set_xlabel('Normalised Frequency (cycles/sample)', fontsize=10)
    ax2.set_title('With RRC Pulse Shaping', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Power Spectral Density: Effect of RRC Filtering', fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ISI quantification: eye opening
# ---------------------------------------------------------------------------

def quantify_isi():
    """Print eye-opening metric for each roll-off factor."""
    print("\nEye Opening Metric (higher = less ISI):")
    print(f"{'Alpha':>8}  {'Eye Opening':>14}")
    print("-" * 26)
    symbols = generate_symbols(N_SYM)
    for alpha in [0.25, 0.50, 0.99]:
        rrc = RRCFilter(alpha=alpha, span=8, sps=SPS)
        tx_wave = rrc.transmit(symbols)
        rx_wave = rrc.receive(tx_wave)
        eye = eye_diagram_data(np.real(rx_wave), SPS, n_traces=200)
        # Eye opening: min(max) - max(min) at centre sample
        centre = eye.shape[1] // 2
        eye_open = np.min(eye[:, centre][eye[:, centre] > 0]) - \
                   np.max(eye[:, centre][eye[:, centre] < 0]) \
                   if np.any(eye[:, centre] > 0) and np.any(eye[:, centre] < 0) \
                   else np.max(eye[:, centre]) - np.min(eye[:, centre])
        print(f"{alpha:>8.2f}  {eye_open:>14.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    fig1 = plot_eye_diagrams()
    out1 = os.path.join(OUT_DIR, 'eye_diagrams.png')
    save_figure(fig1, out1)
    print(f"Eye diagrams saved to {out1}")

    fig2 = plot_psd_comparison()
    out2 = os.path.join(OUT_DIR, 'psd_comparison.png')
    save_figure(fig2, out2)
    print(f"PSD comparison saved to {out2}")

    quantify_isi()
