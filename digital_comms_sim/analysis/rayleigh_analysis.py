"""
rayleigh_analysis.py — AWGN vs Rayleigh fading BER comparison.

Demonstrates:
  1. BER degradation under Rayleigh fading vs AWGN (BPSK).
  2. Diversity gain concept (1 vs 2-branch MRC, approximated).
  3. Fading channel coefficient realisations.

Run from the repository root:
    python analysis/rayleigh_analysis.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from error_analysis import MonteCarloSimulator
from theoretical import bpsk_ber_awgn, bpsk_ber_rayleigh
from channel import RayleighChannel
from modulator import ConstellationMapper
from visualizer import COLORS, save_figure

SEED    = 42
N_BITS  = 200_000
OUT_DIR = os.path.join(_ROOT, 'results', 'plots')
EBN0_RANGE = np.arange(-4, 21, 1)


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------

def run_simulations():
    results = {}

    print("Simulating BPSK/AWGN ...")
    sim_awgn = MonteCarloSimulator('bpsk', 'awgn', N_BITS, seed=SEED)
    results['awgn'] = sim_awgn.run_sweep(EBN0_RANGE)

    print("Simulating BPSK/Rayleigh ...")
    sim_ray = MonteCarloSimulator('bpsk', 'rayleigh', N_BITS, seed=SEED)
    results['rayleigh'] = sim_ray.run_sweep(EBN0_RANGE)

    return results


# ---------------------------------------------------------------------------
# 2-branch MRC diversity (analytical approximation)
# ---------------------------------------------------------------------------

def bpsk_mrc2_rayleigh(EbN0_dB):
    """
    Exact BER for 2-branch MRC with BPSK over Rayleigh fading:

        BER = [(1-μ)/2]² (1 + 2·(1+μ)/2) ... simplified to first order.

    Using the exact formula:
        BER = p² (3 - 2p) / 2  where p = (1 − √(ρ/(1+ρ))) / 2

    Reference: Proakis & Salehi, Eq 14-4-15.
    """
    EbN0_lin = 10 ** (np.asarray(EbN0_dB, dtype=float) / 10)
    mu = np.sqrt(EbN0_lin / (1 + EbN0_lin))
    p = (1 - mu) / 2
    return p ** 2 * (3 - 2 * p)


# ---------------------------------------------------------------------------
# Plot BER comparison
# ---------------------------------------------------------------------------

def plot_ber_comparison(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    # AWGN theoretical
    ax.semilogy(EBN0_RANGE, bpsk_ber_awgn(EBN0_RANGE),
                '--', color=COLORS['bpsk'], linewidth=1.8,
                label='BPSK/AWGN (Theory)')
    # AWGN simulated
    r = results['awgn']
    mask = r['BER'] > 0
    ax.semilogy(r['EbN0_dB'][mask], r['BER'][mask],
                'o-', color=COLORS['bpsk'], markersize=4, linewidth=1.5,
                label='BPSK/AWGN (Sim)')

    # Rayleigh theoretical
    ax.semilogy(EBN0_RANGE, bpsk_ber_rayleigh(EBN0_RANGE),
                '--', color=COLORS['rayleigh'], linewidth=1.8,
                label='BPSK/Rayleigh (Theory)')
    # Rayleigh simulated
    r2 = results['rayleigh']
    mask2 = r2['BER'] > 0
    ax.semilogy(r2['EbN0_dB'][mask2], r2['BER'][mask2],
                's-', color=COLORS['rayleigh'], markersize=4, linewidth=1.5,
                label='BPSK/Rayleigh (Sim)')
    ax.fill_between(r2['EbN0_dB'][mask2],
                    np.clip(r2['CI_lower'][mask2], 1e-10, None),
                    r2['CI_upper'][mask2],
                    alpha=0.15, color=COLORS['rayleigh'])

    # 2-branch MRC diversity
    ax.semilogy(EBN0_RANGE, bpsk_mrc2_rayleigh(EBN0_RANGE),
                ':', color='#7f7f7f', linewidth=2,
                label='BPSK/Rayleigh MRC-2 (Theory)')

    # Annotate SNR gap at BER=1e-3
    target_ber = 1e-3
    ber_awgn  = bpsk_ber_awgn(EBN0_RANGE)
    ber_ray   = bpsk_ber_rayleigh(EBN0_RANGE)
    try:
        from scipy.interpolate import interp1d
        f_awgn = interp1d(np.log10(ber_awgn + 1e-15), EBN0_RANGE,
                          bounds_error=False, fill_value='extrapolate')
        f_ray  = interp1d(np.log10(ber_ray  + 1e-15), EBN0_RANGE,
                          bounds_error=False, fill_value='extrapolate')
        snr_awgn = float(f_awgn(np.log10(target_ber)))
        snr_ray  = float(f_ray(np.log10(target_ber)))
        gap = snr_ray - snr_awgn
        ax.annotate(
            f'≈{gap:.1f} dB\nfading penalty',
            xy=((snr_awgn + snr_ray) / 2, target_ber),
            xytext=((snr_awgn + snr_ray) / 2, target_ber * 5),
            ha='center', fontsize=9, color='dimgray',
            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8),
        )
    except Exception:
        pass

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title('BPSK: AWGN vs Rayleigh Fading (with MRC-2 Diversity)', fontsize=13)
    ax.set_xlim(-4, 20)
    ax.set_ylim(1e-5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot fading realisations
# ---------------------------------------------------------------------------

def plot_fading_realisations():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    rng = np.random.default_rng(SEED)
    n = 500
    for (ax, EbN0_dB, label) in [
        (axes[0], 0,  'Eb/N0 = 0 dB'),
        (axes[1], 10, 'Eb/N0 = 10 dB'),
    ]:
        # Rayleigh channel coefficients
        h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        h_dB = 20 * np.log10(np.abs(h) + 1e-20)
        ax.plot(np.arange(n), h_dB, color='steelblue', linewidth=0.8)
        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.fill_between(np.arange(n), h_dB, 0,
                        where=(h_dB < 0), alpha=0.3, color='red',
                        label='Deep fades (|h|<1)')
        ax.set_ylabel('|h| (dB)', fontsize=10)
        ax.set_title(f'Rayleigh Fading Realisations — {label}', fontsize=11)
        ax.set_ylim(-25, 15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Symbol Index', fontsize=10)
    fig.suptitle('Rayleigh Fading Channel Realisations', fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    data = run_simulations()

    fig1 = plot_ber_comparison(data)
    out1 = os.path.join(OUT_DIR, 'rayleigh_vs_awgn_ber.png')
    save_figure(fig1, out1)
    print(f"BER comparison saved to {out1}")

    fig2 = plot_fading_realisations()
    out2 = os.path.join(OUT_DIR, 'fading_realisations.png')
    save_figure(fig2, out2)
    print(f"Fading realisations saved to {out2}")
