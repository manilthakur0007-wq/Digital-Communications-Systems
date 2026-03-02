"""
compare_modulations.py — BER curves for BPSK, QPSK, and 16-QAM.

Plots simulated and theoretical BER curves on the same axes,
annotates the 3 dB gap between modulations, and saves the figure.

Run from the repository root:
    python analysis/compare_modulations.py
"""

import os
import sys

# Allow imports from src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from error_analysis import MonteCarloSimulator
from theoretical import bpsk_ber_awgn, qpsk_ber_awgn, qam16_ber_awgn
from visualizer import COLORS, save_figure


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EBN0_RANGE = np.arange(-4, 17, 1)   # dB
N_BITS     = 200_000
SEED       = 42
OUT_DIR    = os.path.join(_ROOT, 'results', 'plots')


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------

def run_all():
    results = {}
    for mod in ('bpsk', 'qpsk', 'qam16'):
        print(f"Simulating {mod.upper()} ...")
        sim = MonteCarloSimulator(mod, 'awgn', N_BITS, seed=SEED)
        r = sim.run_sweep(EBN0_RANGE)
        results[mod] = r
        print(f"  Done. Min BER = {min(r['BER'][r['BER']>0]):.2e}")
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    theory = {
        'bpsk':  bpsk_ber_awgn(EBN0_RANGE),
        'qpsk':  qpsk_ber_awgn(EBN0_RANGE),
        'qam16': qam16_ber_awgn(EBN0_RANGE),
    }

    labels = {
        'bpsk':  'BPSK',
        'qpsk':  'QPSK',
        'qam16': '16-QAM',
    }

    # ---------- theoretical ----------
    for mod, ber in theory.items():
        ax.semilogy(EBN0_RANGE, ber,
                    linestyle='--', linewidth=1.6,
                    color=COLORS[mod], alpha=0.85,
                    label=f'{labels[mod]} (Theory)')

    # ---------- simulated + CI ----------
    for mod, r in results.items():
        snr = r['EbN0_dB']
        ber = r['BER']
        mask = ber > 0
        ax.semilogy(snr[mask], ber[mask],
                    marker='o', markersize=4, linestyle='-',
                    color=COLORS[mod], linewidth=1.6,
                    label=f'{labels[mod]} (Sim)')
        ax.fill_between(snr[mask],
                        np.clip(r['CI_lower'][mask], 1e-10, None),
                        r['CI_upper'][mask],
                        alpha=0.15, color=COLORS[mod])

    # ---------- annotate 3 dB gap ----------
    # At BER = 1e-3, annotate gap between BPSK and QPSK (same!) and QPSK/16-QAM
    target_ber = 1e-3
    # BPSK/QPSK have same BER → use BPSK curve
    from scipy.interpolate import interp1d
    for (mod_a, mod_b, gap_note) in [('qpsk', 'qam16', '≈4 dB gap')]:
        ber_a = theory[mod_a]
        ber_b = theory[mod_b]
        mask_a = ber_a > 0
        mask_b = ber_b > 0
        try:
            f_a = interp1d(np.log10(ber_a[mask_a]), EBN0_RANGE[mask_a],
                           bounds_error=False, fill_value='extrapolate')
            f_b = interp1d(np.log10(ber_b[mask_b]), EBN0_RANGE[mask_b],
                           bounds_error=False, fill_value='extrapolate')
            snr_a = f_a(np.log10(target_ber))
            snr_b = f_b(np.log10(target_ber))
            ax.annotate(
                '',
                xy=(snr_b, target_ber), xytext=(snr_a, target_ber),
                arrowprops=dict(arrowstyle='<->', color='dimgray', lw=1.2),
            )
            ax.text((snr_a + snr_b) / 2, target_ber * 1.8,
                    gap_note, ha='center', fontsize=9, color='dimgray')
        except Exception:
            pass

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER Comparison: BPSK vs QPSK vs 16-QAM (AWGN)', fontsize=13)
    ax.set_xlim(-4, 16)
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    note = (
        'Dashed = theory, Solid = Monte Carlo simulation\n'
        'Shading = 95% Wilson confidence intervals'
    )
    ax.text(0.02, 0.03, note, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', color='gray')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    data = run_all()
    fig = plot(data)
    out = os.path.join(OUT_DIR, 'compare_modulations.png')
    save_figure(fig, out)
    print(f"\nFigure saved to {out}")
