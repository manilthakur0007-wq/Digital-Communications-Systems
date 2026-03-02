"""
capacity_analysis.py — Shannon capacity vs achievable rates.

Plots the Shannon capacity limit curve and overlays the achievable
spectral efficiencies for BPSK, QPSK, and 16-QAM, showing the gap
to the Shannon limit.

Run from the repository root:
    python analysis/capacity_analysis.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from theoretical import shannon_capacity, bpsk_ber_awgn, qpsk_ber_awgn, qam16_ber_awgn
from visualizer import COLORS, save_figure

OUT_DIR = os.path.join(_ROOT, 'results', 'plots')


# ---------------------------------------------------------------------------
# Helper: minimum Eb/N0 for BER = 1e-5 per modulation
# ---------------------------------------------------------------------------

def required_snr(ber_fn, target_ber=1e-5,
                 snr_range=np.linspace(-4, 30, 10000)):
    """Find the Eb/N0 (dB) where theoretical BER first drops below target."""
    ber = ber_fn(snr_range)
    idx = np.where(ber <= target_ber)[0]
    if len(idx) == 0:
        return None
    return snr_range[idx[0]]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_capacity_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel 1: Shannon limit curve ----
    ax = axes[0]
    ebno_sh, eta_sh = shannon_capacity(None)

    ax.plot(ebno_sh, eta_sh, color=COLORS['shannon'], linewidth=2.5,
            label='Shannon Limit C/B = log₂(1 + η·Eb/N0)', zorder=5)
    ax.fill_betweenx(eta_sh, ebno_sh, -2 * np.ones_like(eta_sh),
                     alpha=0.08, color=COLORS['shannon'])

    # Modulation operating points
    mod_info = {
        'bpsk':  (1,  COLORS['bpsk'],  'BPSK\n(1 b/s/Hz)'),
        'qpsk':  (2,  COLORS['qpsk'],  'QPSK\n(2 b/s/Hz)'),
        'qam16': (4,  COLORS['qam16'], '16-QAM\n(4 b/s/Hz)'),
    }
    for mod, (eta, color, label) in mod_info.items():
        # Minimum Eb/N0 from Shannon: (2^η − 1)/η
        ebno_min_lin = (2 ** eta - 1) / eta
        ebno_min_dB  = 10 * np.log10(ebno_min_lin)
        ax.scatter([ebno_min_dB], [eta], s=80, color=color, zorder=6)
        ax.annotate(label,
                    xy=(ebno_min_dB, eta),
                    xytext=(ebno_min_dB + 1.5, eta - 0.3),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.axvline(-1.59, color='red', linestyle=':', linewidth=1.2,
               label='Shannon limit (−1.59 dB)')
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Spectral Efficiency η = C/B (bits/s/Hz)', fontsize=12)
    ax.set_title('Shannon Capacity Limit', fontsize=13)
    ax.set_xlim(-2, 20)
    ax.set_ylim(0, 7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Achievable rate with BER constraint ----
    ax2 = axes[1]
    snr_dB = np.linspace(-4, 30, 2000)

    # Shannon capacity
    snr_lin = 10 ** (snr_dB / 10)
    # Here SNR = Eb/N0 · η, but for a simple illustration we just plot C/B = log2(1+Eb/N0)
    # as a function of Eb/N0 (treating SNR = Eb/N0 for unit bandwidth)
    C_B = np.log2(1 + snr_lin)
    ax2.plot(snr_dB, C_B, color=COLORS['shannon'], linewidth=2,
             label='Shannon: C/B = log₂(1+SNR)')

    # Achievable rates with BER ≤ 1e-5 threshold
    target = 1e-5
    ber_fns = {
        'bpsk':  (bpsk_ber_awgn, 1),
        'qpsk':  (qpsk_ber_awgn, 2),
        'qam16': (qam16_ber_awgn, 4),
    }

    for mod, (fn, bits) in ber_fns.items():
        color = COLORS[mod]
        ber   = fn(snr_dB)
        label = mod.upper().replace('QAM16', '16-QAM')
        # Achievable rate = bits/symbol if BER < target, else 0
        rate = np.where(ber <= target, float(bits), np.nan)
        # Draw horizontal line at bits/symbol above the min Eb/N0
        snr_req = required_snr(fn, target, snr_dB)
        if snr_req is not None:
            ax2.axvline(snr_req, color=color, linestyle='--', linewidth=1,
                        alpha=0.7)
            ax2.axhline(bits, xmin=0, xmax=1, color=color, linestyle=':',
                        linewidth=0.8, alpha=0.6)
            ax2.scatter([snr_req], [bits], s=60, color=color, zorder=5)
            ax2.annotate(f'{label}\n{snr_req:.1f} dB',
                         xy=(snr_req, bits),
                         xytext=(snr_req + 1, bits + 0.3),
                         fontsize=8, color=color)

    ax2.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax2.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
    ax2.set_title('Achievable Rate at BER ≤ 10⁻⁵', fontsize=13)
    ax2.set_xlim(-2, 20)
    ax2.set_ylim(0, 6)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Capacity Analysis: Shannon Limit vs Practical Modulations',
                 fontsize=14)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plot_capacity_analysis()
    out = os.path.join(OUT_DIR, 'capacity_analysis.png')
    save_figure(fig, out)
    print(f"Figure saved to {out}")
