"""
visualizer.py — Publication-quality figures for digital communications.

All plotting functions accept an optional ``ax`` parameter so they can
be embedded in larger figure layouts.  When ``save_path`` is given the
figure is written to disk at 300 dpi.

Functions
---------
plot_constellation
plot_ber_curves
plot_eye_diagram
plot_psd
plot_channel_response
plot_capacity
plot_confidence_bands
create_summary_figure
save_figure
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

COLORS = {
    'bpsk':    '#1f77b4',
    'qpsk':    '#ff7f0e',
    'qam16':   '#2ca02c',
    'rayleigh': '#d62728',
    'shannon': '#9467bd',
    'awgn':    '#8c564b',
}

LINESTYLES = {
    'simulated':   '-',
    'theoretical': '--',
}

_FONT = {'family': 'DejaVu Sans', 'size': 11}
matplotlib.rc('font', **_FONT)
matplotlib.rc('axes', titlesize=12, labelsize=11)
matplotlib.rc('legend', fontsize=9)
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str, dpi: int = 300):
    """Save figure, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Constellation diagram
# ---------------------------------------------------------------------------

def plot_constellation(
    mapper,
    received_symbols=None,
    n_noisy: int = 500,
    title: str = None,
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot a constellation diagram.

    Parameters
    ----------
    mapper : ConstellationMapper
        Provides the ideal constellation points.
    received_symbols : ndarray of complex or None
        Noisy received symbols to scatter-plot (up to n_noisy shown).
    n_noisy : int
        Maximum noisy samples shown.
    title : str or None
    ax : matplotlib Axes or None
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    mod = mapper.modulation.upper().replace('QAM16', '16-QAM')
    color = COLORS.get(mapper.modulation, '#1f77b4')

    # Noisy cloud
    if received_symbols is not None:
        sym = np.asarray(received_symbols, dtype=complex)[:n_noisy]
        ax.scatter(sym.real, sym.imag,
                   s=4, alpha=0.35, color='lightgray', label='Received')

    # Ideal constellation points
    pts = mapper.get_constellation()
    ax.scatter(pts.real, pts.imag,
               s=80, color=color, zorder=5, label='Ideal')

    # Annotate symbol indices
    for idx, p in enumerate(pts):
        bits = mapper.indices_to_bits(np.array([idx], dtype=np.uint8))
        label = ''.join(map(str, bits))
        ax.annotate(label, (p.real, p.imag),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7, color='dimgray')

    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title(title or f'{mod} Constellation')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# BER curves
# ---------------------------------------------------------------------------

def plot_ber_curves(
    simulated: dict = None,
    theoretical: dict = None,
    title: str = 'BER vs Eb/N0',
    ax: plt.Axes = None,
    save_path: str = None,
    show_ci: bool = True,
):
    """
    Plot BER vs Eb/N0 curves.

    Parameters
    ----------
    simulated : dict or None
        Keys are modulation names (e.g. ``'bpsk'``); values are dicts
        with keys ``'EbN0_dB'``, ``'BER'``, ``'CI_lower'``, ``'CI_upper'``.
    theoretical : dict or None
        Same key structure; values are dicts with ``'EbN0_dB'``, ``'BER'``.
    title : str
    ax : Axes or None
    save_path : str or None
    show_ci : bool
        Whether to draw confidence-interval shading.

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    # Plot theoretical curves first (dashed)
    if theoretical:
        for mod, data in theoretical.items():
            color = COLORS.get(mod, None)
            label = f'{mod.upper().replace("QAM16","16-QAM")} Theory'
            ax.semilogy(data['EbN0_dB'], data['BER'],
                        linestyle='--', color=color,
                        linewidth=1.5, label=label, zorder=3)

    # Plot simulated curves
    if simulated:
        for mod, data in simulated.items():
            color = COLORS.get(mod, None)
            snr = np.asarray(data['EbN0_dB'])
            ber = np.asarray(data['BER'])
            label = f'{mod.upper().replace("QAM16","16-QAM")} Sim'

            # Mask BER = 0 (log scale)
            mask = ber > 0
            ax.semilogy(snr[mask], ber[mask],
                        marker='o', markersize=4, linestyle='-',
                        color=color, linewidth=1.5, label=label, zorder=4)

            if show_ci and 'CI_lower' in data and 'CI_upper' in data:
                ci_lo = np.asarray(data['CI_lower'])
                ci_hi = np.asarray(data['CI_upper'])
                ci_lo_safe = np.where(ci_lo > 0, ci_lo, 1e-10)
                ax.fill_between(snr[mask], ci_lo_safe[mask], ci_hi[mask],
                                alpha=0.18, color=color)

    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Bit Error Rate')
    ax.set_title(title)
    ax.set_ylim(1e-6, 1.0)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', ncol=2)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Eye diagram
# ---------------------------------------------------------------------------

def plot_eye_diagram(
    eye_matrix: np.ndarray,
    sps: int,
    title: str = 'Eye Diagram',
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot an eye diagram from a pre-extracted eye matrix.

    Parameters
    ----------
    eye_matrix : ndarray, shape (n_traces, 2*sps)
    sps : int
        Samples per symbol.
    title : str
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    t = np.linspace(-1, 1, eye_matrix.shape[1])
    for trace in eye_matrix:
        ax.plot(t, trace, color='royalblue', alpha=0.12, linewidth=0.6)

    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time / T_s')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Power spectral density
# ---------------------------------------------------------------------------

def plot_psd(
    signals: dict,
    fs: float = 1.0,
    title: str = 'Power Spectral Density',
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot PSDs using Welch's method.

    Parameters
    ----------
    signals : dict
        Keys are trace labels; values are 1-D arrays.
    fs : float
        Sampling frequency.
    title : str
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(signals), 1)))
    for (label, sig), color in zip(signals.items(), color_cycle):
        sig = np.asarray(sig, dtype=complex)
        # Welch PSD
        f, Pxx = welch(sig, fs=fs, nperseg=min(256, len(sig) // 4),
                       return_onesided=False)
        idx = np.argsort(f)
        Pxx_dB = 10 * np.log10(np.abs(Pxx[idx]) + 1e-20)
        ax.plot(f[idx], Pxx_dB, label=label, color=color, linewidth=1.4)

    ax.set_xlabel('Normalised Frequency (cycles/sample)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Channel frequency response
# ---------------------------------------------------------------------------

def plot_channel_response(
    h,
    n_fft: int = 512,
    title: str = 'Channel Frequency Response',
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot the magnitude frequency response of a channel impulse response.

    Parameters
    ----------
    h : array_like
        CIR coefficients.
    n_fft : int
    title : str
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    h = np.asarray(h, dtype=complex)
    H = np.fft.fftshift(np.fft.fft(h, n=n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))
    H_dB = 20 * np.log10(np.abs(H) + 1e-20)

    ax.plot(freqs, H_dB, color='navy', linewidth=1.5)
    ax.set_xlabel('Normalised Frequency (cycles/sample)')
    ax.set_ylabel('|H(f)| (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Shannon capacity
# ---------------------------------------------------------------------------

def plot_capacity(
    EbN0_dB_shannon,
    eta_shannon,
    modulation_points: dict = None,
    title: str = 'Shannon Capacity vs Eb/N0',
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Plot the Shannon capacity curve and overlay modulation operating points.

    Parameters
    ----------
    EbN0_dB_shannon : array_like
        Eb/N0 values (dB) for the Shannon limit curve.
    eta_shannon : array_like
        Spectral efficiency (bits/s/Hz) values for the Shannon limit.
    modulation_points : dict or None
        Optional: keys are modulation names, values are (EbN0_dB, eta)
        operating points to annotate.
    title : str
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    # Shannon limit
    ax.plot(EbN0_dB_shannon, eta_shannon,
            color=COLORS['shannon'], linewidth=2,
            label='Shannon Limit', zorder=5)
    ax.fill_betweenx(eta_shannon, EbN0_dB_shannon,
                     np.full_like(eta_shannon, ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -5),
                     alpha=0.1, color=COLORS['shannon'], label='Capacity region')

    # Modulation operating points
    if modulation_points:
        for mod, (snr, eta) in modulation_points.items():
            color = COLORS.get(mod, 'black')
            label = mod.upper().replace('QAM16', '16-QAM')
            ax.scatter([snr], [eta], s=80, color=color, zorder=6)
            ax.annotate(label, (snr, eta),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=9, color=color)

    # Shannon limit at 0 bits/s/Hz: Eb/N0 = ln(2) ≈ −1.59 dB
    ax.axvline(-1.59, color='red', linestyle=':', linewidth=1,
               label='Shannon limit (−1.59 dB)')

    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Spectral Efficiency C/B (bits/s/Hz)')
    ax.set_title(title)
    ax.set_xlim(-2, 20)
    ax.set_ylim(0, 7)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Summary (4-panel) figure
# ---------------------------------------------------------------------------

def create_summary_figure(
    mappers: list,
    sim_results: dict,
    theoretical_results: dict,
    save_path: str = None,
):
    """
    Create a 4-panel summary figure.

    Panels
    ------
    1 (top-left)   : BPSK constellation
    2 (top-right)  : BER curves (all modulations)
    3 (bottom-left): QPSK constellation
    4 (bottom-right): 16-QAM constellation

    Parameters
    ----------
    mappers : list of ConstellationMapper
        In order [BPSK, QPSK, 16-QAM].
    sim_results : dict
        As returned by the Monte Carlo sweep.
    theoretical_results : dict
        Theoretical BER dicts keyed by modulation.
    save_path : str or None

    Returns
    -------
    fig : Figure
    """
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_bpsk  = fig.add_subplot(gs[0, 0])
    ax_ber   = fig.add_subplot(gs[:, 1:])
    ax_qpsk  = fig.add_subplot(gs[1, 0])

    # Constellations
    if len(mappers) >= 1:
        plot_constellation(mappers[0], ax=ax_bpsk,
                           title='BPSK Constellation')
    if len(mappers) >= 2:
        plot_constellation(mappers[1], ax=ax_qpsk,
                           title='QPSK Constellation')

    # BER curves
    plot_ber_curves(sim_results, theoretical_results,
                    ax=ax_ber, show_ci=True)

    fig.suptitle('Digital Communications Simulator — Summary', fontsize=14)
    if save_path:
        save_figure(fig, save_path)
    return fig
