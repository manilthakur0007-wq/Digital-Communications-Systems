"""
main.py — CLI entry point for the Digital Communications Simulator.

Usage examples
--------------
    python main.py --modulation bpsk --channel awgn --bits 200000 --plot
    python main.py --modulation all --channel awgn --snr-min -4 --snr-max 16
    python main.py --modulation qpsk --channel rayleigh --seed 42 --plot --save
"""

import argparse
import os
import sys

# Ensure src/ is on the path when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from simulation import SimulationConfig, SimulationRunner
from theoretical import (
    bpsk_ber_awgn, qpsk_ber_awgn, qam16_ber_awgn,
    bpsk_ber_rayleigh, shannon_capacity,
)
from modulator import ConstellationMapper
from channel import AWGNChannel
from visualizer import (
    plot_ber_curves, plot_constellation,
    plot_capacity, save_figure,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Digital Communications Systems Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--modulation', '-m',
        choices=['bpsk', 'qpsk', 'qam16', 'all'],
        default='bpsk',
        help='Modulation scheme(s) to simulate.',
    )
    p.add_argument(
        '--channel', '-c',
        choices=['awgn', 'rayleigh', 'isi'],
        default='awgn',
        help='Channel model.',
    )
    p.add_argument(
        '--bits', '-b',
        type=int,
        default=200_000,
        help='Number of bits per Eb/N0 point.',
    )
    p.add_argument(
        '--snr-min',
        type=float,
        default=-4.0,
        dest='snr_min',
        help='Minimum Eb/N0 (dB).',
    )
    p.add_argument(
        '--snr-max',
        type=float,
        default=16.0,
        dest='snr_max',
        help='Maximum Eb/N0 (dB).',
    )
    p.add_argument(
        '--snr-step',
        type=float,
        default=1.0,
        dest='snr_step',
        help='Eb/N0 step size (dB).',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility.',
    )
    p.add_argument(
        '--plot',
        action='store_true',
        help='Generate and display plots.',
    )
    p.add_argument(
        '--save',
        action='store_true',
        help='Save CSV results and PNG plots to results/.',
    )
    p.add_argument(
        '--output-dir',
        default='results',
        dest='output_dir',
        help='Directory for output files.',
    )
    return p


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_modulation(mod: str, args) -> pd.DataFrame:
    """Run a single modulation and return the results DataFrame."""
    cfg = SimulationConfig(
        modulation=mod,
        channel=args.channel,
        n_bits=args.bits,
        EbN0_min=args.snr_min,
        EbN0_max=args.snr_max,
        EbN0_step=args.snr_step,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    runner = SimulationRunner(cfg, verbose=True)
    df = runner.run()
    if args.save:
        runner.save_results(df)
    return df


def build_theoretical(mods, EbN0_range, channel):
    """Build theoretical BER dict for the given modulations."""
    theory = {}
    for mod in mods:
        if channel == 'rayleigh':
            ber = bpsk_ber_rayleigh(EbN0_range)
        elif mod == 'bpsk':
            ber = bpsk_ber_awgn(EbN0_range)
        elif mod == 'qpsk':
            ber = qpsk_ber_awgn(EbN0_range)
        elif mod == 'qam16':
            ber = qam16_ber_awgn(EbN0_range)
        else:
            continue
        theory[mod] = {'EbN0_dB': EbN0_range, 'BER': ber}
    return theory


def main():
    parser = build_parser()
    args = parser.parse_args()

    mods = ['bpsk', 'qpsk', 'qam16'] if args.modulation == 'all' else [args.modulation]
    EbN0_range = np.arange(args.snr_min, args.snr_max + args.snr_step / 2,
                           args.snr_step)

    # --- Run simulations ---
    all_results = {}
    for mod in mods:
        df = run_modulation(mod, args)
        all_results[mod] = {
            'EbN0_dB': df['EbN0_dB'].values,
            'BER':     df['BER'].values,
            'CI_lower': df['CI_lower'].values,
            'CI_upper': df['CI_upper'].values,
        }

    if not (args.plot or args.save):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    # --- BER curves ---
    theory = build_theoretical(mods, EbN0_range, args.channel)
    fig_ber = plot_ber_curves(
        simulated=all_results,
        theoretical=theory,
        title=f'BER vs Eb/N0 — {args.channel.upper()} Channel',
    )
    if args.save:
        ber_path = os.path.join(args.output_dir, 'plots', 'ber_curves.png')
        save_figure(fig_ber, ber_path)
        print(f"BER plot saved to {ber_path}")
    if args.plot:
        plt.show()
    plt.close(fig_ber)

    # --- Constellation diagrams ---
    for mod in mods:
        mapper = ConstellationMapper(mod)
        # Add some noise for scatter
        ch = AWGNChannel(10.0, mapper.bits_per_symbol, seed=args.seed)
        rx_syms = ch.corrupt(
            mapper.map_bits(
                np.random.default_rng(args.seed).integers(
                    0, 2, 1000 * mapper.bits_per_symbol
                )
            )
        )
        fig_c = plot_constellation(mapper, received_symbols=rx_syms)
        if args.save:
            c_path = os.path.join(args.output_dir, 'plots',
                                  f'constellation_{mod}.png')
            save_figure(fig_c, c_path)
            print(f"Constellation saved to {c_path}")
        if args.plot:
            plt.show()
        plt.close(fig_c)

    # --- Shannon capacity ---
    ebno_sh, eta_sh = shannon_capacity(EbN0_range)
    # Modulation operating points (ideal bits/s/Hz)
    mod_pts = {
        'bpsk':  (-1.59 + 10 * np.log10(1), 1.0),
        'qpsk':  (-1.59 + 10 * np.log10(2), 2.0),
        'qam16': (-1.59 + 10 * np.log10(4), 4.0),
    }
    fig_cap = plot_capacity(ebno_sh, eta_sh,
                            modulation_points={m: mod_pts[m] for m in mods if m in mod_pts})
    if args.save:
        cap_path = os.path.join(args.output_dir, 'plots', 'capacity.png')
        save_figure(fig_cap, cap_path)
        print(f"Capacity plot saved to {cap_path}")
    if args.plot:
        plt.show()
    plt.close(fig_cap)


if __name__ == '__main__':
    main()
