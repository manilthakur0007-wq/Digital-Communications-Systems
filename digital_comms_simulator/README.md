# Digital Communications Systems Simulator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-103%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A modular, production-quality digital communications simulator in Python. Implements **BPSK, QPSK, and 16-QAM** modulation with **AWGN and Rayleigh fading** channels, Monte Carlo BER analysis, and Shannon capacity validation — all results validated against closed-form theoretical limits.

---

## Quick Start

```bash
git clone <your-repo-url>
cd digital_comms_simulator
pip install -r requirements.txt

# Run BPSK over AWGN
python src/main.py --modulation bpsk --channel awgn --bits 200000 --save

# Compare all modulations
python src/main.py --modulation all --channel awgn --bits 200000 --save

# Run full test suite
pytest
```

Output files (CSV + plots) are written to `results/`.

---

## Contents

- [Theory](#theory)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [Analysis Scripts](#analysis-scripts)
- [Test Suite](#test-suite)
- [Implementation Notes](#implementation-notes)

---

## Theory

### Modulation Schemes

All constellations are normalised to **unit average symbol energy (Es = 1)**. Gray coding is applied to QPSK and 16-QAM so that adjacent constellation points differ by exactly one bit, minimising BER at moderate-to-high SNR.

#### BPSK — 1 bit/symbol

```
         Q
         |
  0      |      1
  ×------+------×--------→ I
        -1      +1
```

Decision: `bit = 1` if `Re(y) >= 0`, else `bit = 0`.

#### QPSK — 2 bits/symbol (Gray coded)

```
         Q
  10     |     00
   ×     |      ×
   -------+-------→ I
   ×     |      ×
  11     |     01
```

Adjacent quadrants differ by 1 bit. The I and Q decisions are independent BPSK decisions, giving the same per-bit BER as BPSK.

#### 16-QAM — 4 bits/symbol (Gray coded)

```
     Q
+3   × × × ×
+1   × × × ×
-1   × × × ×
-3   × × × ×
    -3-1+1+3  → I

Gray code per axis: 00→-3, 01→-1, 11→+1, 10→+3
Normalisation: divide by sqrt(10) so Es = 1.
```

### BER Derivations

All formulas assume coherent detection with perfect channel knowledge.

**AWGN noise model** (complex baseband):

```
n = nI + j·nQ,   nI, nQ ~ N(0, σ²)
σ² = 1 / (2·k·Eb/N0)          k = bits per symbol, Eb/N0 in linear
```

| Modulation | BER formula |
|---|---|
| BPSK | `Q(sqrt(2·Eb/N0)) = 0.5·erfc(sqrt(Eb/N0))` |
| QPSK | `Q(sqrt(2·Eb/N0))` — same as BPSK per bit |
| 16-QAM | `(3/8)·erfc(sqrt(2·Eb/N0 / 5))` |
| BPSK / Rayleigh | `0.5·[1 − sqrt(Eb/N0 / (1 + Eb/N0))]` |

**Spot checks (simulated BER falls within 95% CI of theory):**

| Condition | Theoretical BER |
|---|---|
| BPSK, Eb/N0 = 8 dB | 1.91 × 10⁻⁴ |
| BPSK, Eb/N0 = 10 dB | 3.87 × 10⁻⁶ |
| 16-QAM, Eb/N0 = 10 dB | 1.75 × 10⁻³ |

### Shannon Capacity

```
C/B = log2(1 + SNR)          bits/s/Hz
```

The **Shannon limit** in the (Eb/N0, η) plane satisfies:

```
η = log2(1 + η·Eb/N0)        η = C/B (spectral efficiency)
```

As η → 0, the minimum required Eb/N0 → ln(2) ≈ **−1.59 dB**. No reliable communication is possible below this value regardless of modulation or coding.

| Modulation | η (bits/s/Hz) | Shannon minimum Eb/N0 |
|---|---|---|
| BPSK | 1 | 0.00 dB |
| QPSK | 2 | 1.76 dB |
| 16-QAM | 4 | 5.74 dB |

### Monte Carlo Methodology

For each Eb/N0 point:

1. Generate N random bits (default: 200,000)
2. Map to symbols via the constellation mapper
3. Pass through the channel model (AWGN / Rayleigh / ISI)
4. Apply equalization (ZF for Rayleigh, frequency-domain ZF for ISI)
5. Demodulate — nearest-neighbour hard decision
6. Count bit errors → BER = errors / N

**Confidence intervals** use the Wilson score formula (valid even near BER = 0):

```
CI = (p̃ ± z·sqrt(p(1−p)/N + z²/(4N²))) / (1 + z²/N)
```

where z = 1.96 for 95% confidence. The theoretical BER must fall within this interval — if it does not, there is a bug.

---

## Project Structure

```
digital_comms_simulator/
├── src/
│   ├── modulator.py         # ConstellationMapper: BPSK, QPSK, 16-QAM
│   ├── channel.py           # AWGNChannel, RayleighChannel, ISIChannel
│   ├── demodulator.py       # Coherent receivers, LLR, ZF equalizer
│   ├── theoretical.py       # Analytical BER, Q-function, Shannon capacity
│   ├── error_analysis.py    # Monte Carlo engine: BER / SER / FER + 95% CI
│   ├── pulse_shaping.py     # Root Raised Cosine filter design
│   ├── channel_estimator.py # Pilot-based LS channel estimation
│   ├── visualizer.py        # BER, constellation, eye diagram, PSD, capacity
│   ├── simulation.py        # SimulationRunner → pandas DataFrame → CSV
│   └── main.py              # CLI entry point (argparse)
├── analysis/
│   ├── compare_modulations.py   # BER curves: BPSK vs QPSK vs 16-QAM
│   ├── capacity_analysis.py     # Shannon limit + achievable rates
│   ├── pulse_shape_analysis.py  # Eye diagrams, PSD, ISI reduction
│   └── rayleigh_analysis.py     # AWGN vs Rayleigh + fading realisations
├── tests/
│   ├── test_modulator.py    # Mapping, Gray coding, energy normalisation
│   ├── test_channel.py      # Noise variance, Rayleigh statistics
│   ├── test_demodulator.py  # Decision boundaries, ZF equalizer
│   └── test_theoretical.py  # Q-function, BER formulas, sim-vs-theory CI
├── results/
│   └── plots/               # Generated PNG figures
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## CLI Reference

```
python src/main.py [OPTIONS]

Options:
  --modulation  bpsk | qpsk | qam16 | all          [default: bpsk]
  --channel     awgn | rayleigh | isi               [default: awgn]
  --bits        Bits per Eb/N0 point                [default: 200000]
  --snr-min     Minimum Eb/N0 (dB)                  [default: -4.0]
  --snr-max     Maximum Eb/N0 (dB)                  [default: 16.0]
  --snr-step    Step size (dB)                      [default: 1.0]
  --seed        Integer random seed                 [default: None]
  --plot        Show interactive plots
  --save        Save results to results/
  --output-dir  Output directory                    [default: results]
```

**Examples:**

```bash
# Reproducible BPSK/AWGN simulation
python src/main.py --modulation bpsk --channel awgn --bits 500000 --seed 42 --save

# QPSK over Rayleigh fading
python src/main.py --modulation qpsk --channel rayleigh --snr-min 0 --snr-max 20 --save

# All modulations, save plots
python src/main.py --modulation all --channel awgn --snr-min -4 --snr-max 16 --save
```

---

## Analysis Scripts

Run from the project root. Each script saves a PNG to `results/plots/`.

| Script | Output |
|---|---|
| `analysis/compare_modulations.py` | BER curves for BPSK, QPSK, 16-QAM with theoretical overlay and 95% CI shading |
| `analysis/capacity_analysis.py` | Shannon limit curve + achievable rates per modulation |
| `analysis/pulse_shape_analysis.py` | Eye diagrams for α = 0.25 / 0.5 / 0.99, PSD with and without RRC filtering |
| `analysis/rayleigh_analysis.py` | AWGN vs Rayleigh BER, MRC-2 diversity curve, fading realisations |

```bash
python analysis/compare_modulations.py
python analysis/capacity_analysis.py
python analysis/pulse_shape_analysis.py
python analysis/rayleigh_analysis.py
```

---

## Test Suite

```bash
pytest               # run all 103 tests
pytest -v            # verbose
pytest --tb=short    # short tracebacks
```

| File | Tests | What is verified |
|---|---|---|
| `test_modulator.py` | 22 | Bit→symbol mapping, Gray coding, unit energy, round-trip zero BER |
| `test_channel.py` | 17 | AWGN σ² matches Eb/N0 formula, Rayleigh KS test, ISI length |
| `test_demodulator.py` | 18 | Zero BER at 100 dB, BPSK decision boundary, LLR formula, ZF exact inversion |
| `test_theoretical.py` | 46 | Q-function values, BER at known SNR, Shannon limit at −1.59 dB, sim CI overlap |

---

## Implementation Notes

### Vectorisation

No Python loops per symbol. Key patterns:

```python
# Modulation: matrix multiply for bit→index, fancy indexing for symbol lookup
indices = bit_matrix @ powers          # (N_sym,)  int
symbols = self.constellation[indices]  # (N_sym,)  complex

# Demodulation: broadcast distance matrix, argmin across symbol dimension
dist2   = np.abs(symbols[:, None] - constellation[None, :]) ** 2
indices = np.argmin(dist2, axis=1)

# Frame error rate: vectorised any() over frame matrix
frame_errors = np.sum(np.any(error_matrix.reshape(n_frames, frame_len), axis=1))
```

### Channel models

| Channel | Model | Equalizer |
|---|---|---|
| AWGN | `y = x + n`, `n ~ CN(0, N0)` | None |
| Rayleigh | `y = h·x + n`, `h ~ CN(0, 1)` | `ŷ = y/h` (perfect CSI) |
| ISI | `y = h * x + n` (FIR) | Frequency-domain ZF block |

### Noise power

```
σ² = 1 / (2·k·Eb/N0_linear)
```

Derived from Es = 1, Eb = Es/k, N0 = Eb/(Eb/N0), σ² = N0/2.

### Root Raised Cosine filter

Closed-form impulse response with special cases at `t = 0` and `t = ±Ts/(4α)` to avoid 0/0. The transmit and receive RRC filters cascade to a raised-cosine response with zero ISI at optimum sampling instants.

---

## Requirements

```
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pandas>=2.0
pytest>=7.4
```
