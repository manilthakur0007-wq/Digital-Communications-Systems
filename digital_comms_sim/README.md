# Digital Communications Systems Simulator

A production-quality, fully-vectorised Python simulator for digital
modulation, fading channels, and Monte Carlo BER analysis — validated
against Shannon's theoretical limits.

---

## Table of Contents

1. [Theory](#theory)
   - [BPSK, QPSK, 16-QAM Constellations](#modulation-schemes)
   - [BER Derivations](#ber-derivations)
   - [Shannon Capacity](#shannon-capacity)
   - [Monte Carlo Methodology](#monte-carlo-methodology)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Running the Tests](#running-the-tests)
6. [Running Analysis Scripts](#running-analysis-scripts)
7. [Sample Output](#sample-output)
8. [Implementation Notes](#implementation-notes)

---

## Theory

### Modulation Schemes

All constellations are normalised to **unit average symbol energy**
(E_s = 1).  Gray coding is used for QPSK and 16-QAM so that adjacent
symbols differ by exactly one bit, minimising BER at moderate-to-high SNR.

#### BPSK

One bit per symbol.  Constellation: `{−1, +1}`.

```
Q
│
│     ×        ×
─────────────────── I
    bit=0     bit=1
```

Decision rule: `b̂ = 1` if `Re(y) ≥ 0`, else `b̂ = 0`.

#### QPSK

Two bits per symbol.  Gray-coded quadrant mapping:

```
Q
│  10  │  00
│  ×   │   ×
────────────── I
│  ×   │   ×
│  11  │  01
```

Each I and Q component is an independent BPSK signal (Es/2 per
component), so the per-bit BER is identical to BPSK.

#### 16-QAM

Four bits per symbol.  Gray-coded two-dimensional 4-PAM:

```
Q
+3  ×   ×   ×   ×
+1  ×   ×   ×   ×
-1  ×   ×   ×   ×
-3  ×   ×   ×   ×
    -3  -1  +1  +3  I

Gray code (I or Q): 00→-3, 01→-1, 11→+1, 10→+3
```

Average symbol energy (unnormalised): E_s = 10.  After normalisation
(÷√10), E_s = 1.

---

### BER Derivations

#### Noise Model

Complex baseband AWGN:

```
n = n_I + j·n_Q,   n_I, n_Q ~ N(0, σ²)
```

Given E_b/N_0 (linear), bit energy E_b = E_s/k, so:

```
σ² = N_0/2 = E_s / (2 k (E_b/N_0)) = 1 / (2k · E_b/N_0)
```

where k = bits per symbol.

#### BPSK

The decision statistic is `Re(y) = ±1 + n_I`.  An error occurs when
the noise exceeds the half minimum-distance (= 1):

```
BER_BPSK = Q(√(2 E_b/N_0)) = ½ erfc(√(E_b/N_0))
```

*Textbook check:* at E_b/N_0 = 10 dB, BER ≈ 3.87 × 10⁻⁶.

#### QPSK

The I and Q components are decoded independently.  Each is a BPSK
decision with the same distance, so:

```
BER_QPSK = Q(√(2 E_b/N_0))    [same as BPSK]
```

#### 16-QAM

The I and Q components are independent 4-PAM signals with levels
`{±1/√10, ±3/√10}` (after normalisation).  Using Gray coding, the
per-bit error probability for 4-PAM is:

```
P_PAM = (3/4) Q(√(4 E_b/N_0 / 5))
```

The overall 16-QAM BER equals the 4-PAM BER:

```
BER_16QAM ≈ (3/8) erfc(√(2 E_b/N_0 / 5))
```

*Textbook check:* at E_b/N_0 = 10 dB, BER ≈ 1.8 × 10⁻³.

#### Rayleigh Fading BPSK

For coherent BPSK over flat Rayleigh fading with perfect CSI:

```
BER_Rayleigh = ½ [1 − √(E_b/N_0 / (1 + E_b/N_0))]
```

At high SNR this decays as 1/(4·E_b/N_0) — much slower than the
Gaussian Q-function.

---

### Shannon Capacity

The AWGN channel capacity is:

```
C = B log₂(1 + SNR)    [bits/s]
```

where `B` is the bandwidth and `SNR = E_s/N_0`.  In terms of spectral
efficiency η = C/B and E_b/N_0:

```
η = log₂(1 + η · E_b/N_0)
```

This implicit equation defines the **Shannon limit curve** in the
(E_b/N_0, η) plane.

**Shannon limit:** as η → 0, the minimum achievable E_b/N_0 →
ln(2) ≈ **−1.59 dB**.  No reliable communication is possible below
this value, regardless of the modulation or coding used.

```
        6 │                       /Shannon
          │                    /
η (b/s/Hz)│               /
        2 │         /
          │   /
        0 └──────────────────────────
         -2    0    5   10   15  Eb/N0 (dB)
```

---

### Monte Carlo Methodology

For each E_b/N_0 point:

1. Generate `N` random bits (typically 200,000+).
2. Map to complex symbols via the constellation mapper.
3. Pass through the channel (AWGN or Rayleigh fading).
4. Demodulate (nearest-neighbour decision).
5. Count bit errors.
6. Compute BER = errors / N.
7. Report 95% **Wilson-score confidence interval**:

```
CI = [p̃ ± z√(p(1−p)/N + z²/(4N²))] / (1 + z²/N)
```

where p = BER, z = 1.96 (95%).  This interval is valid even for BER
near 0 or 1.

**Validation criterion:** the theoretical BER must fall within the
95% CI.  If it does not, there is a bug in the simulation or the
theoretical formula.

---

## Project Structure

```
digital_comms_sim/
├── src/
│   ├── modulator.py         Constellation mapping (BPSK, QPSK, 16-QAM)
│   ├── channel.py           AWGN, Rayleigh fading, ISI channel models
│   ├── demodulator.py       Coherent receivers, LLR, ZF equaliser
│   ├── theoretical.py       Analytical BER, Q-function, Shannon capacity
│   ├── error_analysis.py    Monte Carlo BER/SER/FER engine
│   ├── pulse_shaping.py     Root Raised Cosine filter
│   ├── channel_estimator.py Pilot-based LS channel estimation
│   ├── visualizer.py        Publication-quality matplotlib figures
│   ├── simulation.py        Master simulation runner (DataFrame output)
│   └── main.py              CLI entry point (argparse)
├── analysis/
│   ├── compare_modulations.py
│   ├── capacity_analysis.py
│   ├── pulse_shape_analysis.py
│   └── rayleigh_analysis.py
├── tests/
│   ├── test_modulator.py
│   ├── test_channel.py
│   ├── test_demodulator.py
│   └── test_theoretical.py
├── results/
│   └── plots/               PNG outputs
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run a quick BPSK simulation

```bash
cd digital_comms_sim
python src/main.py --modulation bpsk --channel awgn --bits 200000 --save
```

### Run all modulations and generate plots

```bash
python src/main.py --modulation all --channel awgn --snr-min -4 --snr-max 16 --save
```

### Compare modulations (analysis script)

```bash
python analysis/compare_modulations.py
```

### Run Rayleigh fading analysis

```bash
python analysis/rayleigh_analysis.py
```

---

## CLI Reference

```
usage: main.py [-h] [--modulation {bpsk,qpsk,qam16,all}]
               [--channel {awgn,rayleigh,isi}]
               [--bits BITS]
               [--snr-min SNR_MIN] [--snr-max SNR_MAX] [--snr-step SNR_STEP]
               [--seed SEED]
               [--plot] [--save]
               [--output-dir OUTPUT_DIR]

Options:
  --modulation    Modulation scheme(s): bpsk, qpsk, qam16, all  [default: bpsk]
  --channel       Channel model: awgn, rayleigh, isi             [default: awgn]
  --bits          Bits per Eb/N0 point                           [default: 200000]
  --snr-min       Minimum Eb/N0 (dB)                             [default: -4.0]
  --snr-max       Maximum Eb/N0 (dB)                             [default: 16.0]
  --snr-step      Step size (dB)                                 [default: 1.0]
  --seed          Random seed for reproducibility                [default: None]
  --plot          Show interactive plots
  --save          Save CSV and PNG to results/
  --output-dir    Output directory                               [default: results]
```

### Examples

```bash
# Reproducible BPSK/AWGN run
python src/main.py --modulation bpsk --channel awgn --bits 500000 --seed 42 --save

# Rayleigh fading for QPSK
python src/main.py --modulation qpsk --channel rayleigh --snr-min 0 --snr-max 20 --save

# All modulations, save only
python src/main.py --modulation all --channel awgn --save
```

---

## Running the Tests

```bash
cd digital_comms_sim
pytest
```

Tests are organised by module:

| File | What it tests |
|---|---|
| `test_modulator.py` | Constellation mapping, Gray coding, energy normalisation |
| `test_channel.py` | Noise variance, Rayleigh statistics, ISI convolution |
| `test_demodulator.py` | Zero BER at 100 dB, decision boundaries, ZF equaliser |
| `test_theoretical.py` | Q-function, BER formulas, Shannon limit, sim-vs-theory CI check |

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Running Analysis Scripts

Each analysis script is self-contained and saves figures to `results/plots/`.

| Script | Output |
|---|---|
| `compare_modulations.py` | BER curves for all modulations + theoretical |
| `capacity_analysis.py` | Shannon limit + achievable rates per modulation |
| `pulse_shape_analysis.py` | Eye diagrams (α=0.25/0.5/0.99) + PSD comparison |
| `rayleigh_analysis.py` | AWGN vs Rayleigh BER + fading realisations |

---

## Sample Output

### BER Curves

Running `compare_modulations.py` produces a figure showing:

- Dashed lines: theoretical BER for BPSK, QPSK, 16-QAM
- Solid lines + markers: Monte Carlo simulated BER
- Shaded bands: 95% Wilson confidence intervals
- Annotation: ~4 dB gap between QPSK and 16-QAM at BER = 10⁻³

At E_b/N_0 = 10 dB:
- BPSK/QPSK BER ≈ 3.87 × 10⁻⁶
- 16-QAM BER ≈ 1.76 × 10⁻³

### Shannon Capacity

The Shannon limit plot shows that:
- At η = 1 bit/s/Hz (BPSK), minimum E_b/N_0 = 0 dB
- At η = 2 bits/s/Hz (QPSK), minimum E_b/N_0 ≈ 1.76 dB
- At η = 4 bits/s/Hz (16-QAM), minimum E_b/N_0 ≈ 5.74 dB
- Absolute Shannon limit: −1.59 dB (any rate, any modulation)

### Eye Diagrams

The eye diagram analysis shows that:
- Small α (0.25): narrow eye due to sharp spectral roll-off
- Large α (0.99): wide eye opening, more bandwidth used
- Without pulse shaping: eye nearly closed due to ISI

---

## Implementation Notes

### Vectorisation

All inner simulation loops are fully NumPy-vectorised.  There are no
Python `for` loops per symbol or per bit.  Key vectorised operations:

- `ConstellationMapper.map_bits`: matrix multiplication for bit-to-index
  conversion; fancy indexing for symbol lookup.
- `ConstellationMapper.demap_symbols`: broadcast subtraction `(N,M)`
  distance matrix; `np.argmin` over symbol dimension.
- `MonteCarloSimulator.run_single`: single call to `np.random.default_rng`
  generates all bits and all noise at once.
- Frame error rate: `np.any(error_matrix, axis=1)` over the frame matrix.

### Reproducibility

Pass `--seed <int>` to `main.py` or set `seed=<int>` in
`SimulationConfig` / `MonteCarloSimulator`.  A fixed seed guarantees
identical bit sequences, channel noise, and fading realisations.

### Confidence Intervals

The Wilson score interval (not the normal approximation) is used because
it remains valid even when `n_errors` is very small (near the noise floor).
At BER = 0 (no observed errors), it returns a meaningful upper bound.

### Channel Models

- **AWGN**: complex CN(0, N_0) noise, one noise sample per symbol.
- **Rayleigh**: CN(0, 1) complex channel coefficient per symbol, flat
  fading (quasi-static within one burst), plus AWGN.  Perfect CSI
  assumed; ZF equalisation = divide by `h`.
- **ISI**: FIR convolution with default taps `[1, 0.5, 0.25]`, plus AWGN.
  ZF equalisation in the frequency domain (block processing).

### Modulation Normalisation

All constellations satisfy E[|s|²] = 1 exactly.  The noise power is
derived from the Eb/N0 specification via:

```
σ² = 1 / (2 · k · E_b/N_0_linear)
```

This ensures that the simulated BER exactly matches the theoretical
formula.
