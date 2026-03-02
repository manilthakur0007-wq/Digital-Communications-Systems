"""
theoretical.py — Closed-form BER and channel-capacity expressions.

All BER formulae assume coherent detection over the AWGN channel with
unit-average-energy, Gray-coded constellations.

References
----------
Proakis & Salehi, "Digital Communications", 5th ed., McGraw-Hill (2008).
Haykin, "Communication Systems", 4th ed., Wiley (2001).
Shannon, "A Mathematical Theory of Communication", Bell Sys. Tech. J. (1948).
"""

import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Q-function and helpers
# ---------------------------------------------------------------------------

def q_function(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Q-function: Q(x) = (1/2) erfc(x / √2).

    Numerically identical to scipy.stats.norm.sf(x) but avoids the
    stats dependency and is fully vectorised.
    """
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


# ---------------------------------------------------------------------------
# AWGN BER expressions
# ---------------------------------------------------------------------------

def bpsk_ber_awgn(EbN0_dB) -> np.ndarray:
    """
    Theoretical BER for coherent BPSK over AWGN.

        BER = Q(√(2 Eb/N0))  =  (1/2) erfc(√(Eb/N0))

    Parameters
    ----------
    EbN0_dB : float or array_like
        Eb/N0 in dB.

    Returns
    -------
    ber : ndarray of float
    """
    EbN0_lin = 10 ** (np.asarray(EbN0_dB, dtype=float) / 10)
    return q_function(np.sqrt(2 * EbN0_lin))


def qpsk_ber_awgn(EbN0_dB) -> np.ndarray:
    """
    Theoretical BER for coherent QPSK (Gray coded) over AWGN.

    QPSK with unit symbol energy has the same per-bit SNR geometry as
    BPSK, so:

        BER = Q(√(2 Eb/N0))

    Parameters
    ----------
    EbN0_dB : float or array_like

    Returns
    -------
    ber : ndarray of float
    """
    return bpsk_ber_awgn(EbN0_dB)


def qam16_ber_awgn(EbN0_dB) -> np.ndarray:
    """
    Approximate BER for Gray-coded 16-QAM over AWGN.

    For square 16-QAM the I and Q components are independent 4-PAM
    signals.  With Gray coding, the per-bit error probability of each
    4-PAM component is:

        P_PAM = (3/4) Q(√(4 Eb/N0 · 1/5))

    Because the I and Q BERs are equal, the overall 16-QAM BER equals
    the 4-PAM BER:

        BER ≈ (3/8) erfc(√(2 Eb/N0 / 5))

    Derivation
    ----------
    Unnormalised 4-PAM levels: {−3, −1, +1, +3}, minimum distance = 2,
    average power = 5 (per component).  After normalising 16-QAM so that
    Es = 1 the average power per PAM component is 10/2 = 5, with Eb = Es/4.
    Half minimum distance in normalised units = 1/√10.

    Let σ² = N0/2 be the noise variance per dimension and σ = √(N0/2).

        P(error | inner point) = 2 Q(1/(√10 σ))
        P(error | outer point) = Q(1/(√10 σ))
        P_PAM_SER = (3/2) Q(1/(√10 σ))

    With Gray coding, BER_PAM = P_PAM_SER / 2 = (3/4) Q(√(2/(10 N0))).
    Substituting N0 = 1/(4 EbN0_lin):

        BER = (3/4) Q(√(4 EbN0_lin / 5))
            = (3/8) erfc(√(2 EbN0_lin / 5))

    Parameters
    ----------
    EbN0_dB : float or array_like

    Returns
    -------
    ber : ndarray of float
    """
    EbN0_lin = 10 ** (np.asarray(EbN0_dB, dtype=float) / 10)
    return (3 / 8) * erfc(np.sqrt(2 * EbN0_lin / 5))


# ---------------------------------------------------------------------------
# Rayleigh-fading BER expressions
# ---------------------------------------------------------------------------

def bpsk_ber_rayleigh(EbN0_dB) -> np.ndarray:
    """
    Exact BER for coherent BPSK over flat Rayleigh fading.

    Assuming perfect channel knowledge at the receiver (coherent
    detection with ideal CSI):

        BER = (1/2) [1 − √(ρ / (1 + ρ))]

    where ρ = Eb/N0 (linear).  At high SNR this decays as 1/(4ρ),
    much slower than the Gaussian Q-function decay in AWGN.

    Parameters
    ----------
    EbN0_dB : float or array_like

    Returns
    -------
    ber : ndarray of float
    """
    EbN0_lin = 10 ** (np.asarray(EbN0_dB, dtype=float) / 10)
    return 0.5 * (1 - np.sqrt(EbN0_lin / (1 + EbN0_lin)))


def qpsk_ber_rayleigh(EbN0_dB) -> np.ndarray:
    """
    Approximate BER for coherent QPSK over flat Rayleigh fading.

    Uses the same per-bit geometry as BPSK (equal Eb/N0), so:

        BER ≈ BER_BPSK_Rayleigh(Eb/N0)

    Parameters
    ----------
    EbN0_dB : float or array_like

    Returns
    -------
    ber : ndarray of float
    """
    return bpsk_ber_rayleigh(EbN0_dB)


# ---------------------------------------------------------------------------
# Shannon capacity
# ---------------------------------------------------------------------------

def shannon_capacity(EbN0_dB, spectral_efficiency=None):
    """
    Shannon channel capacity (AWGN channel).

    Two modes:

    1.  If ``spectral_efficiency`` is None, returns the capacity–SNR
        relationship for a fixed bandwidth:

            C/B = log₂(1 + SNR)   [bits/s/Hz]

        where SNR = Eb/N0 · (C/B).  This implicit equation defines the
        Shannon limit curve in the (Eb/N0, C/B) plane.

        Returned: ``(EbN0_dB_array, spectral_efficiency_array)``

    2.  If ``spectral_efficiency`` (η = C/B) is provided as a scalar or
        array, returns the minimum Eb/N0 (in dB) required to achieve
        that efficiency:

            EbN0_min = (2^η − 1) / η  [linear]

        Returned: ``EbN0_dB_min`` (scalar or array matching η).

    Parameters
    ----------
    EbN0_dB : float or array_like
        Eb/N0 range (dB) for mode 1.
    spectral_efficiency : float or array_like, optional
        Target η (bits/s/Hz) for mode 2.

    Returns
    -------
    Depends on mode (see above).
    """
    EbN0_dB = np.asarray(EbN0_dB, dtype=float)

    if spectral_efficiency is None:
        # Solve η = log2(1 + η · EbN0_lin) numerically on a fine grid.
        # For plotting, generate the Shannon-limit locus.
        eta_vals = np.linspace(1e-3, 12.0, 2000)  # spectral efficiencies
        # EbN0_lin required for each η: EbN0_lin = (2^η − 1) / η
        EbN0_lin_req = (2 ** eta_vals - 1) / eta_vals
        EbN0_dB_req = 10 * np.log10(EbN0_lin_req)
        return EbN0_dB_req, eta_vals
    else:
        eta = np.asarray(spectral_efficiency, dtype=float)
        EbN0_lin_min = (2 ** eta - 1) / eta
        return 10 * np.log10(EbN0_lin_min)


def channel_capacity_vs_snr(snr_dB):
    """
    Plain AWGN capacity C/B = log₂(1 + SNR) as a function of SNR (dB).

    Parameters
    ----------
    snr_dB : float or array_like
        SNR = Es/N0 in dB (not Eb/N0).

    Returns
    -------
    capacity : ndarray of float
        Spectral efficiency in bits/s/Hz.
    """
    snr_lin = 10 ** (np.asarray(snr_dB, dtype=float) / 10)
    return np.log2(1 + snr_lin)


def ber_table() -> dict:
    """
    Return a dict of BER values at standard Eb/N0 check-points.

    Useful for quick sanity checks.

    Returns
    -------
    table : dict
        Keys are (modulation, EbN0_dB) tuples; values are BER floats.
    """
    check_points = {
        ('bpsk', 0): bpsk_ber_awgn(0),
        ('bpsk', 4): bpsk_ber_awgn(4),
        ('bpsk', 8): bpsk_ber_awgn(8),
        ('bpsk', 10): bpsk_ber_awgn(10),
        ('qpsk', 10): qpsk_ber_awgn(10),
        ('qam16', 10): qam16_ber_awgn(10),
        ('qam16', 14): qam16_ber_awgn(14),
    }
    return {k: float(v) for k, v in check_points.items()}
