"""
Tests for the NPA (Non-Performing Asset) probability algorithm.

Gap identified: this is the highest-business-value calculation in the pipeline.
Any weight drift (e.g. 0.5 → 0.05) silently corrupts every risk label.
"""

import numpy as np
import pytest

from src.pipeline import calculate_npa_probability


def _npa(foir, cibil, emp):
    return calculate_npa_probability(
        np.array([foir]), np.array([cibil]), [emp]
    )[0]


# ── Component weight verification ──────────────────────────────────────────────

def test_foir_weight_is_0_5():
    """Δfoir = 1.0 → Δnpa = 0.5 (after clip, using a mid-range starting point)."""
    base = _npa(0.0, 700, "Salaried")
    high = _npa(0.5, 700, "Salaried")
    assert abs(high - base - 0.25) < 0.001


def test_cibil_weight_at_extremes():
    """CIBIL 550 vs 850 with foir=0, salaried: diff = (300/850)*0.4."""
    min_cibil = _npa(0.0, 550, "Salaried")
    max_cibil = _npa(0.0, 850, "Salaried")
    expected_diff = (300 / 850) * 0.4
    assert abs(min_cibil - max_cibil - expected_diff) < 0.001


def test_non_salaried_adds_0_1_premium():
    """Non-salaried employment adds exactly 0.1 to NPA probability."""
    salaried = _npa(0.40, 700, "Salaried")
    non_sal = _npa(0.40, 700, "Self-Employed Non-Professional")
    assert abs(non_sal - salaried - 0.1) < 0.001


def test_self_employed_professional_also_adds_premium():
    salaried = _npa(0.40, 700, "Salaried")
    prof = _npa(0.40, 700, "Self-Employed Professional")
    assert abs(prof - salaried - 0.1) < 0.001


# ── Boundary / clipping ────────────────────────────────────────────────────────

def test_npa_never_below_zero():
    """Even with zero FOIR and maximum CIBIL, output must be >= 0."""
    assert _npa(0.0, 850, "Salaried") >= 0.0


def test_npa_never_above_one():
    """Unrealistically high FOIR must be clipped at 1.0."""
    assert _npa(5.0, 550, "Self-Employed Non-Professional") <= 1.0


def test_exact_boundary_value_0_65():
    """FOIR=0.8647, CIBIL=600, Non-Salaried yields NPA ≈ 0.65 (boundary)."""
    npa = _npa(0.8647, 600, "Self-Employed Non-Professional")
    assert npa == pytest.approx(0.65, abs=0.0001)


# ── Monotonicity ───────────────────────────────────────────────────────────────

def test_higher_foir_increases_npa():
    assert _npa(0.60, 700, "Salaried") > _npa(0.30, 700, "Salaried")


def test_lower_cibil_increases_npa():
    assert _npa(0.40, 600, "Salaried") > _npa(0.40, 800, "Salaried")


def test_vectorised_monotonicity():
    """NPA must increase as risk profile worsens across a batch."""
    foir = np.array([0.20, 0.45, 0.80])
    cibil = np.array([820, 710, 590])
    emp = ["Salaried", "Salaried", "Self-Employed Non-Professional"]
    npa = calculate_npa_probability(foir, cibil, emp)
    assert npa[0] < npa[1] < npa[2]
