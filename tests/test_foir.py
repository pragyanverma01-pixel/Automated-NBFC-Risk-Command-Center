"""
Tests for FOIR (Fixed Obligation to Income Ratio) calculation.

Gap identified: FOIR feeds directly into the NPA score with a weight of 0.5,
so an off-by-12 error (annual vs monthly income mix-up) would double every risk score.
"""

import numpy as np
import pytest

from src.pipeline import calculate_foir


def test_foir_exact_value():
    """EMI 10 000 / monthly income 10 000 → FOIR = 1.0."""
    foir = calculate_foir(np.array([10_000.0]), np.array([120_000]))
    assert foir[0] == pytest.approx(1.0, abs=0.001)


def test_foir_quarter_income():
    """EMI = 25 % of monthly income."""
    foir = calculate_foir(np.array([5_000.0]), np.array([240_000]))
    assert foir[0] == pytest.approx(0.25, abs=0.001)


def test_foir_uses_monthly_not_annual_income():
    """If annual income were used directly FOIR would be 12× too small."""
    emi = np.array([20_000.0])
    annual = np.array([960_000])          # monthly = 80 000
    foir = calculate_foir(emi, annual)
    assert foir[0] == pytest.approx(0.25, abs=0.001)   # not 0.0208


def test_higher_income_yields_lower_foir():
    emi = np.array([20_000.0])
    assert calculate_foir(emi, np.array([600_000]))[0] > calculate_foir(emi, np.array([1_200_000]))[0]


def test_higher_emi_yields_higher_foir():
    income = np.array([600_000])
    assert calculate_foir(np.array([25_000.0]), income)[0] > calculate_foir(np.array([15_000.0]), income)[0]


def test_foir_always_positive():
    foir = calculate_foir(np.array([15_000.0, 25_000.0]), np.array([600_000, 1_200_000]))
    assert np.all(foir > 0)


def test_foir_rounded_to_3_decimal_places():
    foir = calculate_foir(np.array([7_777.77]), np.array([333_333]))
    assert foir[0] == round(foir[0], 3)
