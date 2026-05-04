"""
Tests for the EMI (Equated Monthly Instalment) calculation.

Gap identified: the core financial formula had no validation, meaning a wrong
exponent or sign flip would go undetected until downstream CSV outputs diverged.
"""

import numpy as np
import pytest

from src.pipeline import calculate_emi


def test_known_value():
    """Validate against an independently computed reference value."""
    # 10 L loan, 10 % p.a., 60-month tenure → ~21 247 INR/month
    emi = calculate_emi(1_000_000, 10.0, 60)
    assert abs(emi - 21_247.04) < 1.0


def test_shorter_tenure_means_higher_emi():
    assert calculate_emi(500_000, 12.0, 24) > calculate_emi(500_000, 12.0, 60)


def test_higher_rate_means_higher_emi():
    assert calculate_emi(1_000_000, 15.5, 60) > calculate_emi(1_000_000, 8.5, 60)


def test_higher_principal_means_higher_emi():
    assert calculate_emi(1_000_000, 10.0, 60) > calculate_emi(500_000, 10.0, 60)


def test_emi_is_always_positive():
    emi = calculate_emi(750_000, 9.0, 48)
    assert emi > 0


def test_vectorised_input():
    loans = np.array([500_000, 1_000_000, 2_000_000])
    rates = np.array([8.5, 10.0, 12.5])
    tenures = np.array([36, 60, 84])
    emis = calculate_emi(loans, rates, tenures)
    assert emis.shape == (3,)
    assert np.all(emis > 0)


def test_emi_proportional_to_principal():
    """Doubling the principal should exactly double the EMI."""
    emi_base = calculate_emi(1_000_000, 10.0, 60)
    emi_double = calculate_emi(2_000_000, 10.0, 60)
    assert abs(emi_double / emi_base - 2.0) < 1e-9
