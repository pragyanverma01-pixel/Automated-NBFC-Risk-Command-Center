"""
Tests for stochastic sampling distributions.

Gap identified: generate_portfolio specifies employment and vehicle probabilities
explicitly. A swapped probability array would change business metrics silently —
e.g. more self-employed borrowers → inflated portfolio-level NPA risk.
A large-sample chi-square–style tolerance check catches this class of bug.
"""

import numpy as np
import pytest

from src.pipeline import (
    EMPLOYMENT_PROBS,
    EMPLOYMENT_TYPES,
    VEHICLE_CATALOG,
    VEHICLE_PROBS,
    generate_portfolio,
)

# Use a large sample so ±5 pp tolerance is reliable without being flaky.
_N = 5_000
_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def large_portfolio():
    return generate_portfolio(num_records=_N, seed=0)


# ── Employment distribution ────────────────────────────────────────────────────

def test_salaried_proportion(large_portfolio):
    actual = (large_portfolio["Employment_Type"] == "Salaried").mean()
    assert abs(actual - EMPLOYMENT_PROBS[0]) < _TOLERANCE


def test_self_employed_professional_proportion(large_portfolio):
    actual = (large_portfolio["Employment_Type"] == "Self-Employed Professional").mean()
    assert abs(actual - EMPLOYMENT_PROBS[1]) < _TOLERANCE


def test_self_employed_non_professional_proportion(large_portfolio):
    actual = (large_portfolio["Employment_Type"] == "Self-Employed Non-Professional").mean()
    assert abs(actual - EMPLOYMENT_PROBS[2]) < _TOLERANCE


def test_employment_proportions_sum_to_one(large_portfolio):
    counts = large_portfolio["Employment_Type"].value_counts(normalize=True)
    assert abs(counts.sum() - 1.0) < 1e-9


# ── Vehicle distribution ───────────────────────────────────────────────────────

@pytest.mark.parametrize("vehicle, expected_prob", zip(VEHICLE_CATALOG.keys(), VEHICLE_PROBS))
def test_vehicle_proportion(large_portfolio, vehicle, expected_prob):
    actual = (large_portfolio["Vehicle_Model"] == vehicle).mean()
    assert abs(actual - expected_prob) < _TOLERANCE, (
        f"{vehicle}: expected ~{expected_prob:.2f}, got {actual:.3f}"
    )


def test_vehicle_proportions_sum_to_one(large_portfolio):
    counts = large_portfolio["Vehicle_Model"].value_counts(normalize=True)
    assert abs(counts.sum() - 1.0) < 1e-9


# ── CIBIL / income distribution sanity ────────────────────────────────────────

def test_cibil_mean_near_730(large_portfolio):
    """Drawn from N(730, 45) clipped to [550, 850]; mean should be close to 730."""
    assert abs(large_portfolio["CIBIL_Score"].mean() - 730) < 10


def test_median_annual_income_reasonable(large_portfolio):
    """Log-normal distribution; median should be between 500 k and 3 M INR."""
    median = large_portfolio["Annual_Income_INR"].median()
    assert 500_000 < median < 3_000_000
