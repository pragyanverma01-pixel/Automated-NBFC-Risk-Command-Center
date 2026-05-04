"""
Tests for CIBIL-tier-based interest rate assignment.

Gap identified: the four-tier rate ladder uses nested np.where. A misplaced
threshold (e.g. 720 vs 270) would silently misprice every loan.
"""

import numpy as np
import pytest

from src.pipeline import assign_interest_rates


def _rates(scores, seed=0):
    return assign_interest_rates(np.array(scores), rng=np.random.default_rng(seed))


# ── Tier range membership ──────────────────────────────────────────────────────

def test_premium_cibil_gets_lowest_rate():
    rates = _rates([800, 790, 785])
    assert np.all((rates >= 8.5) & (rates <= 9.2))


def test_good_cibil_gets_second_tier_rate():
    rates = _rates([750, 730, 720])
    assert np.all((rates >= 9.5) & (rates <= 10.5))


def test_fair_cibil_gets_third_tier_rate():
    rates = _rates([680, 660, 650])
    assert np.all((rates >= 11.0) & (rates <= 12.5))


def test_poor_cibil_gets_highest_rate():
    rates = _rates([640, 600, 560])
    assert np.all((rates >= 13.0) & (rates <= 15.5))


# ── Exact boundary conditions ──────────────────────────────────────────────────

def test_boundary_780_is_premium():
    rate = _rates([780])[0]
    assert 8.5 <= rate <= 9.2


def test_boundary_779_is_second_tier():
    rate = _rates([779])[0]
    assert 9.5 <= rate <= 10.5


def test_boundary_720_is_second_tier():
    rate = _rates([720])[0]
    assert 9.5 <= rate <= 10.5


def test_boundary_719_is_third_tier():
    rate = _rates([719])[0]
    assert 11.0 <= rate <= 12.5


def test_boundary_650_is_third_tier():
    rate = _rates([650])[0]
    assert 11.0 <= rate <= 12.5


def test_boundary_649_is_highest_tier():
    rate = _rates([649])[0]
    assert 13.0 <= rate <= 15.5


# ── Output properties ──────────────────────────────────────────────────────────

def test_rates_rounded_to_two_decimals():
    rates = _rates([780, 720, 650, 600])
    assert np.all(rates == np.round(rates, 2))


def test_higher_cibil_never_gets_higher_rate_than_lower_cibil():
    """A borrower with a better CIBIL score must not pay more than one with worse."""
    rate_good = _rates([780])[0]
    rate_poor = _rates([560])[0]
    assert rate_good < rate_poor
