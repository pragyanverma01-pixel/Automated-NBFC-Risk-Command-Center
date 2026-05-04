"""
Tests for risk tier classification (High / Medium / Low).

Gap identified: the boundary conditions at 0.45 and 0.65 use strict inequality
(>) which is easy to accidentally flip. These tests lock in the exact semantics.
"""

import numpy as np
import pytest

from src.pipeline import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD, classify_risk


def test_above_high_threshold_is_high_risk():
    result = classify_risk(np.array([0.66, 0.80, 1.00]))
    assert all(r == "High Risk (Intervene)" for r in result)


def test_between_thresholds_is_medium_risk():
    result = classify_risk(np.array([0.46, 0.55, 0.64]))
    assert all(r == "Medium Risk (Monitor)" for r in result)


def test_at_or_below_medium_threshold_is_low_risk():
    result = classify_risk(np.array([0.00, 0.30, RISK_MEDIUM_THRESHOLD]))
    assert all(r == "Low Risk (Safe)" for r in result)


def test_exactly_at_high_threshold_is_medium_not_high():
    """0.65 is NOT strictly > 0.65, so it falls into Medium Risk."""
    result = classify_risk(np.array([RISK_HIGH_THRESHOLD]))
    assert result[0] == "Medium Risk (Monitor)"


def test_exactly_at_medium_threshold_is_low_not_medium():
    """0.45 is NOT strictly > 0.45, so it falls into Low Risk."""
    result = classify_risk(np.array([RISK_MEDIUM_THRESHOLD]))
    assert result[0] == "Low Risk (Safe)"


def test_just_above_high_threshold():
    result = classify_risk(np.array([RISK_HIGH_THRESHOLD + 0.0001]))
    assert result[0] == "High Risk (Intervene)"


def test_just_above_medium_threshold():
    result = classify_risk(np.array([RISK_MEDIUM_THRESHOLD + 0.0001]))
    assert result[0] == "Medium Risk (Monitor)"


def test_only_three_valid_categories():
    npa = np.linspace(0, 1, 200)
    categories = set(classify_risk(npa))
    assert categories == {"High Risk (Intervene)", "Medium Risk (Monitor)", "Low Risk (Safe)"}
