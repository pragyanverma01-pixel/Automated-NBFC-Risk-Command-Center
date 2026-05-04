"""
Tests for the Intervention Hitlist extraction logic.

Gap identified: the collections team acts directly on this output. A bug that
includes low-risk customers or drops high-risk ones has real operational consequences.
"""

import pandas as pd
import pytest

from src.pipeline import ACTIONABLE_COLUMNS, extract_hitlist


@pytest.fixture
def mixed_portfolio():
    return pd.DataFrame({
        "LoanID": [1, 2, 3, 4, 5],
        "CustomerID": [101, 102, 103, 104, 105],
        "CIBIL_Score": [600, 750, 800, 580, 720],
        "Annual_Income_INR": [500_000, 800_000, 1_200_000, 450_000, 900_000],
        "Employment_Type": ["Salaried", "Salaried", "Salaried", "Self-Employed Non-Professional", "Salaried"],
        "Vehicle_Model": ["Model A", "Model B", "Model C", "Model D", "Model E"],
        "Loan_Amount_INR": [700_000, 900_000, 1_200_000, 650_000, 1_100_000],
        "Interest_Rate_Pct": [12.0, 10.0, 9.0, 14.0, 10.5],
        "Tenure_Months": [60, 48, 60, 72, 60],
        "Monthly_EMI_INR": [15_000, 20_000, 25_000, 14_000, 24_000],
        "FOIR_Pct": [0.36, 0.30, 0.25, 0.37, 0.32],
        "NPA_Probability": [0.70, 0.35, 0.20, 0.68, 0.40],
        "Risk_Category": [
            "High Risk (Intervene)",
            "Low Risk (Safe)",
            "Low Risk (Safe)",
            "High Risk (Intervene)",
            "Low Risk (Safe)",
        ],
    })


def test_only_high_risk_records_included(mixed_portfolio):
    hitlist = extract_hitlist(mixed_portfolio)
    assert len(hitlist) == 2
    assert set(hitlist["CustomerID"]) == {101, 104}


def test_sorted_by_npa_probability_descending(mixed_portfolio):
    hitlist = extract_hitlist(mixed_portfolio)
    probabilities = hitlist["NPA_Probability"].tolist()
    assert probabilities == sorted(probabilities, reverse=True)


def test_contains_only_actionable_columns(mixed_portfolio):
    hitlist = extract_hitlist(mixed_portfolio)
    assert list(hitlist.columns) == ACTIONABLE_COLUMNS


def test_no_internal_index_leakage(mixed_portfolio):
    hitlist = extract_hitlist(mixed_portfolio)
    assert list(hitlist.index) == list(range(len(hitlist)))


def test_empty_hitlist_when_no_high_risk_records():
    df = pd.DataFrame({
        "LoanID": [1, 2],
        "CustomerID": [101, 102],
        "CIBIL_Score": [800, 750],
        "Annual_Income_INR": [1_200_000, 900_000],
        "Employment_Type": ["Salaried", "Salaried"],
        "Vehicle_Model": ["Model A", "Model B"],
        "Loan_Amount_INR": [800_000, 600_000],
        "Interest_Rate_Pct": [9.0, 10.0],
        "Tenure_Months": [60, 48],
        "Monthly_EMI_INR": [18_000, 14_000],
        "FOIR_Pct": [0.18, 0.19],
        "NPA_Probability": [0.20, 0.30],
        "Risk_Category": ["Low Risk (Safe)", "Low Risk (Safe)"],
    })
    hitlist = extract_hitlist(df)
    assert len(hitlist) == 0
    assert list(hitlist.columns) == ACTIONABLE_COLUMNS


def test_all_records_included_when_all_are_high_risk():
    df = pd.DataFrame({
        "LoanID": [1, 2, 3],
        "CustomerID": [101, 102, 103],
        "CIBIL_Score": [570, 580, 590],
        "Annual_Income_INR": [450_000, 460_000, 480_000],
        "Employment_Type": ["Self-Employed Non-Professional"] * 3,
        "Vehicle_Model": ["A", "B", "C"],
        "Loan_Amount_INR": [700_000, 680_000, 660_000],
        "Interest_Rate_Pct": [14.0, 14.0, 14.0],
        "Tenure_Months": [72, 72, 72],
        "Monthly_EMI_INR": [16_000, 15_500, 15_000],
        "FOIR_Pct": [0.89, 0.81, 0.75],
        "NPA_Probability": [0.90, 0.80, 0.71],
        "Risk_Category": ["High Risk (Intervene)"] * 3,
    })
    hitlist = extract_hitlist(df)
    assert len(hitlist) == 3
