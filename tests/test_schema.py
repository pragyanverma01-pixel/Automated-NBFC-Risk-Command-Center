"""
Tests for the generated portfolio's data schema and value distributions.

Gap identified: the data generation phase uses stochastic sampling with hard
business constraints (e.g. CIBIL 550–850). Without schema tests, a changed clip
or dtype would corrupt all downstream calculations silently.
"""

import pytest

from src.pipeline import generate_portfolio

REQUIRED_COLUMNS = [
    "LoanID", "CustomerID", "CIBIL_Score", "Annual_Income_INR",
    "Employment_Type", "Vehicle_Model", "Loan_Amount_INR",
    "Interest_Rate_Pct", "Tenure_Months", "Monthly_EMI_INR",
    "FOIR_Pct", "NPA_Probability", "Risk_Category",
]

VALID_EMPLOYMENT_TYPES = {"Salaried", "Self-Employed Professional", "Self-Employed Non-Professional"}
VALID_TENURES = {36, 48, 60, 72, 84}
VALID_RISK_CATEGORIES = {"High Risk (Intervene)", "Medium Risk (Monitor)", "Low Risk (Safe)"}


@pytest.fixture(scope="module")
def portfolio():
    return generate_portfolio(num_records=300, seed=0)


# ── Column / dtype presence ────────────────────────────────────────────────────

def test_all_required_columns_present(portfolio):
    for col in REQUIRED_COLUMNS:
        assert col in portfolio.columns, f"Missing column: {col}"


def test_no_null_values(portfolio):
    null_counts = portfolio.isnull().sum()
    assert null_counts.sum() == 0, f"Null values found:\n{null_counts[null_counts > 0]}"


# ── Record count ───────────────────────────────────────────────────────────────

def test_correct_record_count():
    df = generate_portfolio(num_records=123, seed=1)
    assert len(df) == 123


# ── Value range constraints ────────────────────────────────────────────────────

def test_cibil_score_within_550_850(portfolio):
    assert portfolio["CIBIL_Score"].between(550, 850).all()


def test_npa_probability_within_0_1(portfolio):
    assert portfolio["NPA_Probability"].between(0.0, 1.0).all()


def test_foir_is_positive(portfolio):
    assert (portfolio["FOIR_Pct"] > 0).all()


def test_loan_amounts_are_positive(portfolio):
    assert (portfolio["Loan_Amount_INR"] > 0).all()


def test_interest_rates_within_valid_range(portfolio):
    assert portfolio["Interest_Rate_Pct"].between(8.5, 15.5).all()


def test_annual_incomes_within_valid_range(portfolio):
    assert portfolio["Annual_Income_INR"].between(450_000, 3_800_000).all()


# ── Categorical constraints ────────────────────────────────────────────────────

def test_tenure_values_are_valid(portfolio):
    assert set(portfolio["Tenure_Months"].unique()).issubset(VALID_TENURES)


def test_employment_types_are_valid(portfolio):
    assert set(portfolio["Employment_Type"].unique()).issubset(VALID_EMPLOYMENT_TYPES)


def test_risk_categories_are_valid(portfolio):
    assert set(portfolio["Risk_Category"].unique()).issubset(VALID_RISK_CATEGORIES)


# ── Uniqueness ─────────────────────────────────────────────────────────────────

def test_loan_ids_are_unique(portfolio):
    assert portfolio["LoanID"].is_unique


def test_customer_ids_are_unique(portfolio):
    assert portfolio["CustomerID"].is_unique


# ── Reproducibility ────────────────────────────────────────────────────────────

def test_same_seed_produces_identical_output():
    df1 = generate_portfolio(num_records=100, seed=7)
    df2 = generate_portfolio(num_records=100, seed=7)
    assert df1.equals(df2)


def test_different_seeds_produce_different_output():
    df1 = generate_portfolio(num_records=100, seed=1)
    df2 = generate_portfolio(num_records=100, seed=2)
    assert not df1.equals(df2)


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_single_record_portfolio():
    df = generate_portfolio(num_records=1, seed=0)
    assert len(df) == 1
    assert df.isnull().sum().sum() == 0
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_zero_records_portfolio():
    df = generate_portfolio(num_records=0, seed=0)
    assert len(df) == 0
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
