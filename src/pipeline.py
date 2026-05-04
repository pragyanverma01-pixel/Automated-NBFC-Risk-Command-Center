import sqlite3

import numpy as np
import pandas as pd

VEHICLE_CATALOG = {
    "Maruti Suzuki Swift (Hatchback)": 750_000,
    "Hyundai Venue (Compact SUV)": 1_100_000,
    "Tata Nexon EV (Electric SUV)": 1_650_000,
    "Mahindra XUV700 (Premium SUV)": 2_300_000,
    "Toyota Innova Hycross (Hybrid MPV)": 2_900_000,
    "Kia Seltos (Mid-Size SUV)": 1_750_000,
    "Honda City (Premium Sedan)": 1_500_000,
}

VEHICLE_PROBS = [0.25, 0.15, 0.15, 0.12, 0.08, 0.15, 0.10]

EMPLOYMENT_TYPES = [
    "Salaried",
    "Self-Employed Professional",
    "Self-Employed Non-Professional",
]
EMPLOYMENT_PROBS = [0.60, 0.25, 0.15]

RISK_HIGH_THRESHOLD = 0.65
RISK_MEDIUM_THRESHOLD = 0.45

ACTIONABLE_COLUMNS = [
    "CustomerID",
    "Vehicle_Model",
    "Loan_Amount_INR",
    "NPA_Probability",
    "FOIR_Pct",
    "CIBIL_Score",
]


def calculate_emi(loan_amounts, annual_rates_pct, tenure_months):
    """Standard reducing-balance EMI: P·r·(1+r)^n / ((1+r)^n - 1)."""
    loan_amounts = np.asarray(loan_amounts, dtype=float)
    annual_rates_pct = np.asarray(annual_rates_pct, dtype=float)
    tenure_months = np.asarray(tenure_months, dtype=float)
    r = (annual_rates_pct / 100) / 12
    factor = (1 + r) ** tenure_months
    return (loan_amounts * r * factor) / (factor - 1)


def calculate_foir(emis, annual_incomes):
    """Fixed Obligation to Income Ratio = monthly_emi / monthly_income."""
    return (np.asarray(emis, dtype=float) / (np.asarray(annual_incomes, dtype=float) / 12)).round(3)


def assign_interest_rates(cibil_scores, rng=None):
    """
    CIBIL-tier-based rate assignment:
      >= 780 → 8.5–9.2 %
      >= 720 → 9.5–10.5 %
      >= 650 → 11.0–12.5 %
         else → 13.0–15.5 %
    Draws exactly n samples by scaling one uniform draw per record to its tier range.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    cibil_scores = np.asarray(cibil_scores, dtype=int)
    n = len(cibil_scores)

    tier = np.where(cibil_scores >= 780, 0,
           np.where(cibil_scores >= 720, 1,
           np.where(cibil_scores >= 650, 2, 3)))

    lows  = np.array([8.5,  9.5, 11.0, 13.0])
    highs = np.array([9.2, 10.5, 12.5, 15.5])

    u = rng.uniform(0.0, 1.0, n)
    return (lows[tier] + u * (highs[tier] - lows[tier])).round(2)


def calculate_npa_probability(foir_pct, cibil_scores, employment_types):
    """
    Weighted NPA score (clipped to [0, 1]):
      FOIR component      : foir * 0.5
      CIBIL deviation     : (850 - cibil) / 850 * 0.4
      Employment volatility: +0.1 for non-salaried
    """
    foir_pct = np.asarray(foir_pct, dtype=float)
    cibil_scores = np.asarray(cibil_scores, dtype=float)
    employment_risk = np.where(np.asarray(employment_types) != "Salaried", 0.1, 0.0)
    return (
        (foir_pct * 0.5)
        + ((850 - cibil_scores) / 850 * 0.4)
        + employment_risk
    ).clip(0, 1).round(4)


def classify_risk(npa_probability):
    """Map NPA probability to a categorical risk tier."""
    npa = np.asarray(npa_probability, dtype=float)
    return np.where(
        npa > RISK_HIGH_THRESHOLD, "High Risk (Intervene)",
        np.where(npa > RISK_MEDIUM_THRESHOLD, "Medium Risk (Monitor)", "Low Risk (Safe)"),
    )


def extract_hitlist(df_master):
    """Return actionable high-risk records sorted by descending NPA probability."""
    mask = df_master["Risk_Category"] == "High Risk (Intervene)"
    return (
        df_master[mask]
        .sort_values("NPA_Probability", ascending=False)
        [ACTIONABLE_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )


def generate_portfolio(num_records=1500, seed=42):
    """Synthesise a full NBFC loan portfolio with NPA scores and risk labels."""
    rng = np.random.default_rng(seed)

    cibil_scores = rng.normal(730, 45, num_records).clip(550, 850).astype(int)
    annual_incomes = rng.lognormal(mean=14.6, sigma=0.55, size=num_records).clip(450_000, 3_800_000).astype(int)
    employment = rng.choice(EMPLOYMENT_TYPES, num_records, p=EMPLOYMENT_PROBS)

    vehicle_models_list = list(VEHICLE_CATALOG.keys())
    chosen_vehicles = rng.choice(vehicle_models_list, num_records, p=VEHICLE_PROBS)
    vehicle_prices = np.array([VEHICLE_CATALOG[v] for v in chosen_vehicles])

    ltv_ratios = rng.uniform(0.75, 0.95, num_records)
    loan_amounts = (vehicle_prices * ltv_ratios).astype(int)

    rates = assign_interest_rates(cibil_scores, rng=rng)
    tenure_months = rng.choice([36, 48, 60, 72, 84], num_records, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    emis = calculate_emi(loan_amounts, rates, tenure_months)
    foir_pct = calculate_foir(emis, annual_incomes)
    npa_probability = calculate_npa_probability(foir_pct, cibil_scores, employment)
    risk_category = classify_risk(npa_probability)

    return pd.DataFrame({
        "LoanID": range(900_001, 900_001 + num_records),
        "CustomerID": range(100_001, 100_001 + num_records),
        "CIBIL_Score": cibil_scores,
        "Annual_Income_INR": annual_incomes,
        "Employment_Type": employment,
        "Vehicle_Model": chosen_vehicles,
        "Loan_Amount_INR": loan_amounts,
        "Interest_Rate_Pct": rates,
        "Tenure_Months": tenure_months,
        "Monthly_EMI_INR": emis.astype(int),
        "FOIR_Pct": foir_pct,
        "NPA_Probability": npa_probability,
        "Risk_Category": risk_category,
    })


def save_to_database(df_master, df_hitlist, db_path):
    """Persist master portfolio and hitlist tables to a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        df_master.to_sql("Portfolio_Master", conn, if_exists="replace", index=False)
        df_hitlist.to_sql("Intervention_Hitlist", conn, if_exists="replace", index=False)
