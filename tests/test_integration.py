"""
End-to-end integration test: generate → extract → persist → query.

Gap identified: unit tests validate each function in isolation, but a wiring
bug at the seam between stages (e.g. wrong column name passed to extract_hitlist)
would only surface when the full pipeline runs.
"""

import sqlite3

import pandas as pd
import pytest

from src.pipeline import (
    ACTIONABLE_COLUMNS,
    extract_hitlist,
    generate_portfolio,
    save_to_database,
)


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("integration")
    df_master = generate_portfolio(num_records=300, seed=42)
    df_hitlist = extract_hitlist(df_master)
    db_path = str(tmp / "nbfc.db")
    save_to_database(df_master, df_hitlist, db_path)
    return df_master, df_hitlist, db_path


def test_db_master_row_count_matches_dataframe(pipeline_output):
    df_master, _, db_path = pipeline_output
    with sqlite3.connect(db_path) as conn:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM Portfolio_Master", conn)["n"][0]
    assert count == len(df_master)


def test_db_hitlist_row_count_matches_dataframe(pipeline_output):
    _, df_hitlist, db_path = pipeline_output
    with sqlite3.connect(db_path) as conn:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM Intervention_Hitlist", conn)["n"][0]
    assert count == len(df_hitlist)


def test_hitlist_customer_ids_are_subset_of_master(pipeline_output):
    df_master, df_hitlist, _ = pipeline_output
    assert set(df_hitlist["CustomerID"]).issubset(set(df_master["CustomerID"]))


def test_all_hitlist_records_are_high_risk_in_master(pipeline_output):
    df_master, df_hitlist, _ = pipeline_output
    matched = df_master[df_master["CustomerID"].isin(df_hitlist["CustomerID"])]
    assert (matched["Risk_Category"] == "High Risk (Intervene)").all()


def test_no_low_or_medium_risk_in_hitlist(pipeline_output):
    _, df_hitlist, db_path = pipeline_output
    with sqlite3.connect(db_path) as conn:
        db_hitlist = pd.read_sql("SELECT * FROM Intervention_Hitlist", conn)
    # Hitlist only has ACTIONABLE_COLUMNS — verify all CustomerIDs are high-risk
    df_master, _, _ = pipeline_output
    matched = df_master[df_master["CustomerID"].isin(db_hitlist["CustomerID"])]
    assert not (matched["Risk_Category"].isin(["Low Risk (Safe)", "Medium Risk (Monitor)"])).any()


def test_db_hitlist_has_correct_columns(pipeline_output):
    _, _, db_path = pipeline_output
    with sqlite3.connect(db_path) as conn:
        db_hitlist = pd.read_sql("SELECT * FROM Intervention_Hitlist LIMIT 1", conn)
    assert set(db_hitlist.columns) == set(ACTIONABLE_COLUMNS)


def test_hitlist_npa_values_match_between_df_and_db(pipeline_output):
    """NPA values written to DB must exactly round-trip back."""
    _, df_hitlist, db_path = pipeline_output
    with sqlite3.connect(db_path) as conn:
        db_hitlist = pd.read_sql(
            "SELECT CustomerID, NPA_Probability FROM Intervention_Hitlist ORDER BY CustomerID",
            conn,
        )
    df_sorted = df_hitlist.sort_values("CustomerID")[["CustomerID", "NPA_Probability"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_sorted, db_hitlist, check_dtype=False)
