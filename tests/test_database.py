"""
Tests for SQLite persistence (save_to_database).

Gap identified: the original notebook used a bare sqlite3.connect() call with no
context manager, risking unclosed connections and partial writes on failure.
The refactored function uses a context manager; these tests verify idempotency,
row counts, and schema integrity.
"""

import os
import sqlite3

import pandas as pd
import pytest

from src.pipeline import extract_hitlist, generate_portfolio, save_to_database


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_nbfc.db")


@pytest.fixture(scope="module")
def small_portfolio():
    return generate_portfolio(num_records=50, seed=7)


@pytest.fixture(scope="module")
def small_hitlist(small_portfolio):
    return extract_hitlist(small_portfolio)


# ── File creation ──────────────────────────────────────────────────────────────

def test_database_file_is_created(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    assert os.path.exists(db_path)


# ── Table existence ────────────────────────────────────────────────────────────

def test_portfolio_master_table_exists(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    assert "Portfolio_Master" in tables


def test_intervention_hitlist_table_exists(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    assert "Intervention_Hitlist" in tables


# ── Row count integrity ────────────────────────────────────────────────────────

def test_portfolio_row_count_matches(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM Portfolio_Master", conn)["n"][0]
    assert count == len(small_portfolio)


def test_hitlist_row_count_matches(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM Intervention_Hitlist", conn)["n"][0]
    assert count == len(small_hitlist)


# ── Schema integrity ───────────────────────────────────────────────────────────

def test_portfolio_columns_match_dataframe(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        db_cols = set(pd.read_sql("SELECT * FROM Portfolio_Master LIMIT 1", conn).columns)
    assert db_cols == set(small_portfolio.columns)


def test_hitlist_columns_match_dataframe(db_path, small_portfolio, small_hitlist):
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        db_cols = set(pd.read_sql("SELECT * FROM Intervention_Hitlist LIMIT 1", conn).columns)
    assert db_cols == set(small_hitlist.columns)


# ── Idempotency ────────────────────────────────────────────────────────────────

def test_saving_twice_does_not_duplicate_rows(db_path, small_portfolio, small_hitlist):
    """if_exists='replace' must overwrite, not append."""
    save_to_database(small_portfolio, small_hitlist, db_path)
    save_to_database(small_portfolio, small_hitlist, db_path)
    with sqlite3.connect(db_path) as conn:
        count = pd.read_sql("SELECT COUNT(*) AS n FROM Portfolio_Master", conn)["n"][0]
    assert count == len(small_portfolio)
