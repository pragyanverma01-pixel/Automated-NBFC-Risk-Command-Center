import pytest

from src.pipeline import extract_hitlist, generate_portfolio


@pytest.fixture(scope="session")
def portfolio():
    return generate_portfolio(num_records=500, seed=42)


@pytest.fixture(scope="session")
def hitlist(portfolio):
    return extract_hitlist(portfolio)
