"""
Pytest configuration and fixtures
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.api_data_fetcher import CMFChileAPIFetcher


@pytest.fixture
def api_fetcher():
    """Create CMFChileAPIFetcher instance for testing"""
    config = {
        'currency_map': {
            'usd': {'api': 'dolar', 'column': 'usdclp_obs'},
            'eur': {'api': 'euro', 'column': 'eurclp_obs'},
            'uf': {'api': 'uf', 'column': 'ufclp'}
        }
    }
    return CMFChileAPIFetcher(api_key=None, config=config)
