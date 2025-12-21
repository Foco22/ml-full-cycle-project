"""
Simple test for data ingestion process - fetch_currency function
"""

import pytest
import pandas as pd
import os
from datetime import datetime
from unittest.mock import Mock, patch
from src.ingestion.api_data_fetcher import CMFChileAPIFetcher


class TestIngestion:
    """Test suite for data ingestion"""

    def test_fetch_currency_usd(self, api_fetcher, monkeypatch):
        """Test fetch_currency for USD"""
        # Mock XML response data
        mock_xml = b"""<?xml version="1.0" encoding="utf-8"?>
        <ObservacionesSeries>
            <Obs>
                <Fecha>2024-01-15</Fecha>
                <Valor>900.50</Valor>
            </Obs>
            <Obs>
                <Fecha>2024-01-16</Fecha>
                <Valor>901.20</Valor>
            </Obs>
        </ObservacionesSeries>"""

        # Mock fetch_data to return mock XML
        monkeypatch.setattr(api_fetcher, 'fetch_data', Mock(return_value=mock_xml))

        # Test fetch_currency
        result = api_fetcher.fetch_currency(
            currency_type='usd',
            year='2024',
            month='01',
            use_posteriores=False
        )

        # Verify result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify column exists
        assert 'usdclp_obs' in result.columns

        # Verify data is present
        assert len(result) == 2

    def test_fetch_currency_invalid_currency(self, api_fetcher):
        """Test fetch_currency with invalid currency type"""
        with pytest.raises(ValueError, match="Invalid currency type"):
            api_fetcher.fetch_currency(currency_type='invalid')

    def test_fetch_currency_default_date(self, api_fetcher, monkeypatch):
        """Test fetch_currency uses current date when not specified"""
        mock_xml = b"""<?xml version="1.0" encoding="utf-8"?>
        <ObservacionesSeries>
            <Obs>
                <Fecha>2024-01-15</Fecha>
                <Valor>1000.00</Valor>
            </Obs>
        </ObservacionesSeries>"""

        monkeypatch.setattr(api_fetcher, 'fetch_data', Mock(return_value=mock_xml))

        # Call without year/month - should use current date
        result = api_fetcher.fetch_currency(currency_type='eur')

        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert 'eurclp_obs' in result.columns


class TestIngestionIntegration:
    """Integration tests using real CMF Chile API"""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv('CMF_API_KEY'),
        reason="CMF_API_KEY environment variable not set"
    )
    def test_fetch_currency_real_api_usd(self):
        """Integration test: fetch USD from real CMF API"""
        # Get API key from environment
        api_key = os.getenv('CMF_API_KEY')

        # Create fetcher without mocking
        fetcher = CMFChileAPIFetcher(api_key=api_key, config={
            'currency_map': {
                'usd': {'api': 'dolar', 'column': 'usdclp_obs'}
            }
        })

        # Call real API for a past month (data should be stable)
        result = fetcher.fetch_currency(
            currency_type='usd',
            year='2024',
            month='01',
            use_posteriores=False
        )

        # Verify structure (not exact values, as data may vary)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "Should return data from API"
        assert 'Fecha' in result.columns
        assert 'usdclp_obs' in result.columns

        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(result['Fecha'])

        # Data might be string or numeric depending on parsing
        # Just verify we have values
        assert result['usdclp_obs'].notna().all(), "Should have no null values"

