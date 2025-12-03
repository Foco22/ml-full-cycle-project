"""
Generic API Data Fetcher
Base class for fetching data from any API source
"""

import urllib.request
import ssl
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import logging
from abc import ABC, abstractmethod


class BaseAPIFetcher(ABC):
    """Base class for API data fetchers"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fetcher

        Args:
            api_key: API key for authentication
            config: Additional configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.context = ssl._create_unverified_context()
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def fetch_data(self, url: str) -> bytes:
        """
        Fetch data from URL

        Args:
            url: URL to fetch

        Returns:
            Response content as bytes
        """
        try:
            request = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(request, context=self.context) as response:
                return response.read()
        except Exception as e:
            self.logger.error(f"Error fetching data from {url}: {str(e)}")
            raise

    @abstractmethod
    def build_url(self, **kwargs) -> str:
        """
        Build API URL with parameters
        Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def parse_response(self, response_data: bytes, **kwargs) -> pd.DataFrame:
        """
        Parse API response into DataFrame
        Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Main fetch method
        Must be implemented by subclasses
        """
        pass

    def prepare_for_bigquery(self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Prepare DataFrame for BigQuery insertion
        Can be overridden by subclasses for custom preparation

        Args:
            df: Input DataFrame
            metadata: Optional metadata to add as columns

        Returns:
            DataFrame formatted for BigQuery
        """
        df_prepared = df.copy()

        # Add standard metadata columns
        df_prepared['ingestion_timestamp'] = datetime.now()
        df_prepared['data_source'] = self.__class__.__name__

        # Add custom metadata if provided
        if metadata:
            for key, value in metadata.items():
                df_prepared[key] = value

        return df_prepared


class CMFChileAPIFetcher(BaseAPIFetcher):
    """
    Fetcher for CMF Chile API (Exchange Rates)
    Specific implementation for exchange rate data
    """

    BASE_URL = "https://api.cmfchile.cl/api-sbifv3/recursos_api"

    def build_url(self, currency: str, year: str, month: str) -> str:
        """Build API URL for CMF Chile"""
        return f"{self.BASE_URL}/{currency}/{year}/{month}?apikey={self.api_key}&formato=xml"

    def parse_response(self, response_data: bytes, currency_code: str = 'valor') -> pd.DataFrame:
        """Parse XML response from CMF Chile API"""
        soup = BeautifulSoup(response_data, features="xml")

        fechas = soup.find_all('Fecha')
        valores = soup.find_all('Valor')

        data = []
        for i in range(len(fechas)):
            rows = [fechas[i].get_text(), valores[i].get_text()]
            data.append(rows)

        df = pd.DataFrame(data, columns=['Fecha', currency_code])

        # Convert Fecha to datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')

        return df

    def fetch_currency(
        self,
        currency_type: str,
        year: Optional[str] = None,
        month: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a specific currency

        Args:
            currency_type: Currency code (e.g., 'usd', 'eur', 'uf')
            year: Year to fetch (defaults to current year)
            month: Month to fetch (defaults to current month)

        Returns:
            DataFrame with currency data
        """
        # Currency mapping (can be configured via config)
        currency_map = self.config.get('currency_map', {
            'usd': {'api': 'dolar', 'column': 'usdclp_obs'},
            'eur': {'api': 'euro', 'column': 'eurclp_obs'},
            'uf': {'api': 'uf', 'column': 'ufclp'}
        })

        if currency_type not in currency_map:
            raise ValueError(f"Invalid currency type: {currency_type}")

        # Use current date if not specified
        if year is None or month is None:
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m")

        api_currency = currency_map[currency_type]['api']
        url = self.build_url(api_currency, year, month)

        self.logger.info(f"Fetching {currency_type.upper()} data for {year}-{month}")

        # Fetch and parse data
        xml_data = self.fetch_data(url)
        df = self.parse_response(xml_data, currency_code='Valor')

        # Rename column to match currency
        df.rename(columns={'Valor': currency_map[currency_type]['column']}, inplace=True)

        self.logger.info(f"Fetched {len(df)} records for {currency_type.upper()}")

        return df

    def fetch(
        self,
        currencies: Optional[List[str]] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        merge: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for multiple currencies

        Args:
            currencies: List of currencies to fetch (default: ['usd', 'eur', 'uf'])
            year: Year to fetch
            month: Month to fetch
            merge: If True, merge all currencies into single DataFrame

        Returns:
            DataFrame with all data
        """
        if currencies is None:
            currencies = ['usd', 'eur', 'uf']

        self.logger.info(f"Fetching data for currencies: {currencies}")

        # Fetch each currency
        dfs = {}
        for currency in currencies:
            try:
                dfs[currency] = self.fetch_currency(currency, year, month)
            except Exception as e:
                self.logger.warning(f"Failed to fetch {currency}: {str(e)}")

        if not dfs:
            # No data fetched for any currency
            self.logger.warning("No data fetched for any currency")
            return pd.DataFrame() if merge else {}

        if merge:
            # Merge all currencies on Fecha
            df_merged = list(dfs.values())[0]
            for df in list(dfs.values())[1:]:
                df_merged = df_merged.merge(df, on='Fecha', how='outer')

            # Sort by date
            df_merged = df_merged.sort_values('Fecha').reset_index(drop=True)

            self.logger.info(f"Merged dataset contains {len(df_merged)} records")
            return df_merged
        else:
            return dfs

    def fetch_date_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        currencies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch data for a date range

        Args:
            start_date: Start date
            end_date: End date (defaults to today)
            currencies: List of currencies to fetch

        Returns:
            DataFrame with all data for the date range
        """
        if end_date is None:
            end_date = datetime.now()

        if currencies is None:
            currencies = ['usd', 'eur', 'uf']

        self.logger.info(f"Fetching data from {start_date} to {end_date}")

        all_data = []

        # Iterate through months
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")

            try:
                df_month = self.fetch(currencies=currencies, year=year, month=month, merge=True)
                # Only append if we got valid DataFrame data
                if isinstance(df_month, pd.DataFrame) and not df_month.empty:
                    all_data.append(df_month)
                elif df_month.empty:
                    self.logger.warning(f"No data returned for {year}-{month}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {year}-{month}: {str(e)}")

            # Move to next month
            current_date = (current_date + timedelta(days=32)).replace(day=1)

        # Combine all data
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Fecha']).reset_index(drop=True)

            # Filter to exact date range
            df_combined = df_combined[
                (df_combined['Fecha'] >= start_date) &
                (df_combined['Fecha'] <= end_date)
            ]

            self.logger.info(f"Total records fetched: {len(df_combined)}")
            return df_combined
        else:
            return pd.DataFrame()

    def prepare_for_bigquery(self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Prepare DataFrame for BigQuery insertion

        Args:
            df: Input DataFrame
            metadata: Optional metadata

        Returns:
            DataFrame formatted for BigQuery
        """
        df_prepared = df.copy()

        # Ensure Fecha is datetime
        if 'Fecha' in df_prepared.columns and df_prepared['Fecha'].dtype != 'datetime64[ns]':
            df_prepared['Fecha'] = pd.to_datetime(df_prepared['Fecha'], format='%Y-%m-%d')

        # Convert numeric columns (replace comma with dot)
        for col in df_prepared.columns:
            if col not in ['Fecha', 'ingestion_timestamp', 'data_source']:
                if df_prepared[col].dtype == 'object':
                    df_prepared[col] = df_prepared[col].astype(str).str.replace(',', '.')
                    df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')

        # Add metadata columns
        df_prepared['ingestion_timestamp'] = datetime.now()
        df_prepared['data_source'] = 'CMF_Chile_API'

        if metadata:
            for key, value in metadata.items():
                df_prepared[key] = value

        return df_prepared
