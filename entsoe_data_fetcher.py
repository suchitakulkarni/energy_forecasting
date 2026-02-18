"""
ENTSO-E Data Fetcher for Energy Price Forecasting
=================================================

Fetches real electricity market data from ENTSO-E Transparency Platform.

Setup:
1. Register at https://transparency.entsoe.eu/
2. Email transparency@entsoe.eu with subject "Restful API access"
3. Get your API key from your account settings
4. Set environment variable: export ENTSOE_API_KEY="your-key-here"

Install: pip install entsoe-py pandas

Author: Based on entsoe-py library
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

class EntsoeDataFetcher:
    """
    Fetch and process ENTSO-E data for physics-informed price forecasting.
    """
    
    # Country/area codes (EIC codes)
    AREAS = {
        'DE': 'DE_LU',  # Germany-Luxembourg (best for duck curve analysis)
        'ES': 'ES',      # Spain (high renewables)
        'NL': 'NL',      # Netherlands
        'FR': 'FR',      # France (nuclear heavy)
        'DK': 'DK_1',    # Denmark West (extreme wind penetration)
        'GB': 'GB',      # Great Britain
        'NO': 'NO_2',    # Norway (hydro heavy)
    }
    
    def __init__(self, api_key=None):
        """
        Initialize ENTSO-E client.
        
        Args:
            api_key: Your ENTSO-E API key. If None, reads from ENTSOE_API_KEY env variable.
        """
        if api_key is None:
            api_key = os.getenv('ENTSOE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "No API key provided. Either:\n"
                "1. Pass api_key to constructor, or\n"
                "2. Set ENTSOE_API_KEY environment variable\n\n"
                "Get your key: https://transparency.entsoe.eu/"
            )
        
        self.client = EntsoePandasClient(api_key=api_key)
    
    def fetch_complete_dataset(
        self, 
        country='DE',
        start_date='2023-01-01',
        end_date='2024-01-01',
        include_forecasts=True
    ):
        """
        Fetch complete dataset for physics-informed price forecasting.
        
        Args:
            country: Country code (DE, ES, NL, FR, DK, GB, NO)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_forecasts: Include day-ahead forecasts (for feature engineering)
            
        Returns:
            DataFrame with columns:
                - timestamp: datetime index
                - price: day-ahead market price (EUR/MWh)
                - demand: actual load (MW)
                - demand_forecast: day-ahead demand forecast (MW)
                - nuclear: nuclear generation (MW)
                - coal: coal generation (MW)
                - gas: gas generation (MW)
                - wind_onshore: onshore wind generation (MW)
                - wind_offshore: offshore wind generation (MW)
                - solar: solar generation (MW)
                - hydro: hydro generation (MW)
                - total_gen: total generation (MW)
                - residual_demand: demand - (wind + solar)
                - capacity_margin: available capacity - demand
        """
        
        country_code = self.AREAS.get(country, country)
        
        # Convert to pandas timestamps with timezone
        start = pd.Timestamp(start_date, tz='Europe/Brussels')
        end = pd.Timestamp(end_date, tz='Europe/Brussels')
        
        print(f"Fetching data for {country} from {start_date} to {end_date}...")
        
        # Initialize result dataframe
        df = pd.DataFrame()
        
        # 1. PRICES (most important!)
        print("  Fetching day-ahead prices...")
        try:
            prices = self.client.query_day_ahead_prices(country_code, start=start, end=end)
            df['price'] = prices
        except Exception as e:
            print(f"    Warning: Could not fetch prices: {e}")
        
        # 2. LOAD (Demand)
        print("  Fetching actual load...")
        try:
            load = self.client.query_load(country_code, start=start, end=end)
            df['demand'] = load
        except Exception as e:
            print(f"    Warning: Could not fetch load: {e}")
        
        # 3. LOAD FORECAST (for features)
        if include_forecasts:
            print("  Fetching load forecast...")
            try:
                load_forecast = self.client.query_load_forecast(country_code, start=start, end=end)
                df['demand_forecast'] = load_forecast
            except Exception as e:
                print(f"    Warning: Could not fetch load forecast: {e}")
        
        # 4. GENERATION BY TYPE
        print("  Fetching generation by production type...")
        try:
            gen = self.client.query_generation(country_code, start=start, end=end, psr_type=None)
            
            # ENTSO-E returns generation by type - map to our categories
            # Available types: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_psrtype
            
            gen_mapping = {
                'Nuclear': 'nuclear',
                'Fossil Hard coal': 'coal',
                'Fossil Gas': 'gas',
                'Wind Onshore': 'wind_onshore',
                'Wind Offshore': 'wind_offshore',
                'Solar': 'solar',
                'Hydro Run-of-river and poundage': 'hydro',
                'Hydro Water Reservoir': 'hydro_reservoir',
            }
            
            for entsoe_name, our_name in gen_mapping.items():
                if entsoe_name in gen.columns:
                    df[our_name] = gen[entsoe_name]
                else:
                    print(f"    Warning: {entsoe_name} not available")
            
            # Aggregate hydro types
            if 'hydro' in df.columns and 'hydro_reservoir' in df.columns:
                df['hydro_total'] = df['hydro'].fillna(0) + df['hydro_reservoir'].fillna(0)
            
            # Total generation
            gen_cols = [c for c in df.columns if c in ['nuclear', 'coal', 'gas', 'wind_onshore', 
                                                         'wind_offline', 'solar', 'hydro_total']]
            if gen_cols:
                df['total_gen'] = df[gen_cols].sum(axis=1)
                
        except Exception as e:
            print(f"    Warning: Could not fetch generation: {e}")
        
        # 5. INSTALLED CAPACITY (for utilization calculation)
        print("  Fetching installed capacity...")
        try:
            capacity = self.client.query_installed_generation_capacity(
                country_code, start=start, end=end, psr_type=None
            )
            # Store as separate dataframe or compute utilization
            # Capacity is typically yearly data, so we'll use most recent value
            self.capacity_data = capacity
        except Exception as e:
            print(f"    Warning: Could not fetch capacity: {e}")
        
        # 6. CROSS-BORDER FLOWS (optional - for advanced physics)
        # Commented out for speed, but useful for interconnected markets
        # flows = self.client.query_crossborder_flows(country_from, country_to, start, end)
        
        # DERIVED FEATURES (Physics-based)
        print("  Computing derived features...")
        
        # Residual demand (what conventional generators must serve)
        if all(c in df.columns for c in ['demand', 'wind_onshore', 'solar']):
            renewable_gen = df[['wind_onshore', 'solar']].fillna(0).sum(axis=1)
            if 'wind_offshore' in df.columns:
                renewable_gen += df['wind_offshore'].fillna(0)
            df['residual_demand'] = df['demand'] - renewable_gen
        
        # Capacity margin (stress indicator)
        if 'total_gen' in df.columns and 'demand' in df.columns:
            df['capacity_margin'] = df['total_gen'] - df['demand']
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        print(f"Done! Fetched {len(df)} hours of data.")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing data summary:")
        print(df.isnull().sum())
        
        return df
    
    def compute_physics_features(self, df):
        """
        Compute advanced physics-based features.
        
        Args:
            df: DataFrame from fetch_complete_dataset()
            
        Returns:
            DataFrame with additional physics features
        """
        
        df = df.copy()
        
        # 1. RAMP RATES (how fast is demand changing?)
        if 'demand' in df.columns:
            df['demand_ramp_1h'] = df['demand'].diff(1)
            df['demand_ramp_3h'] = df['demand'].diff(3)
        
        # 2. RENEWABLE VOLATILITY
        if 'wind_onshore' in df.columns:
            df['wind_volatility'] = df['wind_onshore'].rolling(24).std()
        
        if 'solar' in df.columns:
            df['solar_volatility'] = df['solar'].rolling(24).std()
        
        # 3. SCARCITY INDICATOR (using capacity data if available)
        if hasattr(self, 'capacity_data') and 'demand' in df.columns:
            total_capacity = self.capacity_data.sum(axis=1).iloc[-1]  # Most recent
            df['utilization'] = df['demand'] / total_capacity
            df['scarcity_indicator'] = (df['utilization'] > 0.85).astype(int)
        
        # 4. DUCK CURVE STRESS (residual demand ramp in evening)
        if 'residual_demand' in df.columns:
            df['residual_ramp'] = df['residual_demand'].diff(1)
            # Evening ramp stress (hour 17-20)
            evening_mask = (df['hour'] >= 17) & (df['hour'] <= 20)
            df['evening_ramp_stress'] = 0
            df.loc[evening_mask, 'evening_ramp_stress'] = df.loc[evening_mask, 'residual_ramp']
        
        # 5. MUST-RUN CONSTRAINTS (e.g., coal minimum stable generation)
        if 'coal' in df.columns:
            # Coal plants typically can't run below 40% of capacity
            df['coal_min_stable'] = df['coal'].rolling(168).max() * 0.4  # Weekly max * 40%
            df['coal_constraint_active'] = (df['coal'] <= df['coal_min_stable'] * 1.1).astype(int)
        
        # 6. RENEWABLE CURTAILMENT RISK (oversupply)
        if 'residual_demand' in df.columns:
            df['curtailment_risk'] = (df['residual_demand'] < 5000).astype(int)  # Arbitrary threshold
        
        return df
    
    def prepare_for_forecasting(self, df, target='price', horizon=24):
        """
        Prepare dataset for ML forecasting (train/test split, lag features, etc.)
        
        Args:
            df: DataFrame from fetch_complete_dataset()
            target: Target variable to forecast
            horizon: Forecast horizon in hours
            
        Returns:
            X_train, y_train, X_test, y_test, feature_names
        """
        
        # Add lagged features
        df = df.copy()
        
        # Price lags
        for lag in [24, 48, 168]:  # 1 day, 2 days, 1 week
            df[f'{target}_lag_{lag}h'] = df[target].shift(lag)
        
        # Demand lags
        if 'demand' in df.columns:
            for lag in [1, 24]:
                df[f'demand_lag_{lag}h'] = df['demand'].shift(lag)
        
        # Drop NaN from lagging
        df = df.dropna()
        
        # Features (exclude target and time index)
        exclude_cols = [target, 'timestamp'] + [c for c in df.columns if 'forecast' in c.lower()]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target]
        
        # Train/test split (e.g., last 30 days as test)
        split_idx = len(df) - 30*24
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        print(f"Train: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        
        return X_train, y_train, X_test, y_test, feature_cols


def example_usage():
    """
    Example: Fetch German data and prepare for forecasting
    """
    
    # Initialize fetcher (make sure ENTSOE_API_KEY is set)
    fetcher = EntsoeDataFetcher()
    
    # Fetch German data (2023)
    df = fetcher.fetch_complete_dataset(
        country='DE',
        start_date='2023-01-01',
        end_date='2024-01-01',
        include_forecasts=True
    )
    
    # Add physics features
    df = fetcher.compute_physics_features(df)
    
    # Save to CSV
    df.to_csv('entsoe_germany_2023.csv')
    print("\nSaved to entsoe_germany_2023.csv")
    
    # Prepare for ML
    X_train, y_train, X_test, y_test, features = fetcher.prepare_for_forecasting(df)
    
    print(f"\nReady for modeling!")
    print(f"Sample features: {features[:10]}")
    
    return df


if __name__ == "__main__":
    # Run example
    df = example_usage()
