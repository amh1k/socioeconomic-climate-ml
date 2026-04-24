import pandas as pd
import numpy as np

def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-capita, ratio, interaction, and decade features."""
    df = df.copy()
    eps = 1e-6  # Prevent division by zero
    
    # Per-capita metrics (normalize for population size)
    df['GDP_per_Capita'] = df['GDP'] / (df['Population'] + eps)
    df['CO2_per_Capita'] = df['CO2_Emissions'] / (df['Population'] + eps)
    
    # Energy transition ratio (proportion of renewables in total mix)
    df['Renewable_Ratio'] = df['Renewable_Energy_Usage'] / (
        df['Fossil_Fuel_Usage'] + df['Renewable_Energy_Usage'] + eps
    )
    
    # Policy-economic interaction (policy effectiveness conditional on wealth)
    df['Policy_GDP_Interaction'] = df['Policy_Score'] * df['GDP_per_Capita']
    
    # Decade indicator (captures structural regime shifts)
    df['Decade'] = (df['Year'] // 10) * 10
    
    return df

def add_temporal_features(df: pd.DataFrame, 
                          target_cols: list = None, 
                          window: int = 5, 
                          lags: list = None) -> pd.DataFrame:
    """Add lag and rolling window features per country to capture temporal dynamics."""
    df = df.copy()
    
    if target_cols is None:
        target_cols = ['GDP', 'CO2_Emissions', 'Renewable_Energy_Usage', 'Fossil_Fuel_Usage']
    if lags is None:
        lags = [1, 5]
        
    new_cols = []
    # CRITICAL: Sort before applying time-series operations
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    for col in target_cols:
        # Lag features
        for lag in lags:
            col_name = f'{col}_lag{lag}'
            df[col_name] = df.groupby('Country')[col].shift(lag)
            new_cols.append(col_name)
            
        # Rolling statistics (smooths noise, captures trends/volatility)
        df[f'{col}_roll{window}_mean'] = df.groupby('Country')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{col}_roll{window}_std'] = df.groupby('Country')[col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        new_cols.extend([f'{col}_roll{window}_mean', f'{col}_roll{window}_std'])
        
    # Drop rows where new temporal features are NaN (first `max(lags)` years per country)
    df = df.dropna(subset=new_cols).reset_index(drop=True)
    print(f"Added {len(new_cols)} temporal features. Dropped initial NaN periods.")
    return df

def engineer_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete feature engineering pipeline."""
    print("Starting feature engineering...")
    df = create_base_features(df)
    df = add_temporal_features(df)
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df