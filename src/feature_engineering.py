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


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to 10 high-value features based on EDA + domain knowledge."""
    keep_cols = [
        'GDP_per_Capita', 'Population', 'Renewable_Ratio', 
        'Fossil_Fuel_Usage', 'Energy_Consumption_Per_Capita',
        'Urbanization', 'Policy_Score', 'CO2_Emissions_lag1',
        'GDP_roll5_mean', 'Decade'
    ]
    
    # Verify all exist
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing}")
        
    print(f"Feature selection: Keeping {len(keep_cols)} features out of {df.shape[1]-2}")
    return df[keep_cols]
def select_features_by_importance(X_train, y_train, n_features=20, random_state=42):
    """
    Select top N features based on Random Forest importance.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training targets (DataFrame, multi-output)
        n_features: Number of features to keep
        random_state: For reproducibility
    
    Returns:
        List of selected feature names
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    # Train a quick RF on full feature set
    rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1)
    )
    rf.fit(X_train, y_train)
    
    # Aggregate importance across both targets
    importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
    
    # Get top N feature names
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    selected = feature_importance.head(n_features)['feature'].tolist()
    print(f"Selected {n_features} features by importance:")
    for i, (feat, imp) in enumerate(zip(feature_importance.head(n_features)['feature'], 
                                       feature_importance.head(n_features)['importance']), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    return selected



def select_socioeconomic_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to retain ONLY socioeconomic, demographic, and policy features.
    Excludes environmental, climate, physical, and energy-system variables.
    """
    # Core socioeconomic/demographic/policy features
    base_cols = [
        'Population', 'GDP', 'Urbanization', 'Policy_Score',
        'Energy_Consumption_Per_Capita', 'Waste_Management',
        'Industrial_Activity', 'GDP_per_Capita', 'Policy_GDP_Interaction', 'Decade'
    ]
    base_cols.extend(['Renewable_Energy_Usage', 'Fossil_Fuel_Usage', 'Solar_Energy_Potential'])
    
    # Include lagged/rolling variants of purely economic/demographic columns
    temporal_bases = ['GDP', 'Population', 'Energy_Consumption_Per_Capita', 'Policy_Score','Renewable_Energy_Usage', 'Fossil_Fuel_Usage']
    suffixes = ['_lag1', '_lag5', '_roll5_mean', '_roll5_std']
    
    all_socio_cols = base_cols.copy()
    for base in temporal_bases:
        for suffix in suffixes:
            all_socio_cols.append(f"{base}{suffix}")
            
    # Filter to columns that actually exist in the input DataFrame
    final_cols = [col for col in all_socio_cols if col in X.columns]
    
    if not final_cols:
        raise ValueError("No socioeconomic features found. Verify column names.")
        
    print(f"✅ Filtered to {len(final_cols)} socioeconomic features: {final_cols}")
    return X[final_cols]