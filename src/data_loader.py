import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset and perform basic validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def aggregate_to_panel(df: pd.DataFrame, id_cols: list = None) -> pd.DataFrame:
    """Aggregate multiple records per (Country, Year) to a single row using mean."""
    if id_cols is None:
        id_cols = ['Country', 'Year']
        
    # Only aggregate numeric columns; keep identifiers intact
    numeric_cols = df.select_dtypes(include='number').columns.difference(id_cols)
    df_agg = df.groupby(id_cols)[numeric_cols].mean().reset_index()
    
    print(f"Aggregated to panel: {df_agg.shape[0]} unique {id_cols} combinations")
    return df_agg

def get_clean_pipeline(raw_path: str, save_path: str = None) -> pd.DataFrame:
    """Complete data loading and aggregation pipeline."""
    df = load_raw_data(raw_path)
    df_clean = aggregate_to_panel(df)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_clean.to_csv(save_path, index=False)
        print(f"Clean panel saved to {save_path}")
        
    return df_clean