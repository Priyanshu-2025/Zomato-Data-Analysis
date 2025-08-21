import pandas as pd

def load_data(path: str):
    """Load dataset"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame):
    """Basic cleaning"""
    df = df.dropna()
    df = df.drop_duplicates()
    return df
