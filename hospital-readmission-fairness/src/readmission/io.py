import pandas as pd
from pathlib import Path
from .config import settings

def load_raw() -> pd.DataFrame:
    return pd.read_csv(settings.raw_data_path)

def save_processed(df: pd.DataFrame, name: str):
    Path(settings.processed_data_dir).mkdir(parents=True, exist_ok=True)
    out = Path(settings.processed_data_dir) / name
    df.to_csv(out, index=False)
    return str(out)
