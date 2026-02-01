import pandas as pd
from typing import List, Optional


def preprocess(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    drop_first: bool = True,
    fill_numeric: bool = True,
) -> pd.DataFrame:

    if categorical_cols is None:
        categorical_cols = ["gender", "race"]

    out = df.copy()

    missing_cols = [c for c in categorical_cols if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required categorical columns: {missing_cols}")

    for col in categorical_cols:
        out[col] = out[col].astype("category")

    if fill_numeric:
        numeric_cols = out.select_dtypes(include="number").columns
        for col in numeric_cols:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].mean())

    out = pd.get_dummies(
        out,
        columns=categorical_cols,
        drop_first=drop_first
    )

    return out
