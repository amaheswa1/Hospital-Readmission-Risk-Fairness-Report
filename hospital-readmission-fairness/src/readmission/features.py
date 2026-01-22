import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gender"] = out["gender"].astype("category")
    out["race"] = out["race"].astype("category")
    out = pd.get_dummies(out, columns=["gender","race"], drop_first=True)
    return out
