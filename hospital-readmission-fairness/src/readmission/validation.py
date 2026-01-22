import pandas as pd

REQUIRED = {
    "age","gender","race","num_procedures","num_medications",
    "time_in_hospital","has_diabetes","readmitted_30"
}

def validate(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = REQUIRED - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    if "age" in df.columns and (df["age"] < 0).any():
        errors.append("Negative age found")
    if "readmitted_30" in df.columns and not set(df["readmitted_30"]).issubset({0,1}):
        errors.append("Label must be 0/1")
    return errors
