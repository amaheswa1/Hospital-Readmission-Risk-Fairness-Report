import pandas as pd
from readmission.validation import validate

def test_ok():
    df = pd.DataFrame([{
        "age":50,"gender":"Male","race":"White","num_procedures":1,
        "num_medications":5,"time_in_hospital":3,"has_diabetes":0,"readmitted_30":0
    }])
    assert validate(df) == []
