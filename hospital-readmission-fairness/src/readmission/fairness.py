import pandas as pd

def group_metrics(df: pd.DataFrame, y_true, y_pred, group_series: pd.Series):
    out = []
    for g in sorted(group_series.unique()):
        idx = group_series == g
        if idx.sum() == 0:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        acc = (yt == yp).mean()
        out.append({
            "group": g,
            "n": int(idx.sum()),
            "accuracy": float(acc)
        })
    return pd.DataFrame(out)
