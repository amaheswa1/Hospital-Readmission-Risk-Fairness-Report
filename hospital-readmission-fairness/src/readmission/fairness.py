import pandas as pd
import numpy as np
from typing import Iterable, List, Dict, Any, Optional


def group_metrics(
    df: pd.DataFrame,
    y_true: Iterable,
    y_pred: Iterable,
    group_series: pd.Series,
    metrics: Optional[List[str]] = None,
    positive_label: Any = 1,
) -> pd.DataFrame:

    if metrics is None:
        metrics = ["accuracy"]

    if not (len(y_true) == len(y_pred) == len(group_series)):
        raise ValueError("y_true, y_pred, and group_series must have the same length")

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    group_series = group_series.reset_index(drop=True)

    results: List[Dict[str, Any]] = []

    for g in sorted(group_series.dropna().unique()):
        idx = group_series == g
        n = int(idx.sum())
        if n == 0:
            continue

        yt = y_true[idx]
        yp = y_pred[idx]

        row: Dict[str, Any] = {
            "group": g,
            "n": n,
        }

        if "accuracy" in metrics:
            row["accuracy"] = float((yt == yp).mean())

        if "precision" in metrics:
            tp = ((yt == positive_label) & (yp == positive_label)).sum()
            fp = ((yt != positive_label) & (yp == positive_label)).sum()
            row["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        if "recall" in metrics:
            tp = ((yt == positive_label) & (yp == positive_label)).sum()
            fn = ((yt == positive_label) & (yp != positive_label)).sum()
            row["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        if "f1" in metrics:
            p = row.get("precision", 0.0)
            r = row.get("recall", 0.0)
            row["f1"] = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0

        results.append(row)

    return pd.DataFrame(results)
