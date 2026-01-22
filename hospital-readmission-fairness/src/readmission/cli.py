import typer
import pandas as pd
from .io import load_raw
from .validation import validate as validate_df
from .features import preprocess
from .modeling import train_model
from .fairness import group_metrics
from .reporting import write_report, make_report

app = typer.Typer()

@app.command()
def validate():
    df = load_raw()
    errs = validate_df(df)
    if errs:
        for e in errs:
            typer.echo(f"ERROR: {e}")
        raise typer.Exit(code=1)
    typer.echo("Validation passed")

@app.command()
def train():
    df = load_raw()
    df2 = preprocess(df)
    model, metrics, _ = train_model(df2)
    typer.echo(f"Trained model. Metrics: {metrics}")

@app.command()
def report():
    df = load_raw()
    raw = df.copy()
    df2 = preprocess(df)

    model, metrics, (X_test, y_test, preds, probs) = train_model(df2)

    # Reconstruct group columns from raw data
    test_idx = X_test.index
    raw_test = raw.loc[test_idx]

    gender_tbl = group_metrics(raw_test, y_test.values, preds, raw_test["gender"])
    race_tbl = group_metrics(raw_test, y_test.values, preds, raw_test["race"])

    text = make_report(metrics, gender_tbl, race_tbl)
    out = write_report(text, filename="report.md")
    typer.echo(f"Wrote report to {out}")
