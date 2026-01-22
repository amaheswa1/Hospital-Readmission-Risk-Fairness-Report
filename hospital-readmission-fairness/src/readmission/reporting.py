from pathlib import Path
import pandas as pd
from .config import settings

def write_report(text: str, filename="report.md"):
    Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
    out = Path(settings.report_dir) / filename
    out.write_text(text, encoding="utf-8")
    return str(out)

def make_report(metrics: dict, gender_table: pd.DataFrame, race_table: pd.DataFrame) -> str:
    lines = []
    lines.append("# Hospital Readmission Risk + Fairness Report")
    lines.append("")
    lines.append("## Overall Metrics")
    for k,v in metrics.items():
        lines.append(f"- **{k}**: {v:.4f}")
    lines.append("")
    lines.append("## Fairness by Gender")
    lines.append(gender_table.to_markdown(index=False))
    lines.append("")
    lines.append("## Fairness by Race")
    lines.append(race_table.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)
