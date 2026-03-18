"""Create sample description and summary statistics table for 401(k) empirical data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "documents/real_data_401k"
OUT_DIR = PROJECT_ROOT / "documents/outputs/real_data_section"


def _find_dataset_file(data_dir: Path) -> Path:
    for pattern in ("*.csv", "*.dta", "*.parquet"):
        files = sorted(data_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No supported data file found in {data_dir}")


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {suffix}")


def _to_markdown(df: pd.DataFrame, title: str) -> str:
    headers = [str(c) for c in df.columns]
    lines = [f"**{title}**", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r" \hline",
        " & ".join(cols) + r" \\",
        r" \hline",
    ]
    for row in df.itertuples(index=False, name=None):
        rendered = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                rendered.append(f"{float(v):.3f}")
            else:
                rendered.append(str(v))
        lines.append(" & ".join(rendered) + r" \\")
    lines += [r" \hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def run() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_file = _find_dataset_file(DATA_DIR)
    df = _load_dataset(data_file)

    # Print requested quick diagnostics.
    print("Dataset file:", data_file)
    print("Observations:", df.shape[0])
    print("Variables:", df.shape[1])
    print("Columns:")
    print(list(df.columns))

    # Use all numeric columns for summary table.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    summary = (
        pd.DataFrame(
            {
                "Variable": numeric_cols,
                "Mean": [df[c].mean() for c in numeric_cols],
                "Std. Dev.": [df[c].std(ddof=1) for c in numeric_cols],
                "P25": [df[c].quantile(0.25) for c in numeric_cols],
                "Median": [df[c].median() for c in numeric_cols],
                "P75": [df[c].quantile(0.75) for c in numeric_cols],
                "Min": [df[c].min() for c in numeric_cols],
                "Max": [df[c].max() for c in numeric_cols],
                "Missing": [int(df[c].isna().sum()) for c in numeric_cols],
            }
        )
        .reset_index(drop=True)
    )

    # Sample description block.
    sample_desc = pd.DataFrame(
        [
            {"Item": "Dataset file", "Value": data_file.name},
            {"Item": "Observations (N)", "Value": int(df.shape[0])},
            {"Item": "Variables (K)", "Value": int(df.shape[1])},
            {"Item": "Numeric variables summarized", "Value": int(len(numeric_cols))},
        ]
    )

    # Persist outputs.
    sample_csv = OUT_DIR / "sample_description_401k.csv"
    sample_md = OUT_DIR / "sample_description_401k.md"
    sample_tex = OUT_DIR / "sample_description_401k.tex"
    summary_csv = OUT_DIR / "summary_statistics_401k.csv"
    summary_md = OUT_DIR / "summary_statistics_401k.md"
    summary_tex = OUT_DIR / "summary_statistics_401k.tex"

    sample_desc.to_csv(sample_csv, index=False, encoding="utf-8")
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    sample_md.write_text(_to_markdown(sample_desc, "Sample Description: 401(k) Dataset"), encoding="utf-8")
    summary_md.write_text(_to_markdown(summary, "Summary Statistics: 401(k) Dataset"), encoding="utf-8")

    sample_tex.write_text(
        _to_latex(sample_desc, "Sample description for the 401(k) empirical dataset.", "tab:sample-401k"),
        encoding="utf-8",
    )
    summary_tex.write_text(
        _to_latex(summary, "Summary statistics for the 401(k) empirical dataset.", "tab:summary-401k"),
        encoding="utf-8",
    )

    return [sample_csv, sample_md, sample_tex, summary_csv, summary_md, summary_tex]


if __name__ == "__main__":
    created = run()
    print("\nCreated files:")
    for path in created:
        print(path)
