"""Pytask step for the 401(k) empirical application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project.real_data import create_coef_plot, run_401k_empirical_estimators
except ModuleNotFoundError:
    from src.dml_project.real_data import create_coef_plot, run_401k_empirical_estimators


def task_real_data_401k(
    path_to_input: Path = PROJECT_ROOT / "documents/real_data_401k/sipp_1991.csv",
    path_to_results: Annotated[Path, Product] = (
        PROJECT_ROOT / "documents/outputs/aggregated/real_data_results.csv"
    ),
    path_to_fig_png: Annotated[Path, Product] = (
        PROJECT_ROOT / "documents/outputs/figures/paper/fig14_real_data_coefplot.png"
    ),
    path_to_fig_pdf: Annotated[Path, Product] = (
        PROJECT_ROOT / "documents/outputs/figures/paper/fig14_real_data_coefplot.pdf"
    ),
) -> None:
    """Run empirical estimators and export requested 401(k) outputs."""

    expected_input = PROJECT_ROOT / "documents/real_data_401k/sipp_1991.csv"
    expected_results = PROJECT_ROOT / "documents/outputs/aggregated/real_data_results.csv"
    expected_png = PROJECT_ROOT / "documents/outputs/figures/paper/fig14_real_data_coefplot.png"
    expected_pdf = PROJECT_ROOT / "documents/outputs/figures/paper/fig14_real_data_coefplot.pdf"
    if path_to_input != expected_input:
        raise ValueError(f"task_real_data_401k must read {expected_input}, got {path_to_input}")
    if path_to_results != expected_results:
        raise ValueError(f"task_real_data_401k must write {expected_results}, got {path_to_results}")
    if path_to_fig_png != expected_png:
        raise ValueError(f"task_real_data_401k must write {expected_png}, got {path_to_fig_png}")
    if path_to_fig_pdf != expected_pdf:
        raise ValueError(f"task_real_data_401k must write {expected_pdf}, got {path_to_fig_pdf}")

    data = pd.read_csv(path_to_input)
    results = run_401k_empirical_estimators(data)

    path_to_results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path_to_results, index=False, encoding="utf-8")

    path_to_fig_png.parent.mkdir(parents=True, exist_ok=True)
    create_coef_plot(
        results=results,
        output_png=str(path_to_fig_png),
        output_pdf=str(path_to_fig_pdf),
    )
