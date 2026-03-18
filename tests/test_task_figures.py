"""Tests for figure-preparation logic in task_figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dml_project import config
from tasks.task_figures import (
    _FIGURE_GROUP_KEYS,
    _centered_learner_positions,
    _prepare_metrics_for_figures,
)


def _mock_main_results() -> pd.DataFrame:
    """Construct one row per design cell for figure tests."""

    rows: list[dict] = []
    for dgp_idx, dgp_name in enumerate(config.DGP_NAMES):
        for p_value in config.P_VALUES:
            for n_value in config.N_VALUES:
                for learner_idx, learner_name in enumerate(config.LEARNERS):
                    rows.append(
                        {
                            "dgp_name": dgp_name,
                            "learner_name": learner_name,
                            "n": n_value,
                            "p": p_value,
                            "matched_specification": True,
                            "coverage": 0.80
                            + 0.01 * learner_idx
                            + 0.00005 * n_value
                            + 0.00002 * p_value
                            - 0.02 * dgp_idx,
                            "rmse": 0.20 + 0.01 * learner_idx + 0.0001 * p_value,
                            "bias": -0.02 + 0.01 * learner_idx + 0.00005 * n_value,
                        }
                    )
    return pd.DataFrame(rows)


def test_prepare_metrics_uses_design_cell_keys_without_collapsing_n_or_p() -> None:
    """Prepared metrics must preserve dgp/learner/n/p/matched cells."""

    df = _mock_main_results()
    prepared = _prepare_metrics_for_figures(df)

    expected_count = (
        len(config.DGP_NAMES) * len(config.LEARNERS) * len(config.N_VALUES) * len(config.P_VALUES)
    )
    assert len(prepared) == expected_count == 36
    assert (prepared.groupby(_FIGURE_GROUP_KEYS).size() == 1).all()
    assert set(prepared["n"]) == set(config.N_VALUES)
    assert set(prepared["p"]) == set(config.P_VALUES)


def test_prepare_metrics_averages_only_within_exact_plotting_cell() -> None:
    """Any averaging should happen only for exact matching figure keys."""

    df = _mock_main_results()
    first_cell = df.iloc[0].copy()
    duplicate = first_cell.copy()
    duplicate["coverage"] = float(first_cell["coverage"]) + 0.10
    duplicate["rmse"] = float(first_cell["rmse"]) + 0.05
    duplicate["bias"] = float(first_cell["bias"]) - 0.05
    df_with_duplicate = pd.concat([df, pd.DataFrame([duplicate])], ignore_index=True)

    prepared = _prepare_metrics_for_figures(df_with_duplicate)
    assert len(prepared) == 36

    match = prepared.loc[
        (prepared["dgp_name"] == first_cell["dgp_name"])
        & (prepared["learner_name"] == first_cell["learner_name"])
        & (prepared["n"] == first_cell["n"])
        & (prepared["p"] == first_cell["p"])
        & (prepared["matched_specification"] == first_cell["matched_specification"])
    ]
    assert len(match) == 1
    assert np.isclose(
        float(match.iloc[0]["coverage"]),
        (float(first_cell["coverage"]) + float(duplicate["coverage"])) / 2.0,
    )
    assert np.isclose(
        float(match.iloc[0]["rmse"]),
        (float(first_cell["rmse"]) + float(duplicate["rmse"])) / 2.0,
    )
    assert np.isclose(
        float(match.iloc[0]["bias"]),
        (float(first_cell["bias"]) + float(duplicate["bias"])) / 2.0,
    )


def test_centered_learner_positions_are_symmetric_for_three_learners() -> None:
    """Learner offsets should be symmetric around each tick center."""

    x_center = 2.0
    width = 0.3
    positions = _centered_learner_positions(x_center=x_center, width=width)
    assert positions == [x_center - width, x_center, x_center + width]
    assert np.isclose(positions[0] + positions[2], 2.0 * x_center)
