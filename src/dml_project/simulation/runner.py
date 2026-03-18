"""Execution helpers for running Monte Carlo simulation scenarios."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd

# Allow direct execution via file path:
# python src/dml_project/simulation/runner.py
if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from dml_project import config
from dml_project.estimators.dml_plr import DMLPLR
from dml_project.learners.tuning import make_main_learner
from dml_project.utils.seeds import make_seed_bundle


class DGPGenerator(Protocol):
    """Callable protocol for DGP generator functions."""

    def __call__(
        self, n: int, p: int, theta: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _get_dgp_generator(dgp_name: str) -> DGPGenerator:
    """Resolve a configured DGP name to its ``generate_data`` callable.

    Args:
        dgp_name: Canonical DGP module name from project configuration.

    Returns:
        Callable with signature ``(n, p, theta, seed) -> (y, d, x)``.

    Raises:
        ValueError: If ``dgp_name`` is unknown or the module API is invalid.
    """

    if dgp_name not in config.DGP_NAMES:
        raise ValueError(f"Unknown DGP name: {dgp_name}")

    module_path = f"dml_project.dgps.{dgp_name}"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f"DGP module not found for configured dgp_name={dgp_name}") from exc

    generator = getattr(module, "generate_data", None)
    if generator is None or not callable(generator):
        raise ValueError(f"DGP module {module_path} must expose callable generate_data")
    return cast(DGPGenerator, generator)


def run_single_replication(scenario, replication: int) -> dict:
    """Run one Monte Carlo replication for a scenario.

    Args:
        scenario: Scenario object defining DGP, learner, and design settings.
        replication: Replication index within the scenario.

    Returns:
        Dictionary containing estimation outputs, diagnostics, and seeds for
        full replication traceability.
    """

    seed_bundle = make_seed_bundle(
        scenario_id=scenario.scenario_id,
        replication=replication,
        base_seed=scenario.base_seed,
    )

    dgp_generator = _get_dgp_generator(scenario.dgp_name)
    y, d, x = dgp_generator(
        n=scenario.n,
        p=scenario.p,
        theta=scenario.theta,
        seed=seed_bundle.data_seed,
    )

    learner_g = make_main_learner(
        scenario.learner_name, random_state=seed_bundle.learner_g_seed
    )
    learner_m = make_main_learner(
        scenario.learner_name, random_state=seed_bundle.learner_m_seed
    )

    estimator = DMLPLR(
        learner_g=learner_g,
        learner_m=learner_m,
        n_folds=config.N_FOLDS,
        random_state=seed_bundle.split_seed,
    )
    estimator.fit(y, d, x)

    theta_hat = estimator.predict_effect()
    if estimator.se_ > 0:
        t_stat = (theta_hat - scenario.theta) / estimator.se_
    else:
        t_stat = np.nan
    ci_length = estimator.ci_upper_ - estimator.ci_lower_
    if ci_length < 0.0:
        raise ValueError(
            "Invalid confidence interval length in run_single_replication: "
            f"scenario_id={scenario.scenario_id}, replication={replication}, "
            f"theta_hat={theta_hat}, se={estimator.se_}, "
            f"ci_lower={estimator.ci_lower_}, ci_upper={estimator.ci_upper_}, "
            f"ci_length={ci_length}."
        )
    error = theta_hat - scenario.theta
    squared_error = error**2
    covered = int(estimator.ci_lower_ <= scenario.theta <= estimator.ci_upper_)

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "replication": replication,
        "dgp_name": scenario.dgp_name,
        "learner_name": scenario.learner_name,
        "n": scenario.n,
        "p": scenario.p,
        "theta_true": scenario.theta,
        "learner_y": estimator.learner_g_name_,
        "learner_d": estimator.learner_m_name_,
        "data_seed": seed_bundle.data_seed,
        "split_seed": seed_bundle.split_seed,
        "learner_g_seed": seed_bundle.learner_g_seed,
        "learner_m_seed": seed_bundle.learner_m_seed,
        "theta_hat": theta_hat,
        "se": estimator.se_,
        "t_stat": t_stat,
        "ci_lower": estimator.ci_lower_,
        "ci_upper": estimator.ci_upper_,
        "ci_length": ci_length,
        "covered": covered,
        "error": error,
        "squared_error": squared_error,
    }


def run_scenario(scenario) -> pd.DataFrame:
    """Run all replications for a scenario and collect row-wise outputs.

    Args:
        scenario: Scenario object with ``n_rep`` replication count.

    Returns:
        DataFrame with one row per replication.
    """

    rows = [
        run_single_replication(scenario=scenario, replication=replication)
        for replication in range(scenario.n_rep)
    ]
    return pd.DataFrame(rows)


def run_scenario_with_runtime(scenario, mode: str) -> tuple[pd.DataFrame, dict]:
    """Run a scenario and append lightweight runtime metadata.

    Args:
        scenario: Scenario object to execute.
        mode: Pipeline mode label (for example ``"FAST"`` or ``"FULL"``).

    Returns:
        Tuple ``(results_df, runtime_row)`` where ``results_df`` contains
        replication outputs and ``runtime_row`` contains scenario-level timing
        metadata.
    """

    start_time = time.time()
    start_time_iso = datetime.now().isoformat()
    results = run_scenario(scenario)
    end_time = time.time()
    end_time_iso = datetime.now().isoformat()

    runtime_row = {
        "scenario_id": scenario.scenario_id,
        "dgp_name": scenario.dgp_name,
        "learner_name": scenario.learner_name,
        "n": scenario.n,
        "p": scenario.p,
        "mode": mode,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "duration_seconds": end_time - start_time,
        "n_rep": scenario.n_rep,
    }
    return results, runtime_row
