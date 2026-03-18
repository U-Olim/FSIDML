"""Import tests for aggregation and metrics modules."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project.simulation.aggregate import aggregate_results
    from dml_project.simulation.runner import run_scenario
    from dml_project.simulation.scenario import Scenario
    from dml_project.utils.seeds import (
        SeedBundle,
        make_numpy_rng,
        make_seed_bundle,
        seed_bundle_to_dict,
    )
except ModuleNotFoundError:
    from src.dml_project.simulation.aggregate import aggregate_results
    from src.dml_project.simulation.runner import run_scenario
    from src.dml_project.simulation.scenario import Scenario
    from src.dml_project.utils.seeds import (
        SeedBundle,
        make_numpy_rng,
        make_seed_bundle,
        seed_bundle_to_dict,
    )


def test_import_aggregation_modules() -> None:
    """Ensure aggregation modules import without errors."""
    importlib.import_module("dml_project.simulation.aggregate")
    importlib.import_module("dml_project.simulation.metrics")


def _scenario_metric(aggregated: pd.DataFrame, scenario_id: int, metric: str) -> float:
    """Return one aggregated metric value for a scenario id."""

    matched = aggregated.loc[aggregated["scenario_id"] == scenario_id, metric]
    if len(matched) != 1:
        raise AssertionError(
            f"Expected one aggregated row for scenario_id={scenario_id}, got {len(matched)}."
        )
    return float(matched.iloc[0])


def test_make_numpy_rng_returns_generator() -> None:
    """RNG helper should return a NumPy Generator."""
    rng = make_numpy_rng(123)
    assert isinstance(rng, np.random.Generator)


def test_seed_bundle_to_dict_roundtrip_fields() -> None:
    """Dictionary conversion should preserve bundle values."""
    bundle = SeedBundle(
        base_seed=123,
        scenario_id=1,
        replication=2,
        data_seed=101,
        split_seed=202,
        learner_g_seed=303,
        learner_m_seed=404,
    )
    as_dict = seed_bundle_to_dict(bundle)
    assert as_dict == {
        "base_seed": 123,
        "scenario_id": 1,
        "replication": 2,
        "data_seed": 101,
        "split_seed": 202,
        "learner_g_seed": 303,
        "learner_m_seed": 404,
    }


def test_make_seed_bundle_is_stable_for_same_inputs() -> None:
    """Seed bundle should be deterministic for fixed inputs."""
    first = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    second = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    assert first == second


def test_make_seed_bundle_streams_are_distinct() -> None:
    """Seed streams should be independent and non-consecutive."""
    bundle = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    seeds = [
        bundle.data_seed,
        bundle.split_seed,
        bundle.learner_g_seed,
        bundle.learner_m_seed,
    ]
    assert len(set(seeds)) == 4
    sorted_seeds = sorted(seeds)
    assert all((b - a) != 1 for a, b in zip(sorted_seeds, sorted_seeds[1:]))


def test_make_seed_bundle_changes_across_replications() -> None:
    """Different replication ids should generate a different bundle."""
    first = make_seed_bundle(scenario_id=2, replication=7, base_seed=123)
    second = make_seed_bundle(scenario_id=2, replication=8, base_seed=123)
    assert first != second


def test_run_scenario_includes_error_columns() -> None:
    """Runner output should include core estimation error columns."""
    scenario = Scenario(
        scenario_id=11,
        name="agg_columns_check",
        dgp_name="linear_sparse_correlated",
        learner_name="lasso",
        n=60,
        p=10,
        theta=1.0,
        n_rep=2,
        base_seed=123,
    )
    results = run_scenario(scenario)
    assert {
        "theta_hat",
        "error",
        "squared_error",
        "se",
        "t_stat",
        "ci_lower",
        "ci_upper",
        "covered",
    }.issubset(results.columns)


def test_inference_columns_have_valid_values() -> None:
    """SE/CI/coverage outputs should satisfy basic validity constraints."""
    scenario = Scenario(
        scenario_id=12,
        name="inference_check",
        dgp_name="linear_baseline",
        learner_name="lasso",
        n=80,
        p=10,
        theta=1.0,
        n_rep=2,
        base_seed=123,
    )
    results = run_scenario(scenario)
    assert np.isfinite(
        results[["theta_hat", "se", "t_stat", "ci_lower", "ci_upper"]].to_numpy()
    ).all()
    assert (results["se"] > 0).all()
    assert (results["ci_upper"] > results["ci_lower"]).all()
    assert ((results["ci_lower"] < results["theta_hat"]) & (results["theta_hat"] < results["ci_upper"])).all()
    assert set(results["covered"].unique()).issubset({0, 1})


def test_aggregate_results_metrics_properties() -> None:
    """Scenario-level aggregates should satisfy core metric constraints."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 1,
                "theta_true": 2.0,
                "theta_hat": 1.8,
                "se": 0.2,
                "ci_lower": 1.4,
                "ci_upper": 2.2,
                "covered": 1,
            },
            {
                "scenario_id": 1,
                "theta_true": 2.0,
                "theta_hat": 2.1,
                "se": 0.25,
                "ci_lower": 1.6,
                "ci_upper": 2.6,
                "covered": 1,
            },
            {
                "scenario_id": 2,
                "theta_true": 0.0,
                "theta_hat": -0.1,
                "se": 0.15,
                "ci_lower": -0.4,
                "ci_upper": 0.2,
                "covered": 1,
            },
            {
                "scenario_id": 2,
                "theta_true": 0.0,
                "theta_hat": 0.3,
                "se": 0.2,
                "ci_lower": 0.05,
                "ci_upper": 0.55,
                "covered": 0,
            },
        ]
    )
    aggregated = aggregate_results(df)
    assert len(aggregated) == 2
    assert ((aggregated["coverage"] >= 0.0) & (aggregated["coverage"] <= 1.0)).all()
    assert (aggregated["mc_se_coverage"] >= 0.0).all()
    assert (aggregated["mc_se_bias"] >= 0.0).all()
    assert (aggregated["mc_se_rmse"] >= 0.0).all()
    assert (aggregated["rmse"] >= aggregated["bias"].abs()).all()
    assert (aggregated["mean_ci_length"] > 0).all()
    assert "variance_ratio" in aggregated.columns
    assert (aggregated["variance_ratio"] >= 0.0).all()
    assert "t_stat_mean" in aggregated.columns
    assert "t_stat_sd" in aggregated.columns
    assert np.isfinite(aggregated["t_stat_mean"]).all()
    assert (aggregated["t_stat_sd"] >= 0.0).all()


def test_aggregate_coverage_matches_ci_definition() -> None:
    """Aggregated coverage must equal mean CI inclusion indicator."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 10,
                "theta_true": 0.0,
                "theta_hat": 0.1,
                "se": 0.2,
                "ci_lower": -0.2,
                "ci_upper": 0.4,
                "covered": 1,
            },
            {
                "scenario_id": 10,
                "theta_true": 0.0,
                "theta_hat": 0.7,
                "se": 0.2,
                "ci_lower": 0.3,
                "ci_upper": 1.1,
                "covered": 0,
            },
        ]
    )

    aggregated = aggregate_results(df)
    expected_coverage = float(
        ((df["ci_lower"] <= df["theta_true"]) & (df["theta_true"] <= df["ci_upper"])).mean()
    )
    assert np.isclose(_scenario_metric(aggregated, scenario_id=10, metric="coverage"), expected_coverage)


def test_aggregate_mean_ci_length_matches_replication_average() -> None:
    """Aggregated mean_ci_length must equal mean(ci_upper - ci_lower)."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 20,
                "theta_true": 2.0,
                "theta_hat": 2.1,
                "se": 0.1,
                "ci_lower": 1.8,
                "ci_upper": 2.4,
                "covered": 1,
            },
            {
                "scenario_id": 20,
                "theta_true": 2.0,
                "theta_hat": 1.9,
                "se": 0.1,
                "ci_lower": 1.5,
                "ci_upper": 2.3,
                "covered": 1,
            },
        ]
    )

    aggregated = aggregate_results(df)
    expected_ci_length = float((df["ci_upper"] - df["ci_lower"]).mean())
    assert np.isclose(
        _scenario_metric(aggregated, scenario_id=20, metric="mean_ci_length"),
        expected_ci_length,
    )


def test_aggregate_variance_ratio_matches_empirical_sd_over_mean_se() -> None:
    """Aggregated variance_ratio must equal empirical_sd / mean_se."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 21,
                "theta_true": 0.0,
                "theta_hat": 1.0,
                "se": 0.5,
                "ci_lower": 0.0,
                "ci_upper": 2.0,
                "covered": 1,
            },
            {
                "scenario_id": 21,
                "theta_true": 0.0,
                "theta_hat": 2.0,
                "se": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 4.0,
                "covered": 1,
            },
        ]
    )

    aggregated = aggregate_results(df)
    expected = float(df["theta_hat"].std(ddof=1) / df["se"].mean())
    assert np.isclose(_scenario_metric(aggregated, scenario_id=21, metric="variance_ratio"), expected)


def test_aggregate_t_stat_diagnostics_match_replication_values() -> None:
    """Aggregated t-stat diagnostics must match per-replication t-stat moments."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 22,
                "theta_true": 1.0,
                "theta_hat": 1.1,
                "se": 0.2,
                "ci_lower": 0.8,
                "ci_upper": 1.4,
                "covered": 1,
            },
            {
                "scenario_id": 22,
                "theta_true": 1.0,
                "theta_hat": 0.8,
                "se": 0.1,
                "ci_lower": 0.6,
                "ci_upper": 1.0,
                "covered": 1,
            },
        ]
    )

    aggregated = aggregate_results(df)
    t_stat = (df["theta_hat"] - df["theta_true"]) / df["se"]
    assert np.isclose(
        _scenario_metric(aggregated, scenario_id=22, metric="t_stat_mean"),
        float(t_stat.mean()),
    )
    assert np.isclose(
        _scenario_metric(aggregated, scenario_id=22, metric="t_stat_sd"),
        float(t_stat.std(ddof=1)),
    )


def test_aggregate_mc_se_coverage_matches_binomial_formula() -> None:
    """Coverage MC SE must use sqrt(p*(1-p)/R)."""
    df = pd.DataFrame(
        [
            {
                "scenario_id": 30,
                "theta_true": 0.0,
                "theta_hat": 0.0,
                "se": 0.2,
                "ci_lower": -0.1,
                "ci_upper": 0.1,
                "covered": 1,
            },
            {
                "scenario_id": 30,
                "theta_true": 0.0,
                "theta_hat": 0.0,
                "se": 0.2,
                "ci_lower": 0.2,
                "ci_upper": 0.3,
                "covered": 0,
            },
            {
                "scenario_id": 30,
                "theta_true": 0.0,
                "theta_hat": 0.0,
                "se": 0.2,
                "ci_lower": -0.2,
                "ci_upper": 0.2,
                "covered": 1,
            },
            {
                "scenario_id": 30,
                "theta_true": 0.0,
                "theta_hat": 0.0,
                "se": 0.2,
                "ci_lower": 0.3,
                "ci_upper": 0.4,
                "covered": 0,
            },
        ]
    )
    aggregated = aggregate_results(df)
    p_hat = 0.5
    expected = np.sqrt(p_hat * (1.0 - p_hat) / 4.0)
    assert np.isclose(_scenario_metric(aggregated, scenario_id=30, metric="mc_se_coverage"), expected)
