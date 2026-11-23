"""Tests for the cross-validated benchmark module."""

import numpy as np
import pytest
from sklearn.model_selection import KFold

import benchmark


@pytest.fixture(scope="module")
def regression_table():
    """Run the regression benchmark once with 3 folds for speed.

    Returns:
        The regression benchmark table.
    """
    return benchmark.run_regression(folds=3)


@pytest.fixture(scope="module")
def classification_table():
    """Run the classification benchmark once with 3 folds for speed.

    Returns:
        The classification benchmark table.
    """
    return benchmark.run_classification(folds=3)


def test_regression_table_has_one_row_per_model(regression_table):
    """Every regressor appears exactly once."""
    assert sorted(regression_table["model"]) == sorted(benchmark.regressors())


def test_classification_table_has_one_row_per_model(classification_table):
    """Every classifier appears exactly once, including logistic regression."""
    assert sorted(classification_table["model"]) == sorted(benchmark.classifiers())
    assert "Logistic Regression" in set(classification_table["model"])


def test_rmse_is_reported_as_positive_error(regression_table):
    """The negated sklearn scorer is flipped back to a positive error."""
    assert (regression_table["rmse_mean"] > 0).all()


def test_tables_are_sorted_best_first(regression_table, classification_table):
    """Rows are ordered by the first metric, best model first."""
    assert regression_table["r2_mean"].is_monotonic_decreasing
    assert classification_table["accuracy_mean"].is_monotonic_decreasing


def test_classification_scores_are_valid_probabilities(classification_table):
    """Accuracy and F1 stay within [0, 1]."""
    for column in ("accuracy_mean", "f1_macro_mean"):
        assert classification_table[column].between(0, 1).all()


def test_lower_is_better_metric_sorts_ascending():
    """A neg_ scorer listed first sorts with the smallest error on top."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 3))
    target = features @ np.array([1.0, 2.0, 3.0]) + rng.normal(scale=0.1, size=60)
    table = benchmark.benchmark(
        benchmark.regressors(),
        features,
        target,
        {"rmse": "neg_root_mean_squared_error"},
        KFold(n_splits=3, shuffle=True, random_state=0),
    )
    assert table["rmse_mean"].is_monotonic_increasing
    assert table.loc[0, "model"] == "Linear Regression"


def test_to_markdown_renders_header_and_rows(classification_table):
    """The Markdown table has a header, divider, and one line per model."""
    lines = benchmark.to_markdown(classification_table).splitlines()
    assert lines[0] == "| Model | accuracy | f1_macro |"
    assert len(lines) == 2 + len(classification_table)
