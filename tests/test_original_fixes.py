"""Regression tests for the bugs fixed in the original 2024 harness."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from regression_classification_models import DataSet


def _write_csv(tmp_path, loader, name):
    """Save a bundled scikit-learn dataset as CSV with the target last.

    Args:
        tmp_path: pytest temporary directory.
        loader: A scikit-learn ``load_*`` function.
        name: File name for the CSV.

    Returns:
        The path to the written CSV as a string.
    """
    data = loader(as_frame=True)
    frame = data.frame
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_set(tmp_path):
    """A DataSet loaded from the diabetes data and split 80/20.

    Returns:
        DataSet: Split and ready for training.
    """
    dataset = DataSet(_write_csv(tmp_path, load_diabetes, "diabetes.csv"))
    dataset.read_csv()
    dataset.get_train_test_split()
    return dataset


@pytest.fixture
def classification_set(tmp_path):
    """A DataSet loaded from the breast cancer data.

    Returns:
        DataSet: Loaded but not yet split.
    """
    dataset = DataSet(_write_csv(tmp_path, load_breast_cancer, "cancer.csv"))
    dataset.read_csv()
    return dataset


def test_regressors_fit_on_train_split_only(regression_set):
    """Regressors see only the training rows, not the whole dataset."""
    tree = regression_set.decision_tree_regressor()
    assert tree.tree_.n_node_samples[0] == len(regression_set.x_train)
    assert len(regression_set.x_train) < len(regression_set.independents)


def test_logistic_block_runs_logistic_regression(classification_set, monkeypatch):
    """The 'Logistic Regression Results' block calls the logistic classifier."""
    calls = []
    original = DataSet.classifier_logistic_regression

    def spy(self, independent_vars, dependent_vars):
        """Record the call, then train as normal.

        Args:
            self: The DataSet instance under test.
            independent_vars: Training features.
            dependent_vars: Training labels.

        Returns:
            The trained logistic regression classifier.
        """
        calls.append(True)
        return original(self, independent_vars, dependent_vars)

    monkeypatch.setattr(DataSet, "classifier_logistic_regression", spy)
    classification_set.run_classification_model_test()
    assert calls == [True]


def test_classification_report_skips_r2(classification_set, monkeypatch):
    """Classification output no longer computes R2 on class labels."""

    def fail(*_args, **_kwargs):
        """Fail the test if R2 is computed for a classifier.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("R2 computed on class labels")

    monkeypatch.setattr(DataSet, "classification_r2_score", fail)
    classification_set.run_classification_model_test()
    assert classification_set.r2_score == []


def test_missing_file_raises(tmp_path):
    """A wrong path fails fast instead of later in read_csv."""
    with pytest.raises(FileNotFoundError):
        DataSet(str(tmp_path / "missing.csv"))


def test_read_csv_puts_target_last(classification_set):
    """The default indices treat the last CSV column as the target."""
    assert classification_set.independents.shape[1] == 30
    assert set(pd.unique(classification_set.dependents)) == {0, 1}
