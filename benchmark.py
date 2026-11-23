"""Cross-validated benchmark of scikit-learn regressors and classifiers.

Every model runs inside a pipeline whose first step is ``StandardScaler``.
``cross_validate`` refits the whole pipeline on each training fold, so the
scaler never sees the validation fold and the scores are free of that leak.

Run ``python benchmark.py`` to benchmark on two datasets bundled with
scikit-learn: diabetes progression (regression) and breast cancer
diagnosis (classification).
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 0

REGRESSION_SCORING = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
CLASSIFICATION_SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}


def regressors(random_state: int = RANDOM_STATE) -> dict:
    """Build the regression models to compare.

    Args:
        random_state: Seed for the models that use randomness.

    Returns:
        A mapping of display name to an unfitted estimator.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (degree 2)": make_pipeline(
            PolynomialFeatures(degree=2), LinearRegression()
        ),
        "SVR (RBF kernel)": SVR(kernel="rbf"),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
    }


def classifiers(random_state: int = RANDOM_STATE) -> dict:
    """Build the classification models to compare.

    Args:
        random_state: Seed for the models that use randomness.

    Returns:
        A mapping of display name to an unfitted estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (linear kernel)": SVC(kernel="linear", random_state=random_state),
        "SVM (RBF kernel)": SVC(kernel="rbf", random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, criterion="entropy", random_state=random_state
        ),
    }


def benchmark(models: dict, features, target, scoring: dict, cv) -> pd.DataFrame:
    """Score each model with k-fold cross-validation.

    Args:
        models: Mapping of display name to an unfitted estimator.
        features: Feature matrix, shape (n_samples, n_features).
        target: Target vector, shape (n_samples,).
        scoring: Mapping of metric name to a scikit-learn scorer string.
            Scorers prefixed ``neg_`` are negated back to positive errors.
        cv: A scikit-learn cross-validation splitter.

    Returns:
        One row per model with the mean and standard deviation of every
        metric across folds, sorted best-first on the first metric.
    """
    rows = []
    for name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_validate(pipeline, features, target, scoring=scoring, cv=cv)
        row = {"model": name}
        for metric, scorer in scoring.items():
            values = scores[f"test_{metric}"]
            if scorer.startswith("neg_"):
                values = -values
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)

    first_metric, first_scorer = next(iter(scoring.items()))
    lower_is_better = first_scorer.startswith("neg_")
    table = pd.DataFrame(rows).sort_values(
        f"{first_metric}_mean", ascending=lower_is_better
    )
    return table.reset_index(drop=True)


def run_regression(folds: int = 5) -> pd.DataFrame:
    """Benchmark the regressors on the diabetes progression dataset.

    Args:
        folds: Number of cross-validation folds.

    Returns:
        The benchmark table from :func:`benchmark`.
    """
    data = load_diabetes()
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return benchmark(
        regressors(), data.data, data.target, REGRESSION_SCORING, splitter
    )


def run_classification(folds: int = 5) -> pd.DataFrame:
    """Benchmark the classifiers on the breast cancer diagnosis dataset.

    Args:
        folds: Number of cross-validation folds.

    Returns:
        The benchmark table from :func:`benchmark`.
    """
    data = load_breast_cancer()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return benchmark(
        classifiers(), data.data, data.target, CLASSIFICATION_SCORING, splitter
    )


def to_markdown(table: pd.DataFrame) -> str:
    """Render a benchmark table as Markdown, pairing each mean with its std.

    Args:
        table: A table returned by :func:`benchmark`.

    Returns:
        A Markdown table string with ``mean ± std`` cells.
    """
    metrics = [col[: -len("_mean")] for col in table.columns if col.endswith("_mean")]
    header = "| Model | " + " | ".join(metrics) + " |"
    divider = "|---|" + "---:|" * len(metrics)
    lines = [header, divider]
    for _, row in table.iterrows():
        cells = [f"{row[m + '_mean']:.3f} ± {row[m + '_std']:.3f}" for m in metrics]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, run both benchmarks, and print Markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    args = parser.parse_args()

    print(f"## Regression: diabetes progression ({args.folds}-fold CV)\n")
    print(to_markdown(run_regression(args.folds)))
    print(f"\n## Classification: breast cancer diagnosis ({args.folds}-fold CV)\n")
    print(to_markdown(run_classification(args.folds)))


if __name__ == "__main__":
    main()
