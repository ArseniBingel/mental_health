"""Baseline models for the burnout regressor.

Fits a mean predictor, a Ridge regression and a RandomForest on the exact
splits produced by features.py, and reports RMSE, MAE and R2 for each.

These baselines exist to make the neural network in model.py falsifiable. On
tabular data with a modest number of features, a linear model or a forest is
frequently competitive with an MLP, and sometimes better. Running this file
first means the MLP has a number to beat rather than a number to report.

The DummyRegressor is the floor: it always predicts the training mean, so its
R2 on the test set is approximately zero by construction. Any model that fails
to clear it is broken, not merely weak.

This file is run once per feature set (A, B, C). The gap between the three sets
is the actual finding of the project — see the module docstring of features.py.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from features import prepare_data, SEED, FEATURE_SETS



def arrays_from_loaders(data:dict):
    """Recover plain NumPy arrays from the DataLoaders' underlying datasets.

    sklearn estimators cannot consume a DataLoader, but re-running the
    preparation would risk a different split. Reading the tensors back out of
    the TensorDatasets guarantees the baselines and the MLP see byte-identical
    data.

    Returns a dict with keys X_train, X_val, X_test, y_train, y_val, y_test.
    The y arrays are flattened to shape (n,): sklearn expects a 1-D target while torch needed (n, 1).
    """
    loader_dict = [("train", "train_loader"),
                       ("val",   "val_loader"),
                       ("test",  "test_loader")]

    out = {}

    for split, key in loader_dict:
        X_t, y_t = data[key].dataset.tensors
        out[f"X_{split}"] = X_t.numpy()
        out[f"y_{split}"] = y_t.numpy().ravel()

    return out


def build_models(seed=SEED):
    """Return a dict mapping model name to an unfitted sklearn estimator.

    Include the mean predictor, a Ridge regression and a RandomForest.

    Constrain the forest: on tens of thousands of rows an unbounded forest with
    many trees takes minutes and gains little. Cap the number of trees and the
    depth, and use every core.
    """
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1),
        "forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
    }


def evaluate(model, X, y):
    """Fit-free evaluation: predict on X and return a dict of metrics.

    Returns {"rmse": ..., "mae": ..., "r2": ...} as plain floats.

    The model must already be fitted. RMSE is in the units of the target, which
    makes it readable next to a burnout score between 0 and 10; R2 is unitless
    and comparable across feature sets.
    """
    preds = model.predict(X)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y,preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y,preds))
    }
    

def cross_validate_model(model, X_train, y_train, cv=5):
    """Mean and standard deviation of the cross-validated RMSE on the training split.

    Returns (mean_rmse, std_rmse) as floats.

    Uses the training data only — the validation and test splits stay untouched,
    so the numbers here remain comparable to the MLP's later results.

    Note that sklearn's scoring strings are oriented so that higher is better,
    which means error metrics come back negative.
    """
    scores = cross_val_score(model, X_train,y_train, scoring="neg_root_mean_squared_error", cv=cv)
    rmse = -scores
    return float(rmse.mean()), float(rmse.std()) 


def run_baselines(feature_set="A", cv=5, seed=SEED):
    """Fit every baseline on one feature set and report the results.

    Prints a table with one row per model: cross-validated training RMSE, then
    validation RMSE, MAE and R2. Evaluates on the VALIDATION split, not the
    test split — the test split is spent once at the end.

    Returns a dict mapping model name to its validation metrics dict.
    """
    loaders_dict = prepare_data(feature_set=feature_set, seed=seed)
    out = arrays_from_loaders(loaders_dict)
    X_train, y_train = out["X_train"], out["y_train"]
    X_val, y_val = out["X_val"], out["y_val"]
    models = build_models(seed=seed)

    results = {}

    for name, est in models.items():
        cv_mean, cv_std = cross_validate_model(est, X_train=X_train, y_train=y_train, cv=cv)

        est.fit(X_train,y_train)
        metrics = evaluate(est, X_val, y_val)

        results[name] = metrics

        print(f"{name:15} cv_rmse {cv_mean:.3f} ± {cv_std:.3f} | "
              f"val rmse {metrics['rmse']:.3f}  mae {metrics['mae']:.3f}  "
              f"r2 {metrics['r2']:.3f}")

    return results




def compare_feature_sets(seed=SEED):
    """Run the Ridge baseline on all three feature sets and print the R2 values.

    Ridge is used rather than the forest because it is fast and because a linear
    model makes the leakage effect easiest to read: if adding a column lifts R2
    from moderate to near-perfect, that column is not a predictor.

    Returns a dict mapping feature set name to its validation R2.
    """
    results = {}

    for feature in FEATURE_SETS:
        models = build_models(seed)
        ridge_model = models["ridge"]
        loaders = prepare_data(feature_set=feature,seed=seed)
        out = arrays_from_loaders(loaders)
        X_train,y_train = out["X_train"], out["y_train"]
        X_val,y_val = out["X_val"], out["y_val"]

        ridge_model.fit(X_train,y_train)
        metrics = evaluate(ridge_model,X_val,y_val)
        print(f"Feature Set: {feature}, r2: {metrics['r2']}")
        results[feature] = metrics["r2"]

    return results

    

if __name__ == "__main__":
    results = run_baselines(feature_set="A")

    assert set(results) >= {"dummy", "ridge", "forest"}
    for name, m in results.items():
        assert set(m) == {"rmse", "mae", "r2"}
        assert m["rmse"] >= 0 and m["mae"] >= 0
        assert m["rmse"] >= m["mae"], f"{name}: RMSE below MAE is impossible"

    assert abs(results["dummy"]["r2"]) < 0.05, "the mean predictor should explain nothing"
    assert results["ridge"]["r2"] > results["dummy"]["r2"], "Ridge lost to the mean"

    print("\nR2 by feature set:")
    r2_by_set = compare_feature_sets()
    for name in ["A", "B", "C"]:
        print(f"  set {name}: {r2_by_set[name]:.3f}")

    assert r2_by_set["C"] >= r2_by_set["A"], "adding columns should not hurt Ridge"

    print("\nbaselines.py ran fine.")