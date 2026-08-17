"""Compares the baselines and the MLP, and runs the leakage experiment.

Three things happen here, each producing an artefact for the report:

1. Every model is scored on the TEST split — the one split not used for training
   or early stopping. Ridge, RandomForest and the MLP are ranked on the same
   data by RMSE, MAE and R2. Reports whichever wins.

2. One model is run on all three feature sets. The gap between the R2 values is
   the finding: set A is lifestyle and academic columns only, B adds the
   psychological scales, and C adds mental_health_index and dropout_risk, which
   were derived from burnout when the data was generated. A near-perfect R2 on C
   is leakage, not skill.

3. The best model's residuals are plotted against its predictions. About a
   quarter of the target is exactly zero (the generator clipped negatives), and
   MSE cannot represent that floor, so the residuals show a hard diagonal edge.

Predictions from the honest model (set A) are written to predictions.csv with the
original id, so the dashboard can show model output next to the survey aggregates.

"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import prepare_data, SEED, FEATURE_SETS
from baselines import build_models
from model import make_model, fit

FIG_DIR = "figures"


def collect_predictions(model, X_test, test_loader):
    """Return 1-D predictions for one model, whichever kind it is.

    sklearn estimators expose a .predict method and take the array X_test.
    The torch model has no .predict and is run over test_loader in eval mode
    without gradients. Detecting the type here keeps the rest of the file from
    branching on it everywhere.

    The row order of the output matches the test rows, so test_loader must not
    shuffle.
    """
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X_test)).ravel()

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds).ravel()


def fit_all_models(data, seed=SEED):
    """Fit every baseline and the MLP on the training split of one prepared dict.

    The baselines fit on the plain training arrays: the MLP is built from
    data["n_features"] and trained with fit() on the loaders. Returns a dict
    mapping model name to the fitted model, with the MLP under "mlp" alongside
    the baseline names.
    """
    models = build_models(seed=seed)
    for model in models.values():
        model.fit(data["X_train"], data["y_train"])

    mlp, criterion, optimizer = make_model(data["n_features"], seed=seed)
    fit(mlp, data["train_loader"], data["val_loader"],
        criterion, optimizer, verbose=False)
    models["mlp"] = mlp

    return models


def evaluate_all(models, X_test, y_test, test_loader):
    """Score every fitted model on the test split.

    Returns a DataFrame indexed by model name with columns rmse, mae, r2,
    sorted so the best model (lowest RMSE) is first.
    """
    rows = []
    for name, model in models.items():
        y_pred = collect_predictions(model, X_test, test_loader)
        rows.append({
            "model": name,
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        })
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("rmse")


def run_leakage_experiment(model_kind="ridge", seed=SEED):
    """Fit one model kind on each feature set and collect its test R2.
    Returns a DataFrame indexed by feature set (A, B, C)
    with an r2 column.
    """
    rows = []
    for name in FEATURE_SETS:
        data = prepare_data(feature_set=name, seed=seed)
        model = build_models(seed=seed)[model_kind]
        model.fit(data["X_train"], data["y_train"])
        y_pred = np.asarray(model.predict(data["X_test"])).ravel()
        rows.append({
            "feature_set": name,
            "r2": float(r2_score(data["y_test"], y_pred)),
        })
    return pd.DataFrame(rows).set_index("feature_set")


def plot_residuals(y_true, y_pred, outdir=FIG_DIR, name="residuals"):
    """Scatter residuals (y_true - y_pred) against y_pred, save to outdir/name.png.

    A horizontal line at zero marks perfect prediction. On this target the
    clipped floor at zero forces a diagonal band of points rather than a
    shapeless cloud. 
    Returns the path.
    """
    os.makedirs(outdir, exist_ok=True)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, s=8, alpha=0.3)
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (true - pred)")
    ax.set_title("Residuals vs predicted")
    fig.tight_layout()

    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_predictions(ids, y_true, y_pred, path="predictions.csv"):
    """Write id, burnout_true, burnout_pred to a CSV, no index column.

    """
    out = pd.DataFrame({
        "id": np.asarray(ids),
        "burnout_true": np.asarray(y_true),
        "burnout_pred": np.asarray(y_pred),
    })
    out.to_csv(path, index=False, sep=";", decimal=",")
    return path


def run_comparison(seed=SEED):
    """Run the whole comparison and return its results.

    Prepares feature set A, fits all models, scores them on the test split and
    prints the table, runs the leakage experiment and prints it, plots the
    residuals of the best model, exports the set-A predictions with their ids.

    Returns a dict with keys comparison, leakage and predictions_path.
    """
    data = prepare_data(feature_set="A", seed=seed)
    models = fit_all_models(data, seed=seed)

    comparison = evaluate_all(models, data["X_test"], data["y_test"],
                              data["test_loader"])
    print("Model comparison on the test split:")
    print(comparison.round(3), "\n")

    leakage = run_leakage_experiment(model_kind="ridge", seed=seed)
    print("Leakage experiment (Ridge R2 by feature set):")
    print(leakage.round(3), "\n")

    best_name = comparison.index[0]
    y_pred_best = collect_predictions(models[best_name],
                                      data["X_test"], data["test_loader"])
    fig_path = plot_residuals(data["y_test"], y_pred_best)
    print(f"Residual plot saved to {fig_path}")

    pred_path = export_predictions(data["id_test"], data["y_test"], y_pred_best)
    print(f"Predictions saved to {pred_path}")

    return {
        "comparison": comparison,
        "leakage": leakage,
        "predictions_path": pred_path,
    }


if __name__ == "__main__":
    data = prepare_data(feature_set="A", seed=SEED)
    models = fit_all_models(data, seed=SEED)

    assert "mlp" in models, "the MLP must be in the comparison"
    assert len(models) >= 3, "expected at least dummy/ridge/forest plus the mlp"

    table = evaluate_all(models, data["X_test"], data["y_test"],
                         data["test_loader"])
    print(table.round(3))

    assert list(table.columns) == ["rmse", "mae", "r2"]
    assert (table["rmse"] >= table["mae"]).all(), "RMSE below MAE is impossible"
    assert table["r2"].max() <= 1.0

    leak = run_leakage_experiment(model_kind="ridge", seed=SEED)
    print(leak.round(3))

    assert set(leak.index) == set(FEATURE_SETS)
    assert leak.loc["C", "r2"] >= leak.loc["A", "r2"] - 1e-9, \
        "set C should not score below set A"

    best_name = table.index[0]
    y_pred_best = collect_predictions(models[best_name],
                                      data["X_test"], data["test_loader"])
    path = plot_residuals(data["y_test"], y_pred_best)
    print(f"residual plot: {path}")

    export_predictions(data["id_test"], data["y_test"], y_pred_best,
                       path="predictions.csv")
    check = pd.read_csv("predictions.csv", sep=";", decimal=",")
    assert list(check.columns) == ["id", "burnout_true", "burnout_pred"]
    assert len(check) == len(data["y_test"])

    print("\ncompare.py ran fine.")