"""PyTorch neural network (MLP) for predicting the burnout score.

This file has the network itself, the training loop with early stopping, and a
helper to make predictions. It uses the DataLoaders from features.py without
changing them, so the neural network and the sklearn baselines are tested on
exactly the same data.

The network is kept small on purpose. With only about fifteen features and a
target between 0 and 10, a bigger network mostly just adds noise. The real
question here is not "how good can a neural network get" but "does it even beat
Ridge regression".

Early stopping watches the validation loss and keeps the best weights, not the
last ones. Without that, the model you end up with is the one from a few epochs
after it already started getting worse.

One thing about the target shapes this whole file: about a quarter of the
burnout scores are exactly zero, because the script that made the data cut off
negative values at zero. MSE can't handle that floor, so the network predicts
values above it and the residuals show a hard edge at zero.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from features import prepare_data, SEED
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BurnoutMLP(nn.Module):
    """Feed-forward regressor: n_features in, one continuous value out.

    hidden is a tuple of layer widths, so the depth of the network comes from
    its length rather than from hardcoded layers.

    The output layer carries no activation. A regression head must be able to
    emit any real number. A ReLU or sigmoid there would silently bound the
    predictions and make the model unable to fit part of the target range.
    """

    def __init__(self, n_features, hidden=(64, 32), p_drop=0.2):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return predictions of shape (batch, 1)."""
        return self.net(x)


def make_model(n_features, lr=1e-3, weight_decay=0.0, seed=SEED, hidden=(64, 32), p_drop=0.2):
    """Build the model, the loss function and the optimizer together.

    Returns (model, criterion, optimizer).

    Seeds torch before building the model so the same seed gives the same
    starting weights. Any difference between runs then comes from the change
    being tested, not from random initialisation.

    Build the optimizer from the returned model's parameters. If it points at a
    different model, it updates weights that are never used and training does
    nothing.
    """
    torch.manual_seed(seed)
    model = BurnoutMLP(n_features, hidden=hidden, p_drop=p_drop)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, criterion, optimizer


def train_one_epoch(model, loader, criterion, optimizer):
    """One full pass over the training data with gradient updates.

    Returns the mean training loss across the epoch as a float.

    The last batch is usually smaller than the rest. Averaging the per-batch
    losses directly would weight it the same as a full batch, so weight each
    loss by its batch size and divide by the total number of rows.
    """
    model.train()
    running, seen = 0.0, 0

    for xb,yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds,yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        seen += xb.size(0)
    return running / seen

@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    """One full pass without gradient updates. Returns the mean loss as a float.

    Switch off two separate things: model.eval() changes how dropout behaves,
    and torch.no_grad() stops the autograd graph being built. Only eval()
    affects the returned loss.

    Must leave the model in a state where train_one_epoch can resume correctly
    on the next call.
    """
    model.eval()
    running, seen = 0.0, 0

    for xb, yb in loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        running += loss.item() * xb.size(0)
        seen += xb.size(0)

    return running / seen
        


def fit(model, train_loader, val_loader, criterion, optimizer,
        max_epochs=100, patience=8, verbose=True):
    """Train until the validation loss stops improving, then restore the best weights.

    Stops after `patience` consecutive epochs without an improvement, or when
    max_epochs is reached — whichever comes first.

    Writes the best epoch's weights back into `model` before returning, so the
    caller gets the winning model, not the one from `patience` epochs later.
    Deep-copy the state dict when saving it — it shares storage with the live
    model, so a plain reference keeps changing as training continues.

    Returns a dict with keys:
        train_loss      list of floats, one per completed epoch
        val_loss        list of floats, same length
        best_epoch      int, zero-based index into those lists
        best_val_loss   float, the minimum of val_loss
    """
    best_loss = float("inf")
    best_state = None
    history = {"train": [], "val": [], "best_epoch": 0}
    wait = 0

    for e in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion,optimizer)
        history["train"].append(train_loss)
        val_loss = evaluate_loss(model, val_loader, criterion)
        history["val"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = e
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"stopped at epoch {e}")
                break

    model.load_state_dict(best_state)

    return {
        "train_loss": history["train"],
        "val_loss": history["val"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": min(history["val"])
    }

@torch.no_grad()
def predict(model, loader):
    """Run the model over a loader and return (y_true, y_pred).

    Both are 1-D NumPy arrays of the same length as the loader's dataset.

    Do not shuffle the loader. The pairs stay aligned regardless, but the row
    order would no longer match the source data, breaking any later join back
    onto the original rows in predictions.csv.
    """
    model.eval()
    trues,preds = [],[]
    for xb,yb in loader:
        trues.append(yb.cpu().numpy())
        preds.append(model(xb).cpu().numpy())

    return np.vstack(trues).ravel(),np.vstack(preds).ravel()
    


def run_model(feature_set="A", batch_size=64,
              max_epochs=60, seed=SEED):
    """Prepare data, build the model, fit it, and report validation metrics.

    Prints RMSE, MAE and R2 on the validation split and returns a dict with
    keys: model, history, val_loader, metrics.

    val_loader is part of the return value because the caller needs to run
    predict() on the same loader instance the model was validated against.
    Rebuilding it would risk a different split.

    Does not touch the test loader. That split is spent once, in compare.py.
    """
    data = prepare_data(feature_set=feature_set, batch_size=batch_size,
                        seed=seed)

    model, criterion, optimizer = make_model(data["n_features"], seed=seed)

    history = fit(model, data["train_loader"], data["val_loader"],
                  criterion, optimizer, max_epochs=max_epochs)

    y_true, y_pred = predict(model, data["val_loader"])

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    print(f"feature set {feature_set}  "
          f"RMSE {metrics['rmse']:.3f}  "
          f"MAE {metrics['mae']:.3f}  "
          f"R2 {metrics['r2']:.3f}")

    return {
        "model": model,
        "history": history,
        "val_loader": data["val_loader"],
        "metrics": metrics,
    }


if __name__ == "__main__":
    net = BurnoutMLP(n_features=12)
    dummy = torch.randn(7, 12)
    out = net(dummy)

    assert out.shape == (7, 1), f"expected (7, 1), got {tuple(out.shape)}"
    assert out.dtype == torch.float32
    print(f"forward pass ok: {tuple(dummy.shape)} -> {tuple(out.shape)}")

    result = run_model(feature_set="A", max_epochs=15)
    hist = result["history"]

    assert len(hist["train_loss"]) == len(hist["val_loss"])
    assert len(hist["val_loss"]) >= 1
    assert all(v > 0 for v in hist["val_loss"]), "MSE cannot be negative"
    assert hist["val_loss"][-1] < hist["val_loss"][0] * 1.5, "training diverged"
    assert hist["best_val_loss"] == min(hist["val_loss"])

    y_true, y_pred = predict(result["model"], result["val_loader"])

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 1
    assert not np.isnan(y_pred).any(), "NaN predictions"
    print(f"predicted {len(y_pred)} rows, "
          f"range {y_pred.min():.2f} to {y_pred.max():.2f}")

    below_zero = (y_pred < 0).mean()
    print(f"predictions below zero: {below_zero:.1%}")

    print("\nmodel.py ran fine.")