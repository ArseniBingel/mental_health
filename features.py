"""Feature preparation for the burnout regressor.

Selects one of three feature sets, splits the data, builds a ColumnTransformer
that scales numeric columns and one-hot encodes categorical ones, and returns
PyTorch DataLoaders plus the fitted transformer.

The three feature sets exist to make the leakage question measurable. Set A contains only lifestyle and academic variables — the
honest model. Set B adds the psychological scales, which are arguably part of
the same construct as burnout. Set C adds mental_health_index and dropout_risk,
which were derived propably from burnout in the data-generating script.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple
from pathlib import Path

PATH = Path(r"C:\Users\arsen\OneDrive\Desktop\DSI\Abschlussprojekt\student_mental_health_100k_sample.csv")
TARGET = "burnout_score"
FIG_DIR = Path("figures")
SEED = 42

LIFESTYLE = [
    "age", "gender", "academic_year", "study_hours_per_day", "exam_pressure",
    "academic_performance", "sleep_hours", "physical_activity",
    "social_support", "screen_time", "internet_usage", "financial_stress",
    "family_expectation",
]
PSYCH = ["stress_level", "anxiety_score", "depression_score"]
DOWNSTREAM = ["mental_health_index", "dropout_risk", "risk_level"]

FEATURE_SETS = {
    "A": LIFESTYLE,
    "B": LIFESTYLE + PSYCH,
    "C": LIFESTYLE + PSYCH + DOWNSTREAM,
}

def load_data(path=PATH):
    """Read the CSV and strip whitespace from column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def select_features(df, feature_set="A", target=TARGET):
    """Split the DataFrame into predictors X and target y for one feature set.

    feature_set is a key of FEATURE_SETS. X contains exactly the columns listed
    there, in that order; every other column is dropped, including the target.
    Returns (X, y) as a DataFrame and a Series.

    Raise a clear error if a listed column is missing from df
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"{feature_set} is not in FEATURE_SETS. Options: {list(FEATURE_SETS)}")

    cols = FEATURE_SETS[feature_set]
    
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"A feature is missing in the df columns: {missing}")
    if target not in df.columns:
        raise KeyError(f"The target {target} is missing in the df columns")

    X = df[cols].copy()
    y = df[target]

    return X,y


def zero_labels(y: pd.Series) -> np.ndarray:
    """Binary indicator: 1 where the target is exactly zero, else 0.

    About a quarter of the target values sit at exactly 0.0 because the
    data-generating script clipped negative values. This label is not used for
    training the regressor; it exists so the later hurdle model and the
    residual analysis can treat that group separately.

    Returns a NumPy array of int, same length as y.
    """
    return np.where(y == 0.0, 1, 0)

def detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Sort the predictor columns into numeric and categorical lists.

    Returns (numeric_cols, categorical_cols) as two lists of column names.
    """
    numeric_cols = X.select_dtypes(include="number").columns.to_list()
    categorical_cols = X.select_dtypes(exclude="number").columns.to_list()

    return numeric_cols,categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """ColumnTransformer: scale the numerics, one-hot encode the categoricals.

    Returns the transformer UNFITTED, so the caller decides what it sees.
    
    """
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(sparse_output=False,handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )

    return prep

def split_data(X, y, seed=SEED):
    """60/20/20 train/validation/test split via two train_test_split calls.

    No stratification: the target is continuous, so there are no classes to
    balance.

    Returns six objects in the order
    X_train, X_val, X_test, y_train, y_val, y_test.
    """

    X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.20, random_state=seed)
    val_size = 20/80 # 20% of the original, taken from the 80% remainder
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test

def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                 preprocessor: ColumnTransformer, batch_size=64, seed=SEED):
    """Fit the preprocessor on train, transform all three splits, wrap in loaders.

    Only the training loader is shuffled; validation and test stay in order so
    that repeated evaluations are comparable.

    Regression targets are float32 with shape (n, 1)

    Returns a dict with keys: train_loader, val_loader, test_loader,
    preprocessor, n_features. n_features comes from the TRANSFORMED array,
    because one-hot encoding changes the width.
    """
    g = torch.Generator().manual_seed(seed)

    X_train_s = preprocessor.fit_transform(X_train)
    X_val_s   = preprocessor.transform(X_val)
    X_test_s  = preprocessor.transform(X_test)

    n_features = X_train_s.shape[1]       

    Xtr = torch.tensor(X_train_s, dtype=torch.float32)
    Xva = torch.tensor(X_val_s,   dtype=torch.float32)
    Xte = torch.tensor(X_test_s,  dtype=torch.float32)

    ytr = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    yva = torch.tensor(y_val.to_numpy(),   dtype=torch.float32).unsqueeze(1)
    yte = torch.tensor(y_test.to_numpy(),  dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(Xtr, ytr)
    val_ds   = TensorDataset(Xva, yva)
    test_ds  = TensorDataset(Xte, yte)

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   drop_last=False, generator=g),
        "val_loader":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                   drop_last=False),
        "test_loader":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                   drop_last=False),
        "preprocessor": preprocessor,     
        "n_features":   n_features,
    }


def prepare_data(feature_set="A", batch_size=64, seed=SEED):
    """Run the whole preparation for one feature set and return the loaders dict.

    Also returns the raw y arrays under keys y_train, y_val, y_test — the
    sklearn baselines in baselines.py need them without the DataLoader wrapper.
    """
    df = load_data(PATH)
    X,y = select_features(df=df,feature_set=feature_set,target=TARGET)
    num_cols, cat_cols = detect_column_types(X=X)
    prep = build_preprocessor(numeric_cols=num_cols,categorical_cols=cat_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y, seed=seed)

    loaders_dict = make_loaders(X_train,X_val,X_test,y_train,y_val,y_test, preprocessor=prep,batch_size=batch_size, seed=seed)
    loaders_dict["y_train"] = y_train
    loaders_dict["y_val"] = y_val
    loaders_dict["y_test"] = y_test

    return loaders_dict

if __name__ == "__main__":
    data = prepare_data(feature_set="A")

    xb, yb = next(iter(data["train_loader"]))
    print(f"batch features: {tuple(xb.shape)}, targets: {tuple(yb.shape)}")
    print(f"features after encoding: {data['n_features']}")
    print(f"train {len(data['train_loader'].dataset)} | "
          f"val {len(data['val_loader'].dataset)} | "
          f"test {len(data['test_loader'].dataset)}")

    assert xb.dtype == torch.float32 and yb.dtype == torch.float32
    assert xb.shape[1] == data["n_features"]
    assert yb.ndim == 2 and yb.shape[1] == 1
    assert not torch.isnan(xb).any(), "NaNs in the features."

    for name in ["A", "B", "C"]:
        d = prepare_data(feature_set=name)      
        print(f"set {name}: {d['n_features']} features after encoding")

    print("\nfeatures.py ran fine.")