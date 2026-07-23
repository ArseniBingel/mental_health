import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from features import PATH, TARGET, load_data

FIG_DIR = Path("figures")

sns.set_theme(style="whitegrid")


def summarize(df):
    """Print shape, dtypes, missing-value counts and numeric summary stats.
    """
    print(f"shape: {df.shape}")
    print("\ndtypes / non-null counts:")
    df.info()                                    # prints directly, returns None
    print("\nmissing values:\n", df.isna().sum())
    print("\nnumeric summary:\n", df.describe())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    print("\nnumeric columns:    ", num_cols)
    print("categorical columns:", cat_cols)

    return num_cols, cat_cols


def plot_target_distribution(df, target=TARGET, outdir=FIG_DIR):
    """Histogram of the target with a KDE overlay.

    Worth looking at before choosing a loss: a roughly symmetric target suits
    MSE, while a heavily skewed one may need a log transform or L1 loss.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(df, x=target, bins=30, kde=True)
    plt.title(f"Distribution of {target}")
    plt.tight_layout()
    plt.savefig(outdir / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df, outdir=FIG_DIR):
    """Heatmap of pairwise correlations between all numeric columns.

    Two things to look for: which features relate to the target, and which
    features are strongly correlated with *each other* (redundant predictors).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num = df.select_dtypes(include="number")

    plt.figure(figsize=(15, 12))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation between numeric features")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_numeric.png", dpi=150, bbox_inches="tight")
    plt.close()


def top_correlations(df, target=TARGET, n=6):
    """The n features most strongly correlated with the target.

    Ranks by absolute correlation, so strong negative relationships (more sleep,
    less burnout) rank alongside positive ones, but returns the SIGNED values so
    the direction stays visible. The target's self-correlation of 1.0 is dropped
    first, otherwise it would always take the top slot.

    Returns a pandas Series indexed by column name.
    """
    num = df.select_dtypes(include="number")
    corr_target = num.corr()[target].drop(target)
    ranked = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
    return ranked[:n]


def plot_top_features(df, target=TARGET, n=4, outdir=FIG_DIR, sample=5000):
    """Scatter plot with trend line for each of the top n correlated features.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feats = top_correlations(df, target, n).index
    plot_df = df.sample(min(sample, len(df)), random_state=0)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, c in zip(axes, feats):
        sns.regplot(data=plot_df, x=c, y=target, ax=ax, ci=None,
                    scatter_kws={"alpha": 0.3, "s": 10},
                    line_kws={"color": "red"})
        ax.set_title(f"{c} vs {target}")

    plt.tight_layout()
    plt.savefig(outdir / "top_features.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    summarize(df)

    assert TARGET in df.columns, \
        f"'{TARGET}' not found. Available columns: {list(df.columns)}"

    print("\nStrongest correlations with the target:")
    print(top_correlations(df).round(3))

    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_features(df)

    print(f"\nDone. {len(df)} rows, {df.shape[1]} columns.")
    print(f"Figures saved to {FIG_DIR}/")